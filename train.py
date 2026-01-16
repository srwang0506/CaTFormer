import os
import sys
import time
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from tqdm import tqdm
from math import log10
from opts import parse_opts
from core.utils import get_mean, get_std
from core.spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor, DriverFocusCrop, DriverCenterCrop)
from core.temporal_transforms import LoopPadding, TemporalRandomCrop, TemporalCenterCrop, UniformIntervalCrop, UniformPadSample, RandomIntervalCrop
from core.target_transforms import ClassLabel, VideoID
from core.target_transforms import Compose as TargetCompose
from core.utils import AverageMeter, Logger, set_random_seed
from dataset import get_training_set, get_validation_set
from CaTFormer import CaTFormer
from tensorboardX import SummaryWriter

intent_weight = 0.1

if __name__ == '__main__':

    opt = parse_opts()
    set_random_seed(opt.rand_seed)
    if opt.root_path != '':
        #opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        opt.result_path = os.path.join(opt.result_path, 'fold' + str(opt.n_fold))
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = 'flowlstm'
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    in_sample_size = (112, 112)
    out_sample_size = (144, 96)

    torch.manual_seed(opt.manual_seed)
#=======================================================
    model = CaTFormer(
        feature_dim=32,
        nclass=5,
        hidden_dim=32,
        batch_size=opt.batch_size,
        outnet_name='resnet18',
        innet_name='mobilefacenet'
    ).cuda()
    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).cuda()

    writer = SummaryWriter()
    criterion0 = nn.CrossEntropyLoss()
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    criterionj = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion0 = criterion0.cuda()
        criterion1 = criterion1.cuda()
        criterion2 = criterion2.cuda()
        criterionj = criterionj.cuda()

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center', 'driver focus']
    if opt.train_crop == 'random':
        crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == 'corner':
        crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == 'center':
        crop_method = MultiScaleCornerCrop(
            opt.scales, opt.sample_size, crop_positions=['c'])
    elif opt.train_crop == 'driver focus':
        crop_method = DriverFocusCrop(opt.scales, opt.sample_size)

    train_spatial_transform_invideo = Compose([
        Scale(in_sample_size),
        ToTensor(opt.norm_value),
    ])
    train_spatial_transform_outvideo = Compose([
        Scale(out_sample_size),
        ToTensor(opt.norm_value),
    ])

    train_temporal_transform = RandomIntervalCrop(opt.sample_duration, opt.interval)
    train_target_transform = Compose([
        Scale(opt.sample_size),
        ToTensor(opt.norm_value)
    ])
    train_horizontal_flip = RandomHorizontalFlip()

    training_data = get_training_set(
        opt,
        train_spatial_transform_invideo,
        train_spatial_transform_outvideo,
        train_horizontal_flip,
        train_temporal_transform,
        train_target_transform
    )

    cpu_per_gpu = 7
    total_workers = torch.cuda.device_count() * cpu_per_gpu

    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=total_workers,
        pin_memory=True
    )

    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening

    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_step, gamma=0.1)

    if not opt.no_val:
        val_spatial_transform = Compose([
            Scale(opt.sample_size),
            ToTensor(opt.norm_value)
        ])
        val_temporal_transform = UniformIntervalCrop(opt.sample_duration, opt.interval)
        val_target_transform = val_spatial_transform
        validation_data = get_validation_set(
            opt,
            val_spatial_transform,
            val_temporal_transform,
            val_target_transform
        )
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=1,
            shuffle=True,
            num_workers=total_workers,
            pin_memory=True
        )

    if opt.train_from_scratch:
        print('loading checkpoint {}'.format(opt.resume_path))
        model = torch.load(opt.resume_path)
        opt.begin_epoch = 103

    print('run')
    global best_loss
    best_loss = torch.tensor(float('inf'))

    total_start = time.time()

    epoch_progress = tqdm(
        range(opt.begin_epoch, opt.n_epochs + 1),
        desc="Training",
        unit="epoch",
        position=0
    )

    for epoch in epoch_progress:
        if not opt.no_train:
            print(f"train at epoch {epoch}")

            model.train()

            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            total_loss = 0

            batch_progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch}",
                leave=False,
                unit="batch"
            )

            begin = time.time()
            for i, data in enumerate(batch_progress):
                train_data = data[0]
                targets = data[1]
                if not opt.no_cuda:
                    targets = targets.cuda(non_blocking=True)

                output, inres, outres, outjoint, intent_log = model(train_data, targets)
                loss0 = criterion0(output,    targets)
                loss1 = criterion1(inres,     targets)
                loss2 = criterion2(outres,    targets)
                lossj = criterionj(outjoint,  targets)
                loss_main = 0.25 * (loss0 + loss1 + loss2 + lossj)
                loss_intent = criterionj(intent_log, targets) * intent_weight
                loss = loss_main + loss_intent


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

            avg_epoch_loss = total_loss / len(train_loader)
            epoch_progress.set_postfix(avg_epoch_loss=avg_epoch_loss)
            scheduler.step()

            if not os.path.exists(opt.result_path):
                os.makedirs(opt.result_path)
            if epoch % opt.checkpoint == 0:
                save_file_path = os.path.join(opt.result_path, f'flstm-save_{epoch}.pkl')
                torch.save(model, save_file_path)

        print('total_loss:  ', total_loss)
        print('epoch_time:  ', time.time() - begin)

    total_run_time = time.time() - total_start
    print('Total training time: ', total_run_time)

    writer.export_scalars_to_json("./train.json")
    writer.close()
