import os
import numpy as np
import torch
from torch import nn
from torch.nn.functional import softmax

from opts import parse_opts
from core.utils import get_mean, get_std
from core.spatial_transforms import (
    Compose, Scale, ToTensor,
    MultiScaleRandomCrop, MultiScaleCornerCrop, DriverFocusCrop
)
from core.temporal_transforms import UniformIntervalCrop
from core.utils import Logger, set_random_seed
from dataset import get_validation_set
from CaTFormer import CaTFormer

def move_to_cuda(x):
    """Recursively move all torch.Tensor in x to GPU."""
    if torch.is_tensor(x):
        return x.cuda(non_blocking=True)
    elif isinstance(x, list):
        return [move_to_cuda(v) for v in x]
    elif isinstance(x, tuple):
        return tuple(move_to_cuda(v) for v in x)
    elif isinstance(x, dict):
        return {k: move_to_cuda(v) for k, v in x.items()}
    else:
        return x

def distance(x1, x2):
    return np.linalg.norm(x1 - x2)

def prediction_test(model_path, val_loader):
    M = 5
    model = torch.load(model_path)
    
    model = nn.DataParallel(model)
    model = model.cuda()
    model.eval()

    TP_num     = np.zeros(M)
    N_num      = np.zeros(M)
    P_num      = np.zeros(M)
    cfu_matrix = np.zeros((M, M))
    Loss       = 0.0
    TP         = 0
    Num        = 0

    for batch in val_loader:
        raw_data, targets = batch[0], batch[1]
        test_data = move_to_cuda(raw_data)
        targets   = move_to_cuda(targets)

        res_concat, res_in, res_out, output, intent_log = model(test_data, targets)

        # detach before numpy
        p_in  = softmax(res_in,   dim=1).cpu().detach().numpy()
        p_out = softmax(res_out,  dim=1).cpu().detach().numpy()
        p_con = softmax(res_concat, dim=1).cpu().detach().numpy()
        Loss += distance(p_in, p_out) + distance(p_in, p_con) + distance(p_con, p_out)

        pred_tensor = softmax(output, dim=1)
        _, idx      = torch.max(pred_tensor, dim=1)
        pred_label  = idx.cpu().detach().numpy()[-1]
        true_label  = targets.cpu().detach().numpy()[-1]

        if true_label != 0:
            N_num[true_label] += 1
            if pred_label == true_label:
                TP_num[true_label] += 1
        if pred_label != 0:
            P_num[pred_label] += 1

        cfu_matrix[true_label, pred_label] += 1
        if pred_label == true_label:
            TP += 1
        Num += 1

    Pr  = np.mean(TP_num[1:] / P_num[1:])
    Re  = np.mean(TP_num[1:] / N_num[1:])
    Acc = TP / Num
    Loss /= Num

    for i in range(M):
        if cfu_matrix[i].sum() > 0:
            cfu_matrix[i] /= cfu_matrix[i].sum()

    return Pr, Re, 0.0, cfu_matrix, Acc, Loss

def prediction_total(epoch, model_dir, loader_list):
    pr_list, re_list, acc_list, loss_list, cm_list = [], [], [], [], []

    for loader in loader_list:
        ckpt_path = os.path.join(model_dir, str(epoch))
        pr, re, _, cm, acc, loss = prediction_test(ckpt_path, loader)
        pr_list.append(pr), re_list.append(re)
        acc_list.append(acc), loss_list.append(loss)
        cm_list.append(cm)

    Pr   = np.mean(pr_list)
    Re   = np.mean(re_list)
    F1   = 2 * Pr * Re / (Pr + Re) if (Pr + Re) > 0 else 0.0
    Acc  = np.mean(acc_list)
    Loss = np.mean(loss_list)
    cm   = np.mean(cm_list, axis=0)

    print(f'Pr per fold: {pr_list}')
    print(f'Re per fold: {re_list}')
    print(f'Loss per fold: {loss_list}')
    print(f' Pr: {Pr:.3f}  Re: {Re:.3f}  F1: {F1:.3f}  Acc: {Acc:.3f}  loss: {Loss:.3f}')
    return F1, Acc, cm

if __name__ == '__main__':
    opt = parse_opts()
    set_random_seed(opt.rand_seed)

    num_gpus    = torch.cuda.device_count()
    num_workers = num_gpus * 7
    batch_size  = max(1, num_gpus)

    if opt.root_path:
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
    threshold = 0.25 + 0.05 * opt.threshold
    opt.scales = [opt.initial_scale * (opt.scale_step ** i) for i in range(opt.n_scales)]
    opt.mean   = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std    = get_std(opt.norm_value)

    loader_list = []
    for i in range(5):
        opt.n_fold = i
        if opt.train_crop == 'random':
            crop = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop = MultiScaleCornerCrop(opt.scales, opt.sample_size, crop_positions=['c'])
        else:
            crop = DriverFocusCrop(opt.scales, opt.sample_size)

        in_tf   = Compose([Scale((112,112)), ToTensor(opt.norm_value)])
        out_tf  = Compose([Scale((144,96)),   ToTensor(opt.norm_value)])
        tmp_tf  = UniformIntervalCrop(opt.sample_duration, opt.interval)

        val_set    = get_validation_set(opt, in_tf, out_tf, tmp_tf, None)
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        loader_list.append(val_loader)

    os.makedirs(f'./logger/{opt.end_second}', exist_ok=True)
    logger = Logger(f'./logger/{opt.end_second}/{threshold}.log', ['best_epoch','acc','F1','cfu'])

    best_f1, best_acc, best_ckpt, best_cm = 0.0, 0.0, None, None
    for ckpt in sorted(os.listdir(opt.resume_path)):
        print('Evaluating checkpoint:', ckpt)
        f1, acc, cm = prediction_total(ckpt, opt.resume_path, loader_list)
        if f1 > best_f1:
            best_f1, best_acc, best_ckpt, best_cm = f1, acc, ckpt, cm

    logger.log({'best_epoch': best_ckpt, 'acc': best_acc, 'F1': best_f1, 'cfu': best_cm})
    print(f'Best → epoch: {best_ckpt}, F1: {best_f1:.3f}, Acc: {best_acc:.3f}')
    print(best_cm)
