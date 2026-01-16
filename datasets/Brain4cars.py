import torch
import torch.utils.data as data
from PIL import Image
import os
import cv2
import math
import functools
import json
import copy
from os.path import *
import numpy as np
import random
from glob import glob
import csv
import time
import pandas as pd
from utils import load_value_file

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pdb

speed_max = 120
speed_min = 0
label_hflip ={'0':0,'1':3,'2':4,'3':1,'4':2}

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def cv2_loader(path):
    return cv2.imread(path)

def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, source, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, '{:d}.jpg'.format(i+source))   #source判断车内/外，车内为1，车外为0
        if os.path.exists(image_path) and len(video)<15:
            video.append(image_loader(image_path))
        else:
            return video
    return video

def feature_normalize(dt,feature_min,feature_max):
    return (dt-feature_min)/(feature_max-feature_min)

def txt_loader(txt_path,frame_indices):
    road_total = []
    car_state = []
    with open(txt_path,'r') as f:
#         pdb.set_trace()
        for roadinfo in f:
            speed = float(roadinfo.split(',',1)[0])
            if speed ==-1:                       #如果是未记录的车速，则赋值为12  可修改
                speed=12
            speed = [feature_normalize(speed,speed_min,speed_max)]
            lane = roadinfo.split(',',1)[1]
            laneinfo_list = lane.split(',')
            laneinfo = []
            laneinfo.append(min(2,int(laneinfo_list[0]))-1)  #右车道是否存在
            laneinfo.append(min(1,int(laneinfo_list[1])-int(laneinfo_list[0])))  #左车道是否存在
            laneinfo.append(int(laneinfo_list[2][0]))
            road_total.append(speed+laneinfo)
            #road_total.append(laneinfo)
    for i in frame_indices:
        car_state.append(road_total[i-1])
    car_state = np.array(car_state)
    car_state = torch.FloatTensor(car_state)
    return car_state

def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=cv2_loader)  #修改图片加载

def load_annotation_data(data_file_path, fold):
    database = {}
    data_file_path = os.path.join(data_file_path, 'fold%d.csv'%fold)       
    print('Load from %s'%data_file_path)
    with open(data_file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            value = {}
            value['subset'] = row[3]
            value['label'] = row[1]
            value['n_frames'] = int(row[2])
            database[row[0]] = value
    return database

def get_class_labels():
#### define the labels map
    class_labels_map = {}
    class_labels_map['end_action'] = 0  #修改
    class_labels_map['lchange'] = 1
    class_labels_map['lturn'] = 2
    class_labels_map['rchange'] = 3
    class_labels_map['rturn'] = 4
    return class_labels_map

def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data.items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['label']
            video_names.append(key[:-1])    ### key = 'rturn/20141220_154451_747_897'
            annotations.append(value)

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video, end_second,
                 sample_duration, fold):

    data = load_annotation_data(annotation_path, fold)
    root_outvideo_path = '/root/autodl-tmp/sirui/TIFN/brain4cars_data/road_camera/flow/'   #车外光流根路径
    root_invideo_path = '/root/autodl-tmp/sirui/TIFN/brain4cars_data/face_camera/'           #车内图像根路径
    root_oriout_path = '/root/autodl-tmp/sirui/TIFN/brain4cars_data/face_camera/'                #brain4cars中的人为车外信息：车道及岔路口信息
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels()
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name  # 对label进行编码 {0: 'end_action', 1: 'lchange', 2: 'lturn', 3: 'rchange', 4: 'rturn'}
    dataset = []
    for i in range(len(video_names)):
        if i % 100 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))
        video_name = video_names[i]
#       outvideo_path = os.path.join(root_outvideo_path, video_name)+'.avi' #光流文件夹处理之后并没有带“.avi”的后缀名
        outvideo_path = os.path.join(root_outvideo_path, video_name)
        invideo_path = os.path.join(root_invideo_path, video_name,"img")
        #vl = os.path.splitext(video_names[i])
        #pdb.set_trace()
        outtxt_path = root_oriout_path+video_name+'/car_state.txt'
        
        if not os.path.exists(outvideo_path):
            print('File does not exists: %s'%outvideo_path)
            continue
        if not os.path.exists(invideo_path):
            print('File does not exists: %s'%invideo_path)
            continue

#        n_frames = annotations[i]['n_frames']
        # count in the dir
        l = os.listdir(outvideo_path)

        numdif = len(os.listdir(invideo_path))-len(l)-1  #判断车内外图片数量是否对齐
        if numdif < 0:
            print('Video length is different: %s'%video_name)
        n_frames = 0
        # If there are other files (e.g. original videos) besides the images in the folder, please abstract.
        if subset == 'validation':   #训练仍使用完整图像，测试时去掉驾驶行为发生前end_second秒
            n_frames = len(l)-max((end_second*25-numdif),0)
        elif subset == 'training':
            n_frames = len(l)

        #if n_frames < sample_duration+1 :#30*end_second                        #帧率是25还是30?
        if n_frames <= 16 :#30*end_second
            print('Video is too short: %s'%video_name)
            continue

        begin_t = 1
        end_t = n_frames
        sample = {
            'invideo': invideo_path,
            'outvideo': outvideo_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'txt_path':outtxt_path,
            'video_name': video_name
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            sample['hflip'] = False
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                for j in range(0, n_samples_for_each_video):
                    sample['frame_indices'] = list(range(1, n_frames+1))        #frame_indices完全相同是为什么
                    if j%2 == 0:
                        sample['hflip'] = True
                    else:
                        sample['hflip'] = False
                    sample_j = copy.deepcopy(sample)
                    dataset.append(sample_j)
#     pdb.set_trace()
    return dataset, idx_to_class


class Brain4cars_Inside(data.Dataset):
    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 nfold,
                 end_second,
                 n_samples_for_each_video=1,
                 spatial_transform_invideo=None,
                 spatial_transform_outvideo=None,
                 horizontal_flip=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=5,
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            end_second, sample_duration, nfold)
        self.subset = subset
        self.spatial_transform_invideo = spatial_transform_invideo
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.horizontal_flip = horizontal_flip
        self.loader = get_loader()
        self.txt_loader = txt_loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is an image.
        """
        outpath = self.data[index]['outvideo']
        inpath = self.data[index]['invideo']
        label = self.data[index]['label']
        txt_path = self.data[index]['txt_path']
        frame_indices = self.data[index]['frame_indices']
        vid = self.data[index]['video_name']
        h_flip = False

        if self.temporal_transform is not None:
            frame_indices,target_idc = self.temporal_transform(frame_indices)
        #print(frame_indices)
        inclip = self.loader(inpath, frame_indices, 1)
        car_state = self.txt_loader(txt_path,frame_indices)
        #if self.horizontal_flip is not None:
        #    p = random.random()
        #    if p < 0.5:
        #        clip = [self.horizontal_flip(img) for img in clip]
        #        target = [self.horizontal_flip(img) for img in target]
        if self.spatial_transform_invideo is not None:
            self.spatial_transform.randomize_parameters()
            inclip = [self.spatial_transform_invideo(img) for img in inclip]

        inclip = torch.stack(inclip, 0)
        outclip = None
        #if self.target_transform is not None:
        #    target = [self.target_transform(img) for img in target]
        #target = torch.stack(target, 0).permute(1, 0, 2, 3).squeeze()
        car_state = torch.FloatTensor(car_state)
        train_data = [inclip, outclip, car_state]

        if self.subset=='validation':
            return train_data, label, vid
        else:
            return train_data, label
    def __len__(self):
        return len(self.data)

class Brain4cars_Outside(data.Dataset):
    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 nfold,
                 end_second,
                 n_samples_for_each_video=1,
                 spatial_transform_invideo=None,
                 spatial_transform_outvideo=None,
                 horizontal_flip=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=5,
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            end_second, sample_duration, nfold)
        self.subset = subset
        self.spatial_transform_outvideo = spatial_transform_outvideo
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.horizontal_flip = horizontal_flip
        self.loader = get_loader()
        self.txt_loader = txt_loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is an image.
        """
        
        outpath = self.data[index]['outvideo']
        inpath = self.data[index]['invideo']
        label = self.data[index]['label']
        txt_path = self.data[index]['txt_path']
        frame_indices = self.data[index]['frame_indices']
        vid = self.data[index]['video_name']
        h_flip = False

        if self.temporal_transform is not None:
            frame_indices,target_idc = self.temporal_transform(frame_indices)
        #print(frame_indices)
        outclip = self.loader(outpath, frame_indices, 0)
        car_state = self.txt_loader(txt_path,frame_indices)
        #if self.horizontal_flip is not None:
        #    p = random.random()
        #    if p < 0.5:
        #        clip = [self.horizontal_flip(img) for img in clip]
        #        target = [self.horizontal_flip(img) for img in target]
        if self.spatial_transform_outvideo is not None:
            self.spatial_transform.randomize_parameters()
            outclip = [self.spatial_transform_outvideo(img) for img in outclip]

        inclip = None
        outclip = torch.stack(outclip, 0)
        print(outclip.shape)

        #if self.target_transform is not None:
        #    target = [self.target_transform(img) for img in target]
        #target = torch.stack(target, 0).permute(1, 0, 2, 3).squeeze()
        car_state = torch.FloatTensor(car_state)
        train_data = [inclip, outclip, car_state]

        if self.subset=='validation':
            return train_data, label, vid
        else:
            return train_data, label
    def __len__(self):
        return len(self.data)

class Brain4cars_Unit(data.Dataset):
    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 nfold,
                 end_second,
                 n_samples_for_each_video=1,
                 spatial_transform_invideo=None,
                 spatial_transform_outvideo=None,
                 horizontal_flip=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=5,
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            end_second, sample_duration, nfold)
        self.subset = subset
        self.spatial_transform_invideo = spatial_transform_invideo
        self.spatial_transform_outvideo = spatial_transform_outvideo
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.horizontal_flip = horizontal_flip
        self.loader = get_loader()
        self.txt_loader = txt_loader


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is an image.
        """
#         pdb.set_trace()
        
        outpath = self.data[index]['outvideo']
        inpath = self.data[index]['invideo']
        label = self.data[index]['label']
        txt_path = self.data[index]['txt_path']
        frame_indices = self.data[index]['frame_indices']
        vid = self.data[index]['video_name']
        h_flip = self.data[index]['hflip']
        if self.temporal_transform is not None:
            frame_indices,target_idc = self.temporal_transform(frame_indices)
        #print(frame_indices)
        outclip = self.loader(outpath, frame_indices, 0)
        inclip = self.loader(inpath, frame_indices, 1)
        car_state = self.txt_loader(txt_path,frame_indices)
        if self.horizontal_flip is not None and h_flip:
            inclip = [self.horizontal_flip(img) for img in inclip]
            outclip = [self.horizontal_flip(img) for img in outclip]
            label = label_hflip[str(label)]
        if self.spatial_transform_invideo is not None and self.spatial_transform_outvideo is not None:
            #self.spatial_transform_invideo.randomize_parameters()
            #self.spatial_transform_outvideo.randomize_parameters()
            inclip = [self.spatial_transform_invideo(img) for img in inclip]
            outclip = [self.spatial_transform_outvideo(img) for img in outclip]

        inclip = torch.stack(inclip, 0)
        outclip = torch.stack(outclip, 0)
        #if self.target_transform is not None:
        #    target = [self.target_transform(img) for img in target]
        #target = torch.stack(target, 0).permute(1, 0, 2, 3).squeeze()
        car_state = torch.FloatTensor(car_state)
       #train_data = [inclip, outclip, car_state]
        
        train_data = [inclip, outclip, car_state,outpath]
            
        
        if self.subset=='validation':
                return train_data, label, vid,frame_indices  #frame_indices在按照brain4cars进行提前预测时间测试时使用
        else:
            return train_data, label
    def __len__(self):
        return len(self.data)