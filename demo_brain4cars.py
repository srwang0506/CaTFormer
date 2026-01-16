import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

from tqdm import tqdm

DEVICE = 'cuda'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(torch.cuda.current_device())
def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo,imfile):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    img_path = imfile.split("/img/")[0]+'/flow/'+imfile.split('/img/')[1]
    cv2.imwrite(img_path,flo[:, :, [2,1,0]])

def demo(model,path):

    with torch.no_grad():
        images = []
        images_list = os.listdir(path)

        images_list.sort(key = lambda x: int(x[:-4]))
        for img in images_list:
            images.append(os.path.join(path,img))
        flow_path = path.split("/img/")[0]+'/flow/'+path.split('/img/')[1]
        if not os.path.exists(flow_path):
            os.makedirs(flow_path)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, flow_up,imfile1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    
    dataset_path='/project/CaTFormer/brain4cars_data/road_camera/img'
    for action in os.listdir(dataset_path):
        print("Currently working on the optical flow data for {}".format(action))
        action_path=os.path.join(dataset_path,action)
        for video_id in tqdm(os.listdir(action_path)):
            video_path=os.path.join(action_path,video_id)
            print(video_path)
            demo(model,video_path)


