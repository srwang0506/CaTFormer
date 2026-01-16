import os
import numpy as np
import scipy.io as scio

speed_min = 200
speed_max = 0
root_path = '/root/autodl-tmp/coding/CaTFormer/brain4cars_data/face_camera'
for action in os.listdir(root_path):
    action_dir = os.path.join(root_path,action)
    for video in os.listdir(action_dir):
        video_dir = os.path.join(action_dir,video)
        file_path = video_dir+'/params_'+video+'.mat'
        save_path = video_dir+'/car_state.txt'
        data = scio.loadmat(file_path)
        start_index = data['params'][0,0][1][0,0] - 1 
        end_index = data['params'][0,0][2][0,0]
        with open(save_path,'w') as f:
            for i in range(start_index,end_index):
                lane = data['params'][0,0][3][0]
                speed = data['params'][0,0][4][0,i][0,0][0][0,0]
                if speed>speed_max:
                    speed_max = speed
                if speed<speed_min and speed!=-1 and speed!=0:
                    speed_min = speed
                f.write(str(speed)+','+lane+'\n')
print(speed_max)
print(speed_min)

#data['params'][0,0][3][0]为车道线及岔路口信息,为str '1,2,0'
