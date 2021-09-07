import numpy as np
import os
import os.path as osp
import random
from scipy import io as sio
import sys
import torch
from torch.utils import data
from PIL import Image, ImageOps
import cv2

from draw_gaussian import draw_gaussian
from models.decode import heatmap_decode
import misc.utils as utils
import pandas as pd

import math
from scipy import spatial

# from config import cfg

class CCLoader(data.Dataset):
    def __init__(self, data_path, mode, main_transform=None, img_transform=None, downscale=4, default_dis=150):
        self.img_path = data_path + '/img'
        self.gt_path = data_path + '/den'
        self.label_path = data_path + '/label'
        self.data_files = [filename for filename in os.listdir(self.img_path) \
                           if os.path.isfile(os.path.join(self.img_path,filename))]
        self.num_samples = len(self.data_files) 
        self.main_transform=main_transform  
        self.img_transform = img_transform
        self.downscale = downscale
        self.mode = mode
        self.default_dis = default_dis
    
    def __getitem__(self, index):
        fname = self.data_files[index]
        img, label, num_people, original_img = self.get_data(fname)      

        img = img.copy()
        if self.img_transform is not None:
            img = self.img_transform(img)         

        if (self.mode == 'test'):
            return {'image': img, 'label': label, 'fname': fname, 'original_img': original_img, 'num_people': num_people}
        else:
            return img, label


    def __len__(self):
        return self.num_samples

    def get_data(self, fname):
        original_img, label = self.read_image_and_gt(fname)
        if self.main_transform is not None:
            img, label = self.main_transform(original_img, label) 

        height, width, _ = img.shape
        if (len(label) > 0):
            label[:, 0] /= self.downscale
            label[:, 1] /= self.downscale

        out_height = height // self.downscale
        out_width = width // self.downscale
        hm = np.zeros((out_height, out_width), dtype=np.float32)
        hm_reg = np.zeros((out_height, out_width), dtype=np.float32)

        positions = label.astype(int)
        num_people = len(positions)

        if (num_people > 0):
            kd_tree = spatial.KDTree(positions)
            for pos in positions:
                distances, _ = kd_tree.query(pos, k=5)
                for i in range(1, 5):
                    if (math.isinf(distances[i])):
                        distances[i] = self.default_dis
                r = min(max(3, math.floor(np.mean(distances[1:4]))), self.default_dis)
                if (r % 2 == 0):
                    r += 1
                draw_gaussian(hm, pos, 101, r / 5, mode='max')
        hm = torch.from_numpy(hm)
        return img, hm, num_people, original_img


    def read_image_and_gt(self,fname):
        img = cv2.imread(osp.join(self.img_path, fname))
        label = np.loadtxt(osp.join(self.label_path, osp.splitext(fname)[0] + '.txt'), dtype=np.float32).reshape(-1, 2)

        return img, label

    def get_num_samples(self):
        return self.num_samples       
            
        

