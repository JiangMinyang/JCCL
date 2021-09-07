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

class DemoLoader(data.Dataset):
    def __init__(self, data_path):
        self.img_path = data_path + '/img'
        self.label_path = data_path + '/label'
        self.data_files = [filename for filename in os.listdir(self.img_path) \
                           if os.path.isfile(os.path.join(self.img_path,filename))]
        self.num_samples = len(self.data_files) 
    
    def __getitem__(self, index):
        fname = self.data_files[index]
        img = cv2.imread(osp.join(self.img_path, fname))
        label = np.loadtxt(osp.join(self.label_path, osp.splitext(fname)[0] + '.txt'), dtype=np.float32).reshape(-1, 2)

        return {'image': img, 'fname': fname, 'num_people': len(label)}

    def __len__(self):
        return self.num_samples

    def get_num_samples(self):
        return self.num_samples       