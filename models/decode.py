import torch
from torch import nn
import numpy as np
from .net import BasicBlock
from misc.utils import initialize_weights
from misc.images import convert_labels

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def heatmap_decode(hm, threshold=0.3):
    batch, height, width = hm.size()
    hm = _nms(hm)

    hm = (hm > threshold).float()
    return torch.sum(hm, (1, 2)).detach()

def heatmap_decode_hm(hm, threshold=0.3):
    batch, height, width = hm.size()
    hm = _nms(hm)

    hm = (hm > threshold).float()
    return torch.sum(hm, (1, 2)).detach(), hm

def heatmap_decode_location(original_imgs, hm, threshold=0.3):
    batch, height, width = hm.size()
    hm = _nms(hm)
    hm = hm.cpu().detach()

    hm = hm * (hm > threshold).float()
    locations = convert_labels(original_imgs, hm[0], hm[0].nonzero().numpy().astype(float), height=height, width=width)

    return locations

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.c1 =  nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1)
        self.base1 = BasicBlock(128, 128)
        self.maxpool = nn.MaxPool2d(3, stride=1, padding=1)
        self.base2 = BasicBlock(128, 128)
        self.fc = nn.Sequential(nn.Linear(128, 128),
                                nn.LeakyReLU(0.01),
                                nn.Linear(128, 1),
                                nn.ReLU())
        initialize_weights(self.modules())

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.c1(x)
        x = self.base1(x)
        x = self.maxpool(x)
        x = self.base2(x)
        x = torch.sum(x, [2, 3])
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x.squeeze(1)
    

