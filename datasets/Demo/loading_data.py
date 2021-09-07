import torchvision.transforms as torch_transforms
from torch.utils.data import DataLoader
import misc.transforms as transforms
from datasets.Demo.demo_dataloader import DemoLoader
import os.path as osp
import torch
from datasets.Demo.dataset_config import config

def loading_data():
    data_path = config.data_path

    demo_set = DemoLoader(osp.join(data_path))
    demo_loader = DataLoader(demo_set, batch_size=1, num_workers=config.dataloader_worker, shuffle=False, drop_last=False)

    return demo_loader