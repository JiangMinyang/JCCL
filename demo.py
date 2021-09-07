import sys
sys.path.append('datasets')

import torch
import os
import os.path as osp
import cv2
import numpy as np
from datetime import datetime
from models.net import Net
from models.decode import Decoder, heatmap_decode, heatmap_decode_hm, heatmap_decode_location

import misc.utils as utils
import torchvision.transforms as torch_transforms
import misc.transforms as transforms
from scipy import signal
from config import config

config.set_scope('Test')
dataset = config.get('dataset')

from datasets.Demo.loading_data import loading_data
from datasets.Demo.dataset_config import config as dataset_config


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

demo_loader = loading_data()

exp_name = config.get('exp_name')
model_name = config.get('model_name')
model_path = osp.join('./exp', exp_name, model_name)
output_dir = './output/'

model_name = os.path.basename(model_path).split('.')[0]
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_dir = os.path.join(output_dir, exp_name)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

localization_output_dir = os.path.join(output_dir, 'loc')
if not os.path.exists(localization_output_dir):
    os.mkdir(localization_output_dir)

model_state_dict = torch.load(model_path)

net = Net()
net.load_state_dict(model_state_dict['net'], strict=False)
net.to(device)
net.eval()

mae = 0.0
mse = 0.0

decoder = None
if (config.getbool('use_decoder')):
    decoder = Decoder()
    decoder.load_state_dict(model_state_dict['decoder'], strict=False)
    decoder.to(device)
    decoder.eval()

def cc_count(model, image, depth, pred_map, decoder=None):
    h, w, _ = image.shape
    if h < w:
        scale_transform = transforms.Scale(height=dataset_config.height, width=dataset_config.width)
    else:
        scale_transform = transforms.Scale(height=dataset_config.width, width=dataset_config.height)

    new_img, _ = scale_transform(image, [])
    to_tensor = torch_transforms.ToTensor()
    img_tensor = to_tensor(new_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        pred_count, hm = heatmap_decode_hm(output)
        pred_count = pred_count.to('cpu').numpy()[0]
        hm = hm.to('cpu').numpy()[0]
        if (decoder is not None and pred_count > dataset_config.decoder_involve_threshold):
            pred_count = decoder(output).to('cpu').numpy()[0]

        hm_h, hm_w = hm.shape
        if pred_count < dataset_config.test_split_threshold or depth > dataset_config.test_split_depth:
            locations = heatmap_decode_location(image, output)
            for location in locations:
                location[0] = max(min(location[0], h - 1), 0)
                location[1] = max(min(location[1], w - 1), 0)
                pred_map[int(location[0]), int(location[1])] = location[2]
            return pred_count
        else:
            if h > w:
                new_h = h // 2
                img1 = image[:new_h, :]
                img2 = image[new_h:, :]
                pred_map1 = pred_map[:new_h, :]
                pred_map2 = pred_map[new_h:, :]
                hm_count1 = max(1, np.sum(hm[:hm_h // 2, :]))
                hm_count2 = max(1, np.sum(hm[hm_h // 2:, :]))
            else:
                new_w = w // 2
                img1 = image[:, :new_w]
                img2 = image[:, new_w:]
                pred_map1 = pred_map[:, :new_w]
                pred_map2 = pred_map[:, new_w:]
                hm_count1 = max(1, np.sum(hm[:, :hm_w // 2]))
                hm_count2 = max(1, np.sum(hm[:, hm_w // 2:]))

            if (pred_count < 500):
                count1 = cc_count(model, img1, depth + 1, pred_map1, decoder)
                count2 = cc_count(model, img2, depth + 1, pred_map2, decoder)
            else:
                count1 = max(hm_count1, cc_count(model, img1, depth + 1, pred_map1, decoder))
                count2 = max(hm_count2, cc_count(model, img2, depth + 1, pred_map2, decoder))

            return count1 + count2

count = 0
for batch_idx, data in enumerate(demo_loader):
    count += 1
    fname = data['fname'][0]
    gt = data['num_people'][0].numpy()
    original_img = data['image'][0].numpy()

    h, w = original_img.shape[:2]
    pred_map = np.zeros((h, w))
    pred = cc_count(net, original_img, 1, pred_map, None)
    mae += abs(gt-pred)
    mse += ((gt-pred)*(gt-pred))

    print(fname, 'pred: %.2f' % pred, 'gt: %.1f' % gt)

    csv_name = fname.replace('jpg', 'csv')
    np.savetxt(osp.join(localization_output_dir, csv_name), pred_map, delimiter=',', fmt='%.3f')
    k = np.ones((3, 3))
    bmap = signal.convolve2d((pred_map > 0), k, boundary='symm', mode='same')
    points = np.transpose(np.nonzero(bmap))
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY);
    original_img = cv2.merge([gray,gray,gray])
    for point in points:
        original_img[point[0]][point[1]] = [0, 0, 255]
    cv2.imwrite(osp.join(output_dir, fname.split('.')[0] + '_output_c_%d.png' % pred), original_img)

mae = mae / count
mse = np.sqrt(mse / count)
print('\nMAE: %0.2f, MSE: %0.2f' % (mae, mse))
