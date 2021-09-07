import sys
sys.path.append('datasets')

import torch
import os
import os.path as osp
import cv2
import numpy as np
from datetime import datetime
import misc.utils as utils
import math
from scipy import spatial
import pandas as pd

img_path = './datasets/Demo/img'
gt_path = './datasets/Demo/label'
pred_path = './output/JHU_CKPT/loc'
pred_files = [filename for filename in os.listdir(pred_path) \
       if os.path.isfile(os.path.join(pred_path, filename))]

sigmas = [40, 20, 5]

for sigma in sigmas:
    print(sigma)
    m_ap = 0
    m_ar = 0

    for thre in np.arange(0.5, 1.0, 0.05):
        max_dis = math.sqrt(-math.log(thre) * 2 * (sigma ** 2))

        print(max_dis)

        count = 0
        precision = 0
        recall = 0

        for f in pred_files:
            count += 1
            preds = np.loadtxt(osp.join(pred_path, f), delimiter=',', dtype=np.float32)
            gts = np.loadtxt(osp.join(gt_path, f.split('.')[0] + '.txt'), dtype=np.float32).reshape(-1, 2)
            img = cv2.imread(osp.join(img_path, f.split('.')[0] + '.jpg'))
            height, width = img.shape[:2]
            gts[:, 0] *= width
            gts[:, 1] *= height
            gts[:, [0, 1]] = gts[:, [1, 0]]

            preds_locs = np.transpose(np.nonzero(preds)).astype(float)
            preds_locs = np.insert(preds_locs, 2, values=[preds[int(preds_locs[i][0]), int(preds_locs[i][1])] for i in range(len(preds_locs))], axis=1)

            preds_locs = preds_locs[(-preds_locs[:, 2]).argsort()]

            if (len(gts) == 0):
                for point in preds_locs:
                    fp += 1
            else:
                kd_tree = spatial.KDTree(gts)
                gts = np.insert(gts, 2, values=0, axis=1)

                tp = fp = fn = 0

                for point in preds_locs:
                    distances, positions = kd_tree.query(point[:2], distance_upper_bound=max_dis)
                    if (distances is not list):
                        distances = [distances]
                        positions = [positions]
                    match = False
                    for i in range(len(distances)):
                        if (math.isinf(distances[i])):
                            break
                        if (gts[positions[i]][2] == 1):
                            continue
                        match = True
                        gts[positions[i]][2] = 1
                        break

                    if match:
                        tp += 1
                    else:
                        fp += 1
            for gt in gts:
                if gt[2] == 0:
                    fn += 1
            if (tp + fp == 0):
                precision += 1
            else:
                precision += tp / (tp + fp)
            if (tp + fn == 0):
                recall += 1
            else:
                recall += tp / (tp + fn)

        avg_precision = precision / count
        avg_recall = recall / count

        print('ap %.2f' % thre, avg_precision)
        print('ar %.2f' % thre, avg_recall)
        m_ap += avg_precision
        m_ar += avg_recall

    print('map', m_ap / 10)
    print('mar', m_ar / 10)
max_dis = math.sqrt(-math.log(thre) * 2 * (sigma ** 2))

print(max_dis)

count = 0
precision = 0
recall = 0

for f in pred_files:
    count += 1
    preds = np.loadtxt(osp.join(pred_path, f), delimiter=',', dtype=np.float32)
    gts = np.loadtxt(osp.join(gt_path, f.split('.')[0] + '.txt'), dtype=np.float32).reshape(-1, 2)
    img = cv2.imread(osp.join(img_path, f.split('.')[0] + '.jpg'))
    height, width = img.shape[:2]
    gts[:, 0] *= width
    gts[:, 1] *= height
    gts[:, [0, 1]] = gts[:, [1, 0]]

    preds_locs = np.transpose(np.nonzero(preds)).astype(float)
    preds_locs = np.insert(preds_locs, 2, values=[preds[int(preds_locs[i][0]), int(preds_locs[i][1])] for i in range(len(preds_locs))], axis=1)

    preds_locs = preds_locs[(-preds_locs[:, 2]).argsort()]

    kd_tree = spatial.KDTree(gts)
    gts = np.insert(gts, 2, values=0, axis=1)

    tp = fp = fn = 0

    for point in preds_locs:
        # print(point[:2])
        distances, positions = kd_tree.query(point[:2], distance_upper_bound=max_dis)
        if (distances is not list):
            distances = [distances]
            positions = [positions]
        match = False
        for i in range(len(distances)):
            if (math.isinf(distances[i])):
                break
            if (gts[positions[i]][2] == 1):
                continue
            match = True
            gts[positions[i]][2] = 1
            break

        if match:
            tp += 1
        else:
            fp += 1
    for gt in gts:
        if gt[2] == 0:
            fn += 1
    print(len(gts), len(preds_locs), tp, fp, fn, tp + fp, tp + fn)
    print('precision', tp / (tp + fp))
    print('recall', tp / (tp + fn))
    precision += tp / (tp + fp)
    recall += tp / (tp + fn)

avg_precision = precision / count
avg_recall = recall / count

print(avg_precision)
print(avg_recall)
