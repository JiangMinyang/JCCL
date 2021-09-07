import numpy as np
import math

def draw_gaussian(heatmap, center, k_size, sigma, mode='sum', scale_factor=1):
    h, w = heatmap.shape

    x = min(w - 1, max(0, np.abs(int(np.floor(center[0])))))
    y = min(h - 1, max(0, np.abs(int(np.floor(center[1])))))

    x1 = x - int(k_size // 2 - (k_size - 1) % 2)
    y1 = y - int(k_size // 2 - (k_size - 1) % 2)
    x2 = x + int(k_size // 2) + 1
    y2 = y + int(k_size // 2) + 1
    dx1 = dy1 = dx2 = dy2 = 0

    if (x1 < 0):
        dx1 = np.abs(x1)
        x1 = 0
    if (y1 < 0):
        dy1 = np.abs(y1)
        y1 = 0
    if (x2 > w):
        dx2 = x2 - w
        x2 = w
    if (y2 > h):
        dy2 = y2 - h
        y2 = h

    xi = np.arange(0, k_size, 1, np.float32)
    yi = xi[:, np.newaxis]
    x0 = y0 = (k_size - 1) / 2
    g = np.exp(- ((xi - x0) ** 2 + (yi - y0) ** 2) / (2 * sigma ** 2))

    g /= scale_factor

    if mode == 'max':
        heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], g[dy1: k_size - dy2, dx1: k_size - dx2])
    else:
        g = g / np.sum(g[dy1: k_size - dy2, dx1: k_size - dx2])
        heatmap[y1:y2, x1:x2] += g[dy1: k_size - dy2, dx1: k_size - dx2]

    return heatmap