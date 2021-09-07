import numpy as np
def convert_labels(original_img, hm, locations, height=768, width=1024):
    shape = original_img.shape[:2]  # shape = [height, width]

    ratio = min(float(height) / shape[0], float(width) / shape[1])

    new_shape = (round(shape[0] * ratio), round(shape[1] * ratio)) # new_shape = [height, width]

    dh = (height - new_shape[0]) / 2  # height padding
    dw = (width - new_shape[1]) / 2  # width padding

    locations = np.insert(locations, 2, values=[hm[int(locations[i][0]), int(locations[i][1])] for i in range(len(locations))], axis=1)

    locations[:, 0] -= dh
    locations[:, 1] -= dw

    locations[:, 0] /= (height - 2 * dh)
    locations[:, 1] /= (width - 2 * dw)

    locations[:, 0] *= shape[0]
    locations[:, 1] *= shape[1]
    return locations
