import os
import numpy as np
from skimage import measure
import torch


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def measure_img(o_img, t_num=1):
    p_img=np.zeros_like(o_img)
    testa1 = measure.label(o_img.astype("bool"))
    props = measure.regionprops(testa1)
    numPix = []
    for ia in range(len(props)):
        numPix += [props[ia].area]
    for i in range(0, t_num):
        index = numPix.index(max(numPix)) + 1
        p_img[testa1 == index]=o_img[testa1 == index]
        numPix[index-1]=0
    return p_img


def instance_max_min_normalize(input_tensor):
    b = input_tensor.shape[0]
    min_vals = torch.min(input_tensor.view(b, -1), dim=1)[0].view(b, 1, 1, 1)
    max_vals = torch.max(input_tensor.view(b, -1), dim=1)[0].view(b, 1, 1, 1)
    normalized_tensor = (input_tensor - min_vals) / (max_vals - min_vals)
    return normalized_tensor


