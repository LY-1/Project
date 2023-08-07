

import torch
def horisontal_flip(images, targets):   # 相当于数据扩增，图片左右翻转，目标的loctation也就翻转
    images = torch.flip(images, [-1])   # 水平反转
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets