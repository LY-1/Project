import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import cv2


#from toPatch1 import toPatch
from utils.aug import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):  # 将图像填充为正方形图
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding,左,右,上,下
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)   # 以最近邻的方式放大图片
    return image


def random_resize(images, min_size=288, max_size=448):    # 随机调整图片size，288-448
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=512):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        # img = transforms.ToTensor()(Image.open(img_path))
        #16
        img = np.array(Image.open(img_path), dtype='float64')
        # print('1',img.shape)
        # img = torch.tensor(img).div(256).div(255)
        img = torch.tensor(img).div(255)
        # print('2',img.shape)
        img = toPatch(img)
        # img = img.unsqueeze(0)
        # 1123改
        # img = img.permute(2, 0, 1)
        # Pad to square resolution


        # img, _ = pad_to_square(img, 0)
        # print('3',img.shape)

        # Resize
        # img = resize(img, self.img_size)
        # print('4', img.shape)
        '''
        img = img.view(1, 2, 256, 2, 256)
        img = img.permute(1, 3, 0, 2, 4)

        # print('5', img.shape)

        img1 = img[0][0]
        img2 = img[0][1]
        img3 = img[1][0]
        img4 = img[1][1]

        img1 = resize(img1, self.img_size)
        img2 = resize(img2, self.img_size)
        img3 = resize(img3, self.img_size)
        img4 = resize(img4, self.img_size)

        # import matplotlib.pyplot as plt
        # plt.imshow(img3[0])
        # plt.set_cmap('gray')
        # plt.axis('off')
        # plt.show()

        # print("6", img1.shape)

        img = [img1, img2, img3, img4]
        # print(img[0].shape)
        '''
        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=480, augment=True, multiscale=True, normalized_labels=True):
        #print(list_path)
        with open(list_path, "r") as file:
            self.img_files = file.readlines()
            #print(self.img_files)
        for path in self.img_files:
            self.label_files = [
                path.replace("train", "target").replace("val","target").replace(".bmp", ".txt").replace(".png", ".txt").replace(".jpg", ".txt")]
                # path.replace("train", "target").replace("val", "target").replace(".png", ".txt")]
            # print(self.label_files)
        self.img_size = img_size
        self.max_objects = 2
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()     # %取余

        # Extract image as PyTorch tensor
        #16
        img = np.array(Image.open(img_path),dtype='float64')
        # print('1',img.shape)
        # img = torch.tensor(img).div(256).div(255)
        img = torch.tensor(img).div(255)
        # print('2',img.shape)
        img = img.unsqueeze(0)
        # img = img.permute(2, 0, 1)


        # img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        #
        # # Handle images with less than three channels
        # if len(img.shape) != 3:
        #     img = img.unsqueeze(0)
        #     img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape
        # print(img.shape)
        # ---------
        #  Label
        # ---------
        label_path = img_path.replace(img_path.split('/')[-2],'gt').replace('bmp','txt').replace('png','txt').replace('jpg','txt')
        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))   # [cls, cx, cy, cw, ch]
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            # 转换到pad后的归一化中心坐标和宽高
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))  # len(boxes)为当前张图片中目标的数量
            targets[:, 1:] = boxes
        # Apply augmentations

        if self.augment:
            if np.random.random() < 0.5 and targets is not None:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))

        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]

        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat((targets), 0)

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)



# 2048 to patch
def toPatch(img):
    [h, w] = img.shape[:2]

    if h > 1000 or w > 1000:
        patch_img0 = img[:int(h / 4), :int(w / 4)]
        patch_img1 = img[:int(h / 4), int(w / 4):int(w / 2)]
        patch_img2 = img[:int(h / 4), int(w / 2):int(w * 3 / 4)]
        patch_img3 = img[:int(h / 4), int(w * 3 / 4):]

        patch_img4 = img[int(h / 4):int(h / 2), :int(w / 4)]
        patch_img5 = img[int(h / 4):int(h / 2), int(w / 4):int(w / 2)]
        patch_img6 = img[int(h / 4):int(h / 2), int(w / 2):int(w * 3 / 4)]
        patch_img7 = img[int(h / 4):int(h / 2), int(w * 3 / 4):]

        patch_img8 = img[int(h / 2):int(h * 3 / 4):, :int(w / 4)]
        patch_img9 = img[int(h / 2):int(h * 3 / 4), int(w / 4):int(w / 2)]
        patch_img10 = img[int(h / 2):int(h * 3 / 4), int(w / 2):int(w * 3 / 4)]
        patch_img11 = img[int(h / 2):int(h * 3 / 4), int(w * 3 / 4):]

        patch_img12 = img[int(h * 3 / 4):, :int(w / 4)]
        patch_img13 = img[int(h * 3 / 4):, int(w / 4):int(w / 2)]
        patch_img14 = img[int(h * 3 / 4):, int(w / 2):int(w * 3 / 4)]
        patch_img15 = img[int(h * 3 / 4):, int(w * 3 / 4):]
        imgs = [patch_img0, patch_img1, patch_img2, patch_img3,
                patch_img4, patch_img5, patch_img6, patch_img7,
                patch_img8, patch_img9, patch_img10, patch_img11,
                patch_img12, patch_img13, patch_img14, patch_img15]
        patch_img = []
        for img in imgs:
            img = transform(img)
            img = img.unsqueeze(0)
            patch_img.append(img)
        patch_img = torch.cat(patch_img, dim=0)
        return patch_img
    else:
        img = transform(img)
        return img

def transform(img):
    img = img.unsqueeze(0)
    img, _ = pad_to_square(img, 0)
    img = resize(img, 512)

    return img


