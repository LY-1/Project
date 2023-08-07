import cv2
import os

import time
import numpy as np

import shutil

def slice_picture(path):
    # 图像文件原始路径
    datas = ['train', 'val', 'test']
    root_path = path
    for data_name in datas:
        path = root_path + data_name
        # resize_path = "D:\\dataset\\data3\\test_512\\"
        listdir = os.listdir(path)
        # 新建split文件夹用于保存
        newdir = root_path + 'real/' + data_name
        if (os.path.exists(newdir) == False):
            os.makedirs(newdir)
        else:
            shutil.rmtree(newdir)
            os.makedirs(newdir)
        for i in listdir:
            if i.split('.')[1] == "png" or i.split('.')[1] == "JPG" or i.split('.')[1] == "jpg":
                filepath = os.path.join(path, i)
                filename = i.split('.')[0]

                # leftpath1 = os.path.join(newdir, filename) + "_left1." + i.split('.')[1]
                # leftpath2 = os.path.join(newdir, filename) + "_left2.." + i.split('.')[1]
                # rightpath1 = os.path.join(newdir, filename) + "_right1." + i.split('.')[1]
                # rightpath2 = os.path.join(newdir, filename) + "_right2." + i.split('.')[1]
                img = cv2.imread(filepath, 0)

                [h, w] = img.shape[:2]
                print(filepath, (h, w))
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
                for index in range(16):
                    save_path = leftpath1 = os.path.join(newdir, filename) + "_{}.".format(index) + i.split('.')[1]
                    cv2.imwrite(save_path, imgs[index])