import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import shutil
import cv2
from PIL import Image
import numpy as np
import random
import collections
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Net(nn.Module):
    def __init__(self):  # 定义神经网络结构, 输入数据 1x10x10
        super(Net, self).__init__()
        # 第一层（卷积层）
        self.conv1 = nn.Conv2d(1, 6, 3)  # 输入频道1， 输出频道6， 卷积3x3
        # 第二层（卷积层）
        self.conv2 = nn.Conv2d(6, 16, 3)  # 输入频道6， 输出频道16， 卷积3x3
        # 第三层（全连接层）
        self.fc1 = nn.Linear(16 * 6 * 6, 512)  # 输入维度16x28x28=12544，输出维度 512
        # 第四层（全连接层）
        self.fc2 = nn.Linear(512, 64)  # 输入维度512， 输出维度64
        # 第五层（全连接层）
        self.fc3 = nn.Linear(64, 2)  # 输入维度64， 输出维度2

    def forward(self, x):  # 定义数据流向
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = x.view(-1, 16 * 6 * 6)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)

        return x






# srcpath = r'F:\data\Guofangkeda\GFdata\labels/'
# picpath = r'F:\data\Guofangkeda\GFdata\allpic/'
# dstpath = r'F:\data\Guofangkeda\GFdata\classifylabels/'
def getimgandtxt_test(srcpath,picpath,img_w,img_h):
    files = os.listdir(srcpath)
    for file in files:
        print(srcpath+file)
        txtfile = srcpath + file
        f1 = open(txtfile, 'r')
        img = Image.open(picpath+file.replace('txt','jpg'))
        for line in f1:
            if line!='\n':
                _,xx,yy,ww,hh = line.strip('\n').split(' ')
                x = float(xx)*img_w
                y = float(yy)*img_h
                w = float(ww)*img_w
                h = float(hh)*img_h
                ###10*10? target area
                xx1 = x - 5
                yy1 = y - 5
                xx2 = x + 5
                yy2 = y + 5
                region = img.crop((xx1, yy1, xx2, yy2))
                pic = torch.from_numpy(np.array(region)[:,:]/255)
                pic = torch.unsqueeze(pic, 0).to(torch.float32)
                X.append(pic)
                Y.append([file.split('.')[0],xx,yy,ww,hh])
    return X,Y



class ImageClassifyDataset(Dataset):
    def __init__(self, imagefile, labelfile, classify_num=1, train=True):
        '''
        这里进行一些初始化操作。
        '''
        self.imagefile = imagefile
        self.labelfile = labelfile
        self.classify_num = classify_num

    def __len__(self):
        return len(self.labelfile)

    def __getitem__(self, item):
        '''
        这个函数是关键，通过item(索引)来取数据集中的数据，
        一般来说在这里才将图像数据加载入内存，之前存的是图像的保存路径
        '''

        label= self.labelfile[item]  # label直接用0,1,2,3,4...表示不同类别
        img = self.imagefile[item]
        return img, label


if __name__ == '__main__':
    net = torch.load('saveGF_0326.pt')
    w = 512
    h = 512
    # w = 320
    # h = 256
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    print(net)
    # srcpath = r'C:\Users\PC\Desktop\result\predict/'
    root_path = r'/home/junjzhan/LY/Infrared_project_v2/data/ALLdata/Sourcedata/GFdata2022.3.24'
    srcpath = root_path + '/result/conf_0.5/predict/'
    picpath = root_path + r'/test/'
    dstpath = root_path + r'/result/conf_0.5/cls_result/'
    if not os.path.exists(dstpath):
        os.makedirs(dstpath)
    files = os.listdir(srcpath)
    X = []
    Y = []
    X,Y = getimgandtxt_test(srcpath,picpath,w,h)
    print(Y)
    # print(len(Y),len(X))
    # print('1')
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    dataset = ImageClassifyDataset(X,Y,1)
    # print('2')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    # print('3')
    print(len(dataloader))
    for index, data in enumerate(dataloader):
        inputs, labels = data
        outputs = net(inputs)
        f = open(dstpath + labels[0][0] + '.txt','a')
        # print(labels)
        print(outputs)
        if outputs[0,0]<=outputs[0,1]:
            print(1)
            s = '0 {} {} {} {}\n'.format(labels[1][0],labels[2][0],labels[3][0],labels[4][0])    # dataloder返回变成5*1
            f.write(s)
        else:
            print(0)
