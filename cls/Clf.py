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
        self.conv1 = nn.Conv2d(1, 6, 3)  # 输入频道1， 输出频道6， 卷积3x3,  无pad,特征图每次变小2
        # 第二层（卷积层）
        self.conv2 = nn.Conv2d(6, 16, 3)  # 输入频道6， 输出频道16， 卷积3x3
        # 第三层（全连接层）
        #512*512
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
def getimgandtxt(srcpath,picpath, img_w, img_h):
    files = os.listdir(srcpath)
    for file in files:
        # if file.split('_')[0] in ['4', '05', '11','18']:
        # if file.split('_')[0] in ['02', '05', '13', '14', '16']:
        if file.split('_')[0] in ['1', '5', '10', '15', '20']:
            print(srcpath+file)
            txtfile = srcpath + file
            f1 = open(txtfile, 'r')
            if os.path.exists(picpath+file.replace('txt','jpg')):
                img = Image.open(picpath+file.replace('txt','jpg'))
                for line in f1:
                    _,x,y,w,h = line.split(' ')
                    x = float(x)*img_w
                    y = float(y)*img_h
                    w = float(w)*img_w
                    h = float(h)*img_h
                    ###10*10? target area
                    xx1 = x - 5
                    yy1 = y - 5
                    xx2 = x + 5
                    yy2 = y + 5
                    region = img.crop((xx1, yy1, xx2, yy2))
                    # print(np.array(region)[:,:,0])
                    pic = torch.from_numpy(np.array(region)[:,:]/255)
                    # print(pic.size())
                    # pic = torch.unsqueeze(pic,0)
                    pic = torch.unsqueeze(pic, 0).to(torch.float32)
                    print(pic.size())
                    if pic.size()[2] !=9 and pic.size()[1]!=9:

                        X.append(pic)
                        # print(np.array(region)[:,:,0]/255)
                        Y.append(1)
                        ran = random.randint(1, 6)
                    if ran % 2 == 0:
                        x += random.randint(int(w),2*int(w))*random.choice((-1, 1))    # 左右两个目标的位置随机采样
                        y += random.randint(int(h),2*int(h))*random.choice((-1, 1))
                        xxx1 = x - 5
                        yyy1 = y - 5
                        xxx2 = x + 5
                        yyy2 = y + 5
                        region2 = img.crop((xxx1, yyy1, xxx2, yyy2))
                        pic2 = torch.from_numpy(np.array(region2)[:, :] / 255)
                        # print(pic.size())
                        # pic2 = torch.unsqueeze(pic2, 0)
                        pic2 = torch.unsqueeze(pic2, 0).to(torch.float32)
                        if pic2.size()[2] != 9 and pic2.size()[1] != 9:
                            X.append(pic2)
                            # print(np.array(region)[:, :, 0] / 255.flatten())
                            Y.append(0)
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


# dataset = ImageClassifyDataset(X,Y,1)
# dataloader = DataLoader(dataset, batch_size=5, shuffle=True,num_workers=5)
# for index, data in enumerate(dataloader):
# 	print(index)	# batch索引
# 	print(data)		# 一个batch的{img,label}



# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
#
# train_loss_hist = []
# test_loss_hist = []
# min_test_loss = 1
# for epoch in tqdm(range(20)):
# #     # 训练
#     net.train()
#     running_loss = 0.0
#     for i in range(len(X_train)):
#         images = X_train[i]
#         labels = y_train[i]
#         outputs = net(images)
#         loss = criterion(outputs, labels)  # 计算损失
# #
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         print(i,':',loss.item())
#         running_loss += loss.item()
#         if (i % 250 == 0):  # 每250 mini batch 测试一次
#             correct = 0.0
#             total = 0.0
#
#             net.eval()
#             with torch.no_grad():
#                 for test_i in range(len(X_test)):
#                     test_images, test_labels = X_test[test_i],y_test[test_i]
#                     test_outputs = net(test_images)
#                     test_loss = criterion(test_outputs, test_labels)
#                 print('test loss:',test_loss.item())
#             train_loss_hist.append(running_loss / 250)
#             if min_test_loss>test_loss.item():
#                 min_test_loss = test_loss.item()
#                 torch.save(net, 'save.pt')
#             test_loss_hist.append(test_loss.item())
#             running_loss = 0.0


os.environ["CUDA_VISIBLE_DEVICES"] = \
    "2"  # #指定gpu
if __name__ == '__main__':
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    print(net)
    date = 'XS(2048)_0331'
    epochs = 200
    w = 512
    h = 512
    # w = 320
    # h = 256
    root_path = r'/home/junjzhan/LY/Infrared_project_v2/data/ALLdata/NEWestdata/newestV1xishu/real'
    srcpath = root_path + '/train_gt/'  # train_gt
    picpath = root_path + '/train/'     # train_pic
    # dstpath = r'F:\data\Guofangkeda\GFdata\classifylabels/'
    files = os.listdir(srcpath)
    X = []
    Y = []
    X,Y = getimgandtxt(srcpath,picpath, w, h)
    # print(len(Y),len(X))
    # print('1')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    traindataset = ImageClassifyDataset(X_train,y_train,1)
    testdataset = ImageClassifyDataset(X_test,y_test,1)
    # print('2')
    train_dataloader = DataLoader(traindataset, batch_size=10, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(traindataset, batch_size=1, shuffle=True, num_workers=0)
    # print('3')
    print(len(train_dataloader))
    # for index, data in enumerate(dataloader):
    #     print(index)  # batch索引
    #     print(data)  # 一个batch的{img,label}
    train_loss_hist = []
    test_loss_hist = []
    min_test_loss = 1
    for epoch in tqdm(range(epochs)):
    #     # 训练
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            # images = X_train[i]
            # labels = y_train[i]
            images,labels = data
            outputs = net(images)
            loss = criterion(outputs, labels)  # 计算损失
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(i,':',loss.item())
            running_loss += loss.item()
            if (i % 250 == 0):  # 每250 mini batch 测试一次
                correct = 0.0
                total = 0.0
                net.eval()
                with torch.no_grad():
                    for i, data in enumerate(test_dataloader):
                        test_images, test_labels = data
                        test_outputs = net(test_images)
                        test_loss = criterion(test_outputs, test_labels)
                    print('test loss:',test_loss.item())
                train_loss_hist.append(running_loss / 250)
                if min_test_loss>test_loss.item():
                    min_test_loss = test_loss.item()
                    torch.save(net, 'save{}.pt'.format(date))
                test_loss_hist.append(test_loss.item())
                running_loss = 0.0