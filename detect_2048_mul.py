from __future__ import division

from model import *
from utils.utils import *
from dataset_process.sl_datasets import *

import os
import sys
import time
import datetime
import argparse
import shutil
import cv2
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from test import evaluate

os.environ["CUDA_VISIBLE_DEVICES"] = \
    "0"  # #指定gpu
root_path = r'/home/junjzhan/LY/Infrared_project_v2/data/ALLdata/NEWestdata/newestV1xishu/real'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default=root_path + '/test', help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3_tiny.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints_tiny_512_part_XS(2048)_0328/yolov3_ckpt_419.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="config/sldata.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.2, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--batch_size_2048", type=int, default=16, help="size of the batches in 2048")
    opt = parser.parse_args()
    print(opt)
    allnum = len(os.listdir(root_path + '/test'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    print(dataloader)
    # for data in enumerate(dataloader):
    #     print(data)
    start = time.time()
    # progresstime=0
    # protime=0
    inf_time=0
    is_2048 = False
    for batch_i, (img_paths, test_imgs) in enumerate(dataloader):
        # print('test_imgs.shape:')
        # print(len(test_imgs.shape))
        if len(test_imgs.shape) == 5:
            is_2048 = True
            test_imgs = test_imgs.view(-1, 1, opt.img_size, opt.img_size)
            for i in range(0, test_imgs.shape[0], opt.batch_size_2048):
                if i + opt.batch_size_2048 >= test_imgs.shape[0]:
                    end = test_imgs.shape[0]
                else:
                    end = i + opt.batch_size_2048
                input_imgs = test_imgs[i:end, :, :, :]
        # print(batch_i)
        # Configure input
        # input_imgs = np.concatenate((input_imgs,input_imgs,input_imgs),axis=1)
        # input_imgs = Variable(torch.from_numpy(input_imgs).to(device))

                start_net = time.time()
                # Get detections
                with torch.no_grad():
                    input_imgs = Variable(input_imgs.to(device)).type(torch.cuda.FloatTensor)    #  如果使用GPU 需要此行  0708type
                    #print("input_imgs.shape:",input_imgs.shape)   #  torch.Size([4, 3, 512, 512])
                    # print("input_imgs:", input_imgs)
                    # start_0 = time.time()
                    detections = model(input_imgs)
                    # print('predict_shape:',detections.shape)
                    # end_1 = time.time()
                    # print(opt.nms_thres)
                    detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
                    # end_0=time.time()
                    # print(detections)

                # Log progress
                current_time = time.time()
                inference_time = datetime.timedelta(seconds=current_time - prev_time).total_seconds()
                prev_time = current_time

                print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))
                # protime +=(end_1-start_0)
                # progresstime +=(end_0-start_0)
                inf_time+=inference_time
                # Save image and detections
                img_paths *= int(len(detections) / len(img_paths))
                imgs.extend(img_paths)
                img_detections.extend(detections)
        else:
            input_imgs = test_imgs
            start_net = time.time()
            # Get detections
            with torch.no_grad():
                input_imgs = Variable(input_imgs.to(device)).type(torch.cuda.FloatTensor)  # 如果使用GPU 需要此行  0708type
                # print("input_imgs.shape:",input_imgs.shape)   #  torch.Size([4, 3, 512, 512])
                # print("input_imgs:", input_imgs)
                # start_0 = time.time()
                detections = model(input_imgs)
                # print('predict_shape:',detections.shape)
                # end_1 = time.time()
                # print(opt.nms_thres)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
                # end_0=time.time()
                # print(detections)

            # Log progress
            current_time = time.time()
            inference_time = datetime.timedelta(seconds=current_time - prev_time).total_seconds()
            prev_time = current_time

            print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))
            # protime +=(end_1-start_0)
            # progresstime +=(end_0-start_0)
            inf_time += inference_time
            # Save image and detections
            imgs.extend(img_paths)
            img_detections.extend(detections)
        # print("Compute mAP...")
        # # 计算map
        # precision, recall, AP, f1, ap_class = evaluate(
        #     model,
        #     path='/home/vlab/workspace/fengxiaoyu/AGV_yolov3/test.txt',
        #     iou_thres=0.5,
        #     conf_thres=opt.conf_thres,
        #     nms_thres=opt.nms_thres,
        #     img_size=opt.img_size,
        #     batch_size=opt.batch_size,
        # )
        #
        # print("Average Precisions:")
        # for i, c in enumerate(ap_class):
        #     print(f"+ Class '{c}' ({classes[c]}) - AP: {AP[i]} -Recall: {recall[i]} -f1: {f1[i]}")
        #
        # print(f"mAP: {AP.mean()}" + f'val_precision {precision.mean()}' + f'val_recall {recall.mean()}')
    endtime_0 = time.time()
    #initialize path
    predictpath = root_path + '/result/predict/'
    predictpath2 = root_path + '/result/predict2/'
    # predictpath = '/media/data/fxy/infrared/infrared_datatset/data716/predict/'
    if not os.path.exists(predictpath):
        os.makedirs(predictpath)
    else:
        shutil.rmtree(predictpath)
        os.makedirs(predictpath)

    if not os.path.exists(predictpath2):
        os.makedirs(predictpath2)
    else:
        shutil.rmtree(predictpath2)
        os.makedirs(predictpath2)

    rectpath = root_path + '/result/rect/'
    if not os.path.exists(rectpath):
        os.makedirs(rectpath)
    else:
        shutil.rmtree(rectpath)
        os.makedirs(rectpath)


    #print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        #print("path:",path)
        predict_txt = predictpath + path.split('/')[-1].split('.')[0] + '.txt'
        predict_txt2 = predictpath2 + path.split('/')[-1].split('.')[0] + '.txt'
        #  将测试的所有预测结果[ 类别，中心点x,中心点y，宽w，高h] 归一化写入predict下的*.txt,,
        #  再在F:\Object-Detection-Metrics-master\pascalvoc.py中比对 预测txt(/predict文件夹内容)和真实txt(/txtgt文件夹内容，代表测试集的gt)


        #print("(%d) Image: '%s'" % (img_i, path))
        img_cv = cv2.imread(path)
        height, width, c = img_cv.shape

        if is_2048:
            img_i = img_i % 16
            h_offset = int(img_i / 4)
            w_offset = img_i - h_offset * 4
            act_height = height / 4     # 512
            act_width = width / 4
        else:
            act_height = height
            act_width = width
        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, (act_height, act_width))
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                f = open(predict_txt, 'a')
                # # s = '{} {:.5f} {:.5f} {:.5f} {:.5f}'.format(int(cls_pred.item()),(x2.item()+x1.item())/480/2,(y2.item()+y1.item())/480/2,(x2.item()-x1.item())/480
                #                                            ,(y2.item()-y1.item())/480)
                if is_2048:
                    s = '{} {:.5f} {:.5f} {:.5f} {:.5f}\n'.format(int(cls_pred.item()),((x2.item()+x1.item())/act_width/2 + w_offset) * 0.25,((y2.item()+y1.item())/act_height/2 + h_offset) * 0.25,(x2.item()-x1.item())/width
                                                            ,(y2.item()-y1.item())/height)
                else:
                    s = '{} {:.5f} {:.5f} {:.5f} {:.5f}\n'.format(int(cls_pred.item()),(x2.item()+x1.item())/width/2,(y2.item()+y1.item())/height/2,(x2.item()-x1.item())/width
                                                           ,(y2.item()-y1.item())/height)
                print(s)
                f.write(s)

                #track_txt: zhenid -1 x y w h 1 -1 -1 -1//// padding=1或0
                f2 = open(predict_txt2, 'a')
                if is_2048:
                    s2 = '{} -1 {} {} {} {} 1 -1 -1 -1\n'.format(int(path.split('/')[-1].split('.')[0].split('_')[-1]),
                                                                 (x2.item() + x1.item()) / 2 + 512 * w_offset,
                                                                 # (y2.item() + y1.item())/2, (x2.item() - x1.item())*2
                                                                 (y2.item() + y1.item()) / 2 + 512 * h_offset, (x2.item() - x1.item())
                                                                 #   , (y2.item() - y1.item())*2)
                                                                 , (y2.item() - y1.item()))

                else:
                    s2 = '{} -1 {} {} {} {} 1 -1 -1 -1\n'.format(int(path.split('/')[-1].split('.')[0].split('_')[-1]), (x2.item()+x1.item())/2,
                                                            # (y2.item() + y1.item())/2, (x2.item() - x1.item())*2
                                                             (y2.item() + y1.item())/2, (x2.item() - x1.item())
                                                            #   , (y2.item() - y1.item())*2)
                                                             , (y2.item() - y1.item()))
                f2.write(s2)

                #print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))


                # Create a Rectangle in cv2_img
                '''
                cv2.rectangle(img_cv, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), thickness=1)
                # cv2.rectangle(img_cv, (x1, y1), (x2, y2), (255, 255, 255), thickness=1)
                # text = classes[int(cls_pred)] + ':' + str(cls_conf.item())
                text='size:'+str(round((x2.item()-x1.item())*(y2.item()-y1.item()),2))
                cv2.putText(img_cv, text, (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 0),
                # cv2.putText(img_cv, text, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 0),
                            thickness=1)

                filename = path.split("/")[-1].split(".")[0]
                save_path = rectpath + filename + '.png'
                # save_path = '/media/data/fxy/infrared/infrared_datatset/data716/rect/' + filename + '.png'
                # print(save_path)
                # cv2.imwrite(save_path, img_cv)  # save picture
                '''

        else:
            f = open(predict_txt, 'a')
            #s = '\n'
            #f.write(s)
            f2 = open(predict_txt2, 'a')
            # s2 = '\n'
            # f2.write(s2)

    end = time.time()
    all_time = endtime_0 - start
    all= end -start
    numbers = len(dataloader)
    print("batch_size:",opt.batch_size,'\n')
    print("\nfps:", allnum / all, '\n')
    print("\nfps:", allnum / inf_time, '\n')
    print("\nfps:", allnum / all_time, '\n')
    # print("\nfps:",allnum/progresstime,'\n')
    # print("\nfps:",allnum/protime,'\n')
    print("dataloader_len:", numbers, '\n')
    print("all_time:", all, '\n')