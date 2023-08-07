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

from sort import *
from cls.Clf import Net

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from test import evaluate

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # #指定gpu
root_path = r'/home/junjzhan/LY/Infrared_project_v2/data/ALLdata/NEWestdata/newestV0miji/test'

if __name__ == "__main__":

    # ToTrack = True  # 是否进行跟踪
    ToTrack = False

    ShowImg = True  # 是否输出图像
    ShowImg = False

    # dectect
    parser = argparse.ArgumentParser(description='Detect')
    parser.add_argument("--image_folder", type=str, default=root_path, help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3_tiny.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints_tiny_512_part_XS(2048)_0328/yolov3_ckpt_419.pth",
                        help="path to weights file")
    parser.add_argument("--class_path", type=str, default="config/sldata.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.2, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--batch_size_2048", type=int, default=16, help="size of the batches in 2048")
    opt = parser.parse_args()

    # track
    parser1 = argparse.ArgumentParser(description='Track')
    parser1.add_argument("--max_age", help="", type=int, default=10)
    parser1.add_argument("--min_hits", help="", type=int, default=0)
    parser1.add_argument("--threshold", help="", type=float, default=0.5)
    args = parser1.parse_args()
    imgsize = (2048, 2048)



    if ToTrack:
        KalmanBoxTracker.count = 0
        mot_tracker = Sort(max_age=args.max_age,
                           min_hits=args.min_hits,
                           iou_threshold=args.threshold,
                           imgsize=imgsize)

    txtPath = os.path.join(root_path, 'result/txtOut/')
    imgPath = os.path.join(root_path, 'result/imgOut/')
    if not os.path.exists(txtPath):
        os.makedirs(txtPath)
    else:
        shutil.rmtree(txtPath)
        os.makedirs(txtPath)

    if not os.path.exists(imgPath):
        os.makedirs(imgPath)
    else:
        shutil.rmtree(imgPath)
        os.makedirs(imgPath)

    allnum = len(os.listdir(root_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # cls
    net = Net()
    net = torch.load('cls/saveXS(2048)_0331.pt')


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

    start = time.time()
    inf_time=0
    imread_time=0
    trans_time=0
    track_time=0
    imwirte_time=0
    patch_time = 0
    is_2048=False

    batch_time=0
    for batch_i, (img_paths, test_imgs) in enumerate(dataloader):
        img_cv = np.array(Image.open(img_paths[0]))
        start1 = time.time()

        if len(test_imgs.shape) == 5:   # 2048
            is_2048=True

            test_imgs = test_imgs.view(-1, 1, opt.img_size, opt.img_size)
            for i in range(0, test_imgs.shape[0], opt.batch_size_2048):
                pretime = time.time()
                if i + opt.batch_size_2048 >= test_imgs.shape[0]:
                    end = test_imgs.shape[0]
                else:
                    end = i + opt.batch_size_2048
                input_imgs = test_imgs[i:end, :, :, :]
                current_time = time.time()
                time1 = datetime.timedelta(seconds=current_time - pretime).total_seconds()
                patch_time += time1

                # Get detections
                pretime=time.time()
                with torch.no_grad():
                    input_imgs = Variable(input_imgs.to(device)).type(torch.cuda.FloatTensor)
                    detections = model(input_imgs)
                    detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
                current_time = time.time()
                inference_time = datetime.timedelta(seconds=current_time - pretime).total_seconds()
                print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))
                inf_time += inference_time

                img_paths = img_paths[0]
                height, width = 2048,2048

                if ShowImg:
                    pretime = time.time()
                    img_cv = cv2.imread(img_paths)
                    current_time = time.time()
                    time1 = datetime.timedelta(seconds=current_time - pretime).total_seconds()
                    imread_time += time1

                sOut = []
                for ii,detection in enumerate(detections):  # detections 16

                    pretime = time.time()
                    img_i = ii % 16
                    h_offset = int(img_i / 4)
                    w_offset = img_i - h_offset * 4
                    act_height = height / 4
                    act_width = width / 4
                    current_time = time.time()
                    time1 = datetime.timedelta(seconds=current_time - pretime).total_seconds()
                    patch_time += time1

                    if detection is not None:
                        pretime = time.time()
                        detection = rescale_boxes(detection, opt.img_size, (act_height, act_width))
                        current_time = time.time()
                        time1 = datetime.timedelta(seconds=current_time - pretime).total_seconds()
                        trans_time += time1

                        pretime = time.time()
                        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                            xc = (x2.item() + x1.item()) / 2 + 512 * w_offset
                            yc = (y2.item() + y1.item()) / 2 + 512 * h_offset
                            s = [int(img_paths.split('/')[-1].split('.')[0].split('_')[-1]),
                                                                         xc,
                                                                         yc,
                                                                         (x2.item() - x1.item()),
                                                                         (y2.item() - y1.item())]
                            x_left = int(xc) - 5 if (xc - 5) > 0 else 0
                            y_top = int(yc) - 5 if (yc - 5) > 0 else 0
                            x_right = int(xc) + 5 if (xc + 5) < width else width
                            y_bottom = int(yc) + 5 if(yc + 5) < height else height

                            region = img_cv[y_top:y_bottom, x_left:x_right]
                            pic = torch.from_numpy(np.array(region)[:, :] / 255)

                            pic = torch.unsqueeze(pic, 0).to(torch.float32)
                            pic = resize(pic, size=[10, 10])
                            pic = torch.unsqueeze(pic, 0)

                            outputs = net(pic)
                            if outputs[0, 0] <= outputs[0, 1]:
                                sOut.append(s)
                        current_time = time.time()
                        time1 = datetime.timedelta(seconds=current_time - pretime).total_seconds()
                        patch_time += time1

                if not ToTrack:
                    pretime = time.time()
                    filename = img_paths.split("/")[-1].split(".")[0]
                    predict_txt = txtPath + filename + '.txt'
                    f = open(predict_txt, 'a')
                    if len(sOut):
                        for s in sOut:
                            text='{} {} {} {} {}\n'.format(s[0],s[1],s[2],s[3],s[4])
                            f.write(text)
                            if ShowImg:
                                cv2.rectangle(img_cv, (int(s[1]-s[3]/2), int(s[2]-s[4]/2)), (int(s[1]+s[3]/2), int(s[2]+s[4]/2)), (255, 255, 255),
                                              thickness=1)
                    else:
                        text=''
                        f.write(text)
                        if ShowImg:
                            save_path = imgPath + filename + '.png'
                            cv2.imwrite(save_path, img_cv)
                    f.close()
                    if ShowImg:
                        save_path = imgPath + filename + '.png'
                        cv2.imwrite(save_path, img_cv)
                    current_time = time.time()
                    time1 = datetime.timedelta(seconds=current_time - pretime).total_seconds()
                    imwirte_time += time1
                else:   # Track
                    pretime = time.time()
                    det = np.array(sOut)
                    det[:,0:3]=det[:,1:4]
                    det[:, 4] = 1
                    det[:, 0:2] -= det[:, 2:4] / 2
                    det[:, 2:4] += det[:, 0:2]
                    trackers = mot_tracker.update(det)
                    current_time = time.time()
                    time1 = datetime.timedelta(seconds=current_time - pretime).total_seconds()
                    track_time += time1

                    pretime = time.time()
                    filename = img_paths.split("/")[-1].split(".")[0]
                    predict_txt = txtPath + filename + '.txt'
                    f = open(predict_txt, 'a')
                    if len(sOut):
                        for x1, y1, x2, y2, id in trackers:
                            s = '{} {} {} {} {}\n'.format(int(id),
                                                          (x2.item() + x1.item()) / 2,
                                                          (y2.item() + y1.item()) / 2,
                                                          (x2.item() - x1.item()),
                                                          (y2.item() - y1.item()))
                            f.write(s)
                            if ShowImg:
                                cv2.rectangle(img_cv, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), thickness=1)
                                cv2.putText(img_cv, str(int(id)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (255, 255, 255), thickness=1)
                        if ShowImg:
                            save_path = imgPath + filename + '.png'
                            cv2.imwrite(save_path, img_cv)
                    else:
                        det = np.zeros((0, 5))
                        trackers = mot_tracker.update(det)
                        s = ''
                        f.write(s)
                        if ShowImg:
                            save_path = imgPath + filename + '.png'
                            cv2.imwrite(save_path, img_cv)
                    f.close()
                    current_time = time.time()
                    time1 = datetime.timedelta(seconds=current_time - pretime).total_seconds()
                    imwirte_time += time1

        else:       #512
            input_imgs = test_imgs

            pretime = time.time()
            with torch.no_grad():
                input_imgs = Variable(input_imgs.to(device)).type(torch.cuda.FloatTensor)
                detections = model(input_imgs)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)[0]
            current_time = time.time()
            inference_time = datetime.timedelta(seconds=current_time - pretime).total_seconds()
            print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))
            inf_time += inference_time

            img_paths = img_paths[0]
            height, width = input_imgs.shape[2:]

            if ShowImg:
                pretime = time.time()
                img_cv = cv2.imread(img_paths)
                current_time = time.time()
                time1 = datetime.timedelta(seconds=current_time - pretime).total_seconds()
                imread_time+=time1

            filename = img_paths.split("/")[-1].split(".")[0]
            predict_txt = txtPath + filename + '.txt'

            if detections is not None:
                pretime = time.time()
                detections = rescale_boxes(detections, opt.img_size, (height, width))
                current_time = time.time()
                time1 = datetime.timedelta(seconds=current_time - pretime).total_seconds()
                trans_time += time1

                if not ToTrack:
                    pretime = time.time()
                    f = open(predict_txt, 'a')
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                        xc = (x2.item() + x1.item()) / 2
                        yc = (y2.item() + y1.item()) / 2
                        s = '{} {} {} {} {}\n'.format(int(img_paths.split('/')[-1].split('.')[0].split('_')[-1]),
                                                      xc,
                                                      yc,
                                                      (x2.item() - x1.item()),
                                                      (y2.item() - y1.item()))

                        x_left = int(xc) - 5 if (xc - 5) > 0 else 0
                        y_top = int(yc) - 5 if (yc - 5) > 0 else 0
                        x_right = int(xc) + 5 if (xc + 5) < width else width
                        y_bottom = int(yc) + 5 if (yc + 5) < height else height
                        region = img_cv[y_top:y_bottom, x_left:x_right]
                        pic = torch.from_numpy(np.array(region)[:, :] / 255)

                        pic = torch.unsqueeze(pic, 0).to(torch.float32)
                        pic = resize(pic, size=[10, 10])
                        pic = torch.unsqueeze(pic, 0)

                        outputs = net(pic)
                        if outputs[0, 0] <= outputs[0, 1]:
                            f.write(s)
                        if ShowImg:
                            cv2.rectangle(img_cv, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), thickness=1)
                    if ShowImg:
                        save_path = imgPath + filename + '.png'
                        cv2.imwrite(save_path, img_cv)  # save picture
                    f.close()
                    current_time = time.time()
                    time1 = datetime.timedelta(seconds=current_time - pretime).total_seconds()
                    imwirte_time += time1
                else:  # Track
                    pretime = time.time()
                    det = detections.numpy()
                    trackers = mot_tracker.update(det[:, 0:5])
                    current_time = time.time()
                    time1 = datetime.timedelta(seconds=current_time - pretime).total_seconds()
                    track_time += time1

                    pretime = time.time()
                    f = open(predict_txt, 'a')
                    for x1, y1, x2, y2, id in trackers:
                        s = '{} {} {} {} {}\n'.format(int(id),
                                                      (x2.item() + x1.item()) / 2,
                                                      (y2.item() + y1.item()) / 2,
                                                      (x2.item() - x1.item()),
                                                      (y2.item() - y1.item()))
                        f.write(s)
                        if ShowImg:
                            cv2.rectangle(img_cv, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), thickness=1)
                            cv2.putText(img_cv, str(int(id)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                        (255, 255, 255), thickness=1)
                    if ShowImg:
                        save_path = imgPath + filename + '.png'
                        cv2.imwrite(save_path, img_cv)  # save picture
                    f.close()
                    current_time = time.time()
                    time1 = datetime.timedelta(seconds=current_time - pretime).total_seconds()
                    imwirte_time += time1
            else:
                pretime = time.time()
                f = open(predict_txt, 'w+')
                s = ''
                f.write(s)
                f.close()
                if ShowImg:
                    save_path = imgPath + filename + '.png'
                    cv2.imwrite(save_path, img_cv)
                current_time = time.time()
                time1 = datetime.timedelta(seconds=current_time - pretime).total_seconds()
                imwirte_time += time1

                if ToTrack:
                    pretime = time.time()
                    det = np.zeros((0, 5))
                    trackers = mot_tracker.update(det)
                    current_time = time.time()
                    time1 = datetime.timedelta(seconds=current_time - pretime).total_seconds()
                    track_time += time1

        end1=time.time()
        time1 = datetime.timedelta(seconds=end1 - start1).total_seconds()
        batch_time += time1
    end = time.time()
    print('\n')
    print('Data:',root_path)
    print('Track:',ToTrack)
    print('Show:',ShowImg)
    print('Infer time:',inf_time/allnum)
    if ShowImg:
        print('imread time:',imread_time/allnum)
    print('scale time:', trans_time / allnum)
    if ToTrack:
        print('track time:', track_time / allnum)
    print('write time:', imwirte_time / allnum)
    if is_2048:
        print('patch time:',patch_time/allnum)
    print('batch time:', batch_time / allnum)
    print('dataloder time:',(end-start)/allnum-batch_time / allnum)
    print('total time:',(end-start)/allnum)
    print('FPS:',allnum/(end-start))
