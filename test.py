from __future__ import division

from model import *
from utils.utils import *
from dataset_process.sl_datasets import *
from utils.parse_cfg import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')

# os.environ["CUDA_VISIBLE_DEVICES"] = \
#     "1"  # #指定gpu


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    # 加载数据
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    # for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        # imgs : [batch_size,channels,width,height]
        # targets : [index,class_id,cx,cy,w,h]
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            # print(nms_thres)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        # print(outputs,'\ngt=',targets)
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
        # print(sample_metrics)
    # 这里需要注意,github上面的代码有错误,需要添加if条件语句，训练才能正常运行
    if len(sample_metrics) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    #print(true_positives, len(pred_scores))
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    # end = time.time()
    # all_time = end - start
    numbers = len(dataloader)
    # print("\nfps:",numbers/all_time,'\n')
    print("numbers:",numbers,'\n')
    # print("all_time:", all_time,'\n')
    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3_tiny.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/sl.data", help="path to data config file")
    # parser.add_argument("--weights_path", type=str, default="checkpoints_save/yolov3_ckpt_189.pth", help="path to weights file")
    parser.add_argument("--weights_path", type=str, default="checkpoints_tiny_512_part_XS(2048)_0328/yolov3_ckpt_419.pth",
                        help="path to weights file")
    parser.add_argument("--class_path", type=str, default="config/sldata.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.2, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
    opt = parser.parse_args()
    #print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 获取数据参数
    data_config = parse_data_cfg(opt.data_config)

    # 验证集的路径
    valid_path = data_config["test"]
    # 类别名称--类别名称文本一定要空出一行，不然读到的类别会少一类从而报错
    class_names = load_classes(data_config["names"])

    #print('valid_path = ', valid_path)

    # Initiate model
    # 加载模型以及初始化
    start = time.time()
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))
        total = sum([param.nelement() for param in model.parameters()])
        #print("Number of parameter: %.2fM" % (total / 1e6))


    #print("Compute mAP...")

    start = time.time()
    #print("start")
    # 计算map
    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
    )



    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]})\n - AP: {AP[i]}\n - Recall: {recall[i]}\n - f1: {f1[i]}\n")

    print(f"mAP: {AP.mean()}\n"+ f'val_precision {precision.mean()}\n' + f'val_recall {recall.mean()}')
