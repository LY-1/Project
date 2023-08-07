from __future__ import division

from model import *
from utils.utils import *
from dataset_process.sl_datasets import *
from utils.parse_cfg import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

from utils.loggor import Logger
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

import warnings
warnings.filterwarnings('ignore')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = \
    "1"  # #指定gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 配合log输出信息来用,  ='2'输出信息：ERROR + FATAL



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3_tiny.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/sl.data", help="path to data config file")
    # parser.add_argument("--pretrained_weights",type=str, default="checkpoints_dense-v3-tiny-spp_all/yolov3_ckpt_87.pth",help="if specified starts from checkpoint model")
    parser.add_argument("--pretrained_weights", type=str,help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
    # parser.add_argument("--checkpoint_interval", type=int, default=20, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)



    logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_cfg(opt.data_config)

    # data_config["train"] = "/media/shuer/3B4EA5535AABF3B6/ld/infrared/infrared_datatset/data910/train.txt"
    # data_config["valid"] = "/media/shuer/3B4EA5535AABF3B6/ld/infrared/infrared_datatset/data910/val.txt"
    # data_config["test"] = "/media/shuer/3B4EA5535AABF3B6/ld/infrared/infrared_datatset/707data/data908/test.txt"
    # data_config["names"] = "/media/shuer/3B4EA5535AABF3B6/ld/infrared/yolov3/config/sldata.names"

    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    # min_val_precision = 0.6
    min_loss_val = 10
    min_epoch = 1   #  100

    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i
            #16
            imgs = Variable(imgs.to(device)).type(torch.cuda.FloatTensor)
            # imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            try:
                loss, outputs = model(imgs, targets)
                loss.backward()
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception
            #
            # loss, outputs = model(imgs, targets)
            # loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()


            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:          # train.py : 调用test.py里的evaluate( val.txt )进行训练，输出平均mAP  //  直接运行test.py: 使用test.txt进行test
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=4,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)


            # Print class APs aand mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"mAP {AP.mean()}" + f'val_precision {precision.mean()}' + f'val_recall {recall.mean()}')
            savename = 'XS(2048)_0328'
            if not os.path.exists('checkpoints_tiny_512_all_' + savename + '/'):
                os.mkdir('checkpoints_tiny_512_all_' + savename + '/')
            if not os.path.exists('checkpoints_tiny_512_part_' + savename + '/'):
                os.mkdir('checkpoints_tiny_512_part_' + savename + '/')
            torch.save(model.state_dict(), f"checkpoints_tiny_512_all_" + savename + "/" + "yolov3_ckpt_%d.pth" % epoch)
            if epoch>min_epoch and loss <= min_loss_val:
                min_loss_val = loss
                print('---saveing model')
                torch.save(model.state_dict(), f"checkpoints_tiny_512_part_" + savename  + "/" + "yolov3_ckpt_%d.pth" % epoch)
            # if precision.mean() > min_val_precision:
            #     min_val_precision = precision.mean()
            #     print('---saveing model')
            #     torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)


        # if epoch % opt.checkpoint_interval == 0:
        # torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
