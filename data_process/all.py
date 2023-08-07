from new_retxt import new_retxt
from dataToTrainTest import move
from slice_picture_v3 import slice_picture
from new_gt_transform import gt_transform_1
from new_gt_transform_v2 import gt_transform_2
from remove_nullpic import remove_nullpic
from remove_val_null import remove_val_null
from list_datas import list_datas
from compose_gt import compose_gt

root_path = r'/home/junjzhan/LY/Infrared_project_v2/6.7_test/'
dataset = '2048'
if dataset == '2048':
    new_retxt(root_path, dataset)
    move(root_path, dataset)
    slice_picture(root_path)
    gt_transform_1(root_path)   # 2048->1024, all:train, val, test
    gt_transform_2(root_path)   # 1024->512, test
    remove_nullpic(root_path)   # 1024->512, train, val, delete trian null pic
    remove_val_null(root_path)  # delete val null pic
    list_datas(root_path, dataset)
    compose_gt(root_path, dataset)
else:
    new_retxt(root_path, dataset)
    move(root_path, dataset)
    list_datas(root_path, dataset)
    compose_gt(root_path, dataset)