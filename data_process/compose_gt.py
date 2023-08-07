import shutil
import os

def compose_gt(path, dataset):

    root_path = path
    if dataset == '2048':
        train_gt_path = os.path.join(root_path, 'real/train_gt')
        test_gt_path = os.path.join(root_path, 'real/test_gt')
        save_path = os.path.join(root_path, 'real/gt')
    else:
        train_gt_path = os.path.join(root_path, 'train_gt')
        test_gt_path = os.path.join(root_path, 'test_gt')
        save_path = os.path.join(root_path, 'gt')


    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    for file in os.listdir(train_gt_path):
        train_gt = os.path.join(train_gt_path, file)
        shutil.copy(train_gt, save_path)

    for file in os.listdir(test_gt_path):
        test_gt = os.path.join(test_gt_path, file)
        shutil.copy(test_gt, save_path)