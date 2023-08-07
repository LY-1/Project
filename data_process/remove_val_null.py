import os

'''
1024_GT->512_GT,delete null pic(val)
'''

def remove_val_null(path):

    data = 'val'
    root_path = path
    #gt_aug_path = os.path.join(root_path, 'new_{}_gt/'.format(data))

    #listdir = os.listdir(gt_aug_path)
    # 新建split文件夹用于保存
    newdir = os.path.join(root_path, 'real/train_gt')
    files = os.listdir(root_path + '/real/{}'.format(data))
    n = 0
    for file in files:
         # print(file)
         if not os.path.exists(newdir + '/' + file.replace('png','txt').replace('jpg','txt')):
             os.remove(root_path + '/real/{}/'.format(data) + file)
             n += 1
    print('loss file num:',n)