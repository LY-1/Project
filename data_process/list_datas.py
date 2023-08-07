
import os

def list_datas(path, dataset):
    if dataset == '2048':
        root_path = os.path.join(path, 'real')
    else:
        root_path = path

    datas = ['train', 'test', 'val']

    for data in datas:
        f = open(root_path + '/{}.txt'.format(data), 'w')
        paths = root_path + "/{}/".format(data)
        # f = open('/media/shuer/3B4EA5535AABF3B6/ld/fxy/infrared/infrared_datatset/data716/test.txt', 'w')
        # paths = "/media/shuer/3B4EA5535AABF3B6/ld/infrared/infrared_datatset/data716/test01/"
        files = os.listdir(paths)
        files.sort()
        for file in files:
            path = paths + file
            f.write(path + '\n')
        f.close()



