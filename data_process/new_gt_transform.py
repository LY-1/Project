import cv2
import os
import shutil

'''
2048_GT->1024_GT
'''

def gt_transform_1(path):

    datas = ['train_gt', 'test_gt']
    root_path = path


    for data in datas:
        gt_path = os.path.join(root_path, '{}/'.format(data))

        listdir = os.listdir(gt_path)
        # 新建split文件夹用于保存
        newdir = os.path.join(root_path, 'inter_gt', data)

        if (os.path.exists(newdir) == False):
            os.makedirs(newdir)
        else:
            shutil.rmtree(newdir)
            os.makedirs(newdir)
        for i in listdir:
            # print(newdir)
            # if i.split('.')[1] == "txt":
            # if ".txt" in i:
            if i[-4:] == ".txt":
                with open(gt_path + i) as f1:
                    lines = f1.readlines()
                    f = open(newdir + '/' + i.split('.')[0] + '_left1.txt', 'a')
                    f.close()
                    f = open(newdir + '/' + i.split('.')[0] + '_left2.txt', 'a')
                    f.close()
                    f = open(newdir + '/' + i.split('.')[0] + '_right1.txt', 'a')
                    f.close()
                    f = open(newdir + '/' + i.split('.')[0] + '_right2.txt', 'a')
                    f.close()

                    for line in lines:
                        if line == '/n':
                            continue
                        x = float(line.split(' ')[1])
                        y = float(line.split(' ')[2])
                        w = float(line.split(' ')[3])
                        h = float(line.split(' ')[4])
                        # print(h)
                        if x < 0.5 and y < 0.5:
                            if x + w/2 > 0.5:
                                w = 0.5 - (x - w / 2)
                                x = 0.5 - w / 2
                            if y + h / 2 > 0.5:
                                h = 0.5 - (y - h / 2)
                                y = 0.5 - h / 2
                            x = 2 * x
                            y = 2 * y
                            w = 2 * w
                            h = 2 * h
                            f = open(newdir + '/' + i.split('.')[0] + '_left1.txt', 'a')
                            l = ' '.join(['0'] + [str(x)] + [str(y)] + [str(w)] + [str(h)])
                            print('left1:', l)
                            f.write(l + '\n')
                            f.close()
                        elif x < 0.5 and y >= 0.5:
                            if x + w / 2 > 0.5:
                                w = 0.5 - (x - w / 2)
                                x = 0.5 - w / 2
                            if y - h / 2 < 0.5:
                                h = (y + h / 2) - 0.5
                                y = 0.5 + h / 2
                            x = 2 * x
                            y = 2 * (y - 0.5)
                            w = 2 * w
                            h = 2 * h
                            f = open(newdir + '/' + i.split('.')[0] + '_left2.txt', 'a')
                            l = ' '.join(['0'] + [str(x)] + [str(y)] + [str(w)] + [str(h)])
                            print('left2:', l)
                            f.write(l + '\n')
                            f.close()
                        elif x >= 0.5 and y < 0.5:
                            if x - w / 2 < 0.5:
                                w = (x + w / 2) - 0.5
                                x = 0.5 + w / 2
                            if y - h / 2 > 0.5:
                                h = (y + h / 2) - 0.5
                                y = 0.5 + h / 2
                            x = 2 * (x - 0.5)
                            y = 2 * y
                            w = 2 * w
                            h = 2 * h
                            f = open(newdir + '/' + i.split('.')[0] + '_right1.txt', 'a')
                            l = ' '.join(['0'] + [str(x)] + [str(y)] + [str(w)] + [str(h)])
                            print('right1:', l)
                            f.write(l + '\n')
                            f.close()
                        elif x >= 0.5 and y >= 0.5:
                            if x - w / 2 < 0.5:
                                w = (x + w / 2) - 0.5
                                x = 0.5 + w / 2
                            if y - h / 2 < 0.5:
                                h = 0.5 - (y - h / 2)
                                y = 0.5 - h / 2
                            x = 2 * (x - 0.5)
                            y = 2 * (y - 0.5)
                            w = 2 * w
                            h = 2 * h
                            f = open(newdir + '/' + i.split('.')[0] + '_right2.txt', 'a')
                            l = ' '.join(['0'] + [str(x)] + [str(y)] + [str(w)] + [str(h)])
                            print('right2:', l)
                            f.write(l + '\n')
                            f.close()
                        else:
                            print('------------------------------------------------------------')
