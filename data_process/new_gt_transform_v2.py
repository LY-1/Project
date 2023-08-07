import cv2
import os
import shutil

'''
1024_GT->512_GT, add null txt(train,test)
'''

def gt_transform_2(path):
    data = 'test'
    root_path = path
    gt_aug_path = os.path.join(root_path, 'inter_gt/{}_gt/'.format(data))

    listdir = os.listdir(gt_aug_path)
    # 新建split文件夹用于保存
    newdir = os.path.join(root_path, 'real/{}_gt'.format(data))

    if (os.path.exists(newdir) == False):
        os.mkdir(newdir)
    else:
        shutil.rmtree(newdir)
        os.mkdir(newdir)

    left1 = [0, 1, 4, 5]
    right1 = [2, 3, 6, 7]
    left2 = [8, 9, 12, 13]
    right2 = [10, 11, 14, 15]
    pos = {'left1':left1, 'left2':left2, 'right1':right1, 'right2':right2}

    for i in listdir:
        # print(newdir)
        # if i.split('.')[1] == "txt":
        # if ".txt" in i:
        patch = i.split('_')[2].split('.')[0]
        img_name = i.split('_')[0] + '_' + i.split('_')[1]
        # for index in pos[patch]:
        #     f = open(newdir + '/' + img_name + '_' + str(index) + '.txt', 'a+')
        #     f.close
        if i[-4:] == ".txt":
            with open(gt_aug_path + i) as f1:
                lines = f1.readlines()
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
                        f = open(newdir + '/' + img_name + '_' + str(pos[patch][0]) + '.txt', 'a+')
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
                        f = open(newdir + '/' + img_name + '_' + str(pos[patch][2]) + '.txt', 'a+')
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
                        f = open(newdir + '/' + img_name + '_' + str(pos[patch][1]) + '.txt', 'a+')
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
                        f = open(newdir + '/' + img_name + '_' + str(pos[patch][3]) + '.txt', 'a+')
                        l = ' '.join(['0'] + [str(x)] + [str(y)] + [str(w)] + [str(h)])
                        print('right2:', l)
                        f.write(l + '\n')
                        f.close()
                    else:
                        print('------------------------------------------------------------')

    files = os.listdir(root_path + '/real/{}'.format(data))
    n = 0
    for file in files:
         # print(file)
         if not os.path.exists(newdir + '/' + file.replace('png','txt').replace('jpg','txt')):
             print(newdir + '/' + file.replace('png','txt').replace('jpg','txt'))
             f2 = open(newdir + '/' + file.replace('png','txt').replace('jpg','txt'), 'a')
             f2.write('\n')
             f2.close()
             n += 1
    print('loss file num:',n)
