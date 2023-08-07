import os
import shutil

# set img_size and img_format and test_rate

def new_retxt(path, dataset):

    srcpath = path
    src_dir =[srcpath +f for f in os.listdir(srcpath) if f.startswith(('data'))]
    des_path = [srcpath+'test/',srcpath+'test_gt/',srcpath+'train/',srcpath+'train_gt/']
    print(src_dir)

    for d in range(len(des_path)):
        if not os.path.exists(des_path[d]):
            os.mkdir(des_path[d])
        else:
            shutil.rmtree(des_path[d])
            os.mkdir(des_path[d])
    # print(des_path[1] + src_dir[14].split('/')[-1][4:])
    if dataset == 'JWS':
        w = 320
        h = 256
    if dataset == 'GF':
        w = 512
        h = 512
    if dataset == '2048':
        w = 2048
        h = 2048
    img_num = 0
    for data_path in src_dir:
        gtfilepath = data_path + '/gt/result.txt'
        files = os.listdir(data_path + '/img1/')
        img_num += len(files)
        index = [int(i.split('.')[0]) for i in files]
        index.sort()
        img_format = files[0].split('.')[1]
        # JWS
        if dataset == 'JWS' or dataset == '2048':
            files = [str(i).zfill(6) + '.' + img_format for i in index]
        # GF
        if dataset == 'GF':
            files = [str(i) + '.' + img_format for i in index]
        if dataset == '2048':
            train_num = int(len(files) * 2 / 3) # 2048
        else:
            train_num = int(len(files) * 0.75) # else

        i = 0
        with open(gtfilepath) as f1:
            lines = f1.readlines()
            idx = [i.strip('\n').split(',')[0] for i in lines]
            idx = list(set(idx))
            idx = [int(i) for i in idx]
            idx.sort()
            #ini_index = int(lines[0].strip('\n').split(',')[0])
            finish_index = int(index[train_num])
            for line in lines:
                if int(line.strip('\n').split(',')[0]) < finish_index:
                    ls = line.strip('\n').split(',')
                    # f2 = open(des_path[1] + src_dir[i].split('/')[-1][4:] +'_'+ str('%06d'%int(ls[0])) + '_2.txt', 'a')
                    f2 = open(des_path[3] + data_path.split('/')[-1][4:] + '_' + str('%06d' % int(ls[0])) + '.txt', 'a')    # 第几个文件夹第几帧
                    # f2 = open(des_path[1] + src_dir[i].split('/')[-1][4:] + '_' + str('%d' % int(ls[0])) + '.txt', 'a')
                    # with open(des_path[1] + src_dir[i].split('/')[-1][4:] +'_'+ str('%06d'%int(ls[0])) + '.txt', 'w') as f2:
                    l =' '.join(['0']+ [str((int(ls[4])+int(ls[2]))/w/2)] + [str((int(ls[5])+int(ls[3]))/h/2)] + [str(abs(int(ls[4])-int(ls[2]))/w)] + [str(abs(int(ls[5])-int(ls[3]))/h)])
                    print(l)
                    f2.write(l + '\n')
                    f2.close()

                else:        # test
                    ls = line.strip('\n').split(',')
                    # f2 = open(des_path[3] + src_dir[i].split('/')[-1][4:] +'_'+ str('%06d'%int(ls[0])) + '_2.txt', 'a')
                    f2 = open(des_path[1] + data_path.split('/')[-1][4:] + '_' + str('%06d' % int(ls[0])) + '.txt', 'a')
                    # f2 = open(des_path[3] + src_dir[i].split('/')[-1][4:] + '_' + str('%d' % int(ls[0])) + '.txt', 'a')
                    # with open(des_path[1] + src_dir[i].split('/')[-1][4:] +'_'+ str('%06d'%int(ls[0])) + '.txt', 'w') as f2:
                    # result中的坐标是目标实际所处的位置，且是左上和右下坐标，经过下面的处理可以转为归一化后的中心点坐标和宽高
                    # ' '.join表示把元素拼接起来，分隔符为空格(引号间的字符)
                    l =' '.join(['0']+ [str((int(ls[4])+int(ls[2]))/w/2)] + [str((int(ls[5])+int(ls[3]))/h/2)] + [str(abs(int(ls[4])-int(ls[2]))/w)] + [str(abs(int(ls[5])-int(ls[3]))/h)])
                    print(l)
                    f2.write(l + '\n')
                    f2.close()

        for file in files:
            if i < train_num:
                i += 1
                #保存图片
                # print('copy {} to {}'.format(ori_path + '/img1/' + file, des_path[0] + src_dir[i].split('/')[-1][4:]+ '_' + file.split('.')[0]+'.'+file.split('.')[1]))
                print('copy {} to {}'.format(data_path + '/img1/' + file, des_path[2] + data_path.split('/')[-1][4:] + '_' + file.split('.')[0].zfill(6) + '.' + file.split('.')[1]))
                # shutil.copy(ori_path + '/img1/' + file, des_path[0] + src_dir[i].split('/')[-1][4:]+ '_' + file.split('.')[0]+'.'+file.split('.')[1])
                shutil.copy(data_path + '/img1/' + file, des_path[2] + data_path.split('/')[-1][4:] + '_' + file.split('.')[0].zfill(6) + '.' + file.split('.')[1])

            else:
                # print('copy {} to {}'.format(ori_path + '/img1/' + file, des_path[2] + src_dir[i].split('/')[-1][4:]+ '_' + file.split('.')[0]+'.'+file.split('.')[1]))
                print('copy {} to {}'.format(data_path + '/img1/' + file,des_path[0] + data_path.split('/')[-1][4:] + '_' + file.split('.')[0].zfill(6) + '.' + file.split('.')[1]))
                # shutil.copy(ori_path + '/img1/' + file, des_path[2] + src_dir[i].split('/')[-1][4:]+ '_' + file.split('.')[0]+'.'+file.split('.')[1])
                shutil.copy(data_path + '/img1/' + file, des_path[0] + data_path.split('/')[-1][4:] + '_' + file.split('.')[0].zfill(6) + '.' + file.split('.')[1])


    files1 = os.listdir(srcpath + '/test')
    n = 0
    for file in files1:
    #     print(file)
         # 有图片但没有标签
         if not os.path.exists(des_path[1]+file.replace('png','txt').replace('jpg','txt')):
             print(des_path[1]+file.replace('png','txt').replace('jpg','txt'))
             f3 = open(des_path[1]+file.replace('png','txt').replace('jpg','txt'), 'a')
             f3.write('\n')
             f3.close()
             n+=1
    print('loss file num:',n)

    files2 = os.listdir(srcpath + '/train')
    n = 0
    for file in files2:
    #     print(file)
         if not os.path.exists(des_path[3]+file.replace('png','txt').replace('jpg','txt')):
             #print(des_path[3]+file.replace('png','txt').replace('jpg','txt'))
             f4 = open(des_path[3]+file.replace('png','txt').replace('jpg','txt'), 'a')
             f4.write('\n')
             f4.close()
             n+=1
    print('loss file num:',n)
    print('total img_num:',img_num)