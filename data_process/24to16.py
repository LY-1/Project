import time

import cv2
import numpy as np
import numpngw
import os

#data_path = ['data1', 'data2', 'data3', 'data4', 'data5', 'data7', 'data8', 'data9', 'data10',
#             'data12', 'data13', 'data14', 'data15', 'data16', 'data17', 'data19', 'data20', 'data21', 'data22',
#             'data24', 'data25']
# data_path = ['data01','data02','data03','data04','data05','data06','data07','data08','data09','data10','data11','data12','data13',
#              'data14','data15','data16','data17','data18','data19',
#               'data20','data21','data22']
data_path = os.listdir(r'/home/junjzhan/LY/Infrared_project/data/test/20210125/2021.1.25V6/sourcedata')

# data_path = ['data1']

for data in data_path:
    root_path = "/home/junjzhan/LY/Infrared_project/data/test/20210125/2021.1.25V6/sourcedata/{}/img1".format(data)
    img_all = os.listdir(root_path)
    #print(img_all)
    start = time.time()
    for img in img_all:
        img_path = os.path.join(root_path,img)
        i = cv2.imread(img_path,0)


        i = np.array(i, dtype='uint8')
        # i *= 256
        numpngw.write_png(img_path, i)
    end = time.time()
    print(data ,"所用时间", (end - start))
