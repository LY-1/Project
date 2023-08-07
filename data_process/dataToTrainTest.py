##深度学习过程中，需要制作训练集和验证集、测试集。
import os, random, shutil
# set val_rate
def moveFile(fileDir, tarDir, dataset):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    if dataset == '2048':
        rate = 0.25  # 2048 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    else:
        rate = 0.3    # else 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片+

    print(sample)
    for name in sample:
        shutil.move(fileDir + name, tarDir + name)
    return

def move(path, dataset):
    root_path = path
    fileDir = root_path + '/train/'  # 源图片文件夹路径  929张  929*0.6=557   929*0.2=185    929*0.2=185
    tarDir = root_path + '/val/'  # 移动到新的文件夹路径
    if os.path.exists(tarDir):
        shutil.rmtree(tarDir)
    os.makedirs(tarDir)
    moveFile(fileDir, tarDir, dataset)


if __name__ == '__main__':
    root_path = r'/home/junjzhan/LY/Infrared_project_v2/6.6_test'
    fileDir = root_path + '/train/'  # 源图片文件夹路径  929张  929*0.6=557   929*0.2=185    929*0.2=185
    tarDir = root_path + '/val/'  # 移动到新的文件夹路径
    if not os.path.exists(tarDir):
        os.makedirs(tarDir)
    moveFile(fileDir, tarDir)