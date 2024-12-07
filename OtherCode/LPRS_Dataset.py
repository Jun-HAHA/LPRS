import os
import random
from PIL import Image

# 获取原始数据
def OriData(ImgData):
    Data1 = ImgData.split("_")
    OD = []
    for i in Data1:
        for j in i.split("&"):
            OD.append(j)
    return list(map(int, OD))

# 数据归一化
def Normalization(PicSize, OriData):
    dx = OriData[2] - OriData[0]
    dy = OriData[3] - OriData[1]
    CenterX = (OriData[2] + OriData[0]) / 2.0
    CenterY = (OriData[3] + OriData[1]) / 2.0
    dw = 1./PicSize[0]
    dh = 1./PicSize[1]
    TargetX = CenterX * dw
    TargetY = CenterY * dh
    TargetW = dx * dw
    TargetH = dy * dh
    return TargetX, TargetY, TargetW, TargetH

# 生成图片标签文件
def LabelTXT(PicName, PicIndex, TargetPath):
    img = Image.open(PicName)
    label = PicName.split("/")[1].split("_")[1]
    ImgData = Normalization(img.size, OriData(PicName.split("/")[-1].split("-")[2]))
    if label == "green":
        name = "green_{c}_{p}".format(c = TargetPath, p = PicIndex)
        TargetStr = "{l} {x} {y} {w} {h}".format(l = 1, x = ImgData[0], y = ImgData[1], w = ImgData[2], h = ImgData[3])
    else:
        name = "blue_{c}_{f}_{i}".format(c = label,f = TargetPath, i = PicIndex)
        TargetStr = "{l} {x} {y} {w} {h}".format(l = 0, x = ImgData[0], y = ImgData[1], w = ImgData[2], h = ImgData[3])
    img.save("LPRS_Dataset/{t}/images/{p}.jpg".format(t = TargetPath, p = name))
    with open("LPRS_Dataset/{t1}/labels/{t2}.txt".format(t1 = TargetPath, t2 = name), "w+") as f:
        f.write(TargetStr)
        f.close()

# 指定生成数据集
def Dataset(path, n, target):
    PicIndex = 0
    for i in random.sample(os.listdir(path), n):
        PicIndex += 1
        Path = "{c}/{p}".format(c = path, p = i)
        LabelTXT(Path, PicIndex, target)

# 随机生成数据集
def RandomDataset():
    GreenPath = "CCPD2020/ccpd_green"
    BlueBasePath = "CCPD2019/ccpd_base"
    BlueBlurPath = "CCPD2019/ccpd_blur"
    BlueChallengePath = "CCPD2019/ccpd_challenge"
    BlueDbPath = "CCPD2019/ccpd_db"
    BlueFnPath = "CCPD2019/ccpd_fn"
    BlueRotatePath = "CCPD2019/ccpd_rotate"
    BlueTiltPath = "CCPD2019/ccpd_tilt"
    BlueWeatherPath = "CCPD2019/ccpd_weather"

    Dataset(BlueBlurPath, 50, "train")
    Dataset(BlueChallengePath, 50, "train")
    Dataset(BlueBlurPath, 50, "train")
    Dataset(BlueDbPath, 20, "train")
    Dataset(BlueFnPath, 20, "train")
    Dataset(BlueRotatePath, 20, "train")
    Dataset(BlueTiltPath, 20, "train")
    Dataset(BlueWeatherPath, 20, "train")

    Dataset(BlueBlurPath, 20, "val")
    Dataset(BlueChallengePath, 20, "val")
    Dataset(BlueBlurPath, 20, "val")
    Dataset(BlueDbPath, 10, "val")
    Dataset(BlueFnPath, 10, "val")
    Dataset(BlueRotatePath, 10, "val")
    Dataset(BlueTiltPath, 10, "val")
    Dataset(BlueWeatherPath, 10, "val")

if __name__ == "__main__":
    RandomDataset()