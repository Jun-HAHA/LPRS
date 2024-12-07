from ultralytics import YOLO
from PIL import Image
import cv2
import os
import random

def GetRandomPic():
    Path = random.sample(os.listdir("LPRS_Dataset/train/images"), 1)

    return "LPRS_Dataset/train/images/{}".format(Path[0])

def ModelTest():
    model = YOLO("runs/detect/train/weights/best.pt")
    TestPic = GetRandomPic()
    res = model.predict(TestPic)
    img = cv2.imread(TestPic, flags=1)

    x = int(list(res[0].boxes.xywh[0])[0])
    y = int(list(res[0].boxes.xywh[0])[1])
    w = int(list(res[0].boxes.xywh[0])[2])
    h = int(list(res[0].boxes.xywh[0])[3])
    print(res[0].boxes.xywh)
    print(x,y,w,h)
    CardPic = img[y-h//2:y+h//2, x-w//2:x+w//2]
    ci = Image.fromarray(CardPic)

    cv2.imshow("pre", res[0].plot())
    cv2.imshow("card", CardPic)
    cv2.waitKey(0)

if __name__ == "__main__":
    ModelTest()