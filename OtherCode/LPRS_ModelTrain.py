from ultralytics import YOLO

def ModelTrain():
    model = YOLO("yolov8n.pt")
    model.train(data = "LPRS.yaml", epochs = 10)

if __name__ == "__main__":
    ModelTrain()