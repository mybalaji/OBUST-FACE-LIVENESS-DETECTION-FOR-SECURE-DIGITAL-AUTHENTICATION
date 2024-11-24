from ultralytics import YOLO


model = YOLO('yolov8n-cls.pt')
def main():
    model.train(data=r'C:\Users\balla\VSMAIN\Dataset\crop_face', epochs=20)

if __name__ == '__main__':
    main()

