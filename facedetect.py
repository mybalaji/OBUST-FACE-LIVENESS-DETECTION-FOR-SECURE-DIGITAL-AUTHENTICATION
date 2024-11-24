from time import time
import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector

####################################
classID = 1 # 0 is fake and 1 is real
outputFolderPath = 'Dataset/DataCollect'
confidence = 0.8
save = True
blurThreshold = 35  # Larger is more focus

debug = False
offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight = 640, 480
floatingPoint = 6
####################################

cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)

detector = FaceDetector()

while True:
    success, img = cap.read()
    imgOut = img.copy()
    img, bboxs = detector.findFaces(img, draw=False)

    listBlur = []  # True False values indicating if the faces are blur or not
    listInfo = []  # The normalized values and the class name for the label txt file

    if bboxs:
        print(f"Detected faces: {len(bboxs)}")  # Check how many faces detected

        for i, bbox in enumerate(bboxs):
            print(f"Processing face {i + 1} with bbox: {bbox['bbox']}")  # Debug bbox

            x, y, w, h = bbox["bbox"]
            score = bbox["score"][0]

            # ------ Check the score --------
            if score > confidence:
                # Adding offset
                offsetW = (offsetPercentageW / 100) * w
                x = int(x - offsetW)
                w = int(w + offsetW * 2)
                offsetH = (offsetPercentageH / 100) * h
                y = int(y - offsetH * 3)
                h = int(h + offsetH * 3.5)

                # Avoid negative or out-of-bounds values
                x = max(0, x)
                y = max(0, y)
                w = max(0, w)
                h = max(0, h)

                # Find Blurriness
                imgFace = img[y:y + h, x:x + w]
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                if blurValue > blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)

                # Normalize Values
                ih, iw, _ = img.shape
                xc, yc = x + w / 2, y + h / 2
                xcn, ycn = round(xc / iw, floatingPoint), round(yc / ih, floatingPoint)
                wn, hn = round(w / iw, floatingPoint), round(h / ih, floatingPoint)

                # Avoid values above 1
                xcn = min(1, xcn)
                ycn = min(1, ycn)
                wn = min(1, wn)
                hn = min(1, hn)

                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")

                # Drawing rectangles on each face
                cv2.rectangle(imgOut, (x, y), (x + w, y + h), (255, 0, 0), 3)
                cvzone.putTextRect(imgOut, f'Score: {int(score * 100)}% Blur: {blurValue}', (x, y - 10),
                                   scale=2, thickness=3)

                if debug:
                    cv2.rectangle(img, (x, y, w, h), (255, 0, 0), 3)
                    cvzone.putTextRect(img, f'Score: {int(score * 100)}% Blur: {blurValue}', (x, y - 10),
                                       scale=2, thickness=3)

        # ------ Save the detected faces and label data --------
        if save:
            if all(listBlur) and listBlur != []:
                # Save Image
                timeNow = str(time()).replace('.', '')
                cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", img)  # Save imgOut instead of img

                # Save Label Text File
                with open(f"{outputFolderPath}/{timeNow}.txt", 'a') as f:
                    for info in listInfo:
                        f.write(info)

    cv2.imshow("Image", imgOut)

    # Press 'q' to break the loop and stop the camera
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
