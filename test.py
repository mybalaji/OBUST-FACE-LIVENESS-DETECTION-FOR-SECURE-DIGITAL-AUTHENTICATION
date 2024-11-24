import cv2
from ultralytics import YOLO
import numpy as np

# Load the trained YOLO model for real/fake classification
model = YOLO(r"C:\Users\balla\VSMAIN\runs\classify\train2\weights\best.pt")

# Load OpenCV's pre-trained face detection model
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Initialize the webcam
cap = cv2.VideoCapture(0)
classNames = ["Fake","Real"]

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the frame for face detection
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Process each detected face
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Filter weak detections
            # Get face bounding box
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (left, top, right, bottom) = box.astype("int")

            # Extract face region
            face_image = frame[top:bottom, left:right]

            # Convert to RGB and save temporarily for YOLO model input
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite("temp_face.jpg", face_rgb)

            # Predict using YOLO model
            results = model("temp_face.jpg")
            # classNames = results[0].names
            # print(class_label)
            # print(results[0].probs)
            class_label = classNames[np.argmax(results[0].probs.data.tolist())]
            # yolo_confidence = results[

            # Display results on the frame
            label = f"{class_label}"
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the output frame
    cv2.imshow('Webcam Feed', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


