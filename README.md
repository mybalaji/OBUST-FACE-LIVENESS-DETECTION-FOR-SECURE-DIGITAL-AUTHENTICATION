# ROBUST-FACE-LIVENESS-DETECTION-FOR-SECURE-DIGITAL-AUTHENTICATION

I Have created a dataset using facedetect.py with classId = 0 as Fake and ClassId = 1 as Real and create Real and Fake Folders to move these collected files with labels to move to their respective folders and crop the faces using create_dataset.py file and then train the images collected and then test it with your webcam for results.

Workflow Summary:
1. Data Collection:

Used facedetect.py to collect images.
Labeled collected images with classId = 0 for "Fake" and classId = 1 for "Real."
Organized images into respective folders (Real and Fake).

2. Preprocessing:

Used create_dataset.py to crop the detected faces from the collected images.
Ensured that the images are properly labeled and formatted for model training.

3. Training:

Trained a YOLO model (or another chosen model) using the preprocessed dataset.
Set up your training pipeline to classify images as Real or Fake.

4. Testing with Webcam:

Integrated webcam input into your pipeline.
Tested the trained model to classify live feed inputs as Real or Fake faces.
