import os
import cv2
from pathlib import Path

# Paths for the dataset
dataset_path = 'Dataset/all'  # Folder with both images and annotations
cropped_path = 'dataset/cropped_faces'      # Directory for cropped images

# Create directories for cropped images
real_cropped_dir = Path(cropped_path) / 'real'
fake_cropped_dir = Path(cropped_path) / 'fake'
real_cropped_dir.mkdir(parents=True, exist_ok=True)
fake_cropped_dir.mkdir(parents=True, exist_ok=True)

# Supported image file extensions
image_extensions = {".jpg", ".jpeg", ".png"}

# Function to crop and save face from image
def crop_and_save_face(image_path, label_path, cropped_dir, class_id):
    # Load the image
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    
    # Read label file to get bounding box
    with open(label_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            label_class_id = int(data[0])

            if label_class_id == class_id:
                # Get bounding box details (normalized)
                x_center, y_center, width, height = map(float, data[1:])
                
                # Convert to pixel values
                x_center, y_center = int(x_center * w), int(y_center * h)
                width, height = int(width * w), int(height * h)
                
                # Calculate top-left and bottom-right coordinates
                x1, y1 = x_center - width // 2, y_center - height // 2
                x2, y2 = x_center + width // 2, y_center + height // 2

                # Crop the face from the image
                cropped_face = image[y1:y2, x1:x2]

                # Save cropped face image
                image_name = Path(image_path).stem
                save_path = cropped_dir / f"{image_name}_cropped.jpg"
                if not cropped_face.any():
                    return False
                cv2.imwrite(str(save_path), cropped_face)

# Iterate through files in the dataset folder
for file in Path(dataset_path).glob("*"):
    if file.suffix in image_extensions:
        # If it's an image, get the corresponding annotation file
        image_file = file
        label_file = file.with_suffix(".txt")
        
        if label_file.exists():
            # Crop and save "real" faces
            crop_and_save_face(str(image_file), str(label_file), real_cropped_dir, class_id=1)
            
            # Crop and save "fake" faces
            crop_and_save_face(str(image_file), str(label_file), fake_cropped_dir, class_id=0)

print("Cropped dataset created successfully.")

