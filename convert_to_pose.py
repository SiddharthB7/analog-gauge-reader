from ultralytics import YOLO
import cv2, os
import numpy as np

# This script automates the conversion of YOLOv8 gauge detection results and keypoint labels
# into cropped images and YOLOv8-Pose label format for training a pose model.

# What it does:
# 1. Loads your YOLOv8 gauge detection model.
# 2. For each image in train/valid/test splits:
#    - Detects the gauge and crops the image to the gauge region.
#    - Saves the crop to crops/{split}/images/.
#    - Reads the corresponding keypoint label file (from labels_old).
#    - Extracts keypoints (center, max, min, tip) and adjusts them to the crop coordinates.
#    - Writes a new label file in YOLOv8-Pose format to crops/{split}/labels/.
# 3. Handles missing detections, missing keypoints, and missing label files gracefully.
# 4. Warns if existing crops will be overwritten.

# In summary:
# - This script prepares your dataset for YOLOv8-Pose training by cropping images and converting keypoint labels.
# - It ensures all images and labels are in the correct format and location for pose training.

# Update model path and check if it exists
model_path = 'runs/detect/gauge-detect6/weights/best.pt'
if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    print("Available models:")
    for dir in os.listdir('runs/detect'):
        weight_path = f"runs/detect/{dir}/weights/best.pt"
        if os.path.exists(weight_path):
            print(f"  - {weight_path}")
    exit(1)

print(f"Loading model from {model_path}")
model = YOLO(model_path)

# Define correct split folder names - 'valid' instead of 'val'
splits = ['train', 'valid', 'test']

# Warning about overwriting
if os.path.exists("crops") and any(os.listdir(f"crops/{s}/images") for s in splits if os.path.exists(f"crops/{s}/images")):
    print("WARNING: Existing crops will be overwritten!")
    confirm = input("Continue? (y/n): ")
    if confirm.lower() != 'y':
        print("Operation cancelled.")
        exit(0)

for split in splits:
    # Check if this split directory exists
    if not os.path.exists(f"{split}/images"):
        print(f"Skipping {split}/images - directory not found")
        continue
        
    print(f"Processing {split} set...")
    os.makedirs(f"crops/{split}/images", exist_ok=True)
    os.makedirs(f"crops/{split}/labels", exist_ok=True)
    
    # Process each image in the split directory
    for imgfile in os.listdir(f"{split}/images"):
        # Skip non-image files
        if not imgfile.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
            
        img = cv2.imread(f"{split}/images/{imgfile}")
        if img is None:
            print(f"Warning: Could not read {split}/images/{imgfile}")
            continue
            
        results = model(img, conf=0.5, verbose=False)
        if results and results[0].boxes and len(results[0].boxes) > 0:
            # take highest‑conf box
            box = results[0].boxes[0].xyxy.cpu().numpy().astype(int)[0]
            x1, y1, x2, y2 = box
            
            # Ensure box coordinates are within image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                print(f"Warning: Empty crop for {imgfile}")
                continue
                
            cv2.imwrite(f"crops/{split}/images/{imgfile}", crop)
            
            # Process the corresponding label file - use labels_old instead of labels
            label_file = os.path.splitext(imgfile)[0] + '.txt'
            label_path = f"{split}/labels_old/{label_file}"
            
            # Removed fallback logic - only look in labels_old
            
            if os.path.exists(label_path):
                try:
                    with open(label_path, 'r') as f:
                        label_lines = f.readlines()
                    
                    # Process all lines and extract keypoints by class ID
                    keypoints_dict = {}
                    for line in label_lines:
                        parts = line.strip().split()
                        if len(parts) != 5:  # Standard YOLO format: class x_center y_center width height
                            print(f"Warning: Unexpected label format: {line.strip()}")
                            continue
                            
                        class_id = int(parts[0])
                        x_center = float(parts[1]) 
                        y_center = float(parts[2])
                        
                        # Skip gauge class (1)
                        if class_id == 1:
                            continue
                            
                        # Store keypoint coordinates with class ID as key
                        # Map: 0=center, 2=max, 3=min, 4=tip
                        keypoints_dict[class_id] = (x_center, y_center)
                    
                    # Check if we have all needed keypoints (center, max, min, tip)
                    required_keypoints = [0, 2, 3, 4]
                    if all(kp in keypoints_dict for kp in required_keypoints):
                        # Create a new label line in YOLO Pose format
                        new_keypoints = []
                        
                        # Add keypoints in order: center, max, min, tip
                        for kp_id in required_keypoints:
                            x, y = keypoints_dict[kp_id]
                            
                            # Adjust keypoint coordinates relative to the crop
                            adj_x = (x * img.shape[1] - x1) / (x2 - x1)
                            adj_y = (y * img.shape[0] - y1) / (y2 - y1)
                            
                            # Ensure keypoints are within [0,1] range
                            adj_x = max(0, min(1, adj_x))
                            adj_y = max(0, min(1, adj_y))
                            
                            # Add visibility (2 = visible)
                            new_keypoints.extend([f"{adj_x:.6f}", f"{adj_y:.6f}", "2"])
                        
                        # Compute new box for the POSE crop
                        new_xc = 0.5
                        new_yc = 0.5
                        new_w = 1.0
                        new_h = 1.0
                        bbox_str = f"{new_xc:.6f} {new_yc:.6f} {new_w:.6f} {new_h:.6f}"
                        
                        # Create label line with class_id always 0 for YOLOv8-Pose
                        new_line = f"0 {bbox_str} " + " ".join(new_keypoints) + "\n"
                        
                        # Write to output file
                        with open(f"crops/{split}/labels/{label_file}", 'w') as f:
                            f.write(new_line)
                    else:
                        missing = [kp for kp in required_keypoints if kp not in keypoints_dict]
                        print(f"Warning: Missing required keypoints {missing} for {imgfile} - skipping label")
                except Exception as e:
                    print(f"Error processing label for {imgfile}: {e}")
            else:
                print(f"Warning: No label file found for {imgfile}")
        else:
            print(f"No detection for {imgfile}")
            
print("Conversion completed!")



