from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os

class GaugeDetector:
    """
    Detects gauges in images and extracts keypoint coordinates
    """
    def __init__(self, detect_model_path, pose_model_path):
        """
        Initialize the gauge detector with detection and pose models
        
        Args:
            detect_model_path: Path to the detection model
            pose_model_path: Path to the pose model
        """
        self.detect_model_path = detect_model_path
        self.pose_model_path = pose_model_path
        
        # Load models
        print("Loading detection model...")
        self.detect_model = YOLO(f"{detect_model_path}/weights/best.pt")
        
        print("Loading pose model...")
        self.pose_model = YOLO(f"{pose_model_path}/weights/best.pt")
        
        print("Models loaded successfully!")
        
    def detect_gauge(self, img_path, visualize=False):
        """
        Detect gauge in image and return cropped image and box coordinates
        
        Args:
            img_path: Path to the image or image array
            visualize: Whether to visualize the detection
            
        Returns:
            Dictionary with cropped image and detection box
        """
        # Load image if path is provided
        if isinstance(img_path, str):
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error: Could not read image at {img_path}")
                return None
        else:
            img = img_path  # Assume image array was passed
            
        # Detect gauge
        detect_results = self.detect_model(img, conf=0.5)
        
        if not detect_results or len(detect_results[0].boxes) == 0:
            print("No gauge detected in the image.")
            return None
        
        # Get the highest confidence detection
        box = detect_results[0].boxes[0].xyxy.cpu().numpy().astype(int)[0]
        x1, y1, x2, y2 = box
        
        # Ensure box coordinates are within image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        
        # Crop the image to the gauge
        gauge_crop = img[y1:y2, x1:x2]
        
        if visualize:
            # Visualize detection
            plt.figure(figsize=(8, 8))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                          fill=False, edgecolor='lime', linewidth=2))
            plt.title("Gauge Detection")
            plt.axis('off')
            plt.show()
            
        return {
            "original_img": img,
            "crop": gauge_crop,
            "box": (x1, y1, x2, y2)
        }
        
    def extract_keypoints(self, crop, visualize=False):
        """
        Extract keypoints from a cropped gauge image
        
        Args:
            crop: Cropped gauge image
            visualize: Whether to visualize the keypoints
            
        Returns:
            Dictionary with keypoint coordinates
        """
        # Detect keypoints
        pose_results = self.pose_model(crop)
        
        if not pose_results or len(pose_results[0].keypoints) == 0:
            print("No keypoints detected in the cropped gauge.")
            return None
            
        # Get keypoints (in crop coordinates)
        keypoints_array = pose_results[0].keypoints.xy.cpu().numpy()[0]
        
        # Convert to normalized coordinates
        h, w = crop.shape[:2]
        keypoints_norm = keypoints_array.copy()
        keypoints_norm[:, 0] = keypoints_array[:, 0] / w
        keypoints_norm[:, 1] = keypoints_array[:, 1] / h
        
        # Convert to dictionary format
        keypoint_names = ['center', 'max', 'min', 'tip']
        keypoints = {}
        keypoints_raw = {}
        
        for i, name in enumerate(keypoint_names):
            if i < len(keypoints_array):
                # Store original pixel coordinates
                keypoints_raw[name] = (float(keypoints_array[i][0]), float(keypoints_array[i][1]))
                # Store normalized coordinates with visibility flag
                keypoints[name] = (
                    float(keypoints_norm[i][0]),
                    float(keypoints_norm[i][1]),
                    1.0  # Visibility flag (1.0 = visible)
                )
                
        if visualize:
            # Visualize keypoints
            plt.figure(figsize=(8, 8))
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            plt.imshow(crop_rgb)
            
            # Colors for different keypoints
            colors = ['red', 'green', 'blue', 'yellow']
            
            # Draw keypoints
            center_x, center_y = None, None
            for i, (name, color) in enumerate(zip(keypoint_names, colors)):
                if i < len(keypoints_array):
                    x, y = keypoints_array[i]
                    plt.plot(x, y, 'o', color=color, markersize=10, label=name)
                    
                    # Store center coordinates for drawing lines
                    if name == 'center':
                        center_x, center_y = x, y
            
            # Draw lines from center to each keypoint
            if center_x is not None:
                for i, name in enumerate(keypoint_names):
                    if i > 0 and i < len(keypoints_array):  # Skip center itself
                        x, y = keypoints_array[i]
                        plt.plot([center_x, x], [center_y, y], '-', color=colors[i], linewidth=2)
                        
            plt.title("Keypoint Detection")
            plt.legend()
            plt.axis('off')
            plt.show()
                
        return {
            "keypoints": keypoints,
            "keypoints_raw": keypoints_raw,
            "keypoint_names": keypoint_names
        }
        
    def process_image(self, img_path, visualize=False):
        """
        Process image to detect gauge and extract keypoints
        
        Args:
            img_path: Path to the image
            visualize: Whether to visualize results
            
        Returns:
            Dictionary with detection and keypoint results
        """
        # Detect gauge and get cropped image
        detection = self.detect_gauge(img_path, visualize=visualize)
        if not detection:
            return None
            
        # Extract keypoints
        keypoints = self.extract_keypoints(detection["crop"], visualize=visualize)
        if not keypoints:
            return None
            
        # Combine results
        return {
            "original_img": detection["original_img"],
            "crop": detection["crop"],
            "box": detection["box"],
            "keypoints": keypoints["keypoints"],
            "keypoints_raw": keypoints["keypoints_raw"],
            "keypoint_names": keypoints["keypoint_names"]
        }

# Test function
def test_detector(img_path=None):
    # Paths to models
    detect_model_path = r"C:\Users\siddharth\OneDrive\ドキュメント\gauge-analog.v1i.yolov11\runs\detect\gauge-detect6"
    pose_model_path = r"C:\Users\siddharth\OneDrive\ドキュメント\gauge-analog.v1i.yolov11\runs\pose\gauge-pose-cuda"
    
    # Create detector
    detector = GaugeDetector(detect_model_path, pose_model_path)
    
    if not img_path:
        # Use default test image
        img_path = r"C:\Users\siddharth\OneDrive\ドキュメント\gauge-analog.v1i.yolov11\bevanimage.jpg"
        
    print(f"Testing with image: {img_path}")
    
    # Process image
    result = detector.process_image(img_path, visualize=True)
    
    if result:
        print("\nKeypoint Coordinates (normalized):")
        for name, (x, y, _) in result["keypoints"].items():
            print(f"{name}: ({x:.4f}, {y:.4f})")
    else:
        print("Processing failed.")

if __name__ == "__main__":
    test_detector()
