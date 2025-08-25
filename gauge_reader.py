import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import our custom modules
from min_max_unit import GaugeDetector
from gauge_scale_reader import GaugeScaleReader

class GaugeReader:
    def __init__(self, detect_model_path, pose_model_path, use_gpu=True):
        """Initialize models for gauge reading"""
        print("Initializing gauge reading system...")
        
        # Check GPU availability
        import torch
        has_gpu = torch.cuda.is_available()
        if use_gpu and not has_gpu:
            print("Warning: GPU requested but not available. Using CPU instead.")
            use_gpu = False
        else:
            print(f"Using GPU: {torch.cuda.get_device_name(0) if has_gpu and torch.cuda.is_available() else 'None'}")
            
        # Initialize detector for gauge and keypoints
        print("Initializing gauge detector...")
        self.detector = GaugeDetector(detect_model_path, pose_model_path)
        
        # Initialize scale reader for min/max values
        print("Initializing scale reader...")
        self.scale_reader = GaugeScaleReader()
        
        print("All components initialized successfully!")
    
    def process_image(self, img_path, visualize=True, confirm_scale=True):
        """Process a gauge image and calculate reading"""
        print(f"\nProcessing image: {img_path}")
        
        # Step 1: Detect gauge and extract keypoints
        print("Detecting gauge and keypoints...")
        detection_result = self.detector.process_image(img_path, visualize=False)
        
        if not detection_result:
            print("Failed to detect gauge or keypoints.")
            return None
            
        # Get components from detection
        original_img = detection_result["original_img"]
        crop = detection_result["crop"]
        box = detection_result["box"]
        keypoints = detection_result["keypoints"]
        keypoints_raw = detection_result["keypoints_raw"]
        keypoint_names = detection_result["keypoint_names"]
        
        # Step 2: Read scale (min, max, unit)
        print("Reading gauge scale...")
        min_val, max_val, unit, ocr_boxes = self.scale_reader.read_scale(crop, visualize=False)
        if min_val is None or max_val is None:
            print("Could not determine scale - skipping visualization and calculation.")
            return None

        # Only now is it safe to calculate reading
        reading = self.calculate_gauge_reading(keypoints, min_val, max_val)

        if visualize:
            self.visualize_results(original_img, crop, box, keypoints_raw,
                                keypoint_names, reading, min_val, max_val, unit, ocr_boxes)


        # Step 3: User confirmation of scale values
        if confirm_scale:
            print(f"\nDetected scale: {min_val} to {max_val} {unit or ''}")
            is_correct = input("Is this scale correct? (y/n): ").strip().lower()
            
            if is_correct != 'y':
                try:
                    min_val = float(input("Enter correct minimum value: "))
                    max_val = float(input("Enter correct maximum value: "))
                    unit = input("Enter unit (leave blank if none): ").strip() or None
                except ValueError:
                    print("Invalid input. Using detected values.")
                    
        # Step 4: Calculate gauge reading
        reading = self.calculate_gauge_reading(keypoints, min_val, max_val)
        
        # Step 5: Visualize results
        if visualize:
            self.visualize_results(original_img, crop, box, keypoints_raw, 
                                  keypoint_names, reading, min_val, max_val, unit)
            
        return {
            'reading': reading,
            'min_val': min_val,
            'max_val': max_val,
            'unit': unit,
            'keypoints': keypoints
        }
    
    def calculate_gauge_reading(self, keypoints, min_val=0, max_val=300):
        """
        Calculate gauge reading based on the angle between min and tip vectors,
        relative to the total min-to-max sweep angle.
        """
        import numpy as np

        # unpack normalized coords
        x_c,  y_c,  _ = keypoints['center']
        x_min,y_min,_ = keypoints['min']
        x_tip,y_tip,_ = keypoints['tip']
        x_max,y_max,_ = keypoints['max']

        # Build vectors from center to each keypoint
        v_min = np.array([x_min - x_c, y_min - y_c])
        v_max = np.array([x_max - x_c, y_max - y_c])
        v_tip = np.array([x_tip - x_c, y_tip - y_c])

        # Get absolute angles in radians (clockwise from positive x-axis)
        def get_angle(v):
            # Convert to clockwise angle from positive x-axis
            angle = -np.arctan2(v[1], v[0])
            # Ensure angle is between 0 and 2π
            if angle < 0:
                angle += 2 * np.pi
            return angle

        # Get absolute angles (clockwise from positive x-axis)
        angle_min = get_angle(v_min)
        angle_tip = get_angle(v_tip)
        angle_max = get_angle(v_max)

        # Adjust angles to ensure they are in correct order for clockwise measurement
        # On most gauges min is left, tip is variable, max is right
        # Make sure we measure clockwise from min to tip
        tip_from_min = (angle_tip - angle_min) % (2 * np.pi)
        max_from_min = (angle_max - angle_min) % (2 * np.pi)

        # Calculate fraction of full scale
        # If max_from_min is very small, the gauge might be a full circle
        if max_from_min < 0.1:
            max_from_min = 2 * np.pi  # Assume full circle
        
        frac = tip_from_min / max_from_min
        
        # Clamp fraction to [0, 1] range
        frac = max(0, min(frac, 1.0))
        
        # Map fraction to reading
        reading = min_val + frac * (max_val - min_val)
        
        # Print debugging info
        print(f"DEBUG: min angle={np.degrees(angle_min):.1f}°, "
              f"tip angle={np.degrees(angle_tip):.1f}°, "
              f"max angle={np.degrees(angle_max):.1f}°")
        print(f"DEBUG: tip_from_min={np.degrees(tip_from_min):.1f}°, "
              f"max_from_min={np.degrees(max_from_min):.1f}°")
        print(f"DEBUG: fraction={frac:.3f}, reading={reading:.1f}")
        
        return reading
    def calculate_gauge_reading(self, keypoints, min_val=0, max_val=300):
        """
        Calculate gauge reading as the fraction of the angle swept by the needle (tip) from min,
        divided by the total angle between min and max, mapped to the scale range.
        """
        # Unpack normalized coordinates
        x_c,  y_c,  _ = keypoints['center']
        x_min, y_min, _ = keypoints['min']
        x_max, y_max, _ = keypoints['max']
        x_tip, y_tip, _ = keypoints['tip']

        # Build vectors from center to each keypoint
        v_min = np.array([x_min - x_c, y_min - y_c])
        v_max = np.array([x_max - x_c, y_max - y_c])
        v_tip = np.array([x_tip - x_c, y_tip - y_c])

        def get_angle(v):
            # Angle from positive x-axis, in [0, 2pi)
            angle = np.arctan2(v[1], v[0])
            if angle < 0:
                angle += 2 * np.pi
            return angle

        # Get angles for min, max, tip
        angle_min = get_angle(v_min)
        angle_max = get_angle(v_max)
        angle_tip = get_angle(v_tip)

        # Always measure sweep in the same direction (counterclockwise)
        sweep = (angle_max - angle_min) % (2 * np.pi)
        tip_sweep = (angle_tip - angle_min) % (2 * np.pi)

        # If sweep is very small, assume full circle
        if sweep < 1e-3:
            sweep = 2 * np.pi

        # Calculate fraction
        frac = tip_sweep / sweep
        # Clamp to [0, 1]
        frac = max(0.0, min(frac, 1.0))

        # Map to scale
        reading = min_val + frac * (max_val - min_val)

        # Debug output
        print("\n--- GAUGE READING DEBUG ---")
        print(f"Keypoints:")
        print(f"  center: ({x_c:.3f}, {y_c:.3f})")
        print(f"  min:    ({x_min:.3f}, {y_min:.3f})  angle={np.degrees(angle_min):.1f}°")
        print(f"  max:    ({x_max:.3f}, {y_max:.3f})  angle={np.degrees(angle_max):.1f}°")
        print(f"  tip:    ({x_tip:.3f}, {y_tip:.3f})  angle={np.degrees(angle_tip):.1f}°")
        print(f"Sweep (min→max): {np.degrees(sweep):.1f}°")
        print(f"Tip from min:    {np.degrees(tip_sweep):.1f}°")
        print(f"Fraction of scale: {frac:.3f}")
        print(f"Reading: {reading:.2f} (scale {min_val} to {max_val})")
        print("--------------------------\n")

        return reading
        
    def angle(self, p, q):
        """Calculate angle between two points (in radians)"""
        return math.atan2(q[1] - p[1], q[0] - p[0])
    
    def interp(self, val, a, b, A, B):
        """Linear interpolation: map val from range [a,b] to range [A,B]"""
        # Handle edge cases: val outside [a,b] range
        if a == b:  # Avoid division by zero
            return (A + B) / 2
        
        mapped = A + ((val-a) / (b-a)) * (B-A)
        
        # Clamp the result to [A,B]
        return max(min(mapped, max(A, B)), min(A, B))
    
    def visualize_results(self, img, crop, box, keypoints, keypoint_names, reading, min_val, max_val, unit, ocr_boxes=None):
        x1, y1, x2, y2 = box

        plt.figure(figsize=(12, 6))

        # 1. Original image with gauge detection
        plt.subplot(1, 3, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                        fill=False, edgecolor='lime', linewidth=2))
        plt.title("Gauge Detection")
        plt.axis('off')

        # 2. Cropped gauge with keypoints and OCR boxes
        plt.subplot(1, 3, 2)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        plt.imshow(crop_rgb)

        # Draw keypoints (original code)
        colors = ['red', 'green', 'blue', 'yellow']
        center_coords = None
        for name, color in zip(keypoint_names, colors):
            if name in keypoints:
                x, y = keypoints[name]
                plt.plot(x, y, 'o', color=color, markersize=10, label=name)
                if name == 'center':
                    center_coords = (x, y)
        if center_coords:
            cx, cy = center_coords
            for name, color in zip(keypoint_names[1:], colors[1:]):
                if name in keypoints:
                    x, y = keypoints[name]
                    plt.plot([cx, x], [cy, y], '-', color=color, linewidth=2)

        # --- DRAW OCR BOXES ON THE CROPPED GAUGE ---
        if ocr_boxes:
            for text, pts, conf in ocr_boxes:
                pts_np = np.array(pts + [pts[0]]) # closes box
                plt.plot(pts_np[:,0], pts_np[:,1], color='orange', linewidth=2)
                x, y = pts[0]
                plt.text(x, y-7, text, color='orange', fontsize=8, bbox=dict(facecolor='white', alpha=0.65, boxstyle='round'))

        plt.title("Keypoint & OCR Detection")
        plt.legend()
        plt.axis('off')

        # 3. Text info panel (unchanged)
        plt.subplot(1, 3, 3)
        plt.axis('off')
        info_text = (
            f"GAUGE READING RESULTS\n\n"
            f"Reading: {reading:.2f} {unit or ''}\n"
            f"Scale: {min_val} to {max_val} {unit or ''}\n\n"
            f"Keypoint Coordinates:\n"
        )
        for name in keypoint_names:
            if name in keypoints:
                x, y = keypoints[name]
                info_text += f"- {name}: ({x:.1f}, {y:.1f})\n"

        plt.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center')

        plt.tight_layout()
        plt.show()

        
    def plot_gauge_dial(self, reading, min_val, max_val, unit, keypoints=None):
        """Create a visual representation of the gauge dial with current reading"""
        from matplotlib.patches import Arc, Circle, Rectangle
        import numpy as np
        
        fig = plt.gca()
        
        # Use a more realistic full circular gauge visualization
        # Draw the outer circle of the gauge
        circle = plt.Circle((0.5, 0.5), 0.4, fill=False, color='gray', lw=2)
        fig.add_patch(circle)
        
        # Draw the gauge face (light color)
        face = plt.Circle((0.5, 0.5), 0.38, fill=True, color='#f8f8f8')
        fig.add_patch(face)
        
        # Default sweep angles if no keypoints provided
        start_angle_deg = -135
        end_angle_deg = 135
        needle_pos_fraction = (reading - min_val) / (max_val - min_val) if max_val > min_val else 0
        
        # Calculate actual angles from keypoints if available
        if keypoints and all(k in keypoints for k in ['center', 'min', 'max', 'tip']):
            # Get coordinates
            center = np.array([keypoints['center'][0], keypoints['center'][1]])
            min_pt = np.array([keypoints['min'][0], keypoints['min'][1]])
            max_pt = np.array([keypoints['max'][0], keypoints['max'][1]])
            tip_pt = np.array([keypoints['tip'][0], keypoints['tip'][1]])
            
            # Calculate vectors from center
            v_min = min_pt - center
            v_max = max_pt - center
            v_tip = tip_pt - center
            
            # Calculate absolute angles (clockwise from x-axis)
            def get_angle(v):
                angle = -np.arctan2(v[1], v[0])  # Negative for clockwise
                if angle < 0:
                    angle += 2 * np.pi
                return angle
                
            angle_min = get_angle(v_min)
            angle_max = get_angle(v_max)
            angle_tip = get_angle(v_tip)
            
            # Ensure clockwise measurement from min to max and tip
            max_from_min = (angle_max - angle_min) % (2 * np.pi)
            tip_from_min = (angle_tip - angle_min) % (2 * np.pi)
            
            # Convert to degrees for visualization
            start_angle_deg = np.degrees(angle_min) - 90  # Adjust for matplotlib's coordinate system
            end_angle_deg = start_angle_deg + np.degrees(max_from_min)
            
            # Calculate needle position as fraction between min and max
            needle_pos_fraction = tip_from_min / max_from_min if max_from_min > 0 else 0
            
            # Debug output
            print(f"Dial visualization: min={start_angle_deg:.1f}°, max={end_angle_deg:.1f}°, needle at {needle_pos_fraction:.2f}")
        
        # For visualization, ensure min is on left and max on right if using default angles
        sweep_angle = end_angle_deg - start_angle_deg
        
        # Draw arc representing the gauge scale
        arc = Arc((0.5, 0.5), 0.7, 0.7, theta1=start_angle_deg, theta2=end_angle_deg, color='black', lw=2)
        fig.add_patch(arc)
        
        # Draw ticks around the gauge
        num_ticks = 10
        for i in range(num_ticks + 1):
            angle_deg = start_angle_deg + sweep_angle * i / num_ticks
            angle_rad = math.radians(angle_deg)
            inner_x = 0.5 + 0.32 * math.cos(angle_rad)
            inner_y = 0.5 + 0.32 * math.sin(angle_rad)
            outer_x = 0.5 + 0.38 * math.cos(angle_rad)
            outer_y = 0.5 + 0.38 * math.sin(angle_rad)
            plt.plot([inner_x, outer_x], [inner_y, outer_y], 'k-', lw=2)
            
            # Add tick labels for major ticks
            if i % 2 == 0:
                tick_value = min_val + (max_val - min_val) * i / num_ticks
                label_x = 0.5 + 0.28 * math.cos(angle_rad)
                label_y = 0.5 + 0.28 * math.sin(angle_rad)
                plt.text(label_x, label_y, f"{tick_value:.0f}", ha='center', va='center', fontsize=8)
        
        # Calculate needle position based on the reading's position in the scale
        needle_angle_deg = start_angle_deg + sweep_angle * needle_pos_fraction
        needle_rad = math.radians(needle_angle_deg)
        
        # Draw needle with a proper pivot point
        needle_x = 0.5 + 0.35 * math.cos(needle_rad)
        needle_y = 0.5 + 0.35 * math.sin(needle_rad)
        plt.plot([0.5, needle_x], [0.5, needle_y], 'r-', lw=3)
        
        # Draw center pivot
        center_circle = plt.Circle((0.5, 0.5), 0.03, color='darkred', fill=True)
        fig.add_patch(center_circle)
        
        # Draw the min and max labels at appropriate positions
        min_rad = math.radians(start_angle_deg)
        max_rad = math.radians(end_angle_deg)
        
        min_x = 0.5 + 0.45 * math.cos(min_rad)
        min_y = 0.5 + 0.45 * math.sin(min_rad)
        max_x = 0.5 + 0.45 * math.cos(max_rad)
        max_y = 0.5 + 0.45 * math.sin(max_rad)
        
        plt.text(min_x, min_y, f"Min: {min_val}", ha='right' if min_x < 0.5 else 'left', 
                 va='center', fontsize=10, color='blue')
        plt.text(max_x, max_y, f"Max: {max_val}", ha='left' if max_x > 0.5 else 'right', 
                 va='center', fontsize=10, color='blue')
        
        # Add unit in the lower part of the gauge
        plt.text(0.5, 0.75, f"Unit: {unit or 'N/A'}", ha='center', fontsize=10)
        
        # Add reading text
        plt.text(0.5, 0.85, f"Reading: {reading:.2f} {unit or ''}", ha='center', fontsize=12, 
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        
        # Set plot limits and turn off axis
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title("Gauge Reading Visualization")

def process_dataset(reader, base_dir='crops', split='test', visualize=False, confirm_scale=False):
    """Process images from a dataset split and calculate gauge readings"""
    split_dir = os.path.join(base_dir, split)
    img_dir = os.path.join(split_dir, 'images')
    
    results = {}
    output_dir = os.path.join(base_dir, f"{split}_results")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {split} dataset...")
    
    if not os.path.exists(img_dir):
        print(f"Error: Image directory {img_dir} not found")
        return results
        
    for img_file in os.listdir(img_dir):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        
        img_path = os.path.join(img_dir, img_file)
        
        try:
            result = reader.process_image(img_path, visualize=visualize, confirm_scale=confirm_scale)
            if result:
                results[img_file] = result
                print(f"{img_file}: {result['reading']:.2f} {result['unit'] or ''}")
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results to CSV
    with open(os.path.join(output_dir, 'readings.csv'), 'w') as f:
        f.write("image,reading,min,max,unit\n")
        for img, result in results.items():
            f.write(f"{img},{result['reading']:.2f},{result['min_val']},{result['max_val']},{result['unit'] or ''}\n")
    
    return results

def main():
    # Paths to models
    detect_model_path = r"C:\Users\siddharth\OneDrive\ドキュメント\gauge-analog.v1i.yolov11\runs\detect\gauge-detect6"
    pose_model_path = r"C:\Users\siddharth\OneDrive\ドキュメント\gauge-analog.v1i.yolov11\runs\pose\gauge-pose-cuda"

    # Create gauge reader
    reader = GaugeReader(detect_model_path, pose_model_path)
    
    # Choose operation mode
    print("\nGauge Reader - Choose Mode:")
    print("1. Process single image")
    print("2. Process dataset folder")
    choice = input("Enter choice (1/2): ").strip()
    
    if choice == '1':
        # Process single image
        img_path = input("Enter path to image: ").strip()
        if not img_path:
            img_path = r"C:\Users\siddharth\OneDrive\ドキュメント\PROJECTS\internship\gauge2.jpg"
            print(f"Using default image: {img_path}")
            
        if not os.path.exists(img_path):
            print(f"Error: Image not found at {img_path}")
            return
            
        result = reader.process_image(img_path, visualize=True, confirm_scale=True)
        if result:
            print(f"\nFinal Reading: {result['reading']:.2f} {result['unit'] or ''}")
        
    elif choice == '2':
        # Process dataset
        base_dir = input("Enter base directory (default: crops): ").strip() or 'crops'
        split = input("Enter split (train/valid/test, default: test): ").strip() or 'test'
        
        confirm = input("Confirm scale for each image? (y/n, default: n): ").strip().lower() == 'y'
        visualize = input("Visualize results? (y/n, default: n): ").strip().lower() == 'y'
        
        process_dataset(reader, base_dir, split, visualize=visualize, confirm_scale=confirm)
        
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()