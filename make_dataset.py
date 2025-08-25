# This script processes up to 200 gauge images, runs the full detection and reading pipeline,
# and saves the results (keypoints, readings, scale info) to an Excel spreadsheet.
# It is used to build a dataset for analysis or machine learning.

import os
import pandas as pd
from gauge_reader import GaugeReader

# 1. Point to your trained models:
DETECT_MODEL = r"C:\Users\siddharth\OneDrive\ドキュメント\gauge-analog.v1i.yolov11\runs\detect\gauge-detect6"
POSE_MODEL   = r"C:\Users\siddharth\OneDrive\ドキュメント\gauge-analog.v1i.yolov11\runs\pose\gauge-pose-cuda"

# 2. Initialize the full pipeline (set use_gpu=True if you have CUDA)
reader = GaugeReader(DETECT_MODEL, POSE_MODEL, use_gpu=False)

# 3. Where your cropped gauge images live
images_dir = "crops/test/images"  # adjust if your folder is different

# 4. Collect the first 200 records
records = []
for fn in sorted(os.listdir(images_dir))[:200]:
    # Only process image files
    if not fn.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
        continue
    img_path = os.path.join(images_dir, fn)
    print(f"Processing {img_path}...")
    # Run the full gauge reading pipeline (detection, keypoints, reading, scale)
    res = reader.process_image(img_path, visualize=False, confirm_scale=False)
    if res:
        k = res['keypoints']
        # Collect results: image name, keypoint coordinates, reading, min/max/unit
        records.append({
            'image'   : fn,
            'center_x': k['center'][0],
            'center_y': k['center'][1],
            'min_x'   : k['min'][0],
            'min_y'   : k['min'][1],
            'max_x'   : k['max'][0],
            'max_y'   : k['max'][1],
            'tip_x'   : k['tip'][0],
            'tip_y'   : k['tip'][1],
            'reading' : res['reading'],
            'min_val' : res['min_val'],
            'max_val' : res['max_val'],
            'unit'    : res['unit'] or ''
        })
    else:
        print(f"Failed to process {fn}")

# 5. Build DataFrame & export
df = pd.DataFrame(records)
excel_out_path = "gauge_keypoints_readings.xlsx"

# Try to save as Excel, install openpyxl if needed, fallback to CSV if all else fails
try:
    df.to_excel(excel_out_path, index=False)
    print(f"Dataset saved to {excel_out_path}")
except ImportError:
    print("Excel export failed: openpyxl not installed.")
    print("Installing openpyxl now...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
    print("openpyxl installed successfully!")
    try:
        df.to_excel(excel_out_path, index=False)
        print(f"Dataset saved to {excel_out_path}")
    except Exception as e:
        print(f"Excel export still failed: {e}")
        csv_out_path = "gauge_keypoints_readings.csv"
        df.to_csv(csv_out_path, index=False)
        print(f"Dataset saved to {csv_out_path} instead")

print(f"Processed {len(records)} images successfully")
