import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import re
from openocr import OpenOCR

# Canonical unit extraction setup
UNIT_VARIANTS = {
    "bar"   : [r"\bbar\b"],
    "mbar"  : [r"\bmbar\b"],
    "kpa"   : [r"\bk[p|f]a\b", r"\bkp[a|4]\b"],
    "pa"    : [r"\bpa\b"],
    "mpa"   : [r"\bm[p|f]a\b"],
    "psi"   : [r"\bp[s5]i\b"],
    "kg/cm²": [r"kg/?c[m|n]2?", r"kg/cm²", r"kgcm2"],
    "lb/in²": [r"(?:lb|1b)/?in2?", r"lb/in²", r"lbin2"],
    "atm"   : [r"\batm\b"],
    "mmhg"  : [r"\bmmh[gq]\b"],
    "torr"  : [r"\btorr\b"],
    "inhg"  : [r"\bin[h|n]g\b"],
    "inh2o" : [r"\bin[h|n]2?o\b"],
    "mmh2o" : [r"\bmmh2?o\b"],
    "hpa"   : [r"\bh[p|f]a\b"],
}
UNIT_REGEX = [(canon, re.compile("|".join(pats), re.IGNORECASE)) for canon, pats in UNIT_VARIANTS.items()]

def normalize_unit(txt):
    s = txt.lower().replace(" ", "")
    s = s.replace("²", "2")
    for canon, rgx in UNIT_REGEX:
        if rgx.search(s):
            return canon
    return None

def _normalise_openocr_output(raw):
    # Compatible with various OpenOCR/PaddleOCR result formats
    if isinstance(raw, dict):
        return next(iter(raw.values()), [])
    elif isinstance(raw, list):
        if isinstance(raw[0], dict):
            return raw
        elif isinstance(raw[0], str) and '\t' in raw[0]:
            _, json_part = raw[0].split('\t', 1)
            import json
            try:
                return json.loads(json_part)
            except Exception:
                print("JSON parse failed")
                return []
    return []

class GaugeScaleReader:
    def __init__(self, backend='torch', device='cpu'):
        self.ocr_engine = OpenOCR(backend=backend, device=device)
    def read_scale(self, image_or_path, visualize=True, score_thresh=0.5):
        if isinstance(image_or_path, str):
            img = cv2.imread(image_or_path)
            if img is None:
                print(f"Error: cannot load image {image_or_path}")
                return None, None, None, []
        else:
            img = image_or_path

        cy, cx = img.shape[0] // 2, img.shape[1] // 2
        tmp_path = "tmp_openocr_input.jpg"
        cv2.imwrite(tmp_path, img)
        try:
            raw, _ = self.ocr_engine(tmp_path)  # This runs OCR on the image using PaddleOCR via OpenOCR.
        except Exception as e:
            print("OpenOCR failed:", e)
            return None, None, None, []
        results = _normalise_openocr_output(raw)

        num_items = []
        unit_items = []
        ocr_boxes = []
        for item in results:
            text = item.get('transcription', '').strip().replace("O", "0").replace("o", "0")
            conf = item.get('score', 0)
            pts = item.get('points', [])
            if len(pts) != 4 or conf < score_thresh:
                continue
            m = re.match(r'^-?\d+(\.\d+)?$', text)
            if m:
                num_items.append((float(text), pts, conf))
            else:
                unit = normalize_unit(text)
                if unit:
                    unit_items.append((unit, pts, conf))
            ocr_boxes.append((text, pts, conf))

        if len(num_items) < 2:
            print("Warning: Less than two numeric values detected.")
            return None, None, None, ocr_boxes

        vals = [item[0] for item in num_items]
        min_val = min(vals)
        max_val = max(vals)

        unit = unit_items[0][0] if unit_items else None

        return min_val, max_val, unit, ocr_boxes

    def _draw_boxes(self, img, results, min_val, max_val, unit, debug_mode=False):
        img_vis = img.copy()
        for item in results:
            text = item.get('transcription', '').strip().replace("O", "0").replace("o", "0")
            conf = item.get('score', 0)
            pts = item.get('points', [])
            if len(pts) != 4:
                continue
            pts_arr = np.array(pts, np.int32).reshape(-1, 1, 2)
            value = None
            try:
                value = float(text)
            except Exception:
                pass
            color = (0,255,0) if value is not None and value == max_val else \
                    (255,0,0) if value is not None and value == min_val else \
                    (0,0,255) if value is not None else (255,128,0)
            cv2.polylines(img_vis, [pts_arr], isClosed=True, color=color, thickness=2)
            x, y = pts[0]
            label = f"{text} ({conf:.2f})" if not debug_mode else text
            cv2.putText(img_vis, label, (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if unit:
            cv2.putText(img_vis, f"UNIT: {unit}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        plt.figure(figsize=(12,8))
        plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        plt.title("Gauge OCR Results (blue=min, green=max, orange=unit, red=other)")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    # Example usage
    img_path = "YOUR_GAUGE_IMAGE.jpg"  # replace with your gauge image path
    reader = GaugeScaleReader(backend='torch', device='cpu')
    min_val, max_val, unit = reader.read_scale(img_path, visualize=True)
    print(f"Min: {min_val}, Max: {max_val}, Unit: {unit}")

# The GaugeScaleReader class is responsible for extracting the scale information from a gauge image.
# Here's what it does and how it relates to keypoints:

# 1. OCR Extraction:
#    - It uses OCR (via OpenOCR/PaddleOCR) to detect all text on the cropped gauge image.
#    - It looks for numeric values (e.g., 0, 100) and units (e.g., bar, psi) in the detected text.

# 2. Scale Value Identification:
#    - It collects all detected numbers and tries to identify the minimum and maximum scale values on the gauge dial.
#    - It also tries to find the unit (bar, psi, etc.) using regex matching.

# 3. Returns OCR Boxes:
#    - For each detected text, it returns the text, its bounding box (polygon), and confidence score.
#    - These are used for visualization and debugging.

# 4. Visualization:
#    - It can draw colored boxes around detected numbers and units on the gauge image.
#    - Blue box for min value, green for max, orange for unit, red for other numbers.

# 5. Integration with Keypoints:
#    - The keypoints (center, min, max, tip) are detected by GaugeDetector (YOLOv8).
#    - GaugeScaleReader does NOT directly use the keypoints for OCR, but:
#      - The keypoints are used by GaugeReader to compute the dial reading (angle mapping).
#      - The min/max values found by OCR are mapped to the corresponding keypoints to interpret the dial position.
#      - The unit is used to label the reading.

# 6. Output:
#    - Returns min_val, max_val, unit, and ocr_boxes to GaugeReader.
#    - GaugeReader uses these to compute the final reading and visualize results.

# In summary:
# - GaugeScaleReader extracts the scale (min, max, unit) from the gauge image using OCR.
# - It does not compute keypoints, but its output is combined with keypoints by GaugeReader to interpret the dial value.
# - It provides visualization and debugging for detected scale values and units.

