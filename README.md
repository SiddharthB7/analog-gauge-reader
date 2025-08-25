

```md
# analog-gauge-reader

Hybrid analog gauge reader: **YOLO detect + YOLO pose** → keypoints → angle → needle fraction,  
**OCR** → scale (min/max) + unit → **final reading**.

> **Note:** OCR for digits/units on mechanical dials is inherently noisy. The angle/needle math is deterministic; OCR is used only to guess min/max/unit and can be corrected interactively.

---

## What this does

- **Finds the gauge** in an image (YOLO detection).
- **Finds 4 keypoints** on the dial (YOLO pose): `center`, `min`, `max`, `tip`.
- **Computes the needle fraction** using angles between `min → tip` over `min → max`.
- **Uses OCR** to estimate **scale** (`min`, `max`) and **unit** (e.g., bar, MPa).
- **Outputs final reading**: `reading = min + fraction * (max - min)` and visualizes everything.

---

## Repo structure

```

.
├─ gauge\_reader.py          # main pipeline & CLI (single image or folder)
├─ min\_max\_unit.py          # GaugeDetector (YOLO detection + YOLO pose)
├─ gauge\_scale\_reader.py    # OCR (OpenOCR backend) + unit parsing
├─ convert\_to\_pose.py       # helper: convert detection labels to YOLO-pose format
├─ make\_dataset.py          # helper: export keypoints/readings to CSV/Excel
├─ make\_gauge\_dataset.py    # helper: remap detection labels for single-class detect
├─ data\_det.yaml            # example YOLO detect data yaml
├─ gauge-pose.yaml          # example YOLO pose data yaml
├─ LICENSE
└─ README.md

````

> We don’t include separate Python “training scripts” for YOLO. You train with the **Ultralytics CLI** (commands below). Each file has comments explaining its role.

---

## Install

Create a virtual env and install the base requirements:

**Windows (PowerShell)**
```pwsh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
````

**Linux/macOS**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### OCR backend (choose one)

This project imports:

```python
from openocr import OpenOCR
```

Two different packages expose that import. **Install only one**:

* **Recommended first:** Topdu’s package

  ```bash
  pip install openocr-python
  ```

* **Alternative:** MarkHoo’s package

  ```bash
  pip install OpenOCR
  ```

> Installing both can cause module shadowing (they both provide `openocr`). If you accidentally installed both, uninstall one:
> `pip uninstall OpenOCR` **or** `pip uninstall openocr-python`.

---

## Run (inference)

1. Put your trained weights somewhere accessible (examples):

```
runs/detect/gauge-detect/weights/best.pt
runs/pose/gauge-pose/weights/best.pt
```

2. Open `gauge_reader.py` and set the two paths in `main()`:

```python
detect_model_path = r"...\runs\detect\gauge-detect"
pose_model_path   = r"...\runs\pose\gauge-pose"
```

3. Launch:

```bash
python gauge_reader.py
```

Choose **1** for a single image or **2** for a folder. You’ll see:

* detection box
* pose keypoints
* OCR boxes (if any)
* computed angle fraction and final value

If OCR guesses the scale wrong, the script will ask you to **confirm/override** min/max/unit and recompute the final reading.

---

## Training (Ultralytics YOLO)

We use Ultralytics for both detection and pose.

### Detection (bounding box)

```bash
yolo detect train \
  model=yolo11s.pt \
  data=data_det.yaml \
  imgsz=640 epochs=50 batch=16 \
  project=runs/detect name=gauge-detect
```

### Pose (4 keypoints = center, max, min, tip)

```bash
yolo pose train \
  model=yolo11s-pose.pt \
  data=gauge-pose.yaml \
  imgsz=640 epochs=60 batch=16 \
  project=runs/pose name=gauge-pose
```

**Dataset note:** We started from the Roboflow *gauge-analog* detection dataset and converted to a 4-keypoint pose dataset.
The helpers `convert_to_pose.py` and `make_gauge_dataset.py` show how labels were adapted.

---

## Known limitations

* **OCR is imperfect.** Reading tiny/curvy/blurred digits on dials is hard; results vary with font, glare, and resolution. We expose a manual override for min/max/unit to keep the final reading stable.
* **Keypoint accuracy drives the final reading.** If predicted `center/min/max/tip` are off, the angle fraction is off. Image clarity and unusual needle shapes affect results.
* **Weights & data are large.** We do not commit model weights or datasets. Share via a release link or drive.

---

## Acknowledgments

* **Dataset:** [Gauge Analog (v1) — Roboflow Universe](https://universe.roboflow.com/khang-nguyen-4jqxa/gauge-analog/dataset/1) by **Khang Nguyen**.
  Used for **gauge detection** (bounding boxes). We converted detections to 4-keypoint **pose** labels for our experiments.
  *Please refer to the dataset page for its license/terms and cite accordingly.*

* **Models & libs:**

  * [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — detection & pose
  * OCR backend: [openocr-python (Topdu)](https://github.com/Topdu/OpenOCR) or [OpenOCR (MarkHoo)](https://github.com/MarkHoo/openocr)

---

## License

This repository is released under the **MIT License** (see `LICENSE`).








