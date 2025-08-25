# This script scans your dataset and flags (or deletes) images and label files that are incomplete or incorrectly formatted.
# Specifically:
# 1. It finds images in the split (train/valid/test) that do not have a corresponding label file and flags them for removal.
# 2. It checks each label file to ensure it has the correct number of columns (for YOLOv8 pose: class + bbox + keypoints).
#    If a label file is malformed, it flags both the label and its corresponding image for removal.
# 3. If DRY_RUN is True, it only prints the files that would be deleted.
#    If DRY_RUN is False, it actually deletes the flagged files.
# Use this script to clean up your dataset before training to ensure all images and labels are valid and complete.

import os

# ─── CONFIG ───────────────────────────────────────────
DATA_ROOT     = r"C:\Users\siddharth\OneDrive\ドキュメント\gauge-analog.v1i.yolov11"
SPLIT         = 'train'             # 'train', 'valid' or 'test'
NUM_KEYPOINTS = 4                   # how many keypoints per object
DRY_RUN       = True                # True → just print, False → actually delete
# ───────────────────────────────────────────────────────

img_dir       = os.path.join(DATA_ROOT, SPLIT, 'images')
lbl_dir       = os.path.join(DATA_ROOT, SPLIT, 'labels')
expected_cols = 5 + 3 * NUM_KEYPOINTS   # class(1) + bbox(4) + (x,y,vis)*K

to_remove = []

# 1) images with no label
for fn in sorted(os.listdir(img_dir)):
    stem, ext = os.path.splitext(fn)
    if ext.lower() not in ('.jpg','.jpeg','.png','.bmp','.tif','.tiff'):
        continue
    lbl = os.path.join(lbl_dir, stem + '.txt')
    if not os.path.exists(lbl):
        to_remove.append(('image', os.path.join(img_dir, fn)))

# 2) labels (and their images) with wrong column count
for fn in sorted(os.listdir(lbl_dir)):
    if not fn.endswith('.txt'):
        continue
    path = os.path.join(lbl_dir, fn)
    with open(path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    for L in lines:
        if len(L.split()) != expected_cols:
            to_remove.append(('label', path))
            stem = os.path.splitext(fn)[0]
            # try common exts
            for ext in ('.jpg','.jpeg','.png'):
                img_file = os.path.join(img_dir, stem + ext)
                if os.path.exists(img_file):
                    to_remove.append(('image', img_file))
            break

if not to_remove:
    print("✅ No files to remove—everything looks good!")
else:
    print(f"⚠️  {len(to_remove)} files flagged for removal:")
    for typ, path in to_remove:
        print(f"   [{typ.upper():5}] {path}")
    if not DRY_RUN:
        for _, path in to_remove:
            try:
                os.remove(path)
            except Exception as e:
                print(f"  ⚠️  Failed to delete {path}: {e}")
        print("\n✅ Deletion complete.")
    else:
        print("\nℹ️  DRY RUN enabled—no files were deleted.")
