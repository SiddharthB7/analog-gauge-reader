# save as make_gauge_labels.py
import os, glob


# It is for converting gauge detection labels (class 1 = gauge) to single-class YOLO detection labels (class 0 = gauge).
# It copies only the gauge bounding box line from each labels_old file, remaps class 1→0, and writes it to labels_gauge.
# This is for object detection training.

for split in ('train','valid','test'):
    src_dir = os.path.join(split, 'labels_old')
    dst_dir = os.path.join(split, 'labels_gauge')
    os.makedirs(dst_dir, exist_ok=True)
    for txt in glob.glob(os.path.join(src_dir,'*.txt')):
        lines = open(txt).read().splitlines()
        # find the line that starts with '1 ' (class 'gauge')
        for L in lines:
            if L.startswith('1 '):
                # remap class 1→0 for single‑class YOLO
                parts = L.split()
                parts[0] = '0'
                open(os.path.join(dst_dir, os.path.basename(txt)),'w').write(' '.join(parts)+'\n')
                break
