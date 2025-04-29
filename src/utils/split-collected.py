import os
import json
import shutil
import random
from glob import glob

#  Load labels from config.json 
with open("config/config.json", "r") as f:
    CONFIG = json.load(f)

LABELS = CONFIG["labels"]
CLASS_NAMES = list(LABELS.keys())

#  Split Config 
DATA_ROOT = "data/data_videos"
COLLECTED_ROOT = os.path.join(DATA_ROOT, "collected")
SPLITS = ["train", "val", "eval"]

RATIOS = {
    "train": 0.7,
    "val": 0.3,
    "eval": 0.0
}

assert abs(sum(RATIOS.values()) - 1.0) < 1e-6, "Ratios must sum to 1.0"
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

def normalize(path):
    return os.path.normpath(path)

#  Step 1: Collect all videos from collected/ 
class_to_files = {}

for class_name in CLASS_NAMES:
    class_dir = os.path.join(COLLECTED_ROOT, class_name)
    if not os.path.isdir(class_dir):
        print(f"⚠️ Skipping missing class folder: {class_name}")
        continue

    for fname in os.listdir(class_dir):
        if fname.lower().endswith(VIDEO_EXTS):
            full_path = normalize(os.path.join(class_dir, fname))
            class_to_files.setdefault(class_name, []).append(full_path)

#  Step 2: Shuffle and assign files 
for class_name, file_list in class_to_files.items():
    random.shuffle(file_list)
    total = len(file_list)

    base_counts = {
        "train": int(total * RATIOS["train"]),
        "val": int(total * RATIOS["val"]),
        "eval": int(total * RATIOS["eval"]),
    }

    assigned = sum(base_counts.values())
    remainder = total - assigned

    for split in SPLITS:
        if remainder == 0:
            break
        if RATIOS[split] > 0.0:
            base_counts[split] += 1
            remainder -= 1

    split_assignments = {}
    start = 0
    for split in SPLITS:
        count = base_counts[split]
        split_assignments[split] = file_list[start:start + count]
        start += count

    #  Step 3: Clear target split folders 
    for split in SPLITS:
        target_dir = os.path.join(DATA_ROOT, split, class_name)
        os.makedirs(target_dir, exist_ok=True)
        for fname in os.listdir(target_dir):
            if fname.lower().endswith(VIDEO_EXTS):
                os.remove(os.path.join(target_dir, fname))

    #  Step 4: Copy files into their new split 
    for split in SPLITS:
        target_dir = os.path.join(DATA_ROOT, split, class_name)
        os.makedirs(target_dir, exist_ok=True)
        for src_path in split_assignments[split]:
            dst_path = os.path.join(target_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dst_path)

    split_summary = ", ".join(f"{k}={v}" for k, v in base_counts.items() if v > 0)
    print(f"[OK] Split '{class_name}': {split_summary}")
