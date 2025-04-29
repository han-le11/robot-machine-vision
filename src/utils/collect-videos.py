import os
import shutil
from glob import glob

# Config
DATA_ROOT = "data/data_videos"
TARGET_ROOT = os.path.join(DATA_ROOT, "collected")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

def normalize(path):
    return os.path.normpath(path)

# Step 1: Find all video files under any split/class
all_video_paths = glob(os.path.join(DATA_ROOT, "*", "*", "*"))

for src_path in all_video_paths:
    src_path = normalize(src_path)
    if not src_path.lower().endswith(VIDEO_EXTS):
        continue

    parts = src_path.split(os.sep)
    if len(parts) < 4:
        continue

    class_name = parts[-2]
    fname = os.path.basename(src_path)

    target_dir = os.path.join(TARGET_ROOT, class_name)
    os.makedirs(target_dir, exist_ok=True)

    dst_path = os.path.join(target_dir, fname)

    # Skip if already moved
    if os.path.exists(dst_path):
        print(f"[SKIP] {fname} already exists in {class_name}")
        continue

    shutil.move(src_path, dst_path)
    print(f"[MOVE] {fname} → {class_name}/")

print("\n✅ All videos moved into 'collected/' folder.")
