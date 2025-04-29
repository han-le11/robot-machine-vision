import os
import json
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from src.utils.pose_extractor import PoseExtractor

with open("config/config.json", "r") as f:
    CONFIG = json.load(f)

SEQ_LEN = CONFIG["seq_len"]
STEP = CONFIG["step"]
LABELS = CONFIG["labels"]
VIDEO_ROOT = "data/data_videos"
OUTPUT_ROOT = "data/data_cached"
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

def process_video(in_path, out_dir, seq_len, step):
    fname = os.path.basename(in_path)
    base_name = os.path.splitext(fname)[0]
    extractor = PoseExtractor()
    sequences = extractor.extract_sliding(in_path, seq_len=seq_len, step=step)

    if not sequences:
        return f"⚠️ Skipped: {in_path} (too short or failed)"

    for i, seq in enumerate(sequences):
        out_name = f"{base_name}_chunk{i:02d}.npy"
        out_path = os.path.join(out_dir, out_name)
        if os.path.exists(out_path):
            continue
        try:
            np.save(out_path, seq)
        except Exception as e:
            return f"Failed to save {out_path}: {e}"

    return None 

def cache_pose_dataset(input_root, output_root, seq_len, step, video_exts=VIDEO_EXTS):
    for class_name in sorted(os.listdir(input_root)):
        class_input_dir = os.path.join(input_root, class_name)
        class_output_dir = os.path.join(output_root, class_name)

        if not os.path.isdir(class_input_dir):
            continue

        os.makedirs(class_output_dir, exist_ok=True)
        video_files = [f for f in os.listdir(class_input_dir) if f.lower().endswith(video_exts)]
        video_paths = [os.path.join(class_input_dir, f) for f in video_files]

        with ProcessPoolExecutor() as executor:
            process = partial(process_video, out_dir=class_output_dir, seq_len=seq_len, step=step)
            results = list(tqdm(executor.map(process, video_paths), total=len(video_paths), desc=f"Caching: {class_name}"))

        for msg in results:
            if msg:
                print(msg)

def cache_all_splits(video_root, output_root, seq_len, step):
    for split in ["train", "val", "eval"]:
        in_dir = os.path.join(video_root, split)
        out_dir = os.path.join(output_root, split)
        print(f"\nCaching split: {split}")
        print(f"Input:  {in_dir}\nOutput: {out_dir}")
        cache_pose_dataset(in_dir, out_dir, seq_len, step)

if __name__ == "__main__":
    print(f"Sequence length:  {SEQ_LEN}")
    print(f"Sliding step:     {STEP}")
    cache_all_splits(VIDEO_ROOT, OUTPUT_ROOT, SEQ_LEN, STEP)
    print("Done.")
