import os
import cv2
import torch
import numpy as np
from torch import nn
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from models.cnn_encoder import CNNEncoder
from models.pose_model import ActionLSTM

# === Settings ===
SEQ_LEN = 32
RESIZE = (224, 224)  # Updated as per your setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoint_epoch05_valacc_95.00.pt"  # Specify the name of the checkpoint file
THRESHOLDS = [round(x, 2) for x in np.arange(0.4, 0.61, 0.02)] # Thresholds to sweep
RECORDINGS_DIR = "recordings" # Directory containing recordings

# === Preprocessing ===
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(RESIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

def extract_uniform_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if len(frames) < SEQ_LEN:
        return None
    idxs = np.linspace(0, len(frames) - 1, SEQ_LEN).astype(int)
    return [frames[i] for i in idxs]

def evaluate_at_threshold(threshold, model, cnn, label_map):
    y_true, y_pred = [], []

    for label_name, label in label_map.items():
        folder = os.path.join(RECORDINGS_DIR, label_name)
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                continue
            path = os.path.join(folder, fname)
            frames = extract_uniform_frames(path)
            if frames is None:
                continue
            frames = [transform(f) for f in frames]
            batch = torch.stack(frames).to(DEVICE)  # [SEQ_LEN, C, H, W]

            with torch.no_grad():
                features = cnn(batch).view(1, SEQ_LEN, -1)
                output = model(features).item()
                pred = 1 if output > threshold else 0

            y_true.append(label)
            y_pred.append(pred)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    accuracy = accuracy_score(y_true, y_pred)
    return accuracy, precision, recall, f1

# === Main Sweep
def run_threshold_sweep():
    label_map = {"positive": 1, "negative": 0}

    print(f"📊 Evaluating checkpoint model at thresholds: {THRESHOLDS}")
    cnn = CNNEncoder(backbone="resnet18").to(DEVICE)
    cnn.eval()

    lstm = ActionLSTM(feature_dim=cnn.feature_dim).to(DEVICE)


    # ✅ Load from checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    lstm.load_state_dict(checkpoint["model_state_dict"])
    lstm.eval()

    for thresh in THRESHOLDS:
        acc, precision, recall, f1 = evaluate_at_threshold(thresh, lstm, cnn, label_map)
        print(f"\nThreshold = {thresh:.2f}")
        print(f"Accuracy:  {acc:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall:    {recall:.2%}")
        print(f"F1 Score:  {f1:.2%}")


if __name__ == "__main__":
    run_threshold_sweep()
