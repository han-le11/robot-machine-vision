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

# === Config ===
SEQ_LEN = 16
RESIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "lstm_mediapipe_model.pt"
RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "recordings")
LABEL_MAP = {"positive": 1, "negative": 0}

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

def evaluate_model():
    # === Load models ===
    cnn = CNNEncoder(backbone="resnet18").to(DEVICE)
    cnn.eval()

    lstm = ActionLSTM(feature_dim=cnn.feature_dim).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        lstm.load_state_dict(checkpoint["model_state_dict"])
        print("✅ Loaded checkpoint state dict")
    else:
        lstm.load_state_dict(checkpoint)
        print("✅ Loaded full model state dict")

    lstm.eval()

    y_true, y_pred = [], []

    for label_name, label in LABEL_MAP.items():
        folder = os.path.join(RECORDINGS_DIR, label_name)
        for fname in tqdm(os.listdir(folder), desc=label_name):
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
                output = lstm(features)
                pred = (output > 0.5).float().item()

            y_true.append(label)
            y_pred.append(int(pred))

    print("\n\U0001F4CA Evaluation Results on Your Recordings:")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.2%}")
    print(f"Precision: {precision_score(y_true, y_pred):.2%}")
    print(f"Recall:    {recall_score(y_true, y_pred):.2%}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.2%}")

if __name__ == "__main__":
    evaluate_model()
