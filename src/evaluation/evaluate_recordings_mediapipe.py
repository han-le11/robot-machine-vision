import os
import cv2
import torch
import numpy as np
import mediapipe as mp
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm
from models.pose_model import ActionLSTM

# === Config ===
SEQ_LEN = 32
BATCH_SIZE = 16
NUM_EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data_custom/train"  # should contain class folders

# === Mediapipe setup ===
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.7)

# === Feature extractor ===
def extract_landmarks(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    if not results.pose_landmarks and not results.left_hand_landmarks and not results.right_hand_landmarks:
        return None  # skip if no landmarks detected

    keypoints = []

    # Pose (33 x 3)
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0.0] * 99)

    # Left hand (21 x 3)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0.0] * 63)

    # Right hand (21 x 3)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0.0] * 63)

    return keypoints  # total length = 225

# === Dataset ===
class GestureDataset(Dataset):
    def __init__(self, root_dir, seq_len):
        self.samples = []
        self.seq_len = seq_len
        self.class_map = {cls: i for i, cls in enumerate(sorted(os.listdir(root_dir)))}

        for cls_name in self.class_map:
            cls_path = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_path):
                if fname.endswith(".mp4"):
                    self.samples.append((os.path.join(cls_path, fname), self.class_map[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        cap = cv2.VideoCapture(path)
        frames = []
        while len(frames) < self.seq_len:
            ret, frame = cap.read()
            if not ret:
                break
            landmarks = extract_landmarks(frame)
            if landmarks:
                frames.append(landmarks)
        cap.release()

        # Pad or sample to SEQ_LEN
        if len(frames) < self.seq_len:
            while len(frames) < self.seq_len:
                frames.append([0.0] * 225)
        else:
            idxs = np.linspace(0, len(frames) - 1, self.seq_len).astype(int)
            frames = [frames[i] for i in idxs]

        return torch.tensor(frames, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# === Training ===
def train():
    dataset = GestureDataset(DATA_DIR, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = ActionLSTM(input_dim=225, hidden_dim=128, num_classes=1).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss, total_correct = 0, 0

        for sequences, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)

            outputs = model(sequences).view(-1)
            labels = labels.view(-1)

            loss = criterion(outputs, labels)
            preds = (outputs > 0.5).float()

            total_correct += (preds == labels).sum().item()
            total_loss += loss.item() * sequences.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = total_correct / len(dataset)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f} | Accuracy = {acc*100:.2f}%")

    torch.save(model.state_dict(), "lstm_mediapipe_model.pt")
    print("✅ Model saved as lstm_mediapipe_model.pt")

if __name__ == "__main__":
    train()
