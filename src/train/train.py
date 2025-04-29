import os
import csv
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from datetime import datetime
from src.models.pose_model import ActionLSTMWithSpatioTemporalAttention

# Load config
with open("config/config.json", "r") as f:
    CONFIG = json.load(f)

CLASS_LABELS = CONFIG["labels"]
SEQ_LEN = CONFIG["seq_len"]
BATCH_SIZE = CONFIG["batch_size"]
NUM_EPOCHS = CONFIG["num_epochs"]
LEARNING_RATE = CONFIG["learning_rate"]
FEATURE_DIM = CONFIG["feature_dim"]
TRAIN_DIR = CONFIG["paths"]["train_dir"]
VAL_DIR = CONFIG["paths"]["val_dir"]
MODEL_DIR = CONFIG["paths"]["model_dir"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
class CachedPoseDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for label_name, label_val in CLASS_LABELS.items():
            class_dir = os.path.join(root_dir, label_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.endswith(".npy"):
                    self.samples.append((os.path.join(class_dir, fname), label_val))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = np.load(path)
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Evaluation
def evaluate(model, loader, dataset_paths):
    model.eval()
    all_labels = []
    all_preds = []
    failed_samples = []

    with torch.no_grad():
        for i, (sequences, labels) in enumerate(loader):
            sequences = sequences.to(DEVICE)
            labels = labels.long().to(DEVICE)

            outputs = model(sequences).squeeze(1)
            preds = torch.argmax(outputs, dim=1)

            for j in range(len(preds)):
                true_label = labels[j].item()
                pred_label = preds[j].item()
                if int(true_label) != int(pred_label):
                    failed_samples.append(dataset_paths[i * loader.batch_size + j])

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, zero_division=0, target_names=list(CLASS_LABELS.keys())))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    return accuracy, precision, recall, f1, failed_samples

# Training
def train(seq_len=SEQ_LEN, batch_size=BATCH_SIZE):
    train_ds = CachedPoseDataset(TRAIN_DIR)
    val_ds = CachedPoseDataset(VAL_DIR)

    # Load hardcoded class weights
    class_weights = torch.zeros(len(CLASS_LABELS), dtype=torch.float32)
    for class_name, class_idx in CLASS_LABELS.items():
        class_weights[class_idx] = CONFIG["class_weights"].get(class_name, 1.0)

    sample_weights = [class_weights[label] for _, label in train_ds.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ActionLSTMWithSpatioTemporalAttention(num_classes=len(CLASS_LABELS)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    config_id = f"pose-seq{seq_len}_batch{batch_size}"
    model_dir = os.path.join(MODEL_DIR, config_id)
    os.makedirs(model_dir, exist_ok=True)
    print(f"Saving model checkpoints to: {model_dir}")

    log_path = os.path.join(model_dir, "training_log_pose.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_acc", "precision", "recall", "f1"])

    best_val_acc = 0
    best_epoch = 0
    best_f1 = 0
    best_ckpt = ""

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for sequences, labels in loop:
            sequences = sequences.to(DEVICE)
            labels = labels.long().to(DEVICE)

            outputs = model(sequences)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            total_correct += (preds == labels).sum().item()
            total_loss += loss.item() * sequences.size(0)
            total_samples += sequences.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item(), acc=(total_correct / total_samples))

        train_acc = total_correct / total_samples
        avg_loss = total_loss / total_samples
        val_paths = [path for path, _ in val_ds.samples]
        val_acc, val_precision, val_recall, val_f1, failed_samples = evaluate(model, val_loader, val_paths)
        gap = train_acc - val_acc

        print(f"Epoch {epoch+1:02d}: Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%} | Gap: {gap:.2%} | Loss: {avg_loss:.4f} | Precision: {val_precision:.2f} | Recall: {val_recall:.2f} | F1: {val_f1:.2f}")

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_loss, train_acc, val_acc, val_precision, val_recall, val_f1])

        ckpt_name = f"pose_epoch{epoch+1:02d}_valacc_{val_acc*100:.2f}.pt"
        ckpt_path = os.path.join(model_dir, ckpt_name)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc
        }, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

        if failed_samples:
            failure_log_path = os.path.join(model_dir, f"val_failures_epoch{epoch+1:02d}.txt")
            with open(failure_log_path, "w") as f:
                for path in failed_samples:
                    f.write(f"{path}\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_f1 = val_f1
            best_ckpt = ckpt_name

    final_path = os.path.join(model_dir, "pose_lstm_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved: {final_path}")

    master_log = os.path.join(MODEL_DIR, "training_master_log.csv")
    if not os.path.exists(master_log):
        with open(master_log, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["config_id", "seq_len", "batch_size", "best_val_acc", "best_f1", "best_epoch", "checkpoint"])

    with open(master_log, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([config_id, seq_len, batch_size, best_val_acc, best_f1, best_epoch, best_ckpt])

    summary_log_path = os.path.join(MODEL_DIR, "training_summary_log.txt")
    with open(summary_log_path, "a") as f:
        f.write("="*80 + "\n")
        f.write(f"Timestamp     : {datetime.now()}\n")
        f.write(f"Config ID     : {config_id}\n")
        f.write(f"SEQ_LEN       : {seq_len}\n")
        f.write(f"BATCH_SIZE    : {batch_size}\n")
        f.write(f"Train Samples : {len(train_ds)}\n")
        f.write(f"Val Samples   : {len(val_ds)}\n")
        f.write(f"Best Epoch    : {best_epoch}\n")
        f.write(f"Val Accuracy  : {best_val_acc:.4f}\n")
        f.write(f"Val F1 Score  : {best_f1:.4f}\n")
        f.write(f"Checkpoint    : {best_ckpt}\n")
        f.write("="*80 + "\n\n")

if __name__ == "__main__":
    train()
