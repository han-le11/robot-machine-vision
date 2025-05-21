import time
import json
import cv2
import torch
import numpy as np
from collections import deque
from src.models.pose_model import ActionLSTMWithSpatioTemporalAttention
from src.utils.pose_extractor import PoseExtractor
from src.utils.socketMessage import send_socket_message

with open("config/config.json", "r") as f:
    CONFIG = json.load(f)

CLASS_LABELS = CONFIG["labels"]
CLASS_NAMES = list(CLASS_LABELS.keys())
SEQ_LEN = CONFIG["seq_len"]
FEATURE_DIM = CONFIG["feature_dim"]
NUM_CLASSES = len(CLASS_LABELS)

DEVICE_INDEX = 0  # Webcam device index
CKPT_PATH = "outputs\models\pose_epoch132_valacc_88.78.pt"

gesture_scores = {label: 0.0 for label in CLASS_NAMES}
SCORE_THRESHOLD = 10.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ActionLSTMWithSpatioTemporalAttention(num_classes=len(CLASS_LABELS)).to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

def run_inference(input_np):
    tensor = torch.tensor(input_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return probs

extractor = PoseExtractor()
buffer = deque(maxlen=SEQ_LEN)

cap = cv2.VideoCapture(DEVICE_INDEX)
cv2.namedWindow("Pose Inference", cv2.WINDOW_NORMAL)

if not cap.isOpened():
    print("Could not open webcam.")
    exit()

print("Webcam started. Press 'q' to quit.")

prev_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    curr_time = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / (curr_time - prev_time))
    prev_time = curr_time

    if not ret:
        print("Failed to grab frame. Exiting.")
        break

    frame = cv2.resize(frame, (480, 480))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    start_mediapipe = time.perf_counter()
    pose_results = extractor.pose.process(frame_rgb)
    hands_results = extractor.hands.process(frame_rgb)
    features = extractor._extract_features(pose_results, hands_results)
    buffer.append(features if features is not None else np.zeros(FEATURE_DIM, dtype=np.float32))
    time_mediapipe = time.perf_counter() - start_mediapipe

    if len(buffer) == SEQ_LEN:
        start_inference = time.perf_counter()
        sequence_np = np.stack(buffer)
        prob = run_inference(sequence_np)

        # Accumulate scores
        for label, p in zip(CLASS_NAMES, prob):
            gesture_scores[label] += p

        # Check for trigger
        for label, score in gesture_scores.items():
            if score >= SCORE_THRESHOLD:
                print(f"Gesture detected: {label} (score={score:.2f})")
                # The following row is used to send the socket messages to the robot but the functionality has not been implemented in this version of the code. The class labels are not the same in this branch and must be matched to the original ones to implement this feature.
                # send_socket_message(label.upper())
                gesture_scores[label] = 0.0  # Reset after sending

        for label in gesture_scores.keys():
            gesture_scores[label] *= 0.95  # Time (frame) decay



        time_inference = time.perf_counter() - start_inference

        start_viz = time.perf_counter()
        top_class = int(np.argmax(prob))
        top_label = CLASS_NAMES[top_class]
        confidence = f"{prob[top_class] * 100:.1f}%"
        color = (0, 255, 0)

        cv2.putText(frame, f"{top_label} ({confidence})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.rectangle(frame, (20, 60), (20 + int(prob[top_class] * 200), 90), color, -1)

        # Optional: print all class confidences to console
        for label, p in zip(CLASS_NAMES, prob):
            print(f"{label}: {p:.2%}", end=" | ")
        print()

        time_viz = time.perf_counter() - start_viz
        print(f"MP: {time_mediapipe:.4f}s, Inf: {time_inference:.4f}s, Viz: {time_viz:.4f}s")

    h, w, _ = frame.shape
    fps_text = f"{fps:.1f} FPS"
    cv2.putText(frame, fps_text, (w - 150, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Pose Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
