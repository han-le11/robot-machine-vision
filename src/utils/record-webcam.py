import cv2
import os
import time

# Config
OUTPUT_DIR = "recordings"       # Root for recordings
LABEL = "no-gesture"              # Label for the clips which determines the folder name
CLIP_DURATION = 5               # Seconds per clip
FRAME_RATE = 30                 # Target frames per second
DEVICE_INDEX = 0                # Webcam index (default is 0)
CLIP_COUNT = 17                 # How many clips to record

save_path = os.path.join(OUTPUT_DIR, LABEL)
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(DEVICE_INDEX)
assert cap.isOpened(), "Failed to open webcam"

print(f"Recording {CLIP_COUNT} x {CLIP_DURATION}s clips labeled as '{LABEL}'...")

for i in range(CLIP_COUNT):
    filename = os.path.join(save_path, f"clip_{int(time.time())}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(filename, fourcc, FRAME_RATE, (width, height))

    print(f"Recording clip {i + 1}/{CLIP_COUNT}: {filename}")
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)
        cv2.imshow("Recording", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Interrupted by user.")
            break

        if time.time() - start_time >= CLIP_DURATION:
            break

    out.release()
    print(f"Saved {filename}")
    time.sleep(0.01)

cap.release()
cv2.destroyAllWindows()
print("🎬 Recording complete.")
