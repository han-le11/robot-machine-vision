import cv2
import mediapipe as mp
from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO("yolo11n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo11n.pt")

# Initialize Mediapipe for pose estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils


def detect_and_wave(frame):
    # Step 1: Use YOLO for human detection
    results = model.predict(frame)
    humans = [res for res in results if res['class'] == 'person']

    if not humans:
        return frame, False  # No humans detected

    # Step 2: Pose estimation. This is not refined yet.
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(img_rgb)

    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract keypoints
        landmarks = results_pose.pose_landmarks.landmark

        # Identify hand positions
        left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        is_waving = False
        # Check if a hand is raised above the shoulder.
        # Detection of an actual waving hand needs to be implemented.
        if (right_hand.y < shoulder.y) or (left_hand.y < shoulder.y):
            # Check for waving motion here, then set is_waving to True
            is_waving = True

        return frame, is_waving

    return frame, False


# Video capture
capture = cv2.VideoCapture(0)

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    frame, is_waving = detect_and_wave(frame)
    if is_waving:
        cv2.putText(frame, "Waving Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
