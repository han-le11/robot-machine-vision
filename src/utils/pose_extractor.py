import cv2
import numpy as np
import mediapipe as mp


class PoseExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands

        self.pose = self.mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )

    def _extract_features(self, pose_results, hands_results):
        def landmark_list(landmarks, count):
            if landmarks and len(landmarks.landmark) == count:
                return [(lm.x, lm.y, lm.z, lm.visibility) for lm in landmarks.landmark]
            else:
                return [(0.0, 0.0, 0.0, 0.0)] * count

        pose = landmark_list(pose_results.pose_landmarks, 33)

        # Initialize both hands to zero
        left = [(0.0, 0.0, 0.0, 0.0)] * 21
        right = [(0.0, 0.0, 0.0, 0.0)] * 21

        if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
            for i, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                label = hands_results.multi_handedness[i].classification[0].label
                if label == 'Left':
                    left = landmark_list(hand_landmarks, 21)
                elif label == 'Right':
                    right = landmark_list(hand_landmarks, 21)

        all_landmarks = pose + left + right
        return np.array(all_landmarks).flatten()  # 132 (pose) + 84 (left) + 84 (right) = 300

    def extract_to_npy(self, input_path, output_path, seq_len=32):
        cap = cv2.VideoCapture(input_path)
        sequence = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (480, 480))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(frame_rgb)
            hands_results = self.hands.process(frame_rgb)
            features = self._extract_features(pose_results, hands_results)

            if features is not None:
                sequence.append(features)

            if len(sequence) >= seq_len:
                break

        cap.release()

        if len(sequence) < seq_len:
            return False

        np.save(output_path, np.array(sequence[:seq_len]))
        return True

    def extract_sliding(self, input_path, seq_len=32, step=16):
        cap = cv2.VideoCapture(input_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (480, 480))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(frame_rgb)
            hands_results = self.hands.process(frame_rgb)
            features = self._extract_features(pose_results, hands_results)

            if features is not None:
                frames.append(features)

        cap.release()

        sequences = []
        for start in range(0, len(frames) - seq_len + 1, step):
            chunk = frames[start:start + seq_len]
            if len(chunk) == seq_len:
                first_shape = len(chunk[0])
                if all(len(f) == first_shape for f in chunk):
                    sequences.append(np.array(chunk))

        return sequences

    def extract_from_frame(self, frame):
        frame = cv2.resize(frame, (480, 480))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(frame_rgb)
        hands_results = self.hands.process(frame_rgb)
        features = self._extract_features(pose_results, hands_results)

        return features if features is not None else np.zeros(300, dtype=np.float32)
