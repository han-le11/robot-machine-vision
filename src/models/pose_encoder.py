import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

class PoseEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Calculate feature dimension:
        # - 33 pose landmarks * 3 coordinates = 99
        # - 468 face landmarks * 3 coordinates = 1404
        # - 21 hand landmarks * 3 coordinates * 2 hands = 126
        # Total: 99 + 1404 + 126 = 1629
        self.feature_dim = 1629

    def forward(self, x):
        # x is a batch of frames [B, T, C, H, W]
        B, T, C, H, W = x.size()
        features = []

        # Process each sample in the batch
        for b in range(B):
            sequence_features = []
            for t in range(T):
                # Convert tensor to numpy and to RGB
                frame = x[b, t].permute(1, 2, 0).cpu().numpy()
                frame = (frame * 255).astype(np.uint8)

                # Process frame with MediaPipe
                results = self.holistic.process(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                # Extract pose landmarks
                pose_landmarks = np.zeros((33, 3))
                if results.pose_landmarks:
                    for i, landmark in enumerate(results.pose_landmarks.landmark):
                        pose_landmarks[i] = [landmark.x, landmark.y, landmark.z]

                # Extract face landmarks (MediaPipe Holistic uses 468 face landmarks)
                face_landmarks = np.zeros((468, 3))
                if results.face_landmarks:
                    for i, landmark in enumerate(results.face_landmarks.landmark):
                        if i < 468:
                            face_landmarks[i] = [landmark.x, landmark.y, landmark.z]

                # Extract hand landmarks
                left_hand = np.zeros((21, 3))
                right_hand = np.zeros((21, 3))
                if results.left_hand_landmarks:
                    for i, landmark in enumerate(results.left_hand_landmarks.landmark):
                        left_hand[i] = [landmark.x, landmark.y, landmark.z]
                if results.right_hand_landmarks:
                    for i, landmark in enumerate(results.right_hand_landmarks.landmark):
                        right_hand[i] = [landmark.x, landmark.y, landmark.z]

                # Concatenate all landmarks
                frame_features = np.concatenate([
                    pose_landmarks.flatten(),
                    face_landmarks.flatten(),
                    left_hand.flatten(),
                    right_hand.flatten()
                ])

                # Normalize feature vector per frame
                mean = np.mean(frame_features)
                std = np.std(frame_features) + 1e-6
                frame_features = (frame_features - mean) / std

                # Optional logging
                # print(f"Frame mean: {mean:.4f}, std: {std:.4f}, max: {frame_features.max():.2f}, min: {frame_features.min():.2f}")

                sequence_features.append(frame_features)

            features.append(sequence_features)

        # After all samples in batch processed
        features = np.array(features)  # [B, T, 1629]
        print("PoseEncoder output shape:", features.shape)
        features = torch.from_numpy(features).float().to(x.device)

        print("Feature mean:", features.mean().item())
        print("Feature std:", features.std().item())
        print("Nonzero ratio:", (features != 0).float().mean().item())

        return features

    def freeze(self):
        # No parameters to freeze since we're using MediaPipe
        self.eval()
