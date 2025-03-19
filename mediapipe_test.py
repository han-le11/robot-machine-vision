import cv2
import mediapipe as mp

class HandData:
    def __init__(self, top, bottom, left, right, centerX):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.centerX = centerX
        self.prevCenterX = 0
        self.isInFrame = False
        self.isWaving = False

    def update(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def check_for_waving(self, centerX):
        self.prevCenterX = self.centerX
        self.centerX = centerX
        if abs(self.centerX - self.prevCenterX) > 0.05:  # Adjust threshold as needed
            self.isWaving = True
        else:
            self.isWaving = False

class HandWavingRecognizer:
    def __init__(self, frame_width=600, frame_height=400, calibration_time=30, bg_weight=0.5, obj_threshold=18):
        self.FRAME_HEIGHT = frame_height
        self.FRAME_WIDTH = frame_width

        # Mediapipe Hands setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

        # Initialize video capture
        self.capture = cv2.VideoCapture(0)

        # Hand data
        self.hand_data = None

        # Hold the background frame for background subtraction.
        self.background = None
        # Variables to count how many frames have passed and to set the size of the window.
        self.frames_elapsed = 0

        # Try editing these if your program has trouble recognizing your skin tone.
        self.CALIBRATION_TIME = calibration_time
        self.BG_WEIGHT = bg_weight
        self.OBJ_THRESHOLD = obj_threshold

    def detect_hand_data(self, landmarks, frame_width, frame_height):
        top = (int(landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * frame_width),
               int(landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * frame_height))
        bottom = (int(landmarks[self.mp_hands.HandLandmark.WRIST].x * frame_width),
                  int(landmarks[self.mp_hands.HandLandmark.WRIST].y * frame_height))
        left = (int(landmarks[self.mp_hands.HandLandmark.THUMB_TIP].x * frame_width),
                int(landmarks[self.mp_hands.HandLandmark.THUMB_TIP].y * frame_height))
        right = (int(landmarks[self.mp_hands.HandLandmark.PINKY_TIP].x * frame_width),
                 int(landmarks[self.mp_hands.HandLandmark.PINKY_TIP].y * frame_height))
        centerX = (left[0] + right[0]) / 2

        if not self.hand_data:
            self.hand_data = HandData(top, bottom, left, right, centerX)
        else:
            self.hand_data.update(top, bottom, left, right)
            self.hand_data.check_for_waving(centerX)

    def process_frame(self):
        # Capture frame from the video
        ret, frame = self.capture.read()
        if not ret:
            print("Failed to capture video frame.")
            return None, None

        # Resize and flip the frame
        frame = cv2.resize(frame, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
        frame = cv2.flip(frame, 1)
        return frame, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def detect_waving(self, landmarks):
        """
        Detect if the hand is waving based on landmarks.
        Here, we'll check if the wrist and fingers' relative motion suggest waving.
        """
        if landmarks:
            wrist_y = landmarks[self.mp_hands.HandLandmark.WRIST].y
            index_tip_y = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            return abs(wrist_y - index_tip_y) > 0.1  # Simple heuristic for waving
        return False

    def run(self):
        while True:
            # Process the current frame
            frame, rgb_frame = self.process_frame()
            if frame is None or rgb_frame is None:
                break

            # Detect hands
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # Check if the hand is waving
                    if self.detect_waving(hand_landmarks.landmark):
                        cv2.putText(frame, "Waving Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("Hand Waving Detection", frame)

            # Check if the user wants to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        self.cleanup()

    def cleanup(self):
        self.capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = HandWavingRecognizer()
    recognizer.run()