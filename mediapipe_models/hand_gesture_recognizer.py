import cv2
import mediapipe as mp
import numpy as np
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from socketMessage import RobotSocketClient
from notification import NotificationListener

from math import acos, degrees
from numpy import array, dot
from numpy.linalg import norm

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

class GestureDetector:
    def __init__(self,
                 model_path="mediapipe_models/gesture_recognizer.task",
                 ):
        """Initialize the GestureDetector with the given model path."""
        self.model_path = model_path
        self.gesture_result = None  # Store detected gesture
        self.last_gesture_sent = None # The gesture that was last sent to the robot
        self.timestamp = 0  # Monotonic timestamp counter

        def _on_notify(msg: str):
            # msg will be like "START_Thumb_Up" or "END_Thumb_Up"
            print("[RAPID→Python]", msg)
            if msg.startswith("START_"):
                self.doing_action = True
            elif msg.startswith("END_"):
                self.doing_action = False

        self.doing_action = False
        self.robot_client = RobotSocketClient(host="192.168.125.1", port=5000)
        self.notifier = NotificationListener(
                            host="192.168.125.1",
                            port=5001,
                            on_message=_on_notify,
                            retry_delay=1.0,
                            recv_timeout=0.1,
                            delimiter="|"   # or "\n" if RAPID uses newline
                        )

        # Initialize MediaPipe Hands for skeleton tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.6)

        # A dictionary that loads images for gestures
        self.gesture_images = {
            "Closed_Fist" : cv2.imread("./pictograms/fist.jpg", cv2.IMREAD_UNCHANGED),
            "Thumb_Up": cv2.imread("./pictograms/thumb_up.jpg", cv2.IMREAD_UNCHANGED),
            "Thumb_Down": cv2.imread("./pictograms/thumb_down.jpg", cv2.IMREAD_UNCHANGED),
            "Victory": cv2.imread("./pictograms/victory.jpg", cv2.IMREAD_UNCHANGED),
            "Open_Palm": cv2.imread("./pictograms/wave.jpg", cv2.IMREAD_UNCHANGED),
            "Middle_Finger": cv2.imread("./pictograms/middle_finger.jpg", cv2.IMREAD_UNCHANGED),
            "None": cv2.imread("./pictograms/none.jpg", cv2.IMREAD_UNCHANGED),
            # Add more gestures and images as needed
        }

        # Define callback function
        def result_callback(result, output_image, timestamp_ms):
            """Callback function to handle gesture recognition results."""
            if result.gestures:
                self.gesture_result = result.gestures[0][0].category_name  # Most confident gesture
            else:
                self.gesture_result = None  # No gesture detected

        # Set up gesture recognizer options
        self.options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_buffer=open(model_path, "rb").read()),
            running_mode=VisionRunningMode.LIVE_STREAM,  # Required for real-time video mode
            result_callback=result_callback  # Callback function for async processing
        )

        # Initialize the gesture recognizer
        self.recognizer = GestureRecognizer.create_from_options(self.options)
        # The camera index 0 is often the built-in webcam. You may need to change it if you have multiple cameras.
        self.cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
        cv2.namedWindow("Gesture Recognition with Hand Tracking", cv2.WINDOW_NORMAL)

    def detect_middle_finger(self, hand_landmarks) -> bool:
        """
        Detect if the middle finger is extended while other fingers are bent.

        Args:
            hand_landmarks: Hand landmarks detected by MediaPipe's Hands.

        Returns:
            bool: True if the middle finger gesture is detected, False otherwise.
        """
        # Finger landmark indices as defined by MediaPipe
        THUMB_TIP, THUMB_BASE = 4, 2
        INDEX_TIP, INDEX_BASE = 8, 6
        MIDDLE_TIP, MIDDLE_DIP, MIDDLE_PIP, MIDDLE_MCP = 12, 11, 10, 9
        RING_TIP, RING_BASE = 16, 14
        PINKY_TIP, PINKY_BASE = 20, 18

        # Helper function with logic to detect middle finger:
        # Middle finger's tip must be above its base, and other fingers' tips should be below their bases.
        def is_extended(tip, base):
            return tip.y < base.y
        
        def angle_between(a, b, c): # Math used to make sure that the finger is straight
            ab = array([a.x - b.x, a.y - b.y, a.z - b.z])
            cb = array([c.x - b.x, c.y - b.y, c.z - b.z])
            cosine_angle = dot(ab, cb) / (norm(ab) * norm(cb))
            return degrees(acos(cosine_angle))

        middle_finger_extended = is_extended(hand_landmarks.landmark[MIDDLE_TIP],
                                             hand_landmarks.landmark[MIDDLE_PIP])
        
        other_fingers_bent = all(
            hand_landmarks.landmark[landmark].y > hand_landmarks.landmark[MIDDLE_PIP].y
            for landmark in [THUMB_BASE, INDEX_BASE, RING_BASE, PINKY_BASE, THUMB_TIP, INDEX_TIP, RING_TIP, PINKY_TIP]
        )

        # Variable definitions for logic
        mcp = hand_landmarks.landmark[MIDDLE_MCP]
        pip = hand_landmarks.landmark[MIDDLE_PIP]
        dip = hand_landmarks.landmark[MIDDLE_DIP]
        tip = hand_landmarks.landmark[MIDDLE_TIP]

        # Middle finger pointing up logic
        finger_vector = array([tip.x - mcp.x, tip.y - mcp.y, tip.z - mcp.z])
        angle0 = degrees(acos(dot(finger_vector, [0, -1, 0]) / norm(finger_vector)))
        middle_finger_pointing_up = angle0 < 10

        # Middle finger straight logic
        angle1 = angle_between(mcp, pip, dip)
        angle2 = angle_between(pip, dip, tip)

        middle_finger_straight = angle1 > 175 and angle2 > 175  # The integers represent the angle of the finger as calculated from the landmarks

        return (middle_finger_extended and
                other_fingers_bent and
                middle_finger_straight and
                middle_finger_pointing_up)

    def draw_hand_skeleton(self, frame, hand_landmarks) -> None:
        """Draw the hand skeleton using MediaPipe landmarks."""
        for landmarks in hand_landmarks:
            for connection in self.mp_hands.HAND_CONNECTIONS:
                start = landmarks.landmark[connection[0]]
                end = landmarks.landmark[connection[1]]

                # Convert to pixel coordinates
                h, w, _ = frame.shape
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))

                # Draw connections between hand joints in white
                cv2.line(frame, start_point, end_point, (255, 255, 255), 2)

            # Draw hand keypoints in red
            for landmark in landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 4, (0, 0, 128), -1)

            # Detect if the middle finger gesture is present
            if self.detect_middle_finger(landmarks):
                self.gesture_result = "Middle_Finger"

    @staticmethod
    def overlay_image(frame, overlay, x=0, y=0, scale=1.0):
        """
        Overlays an image (with or without transparency) on the video feed at the specified position.

        Args:
            frame: The video frame (background).
            overlay: The overlay image (can have alpha channel or black background).
            x: x-coordinate for the top-left corner of the overlay.
            y: y-coordinate for the top-left corner of the overlay.
            scale: Scaling factor for the overlay image.
        """
        # Resize overlay based on scale
        overlay = cv2.resize(overlay, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        overlay_h, overlay_w, _ = overlay.shape

        # Get frame dimensions
        frame_h, frame_w, _ = frame.shape

        # Ensure the overlay fits within the frame boundaries
        y1, y2 = max(0, y), min(frame_h, y + overlay_h)
        x1, x2 = max(0, x), min(frame_w, x + overlay_w)

        # Check for valid overlay position
        if y1 >= y2 or x1 >= x2:
            print(f"Overlay out of bounds: x={x}, y={y}, frame dimensions={frame.shape}")
            return  # Skip overlay

        # Crop the overlay if necessary
        overlay = overlay[0:y2 - y1, 0:x2 - x1]

        # Copy overlay onto the frame
        roi = frame[y1:y2, x1:x2]
        try:
            np.copyto(roi, overlay, casting="same_kind")
        except Exception as e:
            print(f"Error during overlay: {e}")

    @staticmethod
    def display_text(frame, text, x=10, y=30, margin=30, line_spacing=30):
        """
        Display multiline text in the top-right corner of the video feed, aligned to the left.

        Args:
            frame: The frame where text is displayed.
            text: The text to display. Use '\n' for new lines.
            x: X-coordinate of the text.
            y: Y-coordinate of the text.
            line_spacing: Spacing between two lines of text.
            align: Alignment of the text ('left' or 'right').
        """
        # Split the text into lines based on '\n'
        lines = text.split('\n')

        for i, line in enumerate(lines):
            # text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_y = y + i * line_spacing
            cv2.putText(frame, line, (x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    def send_gesture_message(self):
        gesture = self.gesture_result
        gesture_text = self.gesture_result.strip().lower()
        """Send socket message only if gesture has changed."""
        if gesture and gesture.strip().lower() != "none" and ((self.last_gesture_sent == "Thumb_Down"
                                                               and gesture == "Thumb_Up")
                                                              or not self.doing_action):
            self.last_gesture_sent = gesture
            self.robot_client.send_message(gesture)
        elif gesture_text == "none" or gesture_text == "pointing_up" or gesture_text == "iloveyou":
            print("Skipping invalid gesture:", gesture)
        else:
            print("Robot is already doing action")

    def run(self):
        """Start real-time gesture recognition with hand tracking."""
        last_message_time = 0  # Last message timestamp
        if not self.cap.isOpened():
            print("Could not open camera. Check the camera connection.")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            # Convert frame to RGB format (MediaPipe expects RGB images)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect hand landmarks
            hand_results = self.hands.process(frame_rgb)

            # Draw hand skeleton if detected hand(s)
            if hand_results.multi_hand_landmarks:
                self.draw_hand_skeleton(frame, hand_results.multi_hand_landmarks)

            # Convert to MediaPipe Image format for gesture recognition
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Send frame for recognition (async mode) with a manual timestamp
            self.timestamp += 30  # Increment timestamp (approx. 30ms per frame)
            self.recognizer.recognize_async(mp_image, self.timestamp)

            current_time = time.time()
            x, y = 0, 0  # Top-left corner coordinates

            # Display detected gesture on the screen
            if self.gesture_result is not None and self.doing_action:
                print(self.gesture_result)
                gesture_image = self.gesture_images.get(self.gesture_result, None)
                if gesture_image is not None:
                    self.overlay_image(frame=frame, overlay=gesture_image, x=x, y=y, scale=0.4)
                    self.display_text(frame=frame,
                                      text=f"Wait! Robot is answering to your gesture: \n"
                                           f"{self.last_gesture_sent}",
                                      )

                if current_time - last_message_time >= 1.5:
                    # TODO: comment out the line below if not connected to server
                    self.send_gesture_message()
                    last_message_time = current_time
            else:
                # Always display blank screen to avoid flickering time
                blank_screen = self.gesture_images.get("None")
                self.overlay_image(frame=frame, overlay=blank_screen, x=x, y=y, scale=0.4)

            # Show the video feed flipped horizontally
            cv2.imshow("Gesture Recognition with Hand Tracking", frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.robot_client.close()
        self.notifier.stop()

# Run the gesture detector
if __name__ == "__main__":
    detector = GestureDetector("mediapipe_models/gesture_recognizer.task")
    detector.run()
