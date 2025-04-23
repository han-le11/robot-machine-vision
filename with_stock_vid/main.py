import sys
import os

current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(current_dir)

sys.path.insert(0, project_dir)

from yolo_models.motion_detector import MotionDetector

if __name__ == "__main__":
    detector = MotionDetector()
    detector.process_videos(folder_path="with_stock_vid/test-videos")