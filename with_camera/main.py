import sys
import os

# Script to run and test WaveDetector with a camera

current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(current_dir)

sys.path.insert(0, project_dir)

from models.wave_detector import WaveDetector

if __name__ == "__main__":
    detector = WaveDetector()
    detector.process_camera()
