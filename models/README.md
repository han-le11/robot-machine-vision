## Class Overview
### `MotionDetector`
- **Methods**:
  - `detect_and_wave(frame)`:
    - Detects a waving motion in a single frame.
  - `process_videos(folder_path)`:
    - Processes all video files in the specified folder for waving detection.
  - `process_camera()`:
    - Uses the Intel RealSense camera to detect waving in real-time.
    
- **Machine learning model**:
  - YOLOv8 for person detection, by [Ultralytics](https://github.com/ultralytics/yolov8). Key features of YOLOv8 model [here](https://docs.ultralytics.com/models/yolov8/#key-features-of-yolov8) 