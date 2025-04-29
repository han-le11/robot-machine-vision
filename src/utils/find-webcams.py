import cv2

def list_available_cameras(max_index=5):
    print("🔍 Scanning for available cameras...")
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera found at index {i}")
            cap.release()
        else:
            print(f"No camera at index {i}")

list_available_cameras()
