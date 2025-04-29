import cv2

cap = cv2.VideoCapture("vid.mp4")
width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print(f"Resolution: {int(width)} x {int(height)}")
cap.release()
