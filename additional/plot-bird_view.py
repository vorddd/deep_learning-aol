import cv2
import numpy as np

VIDEO_PATH = "video_data/video1.mp4"
WINDOW_NAME = "Pick 4 SOURCE Points (Click TL, TR, BR, BL)"

points = []

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError("Video tidak bisa dibuka")

cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Gagal membaca frame")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

while True:
    vis = frame.copy()

    # gambar titik & garis
    for i, (x, y) in enumerate(points):
        cv2.circle(vis, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(
            vis, f"{i+1}", (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2
        )

    if len(points) == 4:
        cv2.polylines(vis, [np.array(points)], True, (0,255,0), 2)

    cv2.imshow(WINDOW_NAME, vis)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("r"): # reset
        points = []
        print("Reset points")
    elif key == ord("q"): # quit
        break

cv2.destroyAllWindows()

if len(points) == 4:
    src = np.array(points, dtype=np.float32)
    print("\n=== COPY THIS INTO YOUR CODE ===")
    print("SOURCE = np.array([")
    for p in src:
        print(f"    [{int(p[0])}, {int(p[1])}],")
    print("], dtype=np.float32)")
else:
    print("Tidak lengkap, butuh 4 titik")
