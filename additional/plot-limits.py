import cv2

VIDEO_PATH = "video_data/video1.mp4"
WINDOW = "Pick 2 Points for limits (click) | r=reset q=quit"

pts = []

def mouse_cb(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pts) < 2:
            pts.append((x, y))
            print(f"Point {len(pts)}: ({x}, {y})")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError("Video tidak bisa dibuka")

cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ret, frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError("Gagal baca frame")

cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, 1280, 720)
cv2.setMouseCallback(WINDOW, mouse_cb)

while True:
    vis = frame.copy()

    for i, (x, y) in enumerate(pts):
        cv2.circle(vis, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(vis, str(i+1), (x+6, y-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    if len(pts) == 2:
        cv2.line(vis, pts[0], pts[1], (0, 0, 255), 5)

    cv2.imshow(WINDOW, vis)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("r"):
        pts = []
        print("Reset points")
    elif key == ord("q"):
        break

cv2.destroyAllWindows()

if len(pts) == 2:
    (x1, y1), (x2, y2) = pts
    print("\n=== COPY THIS INTO YOUR CODE ===")
    print(f"limits = [{x1}, {y1}, {x2}, {y2}]")
else:
    print("Belum lengkap, butuh 2 titik")
