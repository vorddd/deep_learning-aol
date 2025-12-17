from ultralytics import YOLO
import torch
import numpy as np
import warnings
from collections import defaultdict, deque
from sort import Sort
import cv2
import csv
from pathlib import Path
import sys
import argparse
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
import traceback

warnings.filterwarnings("ignore", category=UserWarning)

# --- FIX torch.load weights_only issue (PyTorch 2.6+) ---
_real_torch_load = torch.load
def torch_load_legacy(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _real_torch_load(*args, **kwargs)
torch.load = torch_load_legacy
# --------------------------------------------------------

def _safe_write_text(path: Path, text: str):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", errors="ignore")
    except Exception:
        pass

def _crash_log_path() -> Path:
    try:
        base = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
    except Exception:
        base = Path.cwd()
    return base / "outputs" / "crash.log"

def pick_video_dialog():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*")]
    )
    root.destroy()
    return path

def show_error_box(title: str, msg: str):
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(title, msg)
        root.destroy()
    except Exception:
        # fallback kalau GUI error
        print(f"[ERROR] {title}\n{msg}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=str, default="", help="Path to input video")
    p.add_argument("--weights", type=str, default="Yolo-Weights/yolov8m.pt", help="Path to YOLO .pt")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.6)
    p.add_argument("--no-ui", action="store_true", help="Disable UI window display")
    p.add_argument("--save-video", action="store_true", help="Save overlay result video to outputs folder")
    p.add_argument("--display-w", type=int, default=1280)
    p.add_argument("--display-h", type=int, default=720)
    return p.parse_args()

def is_frozen():
    return getattr(sys, "frozen", False)

def exe_dir() -> Path:
    return Path(sys.executable).parent if is_frozen() else Path(__file__).parent

def bundled_dir() -> Path | None:
    if is_frozen() and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return None

def resolve_weights_path(rel_or_user_path: str) -> Path:
    """
    Priority:
    1) If user gives an absolute path and exists -> use it
    2) If bundled in PyInstaller via --add-data -> use sys._MEIPASS
    3) If exists next to app.exe (VehicleApp/Yolo-Weights/...) -> use exe_dir
    4) If exists relative to current working dir -> use it (dev mode)
    Else -> fail (NO DOWNLOAD)
    """
    p = Path(rel_or_user_path)

    if p.is_absolute() and p.exists():
        return p

    rel = Path("Yolo-Weights") / p.name if p.name.endswith(".pt") else p

    bdir = bundled_dir()
    if bdir is not None:
        cand = bdir / rel
        if cand.exists():
            return cand

    cand = exe_dir() / rel
    if cand.exists():
        return cand

    cand = Path.cwd() / rel
    if cand.exists():
        return cand

    return Path("")  # not found

class ViewTransformer:
    def __init__(self, source, target):
        self.m = cv2.getPerspectiveTransform(source.astype(np.float32), target.astype(np.float32))

    def transform_points(self, points):
        if points is None or len(points) == 0:
            return points
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        out = cv2.perspectiveTransform(pts, self.m)
        return out.reshape(-1, 2)

def in_count_zone(cx, cy, limits, tol):
    xA, yA, xB, yB = limits
    return (min(xA, xB) < cx < max(xA, xB)) and (yA - tol < cy < yA + tol)

def iou_xyxy(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0
    areaA = max(0, (a[2]-a[0])) * max(0, (a[3]-a[1]))
    areaB = max(0, (b[2]-b[0])) * max(0, (b[3]-b[1]))
    return inter / (areaA + areaB - inter + 1e-9)


# =========================
# Main
# =========================
def main():
    args = parse_args()

    launched_by_double_click = (len(sys.argv) == 1)

    if launched_by_double_click:
        video_path = pick_video_dialog()
    else:
        video_path = args.video.strip()
        if not video_path:
            video_path = pick_video_dialog()

    if not video_path:
        print("[EXIT] No video selected.")
        return

    weights_path = resolve_weights_path(args.weights)
    if not weights_path or not weights_path.exists():
        msg = (
            "YOLO weights not found.\n\n"
            "Expected one of these locations:\n"
            f"1) Bundled: <PyInstaller>\\Yolo-Weights\\*.pt\n"
            f"2) Next to app.exe: {exe_dir()}\\Yolo-Weights\\yolov8m.pt\n\n"
            "Fix:\n"
            "- Put yolov8m.pt inside VehicleApp\\Yolo-Weights\\\n"
            "OR\n"
            "- Bundle it using PyInstaller --add-data.\n"
        )
        show_error_box("Missing YOLO Weights", msg)
        return

    # ---------------- CONFIG ----------------
    CONF_TH = float(args.conf)
    IOU_TH  = float(args.iou)

    limits = [1248, 1152, 3654, 900]
    LINE_TOL = 150

    SOURCE = np.array([
        [1713, 480],
        [2406, 456],
        [3834, 1131],
        [897, 1746],
    ], dtype=np.float32)

    REAL_DISTANCE_METER = 140.0
    TARGET = np.array([
        [0, 0],
        [100, 0],
        [100, REAL_DISTANCE_METER],
        [0, REAL_DISTANCE_METER],
    ], dtype=np.float32)

    INFER_SCALE = 0.75
    INFER_EVERY = 1
    IMG_SIZE = 960

    BIN_SEC = 0.5

    SHOW_UI = (not args.no_ui)
    DISPLAY_SIZE = (int(args.display_w), int(args.display_h))

    STOP_SEC = 1.0
    MIN_MOVE_M  = 1.2
    MIN_MOVE_PX = 22
    STILL_NEED_CONSISTENT = 1

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = exe_dir() / "outputs" / f"run_{run_tag}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    TRACKS_CSV  = RUN_DIR / "tracks_log.csv"
    TS_CSV      = RUN_DIR / "timeseries_0p5s.csv"
    SUMMARY_TXT = RUN_DIR / "summary.txt"
    OUT_VIDEO   = RUN_DIR / "result_overlay.mp4"

    device = 0 if torch.cuda.is_available() else "cpu"
    use_half = torch.cuda.is_available()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        show_error_box("Video Error", f"Video not found / can't be opened:\n{video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 0 else 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = YOLO(str(weights_path))

    tracker = Sort(max_age=35, min_hits=2, iou_threshold=0.3)
    view_transformer = ViewTransformer(SOURCE, TARGET)

    SAVE_VIDEO_DEFAULT = True

    writer = None
    if args.save_video or SAVE_VIDEO_DEFAULT:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(OUT_VIDEO), fourcc, fps, (w, h))

    track_yhist = defaultdict(lambda: deque(maxlen=int(max(10, fps))))
    track_pixhist = defaultdict(lambda: deque(maxlen=int(max(10, fps))))
    speed_ema = {}
    SPEED_BETA = 0.85

    stop_frames_th = max(1, int(round(STOP_SEC * fps)))
    still_windows = defaultdict(int)

    VEHICLE_CLS = {2, 3, 5, 7}
    CLS_NAME = {2: "car", 3: "motorbike", 5: "bus", 7: "truck"}

    counts_by_class = {"car": 0, "motorbike": 0, "bus": 0, "truck": 0}
    track_class = {}

    totalCount = set()
    frame_idx = 0
    last_dets = np.empty((0, 5), dtype=np.float32)
    last_det_cls = []

    bin_frames = max(1, int(round(BIN_SEC * fps)))
    bin_density_ids = set()
    bin_speed_values = []
    bin_flow = 0
    ts_rows = []

    tracks_f = open(TRACKS_CSV, "w", newline="", encoding="utf-8")
    tracks_writer = csv.writer(tracks_f)
    tracks_writer.writerow([
        "frame_idx", "time_s", "track_id", "class",
        "cx", "cy", "speed_kmh", "counted_event", "is_moving"
    ])

    if SHOW_UI:
        cv2.namedWindow("Vehicle Counter", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Vehicle Counter", DISPLAY_SIZE[0], DISPLAY_SIZE[1])

    print("[INFO] video:", video_path)
    print("[INFO] weights:", str(weights_path))
    print(f"[INFO] fps≈{fps:.2f} | size={w}x{h}")
    print("[INFO] outputs:", RUN_DIR)

    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frame_idx += 1
            time_s = frame_idx / fps

            imgRegion = img

            if frame_idx % INFER_EVERY == 0:
                if INFER_SCALE != 1.0:
                    small = cv2.resize(imgRegion, None, fx=INFER_SCALE, fy=INFER_SCALE, interpolation=cv2.INTER_LINEAR)
                else:
                    small = imgRegion

                res = model.predict(
                    source=small,
                    conf=CONF_TH,
                    iou=IOU_TH,
                    imgsz=IMG_SIZE,
                    device=device,
                    half=use_half,
                    verbose=False
                )[0]

                dets_list = []
                det_cls_list = []

                if res.boxes is not None and len(res.boxes) > 0:
                    boxes = res.boxes
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    clss  = boxes.cls.cpu().numpy().astype(int)

                    if INFER_SCALE != 1.0:
                        xyxy /= INFER_SCALE

                    for (x1, y1, x2, y2), c, cls in zip(xyxy, confs, clss):
                        if cls not in VEHICLE_CLS:
                            continue
                        dets_list.append([x1, y1, x2, y2, c])
                        det_cls_list.append(int(cls))

                last_dets = np.array(dets_list, dtype=np.float32) if dets_list else np.empty((0, 5), dtype=np.float32)
                last_det_cls = det_cls_list

            tracks = tracker.update(last_dets)

            cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 4)

            for x1, y1, x2, y2, track_id in tracks:
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                tid = int(track_id)

                cx = (x1 + x2) // 2
                cy = y2

                cls_id = track_class.get(tid, None)
                best_iou = 0.0
                best_cls = None

                if len(last_dets) > 0:
                    tb = (x1, y1, x2, y2)
                    for j, det in enumerate(last_dets):
                        db = (int(det[0]), int(det[1]), int(det[2]), int(det[3]))
                        v = iou_xyxy(tb, db)
                        if v > best_iou:
                            best_iou = v
                            best_cls = last_det_cls[j] if j < len(last_det_cls) else None

                if best_cls is not None and best_iou >= 0.30:
                    track_class[tid] = best_cls
                    cls_id = best_cls

                cls_name = CLS_NAME.get(cls_id, None)

                speed_kmh = None

                ph = track_pixhist[tid]
                ph.append((cx, cy))

                disp_px = None
                if len(ph) >= stop_frames_th:
                    win = list(ph)[-stop_frames_th:]
                    xs = np.array([p[0] for p in win], dtype=np.float32)
                    ys = np.array([p[1] for p in win], dtype=np.float32)
                    disp_px = float(np.hypot(xs.max()-xs.min(), ys.max()-ys.min()))

                disp_m = None
                pt = np.array([[cx, cy]], dtype=np.float32)
                tpt = view_transformer.transform_points(pt)

                if tpt is not None and len(tpt) > 0:
                    ty = float(tpt[0][1])
                    hist = track_yhist[tid]
                    hist.append(ty)

                    if len(hist) >= stop_frames_th:
                        hwin = np.array(list(hist)[-stop_frames_th:], dtype=np.float32)
                        disp_m = float(np.max(hwin) - np.min(hwin))

                    min_len = max(8, int(fps // 3))
                    if len(hist) >= min_len:
                        diffs = np.diff(np.array(hist, dtype=np.float32))
                        dy_per_frame_m = float(np.median(np.abs(diffs)))
                        raw = (dy_per_frame_m * fps) * 3.6
                        prev = speed_ema.get(tid, raw)
                        speed_kmh = SPEED_BETA * prev + (1 - SPEED_BETA) * raw
                        speed_ema[tid] = speed_kmh

                is_moving = True
                if disp_px is not None or disp_m is not None:
                    px_ok = (disp_px is not None and disp_px >= MIN_MOVE_PX)
                    m_ok  = (disp_m  is not None and disp_m  >= MIN_MOVE_M)
                    is_moving = (px_ok or m_ok)

                if (disp_px is not None or disp_m is not None) and (not is_moving):
                    still_windows[tid] += 1
                else:
                    still_windows[tid] = 0

                if still_windows[tid] >= STILL_NEED_CONSISTENT:
                    continue

                if is_moving:
                    bin_density_ids.add(tid)

                if is_moving and speed_kmh is not None and 0 < speed_kmh < 170:
                    bin_speed_values.append(float(speed_kmh))

                counted_event = 0
                if is_moving and in_count_zone(cx, cy, limits, LINE_TOL):
                    if tid not in totalCount:
                        totalCount.add(tid)
                        counted_event = 1
                        bin_flow += 1

                        if cls_name in counts_by_class:
                            counts_by_class[cls_name] += 1

                        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 4)

                tracks_writer.writerow([
                    frame_idx,
                    round(time_s, 4),
                    tid,
                    cls_name,
                    cx, cy,
                    None if speed_kmh is None else round(float(speed_kmh), 3),
                    counted_event,
                    int(is_moving)
                ])

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                label = f"ID {tid}"
                if cls_name is not None:
                    label += f" | {cls_name}"
                cv2.putText(img, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

                if speed_kmh is not None and 0 < speed_kmh < 170:
                    cv2.putText(img, f"{int(speed_kmh)} km/h", (x1, y2 + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            if frame_idx % bin_frames == 0:
                t_end = time_s
                t_start = t_end - BIN_SEC

                density = len(bin_density_ids)
                avg_speed = float(np.mean(bin_speed_values)) if len(bin_speed_values) > 0 else np.nan

                ts_rows.append([
                    round(t_start, 3),
                    round(t_end, 3),
                    int(bin_flow),
                    int(density),
                    None if np.isnan(avg_speed) else round(avg_speed, 3)
                ])

                bin_density_ids.clear()
                bin_speed_values.clear()
                bin_flow = 0

            cv2.putText(img, f"Total: {len(totalCount)}", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 3, cv2.LINE_AA)
            cv2.putText(
                img,
                f"car:{counts_by_class['car']}  moto:{counts_by_class['motorbike']}  bus:{counts_by_class['bus']}  truck:{counts_by_class['truck']}",
                (40, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA
            )

            if writer is not None:
                writer.write(img)

            if SHOW_UI:
                show = cv2.resize(img, DISPLAY_SIZE, interpolation=cv2.INTER_LINEAR)
                cv2.imshow("Vehicle Counter", show)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    cv2.waitKey(0)

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if SHOW_UI:
            cv2.destroyAllWindows()
        tracks_f.close()

        with open(TS_CSV, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["t_start", "t_end", "flow", "density", "avg_speed"])
            wcsv.writerows(ts_rows)

        with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
            f.write(f"video: {video_path}\n")
            f.write(f"weights: {str(weights_path)}\n")
            f.write(f"fps: {fps:.3f}\n")
            f.write(f"total_count: {len(totalCount)}\n")
            f.write(f"counts_by_class: {counts_by_class}\n")
            f.write(f"tracks_csv: {TRACKS_CSV}\n")
            f.write(f"timeseries_csv: {TS_CSV}\n")
            if args.save_video:
                f.write(f"overlay_video: {OUT_VIDEO}\n")

        print(f"[OK] Saved run folder : {RUN_DIR}")
        print(f"[OK] Saved per-track  : {TRACKS_CSV}")
        print(f"[OK] Saved time-series: {TS_CSV}")
        if args.save_video:
            print(f"[OK] Saved overlay   : {OUT_VIDEO}")
        print("[OK] Counts by class:", counts_by_class)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        err = traceback.format_exc()
        _safe_write_text(_crash_log_path(), err)
        try:
            show_error_box("App Crashed", f"Something went wrong.\n\nCrash log saved to:\n{_crash_log_path()}")
        except Exception:
            pass
        try:
            input("\nPress Enter to exit...")
        except Exception:
            pass
        raise
