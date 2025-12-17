# Vehicle Counter & Traffic Analytics (YOLOv8 + Tracking)

This project is an **experimental deep learning–based vehicle counting and traffic analytics system** built using **YOLOv8** for object detection and **tracking-based line crossing** for vehicle counting.

**Important Notice**  
This project is **still in the experimental stage** and **NOT reliable nor production-ready**.  
Results may vary significantly depending on camera angle, lighting conditions, occlusion, traffic density, and video quality.

---

## Features
- Vehicle detection using **YOLOv8**
- Multi-object tracking using **SORT**
- Line-crossing based vehicle counting
- Per-class vehicle count:
  - Car
  - Motorbike
  - Bus
  - Truck
- Traffic analytics output:
  - Vehicle flow
  - Traffic density
  - Approximate speed estimation
- Offline deployment support (EXE via PyInstaller)

---

## Project Structure
```
FINAL_AOL_DEEP-LEARNING/
│
├── additional/
│   ├── plot-bird_view.py        # Bird-eye view visualization
│   └── plot-limits.py           # Counting line visualization
│
├── assets/
│   └── mask.png                 # Optional ROI / mask
│
├── training/
│   ├── train_anomaly_lstm_ae.py # Experimental anomaly detection
│   └── train_overspeed_gru.py   # Experimental overspeed detection
│
├── vehicle-counter-tracker/
│   ├── app.py                   # Main vehicle counter application
│   ├── sort.py                  # SORT tracking algorithm
│   └── outputs/
│       └── run_YYYYMMDD_HHMMSS/
│           ├── summary.txt
│           ├── tracks_log.csv
│           └── timeseries_0p5s.csv
│
├── video_data/
│   └── download_data.py         # Dataset download script
│
├── Yolo-Weights/
│   └── yolov8m.pt               # YOLOv8 pretrained weights
│
├── requirements.txt
└── README.md
```

---

## Output Files
Each execution generates an output folder inside:

```
vehicle-counter-tracker/outputs/
```

Contents:
- **summary.txt**  
  Overall statistics including total count and per-class count

- **tracks_log.csv**  
  Per-frame tracking log containing:
  - Track ID  
  - Object class  
  - Position  
  - Speed estimation  
  - Counting events  

- **timeseries_0p5s.csv**  
  Aggregated traffic metrics every 0.5 seconds:
  - Flow  
  - Density  
  - Average speed  

> Annotated video output is optional and may not be enabled by default.

---

## Dataset
This repository does **NOT** include raw video datasets by default.

To download sample datasets, run:
```bash
python video_data/download_data.py
```

Dataset sources may include:
- Public traffic surveillance videos
- Open datasets
- Sample recordings for academic purposes

Please ensure you have sufficient disk space before downloading.

---

## How to Run (Development Mode)
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python vehicle-counter-tracker/app.py
```

3. Select a video file when prompted.

---

## Experimental Status & Limitations
This project is **experimental** and has multiple known limitations:

- ❌ Not robust to heavy occlusion or crowded traffic scenes
- ❌ Speed estimation is approximate and camera-dependent
- ❌ Sensitive to camera angle and perspective calibration
- ❌ No large-scale benchmarking or validation
- ❌ Not optimized for real-time multi-camera deployment

This project is intended for:
- Academic coursework
- Proof-of-concept experiments
- Learning and exploration in computer vision and deep learning

---

## Disclaimer
This repository is provided **as-is**.  
The author **does not guarantee accuracy, reliability, or suitability for production use**.

---

## Author
Developed as part of an **Applied Online Learning (AOL) – Deep Learning** project.

---

## License
This project is intended for **educational use only**.
