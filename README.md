# Football Video Assistant Referee System (VAR)

A unified desktop application that brings Video Assistant Referee technology to football broadcasts. Analyze games with automated player tracking, foul detection, and offside detection — all in one clean interface.

**What this is:** A PyQt5-based GUI tool that processes recorded football matches and provides deterministic AI-powered decisions for critical on-field moments.

---

## Quick Start

```bash
# Activate the environment
<path to the cloned directory>/venv/Scripts/activate

# Run the application
python main.py
```

The GUI opens to a video preview where you can load any match recording.

---

## Features

| Tool | What it does | What you see |
|------|--------------|--------------|
| **Player Tracking** | Detects and tracks every player on the pitch | Bounding boxes with IDs showing all active players |
| **Foul Detection** | Analyzes multi-camera angles for misconduct | AI predictions for offensive vs defensive actions |
| **Offside Detection** | Catches players in illegal positions | Team classification + offside markers on screenshots |

---

## What You Need

- **Windows** (tested on Windows 11)
- **Python 3.12** with virtual environment
- **PyTorch 2.5.1** with CUDA 12.1 support

### Installation

```bash
# 1. Install PyTorch first (preserves CUDA builds)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 2. Install project dependencies
pip install -r requirements.txt
```

---

## How It Works

### 1. Load a Clip

Click **Open File** or press `O` to select any MP4, MOV, or image file from your broadcast.

### 2. Player Tracking

Navigate to the **Tracking** tab. The system:

- Uses YOLOv8 to detect players, referees, and the ball
- Tracks each player across frames using ByteTrack
- Stablizes detections for smooth video output

Drag the timeline to scrub through the tracked video.

### 3. Foul Detection

Load up to **4 camera angles** in the **Foul** tab:

- Feed all angles to the MViT-ViT model
- Get real-time predictions for offensive actions
- Compare against ground truth annotations

Click **Run Prediction** and let the AI classify the incident.

### 4. Offside Detection

In the **Offside** tab:

- Capture a single frame
- Classify players by team
- Detect which players were offside at the moment of a pass

A visual overlay shows which players were correctly/incorrectly flagged.

---

## Project Structure

```
VAR/
├── main.py                    # Unified GUI entry point
├── requirements.txt           # Python dependencies
├── venv/                      # Virtual environment
│
├── Foul Detection/
│   └── VAR model/             # Foul classification model
│   └── VAR interface/         # Pre-trained weights
│
├── Offside Detection/
│   └── GUI.py                 # Offside analysis tools
│   └── interface/             # Team classification model
│
└── Player Tracking/
    └── short_track.py         # YOLOv8 + ByteTrack pipeline
    └── player_tracking_gui.py # Tracking interface
```

---

## Models Used

| Model | Purpose | Input |
|-------|---------|-------|
| **MViT-ViT** | Foul classification | 4 simultaneous camera angles |
| **YOLOv8s** | Player detection | Single video frame |
| **ByteTrack** | Player ID persistence | YOLO detections |

---

## Key Design Decisions

- **3-6 second clips only** — Full matches are too large; the system focuses on the critical moments that need review.
- **Deterministic outcomes** — Each tool produces confidence scores that help referees make faster decisions.
- **All-local processing** — No cloud needed; runs entirely on your machine with CUDA acceleration.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| App crashes on startup | Run `venv/Scripts/activate` first |
| No CUDA available | Install torch with CPU-only: `pip install torch torchvision` |
| Model load fails | Check `Foul Detection/VAR interface/14_model.pth.tar` exists |
| Video lags | Try a shorter clip or reduce playback speed with the speed dropdown |

---

## License

This project is for educational and research purposes. The underlying models are based on open research in computer vision and sports analytics.
