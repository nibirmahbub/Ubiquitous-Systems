# OhBot Vision Tracker

A real-time face tracking system that uses a webcam and OpenCV to detect faces and drive OhBot servo motors, making the robot's head and eyes follow a person's movements smoothly.

---

## Features

- **Real-time face detection** using OpenCV's Haar Cascade classifier
- **Smooth servo tracking** with an exponential smoothing filter (no jerky movements)
- **Synchronized head and eye movement** — both HEADTURN/HEADNOD and EYETURN/EYETILT servos move together
- **Pause/Resume** tracking at any time without stopping the program
- **On-screen HUD** showing current mode and available controls
- **Clean shutdown** that recentres the robot's head on exit

---

## Requirements

### Hardware
- OhBot robot
- USB webcam (configured as device index `1`)

### Software
- Python 3.x
- [OpenCV](https://pypi.org/project/opencv-python/) (`cv2`)
- [ohbot](https://pypi.org/project/ohbot/) Python library

Install dependencies:
```bash
pip install opencv-python ohbot
```

---

## Usage

```bash
python ohbot_camera_tracker_V2_1.py
```

### Controls

| Key | Action |
|-----|--------|
| `P` | Pause / Resume tracking |
| `Q` | Quit and shut down |

---

## How It Works

1. **Initialization** — OhBot servos are centred (position 5) at startup.
2. **Face Detection** — Each frame is converted to greyscale and scanned with a Haar Cascade frontal-face classifier.
3. **Coordinate Mapping** — The detected face's centre pixel is normalised to the range `[0, 10]` to match OhBot's servo scale.
4. **Smoothing** — A smoothing factor of `0.20` blends the current servo position toward the target each frame, preventing abrupt jumps.
5. **Servo Limits** — Tilt is clamped between `3.0` and `8.0` to protect the robot's physical range of motion.
6. **Pause Mode** — When paused, the face bounding box is still drawn in orange but servos do not move.
7. **Shutdown** — On exit (`Q` or interrupt), the camera is released, windows are closed, and the head is recentred.

---

## Configuration

You can tune the following variables at the top of the script:

| Variable | Default | Description |
|----------|---------|-------------|
| `smoothing_factor` | `0.20` | Higher = faster but jerkier tracking (range: 0–1) |
| `cv2.VideoCapture(1, ...)` | `1` | Change to `0` if your webcam is on a different index |
| `CAP_PROP_FRAME_WIDTH/HEIGHT` | `640 × 480` | Camera resolution |
| `minSize` | `(30, 30)` | Minimum face size in pixels to detect |

---

## Project Context

This script was developed as part of the **"Teaching a Robot to See You"** robotics project (Group 1), exploring real-time computer vision and servo control with OhBot.

---

## License

This project is for educational use.
# Ubiquitous-Systems
