## Description

**OhBot Tracker v7.3** is a real-time human-robot interaction system that combines computer vision, robotic servo control, and ultrasonic sensing.

The system uses a camera and OpenCV-based face detection to track human faces and control the OhBot's head and eye movements. Two ultrasonic sensors connected through an ATmega32U4 provide additional spatial awareness, allowing the robot to detect nearby objects and react when visual tracking is temporarily lost.

The project also includes an autonomous environmental scanning mode and a radar-style visualization that displays detected faces, nearby objects, scanning direction, and sensor distances in real time.

The main focus of the project is to create smoother, more natural robotic tracking while improving reliability in real-world conditions such as servo vibration, temporary face loss, duplicate detections, and additional mechanical load from attached sensors.

## Features

- Real-time face detection and tracking
- YuNet face detector with Haar Cascade fallback
- Automatic target selection when multiple faces are detected
- Separate eye-only and head-tracking zones
- Smooth head and eye movement using interpolation
- Dead-zone control to reduce unnecessary servo movement
- Rolling position buffer for more stable tracking
- Four-axis OhBot control:
  - Head pan
  - Head nod
  - Eye pan
  - Eye tilt
- Dual ultrasonic sensor integration
- ATmega32U4 serial communication
- DTR/RTS handling for reliable native USB communication
- Automatic reaction to left and right ultrasonic detections
- 2.5-second lost-face recovery logic
- Automatic startup environmental scan
- Manual rescan support
- 8-position left-to-right scanning sequence
- Servo settling delay to reduce camera vibration
- Camera frame-buffer flushing to remove motion-blurred frames
- Double-frame face confirmation during scanning
- Duplicate face clustering during radar scans
- Real-time radar visualization
- Face and object markers on radar
- 50 cm, 100 cm, 150 cm, and 200 cm radar distance guides
- Live left and right ultrasonic distance indicators
- Safe servo movement limits
- Head-nod compensation for additional camera and sensor weight
- Live OpenCV HUD with system and connection status
- Keyboard controls for reset, pause, rescan, and exit

## Main Technologies

- Python
- OpenCV
- NumPy
- PySerial
- YuNet
- OhBot Python API
- ATmega32U4
- HC-SR04 ultrasonic sensors

## Keyboard Controls

- `Q` - Quit
- `R` - Reset OhBot to neutral position
- `P` - Pause or resume tracking
- `S` - Start a new environmental scan

## Example System Flow

Camera  
→ Face Detection  
→ Target Selection  
→ Tracking Error Calculation  
→ Eye / Head Movement

Ultrasonic Sensors  
→ ATmega32U4  
→ Serial Communication  
→ Python  
→ Lost-Target Recovery / Object Detection

Both perception sources are also combined in the real-time radar interface.
