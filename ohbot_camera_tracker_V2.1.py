import cv2
from ohbot import ohbot
import time

# --- 1. Initialize OhBot ---
print("Initializing OhBot servos...")
ohbot.init()

ohbot.move(ohbot.HEADTURN, 5)
ohbot.move(ohbot.HEADNOD, 5)
ohbot.move(ohbot.EYETURN, 5)
ohbot.move(ohbot.EYETILT, 5)
ohbot.wait(1)

# --- 2. Setup OpenCV Face Detection ---
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise IOError("Unable to load the face cascade classifier xml file.")

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera active. Look into the camera to begin tracking.")
print("Controls: Q = quit | P = pause/resume")

# Smooth tracking variables
current_pan      = 5.0
current_tilt     = 5.0
smoothing_factor = 0.20
tracking_paused  = False          # ← pause state

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab camera frame.")
            continue

        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0 and not tracking_paused:   # ← skip servo when paused
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            face_center_x = x + (w / 2.0)
            face_center_y = y + (h / 2.0)

            normalized_x = face_center_x / frame_width
            normalized_y = face_center_y / frame_height

            target_pan  = normalized_x * 10
            target_tilt = (1.0 - normalized_y) * 10
            target_tilt = max(3.0, min(8.0, target_tilt))

            current_pan  += (target_pan  - current_pan)  * smoothing_factor
            current_tilt += (target_tilt - current_tilt) * smoothing_factor
            current_pan   = max(0.0, min(10.0, current_pan))
            current_tilt  = max(3.0, min(8.0,  current_tilt))

            ohbot.move(ohbot.HEADTURN, current_pan)
            ohbot.move(ohbot.EYETURN,  current_pan)
            ohbot.move(ohbot.HEADNOD,  current_tilt)
            ohbot.move(ohbot.EYETILT,  current_tilt)

        elif len(faces) > 0 and tracking_paused:
            # Still draw the box so you can see the face, just don't move
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)

        # ── HUD ─────────────────────────────
        if tracking_paused:
            cv2.putText(frame, "PAUSED — press P to resume",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        else:
            cv2.putText(frame, "Tracking  P=pause  Q=quit",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 1)

        cv2.imshow('OhBot Vision Tracker', frame)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            tracking_paused = not tracking_paused
            print("Tracking", "PAUSED" if tracking_paused else "RESUMED")

finally:
    cap.release()
    cv2.destroyAllWindows()
    ohbot.move(ohbot.HEADTURN, 5)
    ohbot.move(ohbot.HEADNOD, 5)
    print("Tracking stopped cleanly.")