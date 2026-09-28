"""
OhBot Tracker v7.3 + ATmega32U4 Ultrasonic Integration
======================================================
Fixes:
  1. ATmega32U4 DTR/RTS assertions added to enable native CDC serial streaming.
  2. Frame buffer flush (cap.grab()) and settling pause to eliminate scan jitter.
  3. Continuous nod hold to prevent the head falling with sensor/camera weight.
  4. Detection clustering to merge adjacent duplicate face sightings on radar.

ATmega32U4 wiring:
  LEFT  sensor: TRIG->Pin9   ECHO->Pin8
  RIGHT sensor: TRIG->Pin11  ECHO->Pin10
"""

import cv2
import numpy as np
import time, sys, os, math, threading, json
import serial, serial.tools.list_ports
from collections import deque

# ══════════════════════════════════════════════════════════
#  CONFIGURATION & HARDWARE PORTS
# ══════════════════════════════════════════════════════════
CAMERA_INDEX = 0       # 0=laptop webcam, 1=external USB

# If head turns RIGHT when face is on LEFT -> set True
INVERT_PAN   = False

# If head nods DOWN when face is at TOP -> set True
INVERT_TILT  = False

# Eye zone: fraction from each edge that triggers HEAD movement
EYE_ZONE_MARGIN = 0.20

# Movement smoothing rates
HEAD_SMOOTH  = 0.08
EYE_SMOOTH   = 0.20
DEAD_ZONE    = 0.07
BUFFER_SIZE  = 8

# If "AUTO" does not pick up your Arduino, put the exact port (e.g., "COM7")
ATMEGA_PORT  = "COM13"
ATMEGA_BAUD  = 115200

ULTRA_THRESHOLD_CM = 250.0

# Physical servo constants adjusted for head weight
HEAD_NOD_NEUTRAL = 6.2   # Shifted up from 5.5 to support front camera/sensors
HEAD_PAN_NEUTRAL = 5.0
EYE_NEUTRAL      = 5.0

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "face_detection_yunet_2023mar.onnx")

# Safe servo ranges to protect linkages
PAN_MIN, PAN_MAX   = 1.0, 9.0
NOD_MIN, NOD_MAX   = 5.2, 7.5   # Higher floor prevents forward sagging
EYE_MIN, EYE_MAX   = 1.0, 9.0
ETLT_MIN, ETLT_MAX = 3.5, 7.0

def clamp(v, lo, hi): 
    return max(lo, min(hi, v))

# ══════════════════════════════════════════════════════════
#  OHBOT INITIALIZATION
# ══════════════════════════════════════════════════════════
OHBOT_CONNECTED = False
try:
    from ohbot import ohbot
    try:
        import ohbot.ohbot as _m
        _m.portList = [f"COM{i}" for i in range(1, 21)]
    except: pass
    ohbot.init()
    ohbot.move(ohbot.HEADTURN, HEAD_PAN_NEUTRAL)
    ohbot.move(ohbot.HEADNOD,  HEAD_NOD_NEUTRAL)
    ohbot.move(ohbot.EYETURN,  EYE_NEUTRAL)
    ohbot.move(ohbot.EYETILT,  EYE_NEUTRAL)
    ohbot.wait(1)
    OHBOT_CONNECTED = True
    print("✅ OhBot connected")
except Exception as e:
    print(f"⚠️  OhBot not found ({e})")

def move_servos(hp, ht, ep, et):
    if not OHBOT_CONNECTED: return
    try:
        ohbot.move(ohbot.HEADTURN, clamp(hp, PAN_MIN,  PAN_MAX))
        ohbot.move(ohbot.HEADNOD,  clamp(ht, NOD_MIN,  NOD_MAX))
        ohbot.move(ohbot.EYETURN,  clamp(ep, EYE_MIN,  EYE_MAX))
        ohbot.move(ohbot.EYETILT,  clamp(et, ETLT_MIN, ETLT_MAX))
    except: pass

def reset_all():
    if OHBOT_CONNECTED:
        try:
            ohbot.move(ohbot.HEADTURN, HEAD_PAN_NEUTRAL)
            ohbot.move(ohbot.HEADNOD,  HEAD_NOD_NEUTRAL)
            ohbot.move(ohbot.EYETURN,  EYE_NEUTRAL)
            ohbot.move(ohbot.EYETILT,  EYE_NEUTRAL)
        except: pass

# ══════════════════════════════════════════════════════════
#  ULTRASONIC COMM THREAD (WITH ATMEGA32U4 FIX)
# ══════════════════════════════════════════════════════════
ultra      = {"left": 999.0, "right": 999.0, "connected": False}
ultra_lock = threading.Lock()

def _find_port():
    if ATMEGA_PORT != "AUTO":
        return ATMEGA_PORT
    keywords = ["Leonardo", "Micro", "32U4", "Arduino", "USB Serial", "ACM"]
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "") + (p.manufacturer or "")
        if any(k.lower() in desc.lower() for k in keywords):
            return p.device
    ohbot_ports = []
    try:
        import ohbot.ohbot as _om
        ohbot_ports = _om.portList
    except: pass
    for p in serial.tools.list_ports.comports():
        if p.device not in ohbot_ports:
            return p.device
    return None

def _ultra_thread():
    port = _find_port()
    if not port:
        print("⚠️  ATmega port not found — ultrasonic sensors disabled")
        return
    try:
        ser = serial.Serial(port, ATMEGA_BAUD, timeout=1)
        
        # ATmega32U4 Native USB handshake:
        ser.dtr = True
        ser.rts = True
        time.sleep(0.5)
        ser.reset_input_buffer()

        with ultra_lock: 
            ultra["connected"] = True
        print(f"✅ ATmega connected and receiving on {port}")
        
        buf = ""
        while True:
            try:
                chunk = ser.read(ser.in_waiting or 1)
                buf  += chunk.decode("utf-8", errors="ignore")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line or not line.startswith("{"): continue
                    
                    d = json.loads(line)
                    with ultra_lock:
                        ultra["left"]  = float(d.get("left",  999.0))
                        ultra["right"] = float(d.get("right", 999.0))
            except json.JSONDecodeError: pass
            except Exception as e:
                print(f"⚠️  Ultra read error: {e}"); time.sleep(0.5)
    except Exception as e:
        print(f"⚠️  Cannot open {port}: {e}")

threading.Thread(target=_ultra_thread, daemon=True).start()
time.sleep(0.8)

def get_ultra():
    with ultra_lock:
        return ultra["left"], ultra["right"], ultra["connected"]

# ══════════════════════════════════════════════════════════
#  FACE DETECTOR
# ══════════════════════════════════════════════════════════
USE_YUNET = False
_det = _haar = None

if os.path.exists(MODEL_PATH):
    try:
        _det = cv2.FaceDetectorYN.create(
            MODEL_PATH, "", (320, 320), 0.82, 0.3, 5000)
        USE_YUNET = True
        print("✅ YuNet face detector loaded")
    except: pass

if not USE_YUNET:
    print("⚠️  Haar Cascade fallback")
    _haar = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces(frame):
    h, w = frame.shape[:2]
    if USE_YUNET:
        _det.setInputSize((w, h))
        _, faces = _det.detect(frame)
        if faces is None or len(faces) == 0: return []
        return sorted([tuple(map(int, f[:4])) for f in faces], key=lambda b: b[0])
    else:
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        found = _haar.detectMultiScale(gray, 1.15, 8, minSize=(60, 60))
        if len(found) == 0: return []
        return sorted([tuple(map(int, f)) for f in found], key=lambda b: b[0])

def middle_face(boxes):
    if not boxes: return None
    if len(boxes) == 1: return boxes[0]
    centres = sorted([(b[0] + b[2] / 2.0, b) for b in boxes])
    return centres[len(centres) // 2][1]

# ══════════════════════════════════════════════════════════
#  CAMERA SETUP
# ══════════════════════════════════════════════════════════
def _open_cam(idx, backend, label):
    c = cv2.VideoCapture(idx, backend)
    if not c.isOpened(): return None
    c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    c.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    c.set(cv2.CAP_PROP_FPS, 30)
    time.sleep(0.5)
    for _ in range(25):
        ok, fr = c.read()
        if ok and fr is not None and fr.mean() > 1.0:
            print(f"✅ Camera idx={idx} {label}")
            return c
        time.sleep(0.05)
    c.release(); return None

cap = None
for idx in ([CAMERA_INDEX] + [i for i in range(4) if i != CAMERA_INDEX]):
    for b, l in [(cv2.CAP_DSHOW, "DirectShow"),
                 (cv2.CAP_MSMF, "MediaFoundation"),
                 (cv2.CAP_ANY,  "Auto")]:
        cap = _open_cam(idx, b, l)
        if cap: break
    if cap: break
if cap is None: print("❌ No camera found"); sys.exit(1)
for _ in range(8): cap.read(); time.sleep(0.05)

# ══════════════════════════════════════════════════════════
#  RADAR MAP
# ══════════════════════════════════════════════════════════
RADAR_SIZE = 600
RADAR_CX   = RADAR_SIZE // 2
RADAR_CY   = RADAR_SIZE - 80
RADAR_R    = 440

scan_hits    = []
scan_running = False
scan_angle   = HEAD_PAN_NEUTRAL
scan_lock    = threading.Lock()

def _pan_to_xy(pan, frac=0.85):
    deg = (pan - 5.0) * 18.0
    rad = math.radians(deg)
    r   = int(RADAR_R * frac)
    return (RADAR_CX + int(r * math.sin(rad)),
            RADAR_CY - int(r * math.cos(rad)))

def draw_radar(hits, sweep_pan=None, face_now=False,
               l_cm=999, r_cm=999, ultra_ok=False):
    img = np.zeros((RADAR_SIZE, RADAR_SIZE, 3), dtype=np.uint8)
    for d in [50, 100, 150, 200]:
        frac = 1.0 - d / ULTRA_THRESHOLD_CM
        r    = int(RADAR_R * max(0.05, frac))
        cv2.circle(img, (RADAR_CX, RADAR_CY), r, (0, 55, 0), 1)
        cv2.putText(img, f"{d}cm", (RADAR_CX + r + 3, RADAR_CY - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 75, 0), 1)
    for deg in range(-90, 91, 30):
        rad = math.radians(deg)
        ex  = RADAR_CX + int(RADAR_R * math.sin(rad))
        ey  = RADAR_CY - int(RADAR_R * math.cos(rad))
        cv2.line(img, (RADAR_CX, RADAR_CY), (ex, ey), (0, 40, 0), 1)
        cv2.putText(img, f"{deg}", (ex - 10, ey - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 65, 0), 1)
    cv2.line(img, (RADAR_CX - RADAR_R, RADAR_CY),
             (RADAR_CX + RADAR_R, RADAR_CY), (0, 70, 0), 1)
    
    for h in hits:
        if h["type"] == "face":
            x, y = _pan_to_xy(h["pan"], 0.78)
            cv2.circle(img, (x, y), 10, (0, 255, 80), -1)
            cv2.circle(img, (x, y), 13, (0, 200, 60), 1)
            cv2.putText(img, "F", (x - 4, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        else:
            frac = max(0.05, 1.0 - h["dist"] / ULTRA_THRESHOLD_CM)
            x, y  = _pan_to_xy(h["pan"], frac)
            cv2.circle(img, (x, y), 7, (0, 220, 220), -1)
            cv2.putText(img, f"{h['dist']:.0f}", (x + 5, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 200, 200), 1)
            
    if sweep_pan is not None:
        x, y = _pan_to_xy(sweep_pan, 1.0)
        cv2.line(img, (RADAR_CX, RADAR_CY), (x, y),
                 (0, 255, 100) if face_now else (0, 160, 60), 2)
    cv2.circle(img, (RADAR_CX, RADAR_CY), 7, (0, 255, 80), -1)
    n_f = sum(1 for h in hits if h["type"] == "face")
    n_o = sum(1 for h in hits if h["type"] == "object")
    cv2.putText(img, "OhBot Radar Map", (RADAR_CX - 70, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 210, 70), 1)
    cv2.putText(img, f"Faces:{n_f}  Objects:{n_o}  "
                f"{'SCANNING' if sweep_pan else 'S=rescan'}",
                (10, RADAR_SIZE - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 150, 60), 1)
    cv2.circle(img, (15, 42), 6, (0, 255, 80), -1)
    cv2.putText(img, "Face", (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 200, 60), 1)
    cv2.circle(img, (15, 58), 5, (0, 220, 220), -1)
    cv2.putText(img, "Object", (24, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 200, 200), 1)
    if ultra_ok:
        for dist, x0, x1, lbl in [(l_cm, 10, RADAR_CX - 10, "L"),
                                  (r_cm, RADAR_CX + 10, RADAR_SIZE - 10, "R")]:
            frac = max(0.0, min(1.0, 1.0 - dist / ULTRA_THRESHOLD_CM))
            bw   = int((x1 - x0) * frac)
            cv2.rectangle(img, (x0, RADAR_SIZE - 40), (x1, RADAR_SIZE - 28), (0, 35, 0), -1)
            if bw > 0:
                cv2.rectangle(img, (x0, RADAR_SIZE - 40), (x0 + bw, RADAR_SIZE - 28),
                              (0, 255, 80), -1)
            cv2.putText(img, f"{lbl}:{dist:.0f}cm", (x0, RADAR_SIZE - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 60), 1)
    return img

# ══════════════════════════════════════════════════════════
#  SCAN METHOD (ANTI-JITTER + DUPLICATE CLUSTERING)
# ══════════════════════════════════════════════════════════
def do_scan(cap_ref, label="Scan"):
    global scan_hits, scan_angle, scan_running
    print(f"🔍 {label} starting...")
    with scan_lock:
        scan_hits    = []
        scan_running = True

    steps = list(np.linspace(2.0, 8.0, 11))
    raw_face_pans = []
    object_hits   = []

    for pan in steps:
        scan_angle = pan
        if OHBOT_CONNECTED:
            try:
                ohbot.move(ohbot.HEADTURN, clamp(pan, PAN_MIN, PAN_MAX))
                ohbot.move(ohbot.HEADNOD,  HEAD_NOD_NEUTRAL)   # Firmly hold nod upright
                ohbot.move(ohbot.EYETURN,  EYE_NEUTRAL)
                ohbot.move(ohbot.EYETILT,  EYE_NEUTRAL)
            except: pass

        # Settle vibration from physical servo movement
        time.sleep(0.55)

        # Discard motion-blurred frames stored in the buffer
        for _ in range(2):
            cap_ref.grab()

        ok, frame = cap_ref.read()
        if ok and frame is not None:
            frame = cv2.flip(frame, 1)
            detected = detect_faces(frame)
            if detected:
                # Double-check confirmation frame to avoid spurious single-frame hits
                time.sleep(0.05)
                cap_ref.grab()
                ok2, frame2 = cap_ref.read()
                if ok2 and frame2 is not None:
                    frame2 = cv2.flip(frame2, 1)
                    if detect_faces(frame2):
                        raw_face_pans.append(pan)
                        print(f"  📷 Face at {(pan-5)*18:.0f}° (pan={pan:.1f})")

        l, r, u_ok = get_ultra()
        if u_ok:
            if l < ULTRA_THRESHOLD_CM:
                object_hits.append({"type": "object", "pan": pan, "dist": l, "side": "left"})
            if r < ULTRA_THRESHOLD_CM:
                object_hits.append({"type": "object", "pan": pan, "dist": r, "side": "right"})

    # Cluster duplicate hits: merge sightings occurring within ~35 degrees (2.0 pan delta)
    clustered_faces = []
    if raw_face_pans:
        clusters = [[raw_face_pans[0]]]
        for p in raw_face_pans[1:]:
            if abs(p - clusters[-1][-1]) <= 2.0:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        
        for c in clusters:
            clustered_faces.append({"type": "face", "pan": sum(c) / len(c)})

    with scan_lock:
        scan_hits = clustered_faces + object_hits

    # Return head to center
    scan_angle = HEAD_PAN_NEUTRAL
    if OHBOT_CONNECTED:
        try:
            ohbot.move(ohbot.HEADTURN, HEAD_PAN_NEUTRAL)
            ohbot.move(ohbot.HEADNOD,  HEAD_NOD_NEUTRAL)
            ohbot.move(ohbot.EYETURN,  EYE_NEUTRAL)
            ohbot.move(ohbot.EYETILT,  EYE_NEUTRAL)
        except: pass

    with scan_lock:
        scan_running = False
        nf = sum(1 for h in scan_hits if h["type"] == "face")
        no = sum(1 for h in scan_hits if h["type"] == "object")
    print(f"✅ {label} done — {nf} face(s), {no} object(s)")

    if label == "Startup scan" and nf > 0:
        with scan_lock:
            fhits = [h for h in scan_hits if h["type"] == "face"]
        target_p = fhits[len(fhits) // 2]["pan"]
        print(f"  ➡️  Centering on face at pan={target_p:.1f}")
        if OHBOT_CONNECTED:
            try: ohbot.move(ohbot.HEADTURN, target_p)
            except: pass

# ══════════════════════════════════════════════════════════
#  TRACKING STATE
# ══════════════════════════════════════════════════════════
head_pan  = HEAD_PAN_NEUTRAL
head_tilt = HEAD_NOD_NEUTRAL
eye_pan   = EYE_NEUTRAL
eye_tilt  = EYE_NEUTRAL

buf_cx = deque(maxlen=BUFFER_SIZE)
buf_cy = deque(maxlen=BUFFER_SIZE)

paused   = False
lost_t   = time.time()
LOST_TIMEOUT = 2.5

print(f"\nDetector : {'YuNet ✅' if USE_YUNET else 'Haar ⚠️'}")
print(f"OhBot    : {'Connected ✅' if OHBOT_CONNECTED else 'Not connected ⚠️'}")
print(f"INVERT_PAN={INVERT_PAN}")
print("Controls : Q=quit  R=reset  P=pause  S=rescan\n")
print("Startup scan in 2s...")
time.sleep(2.0)
threading.Thread(target=do_scan, args=(cap, "Startup scan"), daemon=True).start()

# ══════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════
try:
    while True:
        ok, frame = cap.read()
        if not ok or frame is None or frame.mean() < 0.5:
            time.sleep(0.02); continue

        with scan_lock:
            scanning  = scan_running
            s_pan     = scan_angle
            cur_hits  = list(scan_hits)

        l_cm, r_cm, ultra_ok = get_ultra()

        frame  = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]

        # Zone boundaries
        lx = int(fw * EYE_ZONE_MARGIN)
        rx = int(fw * (1.0 - EYE_ZONE_MARGIN))

        # Draw visual cues
        cv2.rectangle(frame, (0, 0), (fw - 1, fh - 1), (180, 70, 0), 1)
        cv2.putText(frame, "HEAD", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 70, 0), 1)
        cv2.putText(frame, "HEAD", (rx + 4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 70, 0), 1)
        cv2.rectangle(frame, (lx, 0), (rx, fh), (0, 200, 100), 2)
        cv2.putText(frame, "EYE ONLY", (lx + 4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 100), 1)
        cv2.line(frame, (fw // 2 - 10, fh // 2), (fw // 2 + 10, fh // 2), (255, 255, 255), 1)
        cv2.line(frame, (fw // 2, fh // 2 - 10), (fw // 2, fh // 2 + 10), (255, 255, 255), 1)

        all_faces = detect_faces(frame)
        target    = middle_face(all_faces)

        for i, (fx, fy, fw2, fh2) in enumerate(all_faces):
            is_t  = target is not None and (fx, fy, fw2, fh2) == target
            color = (0, 255, 0) if is_t else (0, 165, 255)
            cv2.rectangle(frame, (fx, fy), (fx + fw2, fy + fh2), color, 2 if is_t else 1)
            if len(all_faces) > 1:
                cv2.putText(frame, "TARGET" if is_t else f"#{i+1}",
                            (fx, fy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

        # ── TRACKING / IDLE (ACTIVE ONLY WHEN NOT SCANNING) ──
        if not scanning and not paused:
            if target is not None:
                fx, fy, fw2, fh2 = target
                lost_t = time.time()

                raw_cx = (fx + fw2 / 2.0) / fw
                raw_cy = (fy + fh2 / 2.0) / fh
                buf_cx.append(raw_cx)
                buf_cy.append(raw_cy)
                scx = sum(buf_cx) / len(buf_cx)
                scy = sum(buf_cy) / len(buf_cy)

                err_x =  (scx - 0.5)
                err_y = -(scy - 0.5)

                if INVERT_PAN:  err_x = -err_x
                if INVERT_TILT: err_y = -err_y

                tgt_pan  = HEAD_PAN_NEUTRAL + err_x * 8.0
                tgt_tilt = HEAD_NOD_NEUTRAL + err_y * 3.0

                face_px     = int(scx * fw)
                in_eye_zone = (lx <= face_px <= rx)
                mode        = "EYE ONLY" if in_eye_zone else "HEAD"

                if in_eye_zone:
                    if abs(err_x) > DEAD_ZONE:
                        eye_pan += (tgt_pan - eye_pan) * EYE_SMOOTH
                    if abs(err_y) > DEAD_ZONE:
                        eye_tilt += (tgt_tilt - eye_tilt) * EYE_SMOOTH
                    head_pan  += (HEAD_PAN_NEUTRAL - head_pan)  * 0.04
                    head_tilt += (HEAD_NOD_NEUTRAL - head_tilt) * 0.04
                else:
                    if abs(err_x) > DEAD_ZONE:
                        head_pan += (tgt_pan - head_pan) * HEAD_SMOOTH
                    if abs(err_y) > DEAD_ZONE:
                        head_tilt += (tgt_tilt - head_tilt) * HEAD_SMOOTH
                    eye_pan  += (head_pan  - eye_pan)  * 0.20
                    eye_tilt += (head_tilt - eye_tilt) * 0.20

                head_pan  = clamp(head_pan,  PAN_MIN,  PAN_MAX)
                head_tilt = clamp(head_tilt, NOD_MIN,  NOD_MAX)
                eye_pan   = clamp(eye_pan,   EYE_MIN,  EYE_MAX)
                eye_tilt  = clamp(eye_tilt,  ETLT_MIN, ETLT_MAX)
                move_servos(head_pan, head_tilt, eye_pan, eye_tilt)

                col = (0, 200, 100) if in_eye_zone else (180, 70, 0)
                cv2.putText(frame, f"Mode:{mode}  err_x:{err_x:+.2f}",
                            (10, fh - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

            else:
                buf_cx.clear(); buf_cy.clear()
                secs = time.time() - lost_t

                if secs > LOST_TIMEOUT:
                    if ultra_ok:
                        if l_cm < r_cm and l_cm < ULTRA_THRESHOLD_CM:
                            tgt = HEAD_PAN_NEUTRAL - 3.0
                            head_pan += (tgt - head_pan) * 0.05
                            cv2.putText(frame, "← Ultrasonic LEFT",
                                        (10, fh - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 200), 1)
                        elif r_cm < ULTRA_THRESHOLD_CM:
                            tgt = HEAD_PAN_NEUTRAL + 3.0
                            head_pan += (tgt - head_pan) * 0.05
                            cv2.putText(frame, "Ultrasonic RIGHT →",
                                        (10, fh - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 200), 1)
                        else:
                            head_pan  += (HEAD_PAN_NEUTRAL - head_pan)  * 0.04
                            head_tilt += (HEAD_NOD_NEUTRAL - head_tilt) * 0.04
                    else:
                        head_pan  += (HEAD_PAN_NEUTRAL - head_pan)  * 0.04
                        head_tilt += (HEAD_NOD_NEUTRAL - head_tilt) * 0.04

                    eye_pan  += (EYE_NEUTRAL - eye_pan)  * 0.06
                    eye_tilt += (EYE_NEUTRAL - eye_tilt) * 0.06
                    head_pan  = clamp(head_pan, PAN_MIN, PAN_MAX)
                    head_tilt = clamp(head_tilt, NOD_MIN, NOD_MAX)
                    move_servos(head_pan, head_tilt, eye_pan, eye_tilt)
                else:
                    cv2.putText(frame, "No face",
                                (10, fh - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

        # Ultrasonic horizontal indicators
        if ultra_ok:
            by = fh - 8
            for dist, x0, x1 in [(l_cm, 10, fw // 2 - 10), (r_cm, fw // 2 + 10, fw - 10)]:
                bw = int(max(0, min(1, 1 - dist / ULTRA_THRESHOLD_CM)) * (x1 - x0))
                cv2.rectangle(frame, (x0, by - 10), (x1, by), (35, 35, 35), -1)
                if bw > 0: cv2.rectangle(frame, (x0, by - 10), (x0 + bw, by), (0, 200, 170), -1)
            cv2.putText(frame, f"L:{l_cm:.0f}cm", (10, by - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 180, 160), 1)
            cv2.putText(frame, f"R:{r_cm:.0f}cm", (fw - 65, by - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 180, 160), 1)

        # HUD Overlay
        oc = (0, 255, 120) if OHBOT_CONNECTED else (0, 165, 255)
        uc = (0, 200, 170) if ultra_ok else (80, 80, 80)
        cv2.putText(frame,
                    "OhBot: Connected" if OHBOT_CONNECTED else "OhBot: Not connected",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, oc, 2)
        cv2.putText(frame,
                    f"{'YuNet' if USE_YUNET else 'Haar'}  "
                    f"H:{head_pan:.1f}/{head_tilt:.1f}  E:{eye_pan:.1f}/{eye_tilt:.1f}"
                    + ("  [PAUSED]" if paused else "")
                    + ("  [SCANNING]" if scanning else ""),
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame,
                    f"Ultra:{'OK' if ultra_ok else 'off'}  "
                    f"INVERT_PAN={INVERT_PAN}",
                    (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.38, uc, 1)
        cv2.putText(frame, "Q=quit  R=reset  P=pause  S=rescan",
                    (fw - 230, fh - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

        cv2.imshow("OhBot Tracker v7+Ultra", frame)

        # Radar screen
        live = list(cur_hits)
        if ultra_ok:
            if l_cm < ULTRA_THRESHOLD_CM:
                live.append({"type": "object", "pan": head_pan, "dist": l_cm, "side": "left"})
            if r_cm < ULTRA_THRESHOLD_CM:
                live.append({"type": "object", "pan": head_pan, "dist": r_cm, "side": "right"})
        radar = draw_radar(live,
                           sweep_pan = s_pan if scanning else None,
                           face_now  = target is not None,
                           l_cm=l_cm, r_cm=r_cm, ultra_ok=ultra_ok)
        cv2.imshow("OhBot Radar", radar)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break
        elif key == ord("r"):
            head_pan  = HEAD_PAN_NEUTRAL
            head_tilt = HEAD_NOD_NEUTRAL
            eye_pan   = EYE_NEUTRAL
            eye_tilt  = EYE_NEUTRAL
            reset_all(); print("Reset")
        elif key == ord("p"):
            paused = not paused
            print("PAUSED" if paused else "RESUMED")
        elif key == ord("s") and not scanning:
            threading.Thread(target=do_scan, args=(cap, "Manual scan"), daemon=True).start()

finally:
    cap.release()
    cv2.destroyAllWindows()
    reset_all()
    print("Stopped.")