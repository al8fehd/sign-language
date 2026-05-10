"""
Gerçek Zamanlı İşaret Dili → Metin Çevirici

Çalıştırma:
    uv run python main.py

Tuşlar:
    q        → çıkış
    f        → ayna modu aç/kapat
    b        → son harfi sil
    c        → metni temizle
    Space    → duraklat / devam et
    .        → ayarlar menüsü
                Sağ el pinç  → onay süresi (kare sayısı)
                Sol el pinç  → minimum güven eşiği
              Onaylamak için tekrar . bas veya iki eli birbirine dokut
"""
from __future__ import annotations

import locale
import os
import sys
import warnings

# Fedora/Turkish locale kombinasyonunda MediaPipe graph parser bozulabiliyor.
# Bunu process başında C locale ile yeniden başlatarak sabitliyoruz.
if os.environ.get("LC_ALL") != "C":
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    env.setdefault("LANG", "C")
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)

locale.setlocale(locale.LC_ALL, "C")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import math
import pathlib
import time

import cv2
import joblib
import numpy as np

from hand_detector import HandDetector, HAND_CONNECTIONS
from sender import CommandSender

# ── Sabitler ──────────────────────────────────────────────────────────────────
CLF_FILE         = pathlib.Path(__file__).parent / "asl_model_xgb.pkl"
LE_FILE          = pathlib.Path(__file__).parent / "label_encoder.pkl"
LANDMARKER_MODEL = pathlib.Path(__file__).parent / "hand_landmarker.task"

STABLE_SEC    = 0.8    # seconds a sign must be held to confirm
COOLDOWN_SEC  = 0.3    # seconds to ignore input after a confirm
MIN_CONF      = 0.70

STABLE_MIN_SEC, STABLE_MAX_SEC = 0.2, 3.0
CONF_MIN,       CONF_MAX       = 0.30, 0.95
PINCH_MIN,      PINCH_MAX      = 0.03, 0.35
TOUCH_THRESHOLD                = 0.045
TOUCH_HOLD_SEC                 = 0.7   # seconds to hold hands together
PAUSE_APART_THRESHOLD          = 0.45  # wrist distance to count as "far apart"
PAUSE_HOLD_SEC                 = 0.8   # seconds to hold the pause gesture
QUIT_HOLD_SEC                  = 2.0
QUIT_CONFIRM_SEC               = 3.0

# ── Model ─────────────────────────────────────────────────────────────────────

def _load_model(clf_path, le_path):
    for p in (clf_path, le_path):
        if not p.exists():
            raise FileNotFoundError(f"Model dosyası bulunamadı: {p}")
    clf = joblib.load(clf_path)
    le  = joblib.load(le_path)
    return clf, le


def normalize_landmarks(lm_list) -> np.ndarray:
    pts = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32)
    pts -= pts[0]
    scale = np.max(np.linalg.norm(pts, axis=1) + 1e-6)
    pts /= scale
    return pts.flatten().reshape(1, -1)

# ── Gesture helpers ────────────────────────────────────────────────────────────

def _pinch_distance(landmarks) -> float:
    t, i = landmarks[4], landmarks[8]
    return math.sqrt((t.x - i.x) ** 2 + (t.y - i.y) ** 2)


def _map(dist: float, out_min: float, out_max: float) -> float:
    t = max(0.0, min(1.0, (dist - PINCH_MIN) / (PINCH_MAX - PINCH_MIN)))
    return out_min + t * (out_max - out_min)


def _hands_touching(hands: list) -> bool:
    if len(hands) < 2:
        return False
    lm_a = [lm for lm, _ in hands][0]
    lm_b = [lm for lm, _ in hands][1]
    for a in lm_a:
        for b in lm_b:
            if math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2) < TOUCH_THRESHOLD:
                return True
    return False


def _is_open_hand(landmarks) -> bool:
    """All four fingers extended (stop-sign pose)."""
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    return all(landmarks[t].y < landmarks[p].y for t, p in zip(tips, pips))


def _pause_gesture(hands: list) -> bool:
    """Both hands open AND wrists far apart."""
    if len(hands) < 2:
        return False
    (lm_a, _), (lm_b, _) = hands[0], hands[1]
    dist = math.sqrt((lm_a[0].x - lm_b[0].x) ** 2 + (lm_a[0].y - lm_b[0].y) ** 2)
    return dist > PAUSE_APART_THRESHOLD and _is_open_hand(lm_a) and _is_open_hand(lm_b)


def _pause_midpoint(hands: list) -> tuple[float, float]:
    """Midpoint between the two wrists."""
    lm_a = hands[0][0]
    lm_b = hands[1][0]
    return (lm_a[0].x + lm_b[0].x) / 2, (lm_a[0].y + lm_b[0].y) / 2


def _contact_midpoint(hands: list) -> tuple[float, float]:
    """Returns normalized (x, y) of the midpoint between the closest landmark pair."""
    lm_a = [lm for lm, _ in hands][0]
    lm_b = [lm for lm, _ in hands][1]
    min_dist = float("inf")
    mx, my = 0.5, 0.5
    for a in lm_a:
        for b in lm_b:
            d = math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
            if d < min_dist:
                min_dist = d
                mx = (a.x + b.x) / 2
                my = (a.y + b.y) / 2
    return mx, my


def _draw_progress_wheel(frame, cx: int, cy: int, progress: float, color_arc, color_done,
                         label: str = "") -> None:
    radius, thick = 36, 5
    cv2.circle(frame, (cx, cy), radius, (40, 40, 40), thick + 2)
    sweep = int(360 * min(progress, 1.0))
    color = color_done if progress >= 1.0 else color_arc
    cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, sweep, color, thick)
    cv2.circle(frame, (cx, cy), 4, color, -1)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(frame, label, (cx - tw // 2, cy - radius - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)


def _draw_touch_progress(frame, hands: list, progress: float, label: str = "") -> None:
    h, w = frame.shape[:2]
    nx, ny = _contact_midpoint(hands)
    _draw_progress_wheel(frame, int(nx * w), int(ny * h),
                         progress, (0, 220, 255), (0, 255, 120), label)


def _draw_quit_confirm(frame, hands: list, progress: float, *_) -> None:
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 80), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    cv2.putText(frame, "EMIN MISINIZ?", (w//2 - 180, h//2 - 110),
                cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 0, 255), 3)
    cv2.putText(frame, "iptal etmek icin elinizi kapatin",
                (w//2 - 230, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)

    _draw_progress_wheel(frame, w//2, h//2, progress, (0, 0, 255), (0, 0, 200), "CIK")

    for lm, _ in hands:
        _draw_hand(frame, lm, (0, 0, 220))


def _draw_pause_progress(frame, hands: list, progress: float, label: str = "") -> None:
    h, w = frame.shape[:2]
    nx, ny = _pause_midpoint(hands)
    _draw_progress_wheel(frame, int(nx * w), int(ny * h),
                         progress, (0, 165, 255), (0, 200, 255), label)

# ── Drawing ────────────────────────────────────────────────────────────────────

def _draw_hand(frame, landmarks, color=(0, 220, 0)) -> None:
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 100, 255), -1)


def _draw_pinch_line(frame, landmarks, color) -> None:
    h, w = frame.shape[:2]
    fx, fy = int(landmarks[4].x * w), int(landmarks[4].y * h)
    ix, iy = int(landmarks[8].x * w), int(landmarks[8].y * h)
    cv2.line(frame, (fx, fy), (ix, iy), color, 3)
    cv2.circle(frame, (fx, fy), 9, color, -1)
    cv2.circle(frame, (ix, iy), 9, color, -1)


def _draw_hud(frame, sign: str, conf: float, stable_progress: float,
              text: str, fps: float, mirror: bool, in_cooldown: bool,
              stable_sec: float, min_conf: float) -> None:
    h, w = frame.shape[:2]

    panel_h = 90
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    label_color = (255, 255, 80)
    if sign in ("del", "space"):
        label_color = (100, 200, 255)

    cv2.putText(frame, sign.upper(), (16, 70),
                cv2.FONT_HERSHEY_DUPLEX, 2.2, label_color, 4)

    bar_x, bar_y, bar_w, bar_h_px = 200, 30, w - 230, 14
    filled = int(bar_w * min(stable_progress, 1.0))
    bar_color = (0, 200, 255) if in_cooldown else (0, 255, 100)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_px),
                  (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h_px),
                  bar_color, -1)
    cv2.putText(frame,
                f"GUVEN: {conf:.0%}  FPS:{fps:4.1f}  ESIK:{min_conf:.0%}  SURE:{stable_sec:.1f}s  {'AYNA' if mirror else ''}",
                (bar_x, bar_y + bar_h_px + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
    cv2.putText(frame, "b=sil  c=temizle  f=ayna  Space=duraklat  .=ayarlar  q=cikis",
                (bar_x, bar_y + bar_h_px + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 160, 160), 1)

    text_panel_h = 60
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - text_panel_h), (w, h), (10, 10, 40), -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)

    display_text = text[-50:] if len(text) > 50 else text
    cursor = "|" if (int(time.time() * 2) % 2 == 0) else " "
    cv2.putText(frame, display_text + cursor, (14, h - 18),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)


def _draw_pause(frame) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(frame, "DURAKLATILDI", (w // 2 - 200, h // 2),
                cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 200, 255), 4)
    cv2.putText(frame, "devam ettirmek icin Space", (w // 2 - 190, h // 2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)


def _draw_settings_bar(frame, cx, cy, label, value_str, t, color) -> None:
    bar_w, bar_h_px = 320, 14
    bx = cx - bar_w // 2
    cv2.putText(frame, label, (bx, cy - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(frame, value_str, (bx, cy - 8),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, color, 2)
    cv2.rectangle(frame, (bx, cy), (bx + bar_w, cy + bar_h_px), (60, 60, 60), -1)
    cv2.rectangle(frame, (bx, cy), (bx + int(bar_w * t), cy + bar_h_px), color, -1)


def _draw_settings(frame, hands, prev_stable, prev_conf):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    new_stable = prev_stable
    new_conf   = prev_conf

    right_lm = next((lm for lm, label in hands if label == "Right"), None)
    left_lm  = next((lm for lm, label in hands if label == "Left"),  None)

    for lm, label in hands:
        color = (0, 220, 255) if label == "Right" else (255, 160, 0)
        _draw_hand(frame, lm, color)
        _draw_pinch_line(frame, lm, color)

    cv2.putText(frame, "AYARLAR", (w // 2 - 95, 60),
                cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 255, 255), 3)

    if right_lm is not None:
        new_stable = round(_map(_pinch_distance(right_lm), STABLE_MIN_SEC, STABLE_MAX_SEC), 1)
    t_stable = (new_stable - STABLE_MIN_SEC) / (STABLE_MAX_SEC - STABLE_MIN_SEC)
    _draw_settings_bar(frame, w // 4, h // 2 - 10,
                       "Sag el  ->  Onay suresi",
                       f"{new_stable:.1f}s",
                       t_stable, (0, 220, 255))

    if left_lm is not None:
        new_conf = round(_map(_pinch_distance(left_lm), CONF_MIN, CONF_MAX), 2)
    t_conf = (new_conf - CONF_MIN) / (CONF_MAX - CONF_MIN)
    _draw_settings_bar(frame, 3 * w // 4, h // 2 - 10,
                       "Sol el  ->  Min guven esigi",
                       f"{new_conf:.0%}",
                       t_conf, (255, 160, 0))

    cv2.putText(frame, ". = onayla ve geri don",
                (w // 2 - 145, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 220, 100), 2)

    return new_stable, new_conf

# ── Ana döngü ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--mirror", action="store_true")
    p.add_argument("--send", choices=["off", "udp", "tcp"], default="off")
    p.add_argument("--host", type=str, default="192.168.1.50")
    p.add_argument("--port", type=int, default=5005)
    p.add_argument("--min_interval_ms", type=int, default=120)
    p.add_argument("--always_send", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    clf, le = _load_model(CLF_FILE, LE_FILE)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Kamera açılamadı. --camera 0/1/2 dene.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mirror = not bool(args.mirror)  # mirrored by default
    sender = CommandSender(
        mode=args.send, host=args.host, port=args.port,
        min_interval_ms=args.min_interval_ms,
        send_only_on_change=not args.always_send,
    )

    stable_sec          = STABLE_SEC
    min_conf            = MIN_CONF
    text_buffer         = ""
    streak_sign         = ""     # sign currently being held
    streak_start        = 0.0    # when current streak began
    cooldown_until      = 0.0    # timestamp when cooldown ends
    prev_sign           = ""
    prev_t              = time.time()
    fps                 = 0.0
    paused              = False
    settings_mode       = False
    settings_stable     = stable_sec
    settings_conf       = min_conf
    touch_trigger_ready = True
    touch_start         = 0.0    # when touching began
    pause_start         = 0.0    # when pause/quit gesture began
    pause_gesture_ready = True
    quit_confirm        = False   # in quit confirmation phase
    quit_confirm_start  = 0.0

    win_title = "Isaret Dili → Metin"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_title, 1280, 720)

    try:
        with HandDetector(str(LANDMARKER_MODEL),
                          min_detection_confidence=0.55,
                          min_presence_confidence=0.55) as detector:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if mirror:
                    frame = cv2.flip(frame, 1)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("."):
                    if settings_mode:
                        stable_sec    = settings_stable
                        min_conf      = settings_conf
                        streak_sign   = ""
                        streak_start  = 0.0
                        settings_mode = False
                    else:
                        settings_stable = stable_sec
                        settings_conf   = min_conf
                        settings_mode   = True
                        paused          = False
                elif key == ord(" "):
                    if not settings_mode:
                        paused = not paused
                        if not paused:
                            streak_sign  = ""
                            streak_start = 0.0
                            prev_sign    = ""
                elif not settings_mode and not paused:
                    if key == ord("f"):
                        mirror = not mirror
                    elif key in (ord("b"), 8):
                        text_buffer  = text_buffer[:-1]
                        prev_sign    = ""
                        streak_sign  = ""
                        streak_start = 0.0
                    elif key == ord("c"):
                        text_buffer  = ""
                        prev_sign    = ""
                        streak_sign  = ""
                        streak_start = 0.0

                now = time.time()

                # MediaPipe for hand landmarks + settings gestures
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hands = detector.detect_all(rgb)
                touching = _hands_touching(hands)

                doing_spread = _pause_gesture(hands) and not touching

                if quit_confirm:
                    # ── Phase 2: confirmation ─────────────────────────────────
                    if doing_spread:
                        held = now - quit_confirm_start
                        _draw_quit_confirm(frame, hands, held / QUIT_CONFIRM_SEC, True)
                        if held >= QUIT_CONFIRM_SEC:
                            break  # confirmed quit
                    else:
                        # dropped stop sign → cancel
                        quit_confirm        = False
                        quit_confirm_start  = 0.0
                        pause_gesture_ready = True

                elif doing_spread:
                    # ── Phase 1: initial gesture ──────────────────────────────
                    if pause_start == 0.0:
                        pause_start = now
                    held = now - pause_start
                    if not settings_mode:
                        _draw_pause_progress(frame, hands, held / PAUSE_HOLD_SEC,
                                             "BASLAT" if paused else "DURDUR")
                    threshold = QUIT_HOLD_SEC if settings_mode else PAUSE_HOLD_SEC
                    if held >= threshold and pause_gesture_ready:
                        pause_gesture_ready = False
                        pause_start         = 0.0
                        if settings_mode:
                            quit_confirm       = True
                            quit_confirm_start = now
                        else:
                            paused = not paused
                            if not paused:
                                streak_sign  = ""
                                streak_start = 0.0
                                prev_sign    = ""

                else:
                    pause_start         = 0.0
                    pause_gesture_ready = True

                if touching and not paused and not quit_confirm:
                    if touch_start == 0.0:
                        touch_start = now
                    held = now - touch_start
                    _draw_touch_progress(frame, hands, held / TOUCH_HOLD_SEC,
                                         "KAPAT" if settings_mode else "AYARLAR")
                    if held >= TOUCH_HOLD_SEC and touch_trigger_ready:
                        touch_trigger_ready = False
                        touch_start         = 0.0
                        if settings_mode:
                            stable_sec    = settings_stable
                            min_conf      = settings_conf
                            streak_sign   = ""
                            streak_start  = 0.0
                            settings_mode = False
                        else:
                            settings_stable = stable_sec
                            settings_conf   = min_conf
                            settings_mode   = True
                else:
                    touch_start         = 0.0
                    touch_trigger_ready = True

                if quit_confirm:
                    pass  # already rendered above in gesture block

                elif settings_mode:
                    settings_stable, settings_conf = _draw_settings(
                        frame, hands, settings_stable, settings_conf)

                    # Line between index fingers — color signals active gesture
                    if len(hands) >= 2:
                        h_px, w_px = frame.shape[:2]
                        lm_a, lm_b = hands[0][0], hands[1][0]
                        p_a = (int(lm_a[8].x * w_px), int(lm_a[8].y * h_px))
                        p_b = (int(lm_b[8].x * w_px), int(lm_b[8].y * h_px))
                        if doing_spread:
                            progress = min((now - pause_start) / QUIT_HOLD_SEC if pause_start else 0.0, 1.0)
                            line_color = (
                                int(0   + progress * 0),
                                int(100 - progress * 100),
                                int(200 + progress * 55),
                            )  # blue → red as quit nears
                            _draw_progress_wheel(frame, (p_a[0]+p_b[0])//2, (p_a[1]+p_b[1])//2,
                                                 progress, line_color, (0, 0, 255), "CIK")
                        elif touching:
                            line_color = (0, 220, 255)   # cyan = confirming settings
                        else:
                            line_color = (180, 180, 180)  # gray = idle
                        cv2.line(frame, p_a, p_b, line_color, 2)

                elif paused:
                    _draw_pause(frame)

                else:  # normal mode
                    hand_lm = hands[0][0] if hands else None
                    sign = ""
                    conf = 0.0

                    if hand_lm is not None:
                        _draw_hand(frame, hand_lm)
                        feat  = normalize_landmarks(hand_lm)
                        proba = clf.predict_proba(feat)[0]
                        top_i = int(np.argmax(proba))
                        conf  = float(proba[top_i])
                        sign  = le.inverse_transform([top_i])[0]
                        if conf < min_conf:
                            sign = "?"

                    in_cooldown = now < cooldown_until

                    if sign and sign != "?" and not in_cooldown:
                        if sign == streak_sign:
                            held = now - streak_start
                            if held >= stable_sec:
                                if sign != prev_sign:
                                    if sign == "del":
                                        text_buffer = text_buffer[:-1]
                                    elif sign == "space":
                                        text_buffer += " "
                                    else:
                                        text_buffer += sign
                                prev_sign      = sign
                                cooldown_until = now + COOLDOWN_SEC
                                streak_sign    = ""
                                streak_start   = 0.0
                        else:
                            streak_sign  = sign
                            streak_start = now
                    else:
                        if not sign or sign == "?":
                            streak_sign  = ""
                            streak_start = 0.0
                            if not in_cooldown:
                                prev_sign = ""

                    dt     = now - prev_t
                    prev_t = now
                    if dt > 0:
                        fps = 0.9 * fps + 0.1 / dt if fps > 0 else 1.0 / dt

                    stable_progress = (now - streak_start) / stable_sec if streak_start else 0.0

                    try:
                        sender.send(sign or "IDLE")
                    except Exception:
                        pass

                    _draw_hud(frame, sign or "-", conf, stable_progress,
                              text_buffer, fps, mirror, in_cooldown,
                              stable_sec, min_conf)

                cv2.imshow(win_title, frame)

    finally:
        sender.close()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
