import cv2
import os
import pathlib

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

from hand_detector import HandDetector, HAND_CONNECTIONS

def _draw_hand(frame, landmarks, color=(0, 220, 0)) -> None:
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 100, 255), -1)

def main() -> int:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera acilamadi.")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    win_title = "El Haritasi"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_title, 1280, 720)

    try:
        with HandDetector(min_detection_confidence=0.55,
                          min_presence_confidence=0.55) as detector:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame = cv2.flip(frame, 1)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hands = detector.detect_all(rgb)

                for lm, label in hands:
                    color = (0, 220, 255) if label == "Right" else (255, 160, 0)
                    _draw_hand(frame, lm, color)

                cv2.imshow(win_title, frame)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
