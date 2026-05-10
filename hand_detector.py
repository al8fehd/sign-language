from __future__ import annotations

import mediapipe as mp

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

_hands = mp.solutions.hands


class HandDetector:
    def __init__(self, task_path: str,
                 min_detection_confidence: float = 0.5,
                 min_presence_confidence: float = 0.5):
        # task_path korunuyor; API uyumu için argüman bırakıldı.
        _ = task_path
        self._detector = _hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_presence_confidence,
        )

    def _process(self, rgb_frame):
        return self._detector.process(rgb_frame)

    def detect(self, rgb_frame):
        """Return landmarks for the first detected hand, or None."""
        result = self._process(rgb_frame)
        if not result.multi_hand_landmarks:
            return None
        return result.multi_hand_landmarks[0].landmark

    def detect_all(self, rgb_frame) -> list[tuple]:
        """Return list of (landmarks, handedness_label) for all detected hands."""
        result = self._process(rgb_frame)
        if not result.multi_hand_landmarks:
            return []
        hands = []
        for i, lm in enumerate(result.multi_hand_landmarks):
            label = "Unknown"
            if result.multi_handedness and i < len(result.multi_handedness):
                label = result.multi_handedness[i].classification[0].label
            hands.append((lm.landmark, label))
        return hands

    def close(self):
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
