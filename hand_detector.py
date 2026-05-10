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

_vision = mp.tasks.vision
_BaseOptions = mp.tasks.BaseOptions


class HandDetector:
    def __init__(self, task_path: str,
                 min_detection_confidence: float = 0.5,
                 min_presence_confidence: float = 0.5):
        options = _vision.HandLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=task_path),
            running_mode=_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_presence_confidence,
        )
        self._landmarker = _vision.HandLandmarker.create_from_options(options)
        self._ts = 0

    def _process(self, rgb_frame):
        self._ts += 1
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self._landmarker.detect_for_video(mp_image, self._ts)

    def detect(self, rgb_frame):
        """Return landmarks for the first detected hand, or None."""
        result = self._process(rgb_frame)
        if not result.hand_landmarks:
            return None
        return result.hand_landmarks[0]

    def detect_all(self, rgb_frame) -> list[tuple]:
        """Return list of (landmarks, handedness_label) for all detected hands."""
        result = self._process(rgb_frame)
        if not result.hand_landmarks:
            return []
        hands = []
        for i, lm in enumerate(result.hand_landmarks):
            label = "Unknown"
            if result.handedness and i < len(result.handedness):
                label = result.handedness[i][0].category_name
            hands.append((lm, label))
        return hands

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
