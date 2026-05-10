from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class GestureResult:
    command: str  # STOP, FORWARD, LEFT, RIGHT, IDLE
    confidence: float  # heuristic [0..1]


def _as_xy(landmark) -> np.ndarray:
    return np.array([float(landmark.x), float(landmark.y)], dtype=np.float32)


def _finger_extended(pts: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> bool:
    """
    Heuristic: finger is 'extended' if tip is farther from wrist than PIP and DIP
    along the finger chain direction. Works decently for a single front camera.
    """
    mcp, pip, dip, tip = pts
    # Use mcp as reference; compare distances to tip vs pip/dip.
    d_tip = np.linalg.norm(tip - mcp)
    d_pip = np.linalg.norm(pip - mcp)
    d_dip = np.linalg.norm(dip - mcp)
    return (d_tip > d_pip * 1.12) and (d_tip > d_dip * 1.08)


def _thumb_extended(wrist: np.ndarray, thumb_cmc: np.ndarray, thumb_mcp: np.ndarray, thumb_tip: np.ndarray) -> bool:
    # Thumb is sideways; use angle-ish heuristic: tip far from wrist and far from mcp.
    d_tip_w = np.linalg.norm(thumb_tip - wrist)
    d_mcp_w = np.linalg.norm(thumb_mcp - wrist)
    d_tip_mcp = np.linalg.norm(thumb_tip - thumb_mcp)
    return (d_tip_w > d_mcp_w * 1.05) and (d_tip_mcp > 0.06)


def classify_hand_gesture(hand_landmarks, handedness_label: Optional[str] = None) -> GestureResult:
    """
    Classifies gesture using MediaPipe Hand landmarks.
    Returns one of: STOP, FORWARD, LEFT, RIGHT, IDLE
    """
    # Accepts either a list of NormalizedLandmark (new tasks API) or a
    # proto with a .landmark attribute (old solutions API).
    lm = hand_landmarks if isinstance(hand_landmarks, (list, tuple)) else hand_landmarks.landmark

    # Indices per MediaPipe Hands:
    # 0 wrist
    # Thumb: 1,2,3,4
    # Index: 5,6,7,8
    # Middle: 9,10,11,12
    # Ring: 13,14,15,16
    # Pinky: 17,18,19,20
    wrist = _as_xy(lm[0])

    thumb = (_as_xy(lm[1]), _as_xy(lm[2]), _as_xy(lm[3]), _as_xy(lm[4]))
    index = (_as_xy(lm[5]), _as_xy(lm[6]), _as_xy(lm[7]), _as_xy(lm[8]))
    middle = (_as_xy(lm[9]), _as_xy(lm[10]), _as_xy(lm[11]), _as_xy(lm[12]))
    ring = (_as_xy(lm[13]), _as_xy(lm[14]), _as_xy(lm[15]), _as_xy(lm[16]))
    pinky = (_as_xy(lm[17]), _as_xy(lm[18]), _as_xy(lm[19]), _as_xy(lm[20]))

    thumb_ext = _thumb_extended(wrist, thumb[0], thumb[1], thumb[3])
    index_ext = _finger_extended(index)
    middle_ext = _finger_extended(middle)
    ring_ext = _finger_extended(ring)
    pinky_ext = _finger_extended(pinky)

    ext = [thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext]
    ext_count = int(sum(1 for v in ext if v))

    # --- Primary gestures ---
    # Fist: none or only thumb ambiguous
    if (not index_ext) and (not middle_ext) and (not ring_ext) and (not pinky_ext) and (not thumb_ext or ext_count <= 1):
        return GestureResult(command="STOP", confidence=0.85)

    # Open palm: 4 or 5 extended (thumb may be unreliable)
    if index_ext and middle_ext and ring_ext and pinky_ext and (thumb_ext or ext_count >= 4):
        return GestureResult(command="FORWARD", confidence=0.85)

    # Pointing: only index extended (thumb may float)
    other_fingers_folded = (not middle_ext) and (not ring_ext) and (not pinky_ext)
    if index_ext and other_fingers_folded:
        # Determine direction based on index finger vector in image coordinates.
        mcp, pip, dip, tip = index
        v = tip - mcp
        dx, dy = float(v[0]), float(v[1])

        # Need finger mostly horizontal for left/right; otherwise idle.
        if abs(dx) > abs(dy) * 1.15 and abs(dx) > 0.03:
            # In image coords: x increases to the right.
            cmd = "RIGHT" if dx > 0 else "LEFT"

            # If we have handedness, slightly boost confidence (but we don't invert).
            conf = 0.80 if handedness_label in ("Left", "Right") else 0.72
            return GestureResult(command=cmd, confidence=conf)

        return GestureResult(command="IDLE", confidence=0.55)

    return GestureResult(command="IDLE", confidence=0.40)

