# D:\snapchat_filters\src\gesture_engine.py
import time
from collections import deque
from typing import List, Tuple, Optional
import numpy as np

from src.config import (
    SWIPE_MIN_DX,
    SWIPE_MAX_DY,
    SWIPE_WINDOW_FRAMES,
    SWIPE_COOLDOWN_SEC,
    PINCH_THRESHOLD_NORM,
    PINCH_COOLDOWN_SEC
)
from src.vision_tracker import HandData

class GestureEngine:
    def __init__(self):
        # Position history: deque of (timestamp, x, y)
        self.history = deque(maxlen=SWIPE_WINDOW_FRAMES)
        self.last_swipe_time = 0.0
        self.last_pinch_time = 0.0
        self.was_pinching = False
        self.active_pinch_pos = None
        self.pinch_metric = 0.0  # 0.0 (open) to 1.0 (closed)

        # UI Feedback message & timestamp
        self.feedback_text = ""
        self.feedback_time = 0.0

    def process(self, hands_data: List[HandData]) -> str:
        current_time = time.time()
        event = "NONE"

        if not hands_data:
            self.history.clear()
            self.was_pinching = False
            self.active_pinch_pos = None
            self.pinch_metric = 0.0
            return event

        # Use primary hand (first detected hand)
        primary_hand = hands_data[0]
        hx, hy = primary_hand.palm_center
        self.history.append((current_time, hx, hy))

        # 1. Pinch Detection
        dist_norm = primary_hand.pinch_dist_norm
        # Normalize pinch progress from 0.12 (open) down to threshold (closed)
        norm_open = PINCH_THRESHOLD_NORM * 2.2
        norm_close = PINCH_THRESHOLD_NORM
        clamped = max(norm_close, min(norm_open, dist_norm))
        self.pinch_metric = 1.0 - (clamped - norm_close) / (norm_open - norm_close)

        mid_pinch = (
            (primary_hand.index_tip[0] + primary_hand.thumb_tip[0]) // 2,
            (primary_hand.index_tip[1] + primary_hand.thumb_tip[1]) // 2
        )
        self.active_pinch_pos = mid_pinch

        is_pinch = (dist_norm < PINCH_THRESHOLD_NORM)
        if is_pinch:
            if not self.was_pinching and (current_time - self.last_pinch_time > PINCH_COOLDOWN_SEC):
                event = "PINCH"
                self.last_pinch_time = current_time
                self.was_pinching = True
                self.feedback_text = "PINCH: FILTER APPLIED!"
                self.feedback_time = current_time
                self.history.clear() # Prevent pinch from triggering swipe
                return event
        elif dist_norm > PINCH_THRESHOLD_NORM * 1.3:
            self.was_pinching = False

        # 2. Swipe Detection (only when not pinching)
        if not is_pinch and len(self.history) >= 4 and (current_time - self.last_swipe_time > SWIPE_COOLDOWN_SEC):
            # Check displacement between oldest point and current point
            old_time, old_x, old_y = self.history[0]
            dt = current_time - old_time

            if 0.05 < dt < 0.6:  # Reasonable duration for a hand swipe gesture
                dx = hx - old_x
                dy = hy - old_y

                if abs(dx) >= SWIPE_MIN_DX and abs(dy) <= SWIPE_MAX_DY:
                    if dx < -SWIPE_MIN_DX:
                        # Leftward motion in mirrored view -> Swipe Left
                        event = "SWIPE_LEFT"
                        self.feedback_text = "SWIPED LEFT [<< PREV]"
                    elif dx > SWIPE_MIN_DX:
                        # Rightward motion in mirrored view -> Swipe Right
                        event = "SWIPE_RIGHT"
                        self.feedback_text = "SWIPED RIGHT [NEXT >>]"

                    self.last_swipe_time = current_time
                    self.feedback_time = current_time
                    self.history.clear()
                    return event

        return event

    def get_feedback(self) -> Optional[str]:
        # Return feedback if within 1.2 seconds of occurrence
        if time.time() - self.feedback_time < 1.2:
            return self.feedback_text
        return None
