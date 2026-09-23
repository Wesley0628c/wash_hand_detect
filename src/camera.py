"""
Camera & Video Source Helper Module
Provides flexible frame capture from webcam, video files, or synthetic test streams.
"""

import time
import cv2
import numpy as np
from typing import Optional, Tuple, Union


class Camera:
    """Wrapper around cv2.VideoCapture with automatic FPS calculation and resource handling."""

    def __init__(self, source: Union[int, str] = 0, width: int = 1280, height: int = 720):
        self.source = source
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.prev_time = 0.0
        self.fps = 0.0
        self._is_synthetic = False

    def open(self) -> bool:
        if isinstance(self.source, str) and self.source == "synthetic":
            self._is_synthetic = True
            return True

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            return False

        if isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._is_synthetic:
            # Generate a clean dark background test canvas
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            frame[:] = (30, 30, 30)
            self._update_fps()
            return True, frame

        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            # If video file reached end, can loop if needed
            if isinstance(self.source, str) and self.source != "synthetic":
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            if not ret:
                return False, None

        self._update_fps()
        return True, frame

    def _update_fps(self):
        curr_time = time.time()
        if self.prev_time > 0:
            dt = curr_time - self.prev_time
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps > 0 else (1.0 / dt)
        self.prev_time = curr_time

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
