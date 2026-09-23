"""
MediaPipe Hand Detector Wrapper Module
Extracts 21 3D landmarks for both Left and Right hands with deterministic ordering.
"""

from typing import Tuple, Optional, List, Dict, Any
import cv2
import numpy as np
import mediapipe as mp


class HandDetector:
    """Detects and separates Left and Right hands using MediaPipe."""

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(
        self, frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Any]:
        """
        Process a BGR OpenCV frame.
        Returns:
            left_hand: np.ndarray of shape (21, 3) or None
            right_hand: np.ndarray of shape (21, 3) or None
            results: raw MediaPipe results object for drawing
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        left_hand = None
        right_hand = None

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = handedness.classification[0].label  # "Left" or "Right"
                score = handedness.classification[0].score

                # Extract 21 x 3 normalized coordinates (x, y, z)
                coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                    dtype=np.float32,
                )

                if label == "Left":
                    left_hand = coords
                elif label == "Right":
                    right_hand = coords

        return left_hand, right_hand, results

    def draw_hands(
        self, frame: np.ndarray, results: Any, draw_styled: bool = True
    ) -> np.ndarray:
        """Draw hand skeleton overlay onto frame."""
        if not results.multi_hand_landmarks:
            return frame

        canvas = frame.copy()
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            label = handedness.classification[0].label
            # Color distinctions for left/right
            if draw_styled:
                self.mp_draw.draw_landmarks(
                    canvas,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )
            else:
                color = (0, 255, 128) if label == "Left" else (255, 128, 0)
                self.mp_draw.draw_landmarks(
                    canvas,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=3),
                    self.mp_draw.DrawingSpec(color=(220, 220, 220), thickness=2),
                )

        return canvas

    def close(self):
        if self.hands:
            self.hands.close()
