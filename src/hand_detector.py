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
        min_detection_confidence: float = 0.25,
        min_tracking_confidence: float = 0.25,
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
        Process a BGR OpenCV frame with adaptive aspect-ratio ROI handling.
        Returns:
            left_hand: np.ndarray of shape (21, 3) or None
            right_hand: np.ndarray of shape (21, 3) or None
            results: raw MediaPipe results object for drawing
        """
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        # Adaptive ROI Fallback when full frame has no hands
        is_roi = False
        xmin, xmax, roi_w = 0, w, w
        ymin, ymax, roi_h = 0, h, h

        if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) == 0:
            # Check 1: Split-screen / Educational video (hands in right half)
            if w > h * 1.3:
                xmin, xmax = int(w * 0.42), w
                roi_w = xmax - xmin
                roi = rgb_frame[:, xmin:xmax]
                roi_results = self.hands.process(roi)
                if roi_results.multi_hand_landmarks:
                    results = roi_results
                    is_roi = True
            # Check 2: Vertical video central crop
            elif h > w * 1.2:
                ymin, ymax = int(h * 0.20), int(h * 0.80)
                roi_h = ymax - ymin
                roi = rgb_frame[ymin:ymax, :]
                roi_results = self.hands.process(roi)
                if roi_results.multi_hand_landmarks:
                    results = roi_results
                    is_roi = True

        left_hand = None
        right_hand = None

        if results.multi_hand_landmarks and results.multi_handedness:
            extracted_coords = []
            extracted_labels = []

            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = handedness.classification[0].label  # "Left" or "Right"
                coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                    dtype=np.float32,
                )
                if is_roi:
                    # Map coordinates back to full image normalized space [0, 1]
                    coords[:, 0] = (coords[:, 0] * roi_w + xmin) / float(w)
                    coords[:, 1] = (coords[:, 1] * roi_h + ymin) / float(h)
                    for idx_lm, lm_obj in enumerate(hand_landmarks.landmark):
                        lm_obj.x = float(coords[idx_lm, 0])
                        lm_obj.y = float(coords[idx_lm, 1])

                extracted_coords.append(coords)
                extracted_labels.append(label)

            if len(extracted_coords) == 1:
                if extracted_labels[0] == "Left":
                    left_hand = extracted_coords[0]
                else:
                    right_hand = extracted_coords[0]
            elif len(extracted_coords) >= 2:
                # If two hands detected, ensure both left and right are assigned even if labels duplicate
                if extracted_labels[0] != extracted_labels[1]:
                    for c, lab in zip(extracted_coords[:2], extracted_labels[:2]):
                        if lab == "Left":
                            left_hand = c
                        else:
                            right_hand = c
                else:
                    # Duplicate label fallback: sort by wrist X-coordinate
                    if extracted_coords[0][0, 0] <= extracted_coords[1][0, 0]:
                        left_hand, right_hand = extracted_coords[0], extracted_coords[1]
                    else:
                        left_hand, right_hand = extracted_coords[1], extracted_coords[0]

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
