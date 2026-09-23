"""
MediaPipe Hand Detector Wrapper Module
Extracts 21 3D landmarks for both Left and Right hands with deterministic ordering.
"""

from typing import Tuple, Optional, List, Dict, Any
import cv2
import numpy as np
import mediapipe as mp


def _enhance_contrast_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """Apply adaptive histogram equalization on L-channel to highlight finger contours under soap foam."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)


class HandDetector:
    """Detects and separates Left and Right hands using MediaPipe with temporal tracking & dropout compensation."""

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.50,
        min_tracking_confidence: float = 0.50,
        ghost_frames_threshold: int = 4,
        crop_split_screen: bool = False,
    ):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.crop_split_screen = crop_split_screen

        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Temporal tracking memory
        self.prev_left_hand: Optional[np.ndarray] = None
        self.prev_right_hand: Optional[np.ndarray] = None
        self.left_lost_count: int = 0
        self.right_lost_count: int = 0
        self.ghost_frames_threshold = ghost_frames_threshold

    def reset_tracking(self):
        """Reset temporal tracking history."""
        self.prev_left_hand = None
        self.prev_right_hand = None
        self.left_lost_count = 0
        self.right_lost_count = 0

    def process(
        self, frame: np.ndarray, crop_split_screen: Optional[bool] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Any]:
        """
        Process a BGR OpenCV frame with full frame MediaPipe tracking and CLAHE enhancement.
        Returns:
            left_hand: np.ndarray of shape (21, 3) or None
            right_hand: np.ndarray of shape (21, 3) or None
            results: raw MediaPipe results object for drawing
        """
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        use_split_crop = self.crop_split_screen if crop_split_screen is None else crop_split_screen
        is_roi = False
        xmin, xmax, roi_w = 0, w, w
        ymin, ymax, roi_h = 0, h, h

        if use_split_crop and w > h * 1.3:
            # Explicit split screen mode: Hands in right 58% of the frame
            xmin, xmax = int(w * 0.42), w
            ymin, ymax = int(h * 0.05), int(h * 0.95)
            roi_w = xmax - xmin
            roi_h = ymax - ymin
            roi = rgb_frame[ymin:ymax, xmin:xmax]
            roi_scaled = cv2.resize(roi, (roi_w * 2, roi_h * 2), interpolation=cv2.INTER_LINEAR)
            results = self.hands.process(roi_scaled)
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                is_roi = True
            else:
                roi_clahe = _enhance_contrast_clahe(roi_scaled)
                results_clahe = self.hands.process(roi_clahe)
                if results_clahe.multi_hand_landmarks and len(results_clahe.multi_hand_landmarks) > 0:
                    results = results_clahe
                    is_roi = True
                else:
                    results = self.hands.process(rgb_frame)
        else:
            # Default: Full frame detection
            results = self.hands.process(rgb_frame)
            # If hands not found or only 1 hand found under foam, fallback to CLAHE contrast enhancement
            if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) < 2:
                enhanced = _enhance_contrast_clahe(rgb_frame)
                results_clahe = self.hands.process(enhanced)
                if results_clahe.multi_hand_landmarks and len(results_clahe.multi_hand_landmarks) > (len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0):
                    results = results_clahe

        curr_left_hand = None
        curr_right_hand = None

        if results.multi_hand_landmarks and results.multi_handedness:
            extracted_coords = []
            extracted_labels = []

            max_dim = float(max(w, h))
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = handedness.classification[0].label  # "Left" or "Right"
                coords = np.array(
                    [[lm.x * (w / max_dim), lm.y * (h / max_dim), lm.z * (w / max_dim)] for lm in hand_landmarks.landmark],
                    dtype=np.float32,
                )
                if is_roi:
                    # Map coordinates from ROI back to full image normalized space
                    coords[:, 0] = (coords[:, 0] * roi_w + xmin) / max_dim
                    coords[:, 1] = (coords[:, 1] * roi_h + ymin) / max_dim

                extracted_coords.append(coords)
                extracted_labels.append(label)

            # 2. Hand ID Assignment & Temporal Tracking
            if len(extracted_coords) == 1:
                hand_cand = extracted_coords[0]
                label_cand = extracted_labels[0]

                # Check previous distance if available
                if self.prev_left_hand is not None and self.prev_right_hand is not None:
                    d_to_left = np.linalg.norm(hand_cand[0] - self.prev_left_hand[0])
                    d_to_right = np.linalg.norm(hand_cand[0] - self.prev_right_hand[0])
                    if d_to_left < d_to_right and d_to_left < 0.25:
                        curr_left_hand = hand_cand
                    elif d_to_right < d_to_left and d_to_right < 0.25:
                        curr_right_hand = hand_cand
                    else:
                        if label_cand == "Left":
                            curr_left_hand = hand_cand
                        else:
                            curr_right_hand = hand_cand
                else:
                    if label_cand == "Left":
                        curr_left_hand = hand_cand
                    else:
                        curr_right_hand = hand_cand

            elif len(extracted_coords) >= 2:
                h1, h2 = extracted_coords[0], extracted_coords[1]
                l1, l2 = extracted_labels[0], extracted_labels[1]

                # If labels are distinct and we have no strong previous history
                if l1 != l2 and (self.prev_left_hand is None or self.prev_right_hand is None):
                    curr_left_hand = h1 if l1 == "Left" else h2
                    curr_right_hand = h2 if l1 == "Left" else h1
                elif self.prev_left_hand is not None and self.prev_right_hand is not None:
                    # Hungarian/Euclidean matching against previous wrist positions
                    cost_standard = (
                        np.linalg.norm(h1[0] - self.prev_left_hand[0])
                        + np.linalg.norm(h2[0] - self.prev_right_hand[0])
                    )
                    cost_swapped = (
                        np.linalg.norm(h2[0] - self.prev_left_hand[0])
                        + np.linalg.norm(h1[0] - self.prev_right_hand[0])
                    )
                    if cost_standard <= cost_swapped:
                        curr_left_hand, curr_right_hand = h1, h2
                    else:
                        curr_left_hand, curr_right_hand = h2, h1
                else:
                    # Fallback when labels duplicate and no history: sort by wrist X
                    if h1[0, 0] <= h2[0, 0]:
                        curr_left_hand, curr_right_hand = h1, h2
                    else:
                        curr_left_hand, curr_right_hand = h2, h1

        # 3. Temporal Dropout Smoothing / Ghosting for Foam & Occlusion
        if curr_left_hand is not None:
            self.prev_left_hand = curr_left_hand.copy()
            self.left_lost_count = 0
            final_left = curr_left_hand
        elif self.prev_left_hand is not None and self.left_lost_count < self.ghost_frames_threshold:
            self.left_lost_count += 1
            final_left = self.prev_left_hand
        else:
            final_left = None
            self.prev_left_hand = None
            self.left_lost_count = 0

        if curr_right_hand is not None:
            self.prev_right_hand = curr_right_hand.copy()
            self.right_lost_count = 0
            final_right = curr_right_hand
        elif self.prev_right_hand is not None and self.right_lost_count < self.ghost_frames_threshold:
            self.right_lost_count += 1
            final_right = self.prev_right_hand
        else:
            final_right = None
            self.prev_right_hand = None
            self.right_lost_count = 0

        return final_left, final_right, results

    def draw_hands(
        self, frame: np.ndarray, results: Any, draw_styled: bool = True
    ) -> np.ndarray:
        """Draw hand skeleton overlay onto frame."""
        if not results or not results.multi_hand_landmarks:
            return frame

        canvas = frame.copy()
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            label = handedness.classification[0].label
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
