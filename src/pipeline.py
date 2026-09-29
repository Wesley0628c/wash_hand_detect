"""
Unified Wash Hand Detection Pipeline Module
Provides a single, standardized pipeline (WashHandPipeline) for real-time video,
offline batch evaluation, and benchmark analysis.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
import cv2
import numpy as np

from src.hand_detector import HandDetector
from src.features import extract_hand_features, feature_dict_to_vector
from src.rule_classifier import WashHandRuleClassifier, LABELS, NAME_TO_LABEL, LABEL_SHORT_ZH, FEEDBACK_ZH
from src.ml_classifier import WashHandMLClassifier
from src.accumulator import TemporalProbabilityAccumulator
from src.state_machine import WashHandStateMachine


@dataclass
class PipelineConfig:
    """Centralized configuration for wash hand detection pipeline."""
    model_type: str = "rule"  # "hybrid", "rule", "ml" — default is pure rule-based
    model_path: str = "models/wash_hand_xgb.joblib"
    min_detection_confidence: float = 0.50
    min_tracking_confidence: float = 0.50
    ghost_frames_threshold: int = 4
    crop_split_screen: bool = False
    window_sec: float = 0.50
    margin_threshold: float = 0.12
    consecutive_frames_required: int = 2
    rule_weight: float = 0.25
    step_duration: float = 1.0
    guide_mode: str = "free"  # "free" or "sequence"



@dataclass
class FrameResult:
    """Structured per-frame output across the detection and classification lifecycle."""
    timestamp_sec: float
    frame_id: int
    left_hand: Optional[np.ndarray]
    right_hand: Optional[np.ndarray]
    num_hands_observed: int
    hand_status: Dict[str, str]
    is_observed: bool
    quality_score: float
    features: Dict[str, Any]
    feature_vector: np.ndarray
    raw_scores: Dict[str, float]
    raw_label: str
    integrated_probs: Dict[str, float]
    display_label: str
    observed_label: str
    confidence: float
    feedback_msg: str
    just_completed_step: Optional[str] = None
    is_session_completed: bool = False


class WashHandPipeline:
    """
    Unified end-to-end pipeline encapsulating hand detection, coordinate normalization,
    feature extraction, hierarchical rule & ML classification, temporal smoothing,
    and wash progress state tracking.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        self.detector = HandDetector(
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            ghost_frames_threshold=self.config.ghost_frames_threshold,
            crop_split_screen=self.config.crop_split_screen,
        )

        self.rule_classifier = WashHandRuleClassifier()

        self.ml_classifier = None
        if self.config.model_type in ("hybrid", "ml"):
            self.ml_classifier = WashHandMLClassifier(
                model_path=self.config.model_path,
                hybrid_with_rules=(self.config.model_type == "hybrid"),
                rule_weight=self.config.rule_weight,
            )

        consecutive_req = 2 if self.config.model_type == "rule" else self.config.consecutive_frames_required
        self.accumulator = TemporalProbabilityAccumulator(
            window_sec=self.config.window_sec,
            margin_threshold=self.config.margin_threshold,
            consecutive_frames_required=consecutive_req,
        )

        self.state_machine = WashHandStateMachine(
            mode=self.config.guide_mode,
            step_duration=self.config.step_duration,
        )
        self.state_machine.start()

        self.prev_left: Optional[np.ndarray] = None
        self.prev_right: Optional[np.ndarray] = None
        self.frame_idx: int = 0
        self._prev_timestamp: Optional[float] = None

    def reset(self):
        """Reset internal pipeline history."""
        self.detector.reset()
        self.accumulator.reset()
        self.state_machine.reset()
        if self.ml_classifier is not None:
            self.ml_classifier.reset()
        self.prev_left = None
        self.prev_right = None
        self.frame_idx = 0
        self._prev_timestamp = None

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_sec: Optional[float] = None,
        crop_split_screen: Optional[bool] = None,
    ) -> Tuple[FrameResult, Any]:
        """
        Process a single BGR video frame through the complete pipeline.
        Returns:
            (FrameResult, mediapipe_results)
        """
        self.frame_idx += 1
        ts = timestamp_sec if timestamp_sec is not None else float(self.frame_idx / 30.0)

        # Compute real dt from consecutive timestamps (avoids hardcoded 1/30)
        if self._prev_timestamp is not None:
            dt = max(1e-4, ts - self._prev_timestamp)
        else:
            dt = 1.0 / 30.0
        self._prev_timestamp = ts

        # 1. Detection
        left_hand, right_hand, results = self.detector.process(
            frame, crop_split_screen=crop_split_screen
        )
        meta = self.detector.last_metadata

        # 2. Features
        features = extract_hand_features(
            left_hand, right_hand, self.prev_left, self.prev_right
        )
        vec_160 = feature_dict_to_vector(features)

        # 3. Model Classification
        # FIX: rule_classifier.predict_probabilities() does not accept timestamp;
        # ml_classifier handles its own buffer reset when no hands.
        if left_hand is not None or right_hand is not None:
            if self.config.model_type in ("hybrid", "ml") and self.ml_classifier is not None:
                frame_probs = self.ml_classifier.predict_probabilities(features)
            else:
                # rule-only: correct API call (no timestamp param)
                frame_probs = self.rule_classifier.predict_probabilities(features)
        else:
            # No hands: reset ML temporal buffer to prevent stale feature carry-over
            if self.ml_classifier is not None:
                self.ml_classifier.reset()
            frame_probs = {k: 0.01 for k in LABELS.values()}
            frame_probs["other"] = 0.93

        raw_label = max(frame_probs, key=frame_probs.get)

        # 4. Temporal Smoothing Accumulator
        label, conf, integrated = self.accumulator.update(frame_probs, timestamp=ts)

        # 5. State Machine Tracking (separate observed evidence from unobserved)
        is_fresh_observation = meta.get("is_observed", False)
        just_completed, completed_name = self.state_machine.update(
            label, dt=dt, is_observed=is_fresh_observation
        )

        feedback_msg = FEEDBACK_ZH.get(label, "動作調整中，請依步驟搓洗")

        self.prev_left = left_hand
        self.prev_right = right_hand

        result = FrameResult(
            timestamp_sec=ts,
            frame_id=self.frame_idx,
            left_hand=left_hand,
            right_hand=right_hand,
            num_hands_observed=meta.get("raw_num_hands", 0),
            hand_status={"left": meta.get("left_status", "missing"), "right": meta.get("right_status", "missing")},
            is_observed=is_fresh_observation,
            quality_score=meta.get("quality_score", 0.0),
            features=features,
            feature_vector=vec_160,
            raw_scores=frame_probs,
            raw_label=raw_label,
            integrated_probs=integrated,
            display_label=label,
            observed_label=label if is_fresh_observation else "other",
            confidence=conf,
            feedback_msg=feedback_msg,
            just_completed_step=completed_name,
            is_session_completed=self.state_machine.is_completed,
        )

        return result, results
