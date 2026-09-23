"""
Temporal Probability Accumulator & Winner-Take-All Decision Module
Integrates frame-level class probabilities over a sliding time window (default 1.0s)
and uses margin hysteresis to output the single dominant wash-hand action.
"""

import time
from collections import deque
from typing import Dict, Tuple, Optional
from src.rule_classifier import LABELS, LABEL_NAMES_ZH


class TemporalProbabilityAccumulator:
    """
    Accumulates multi-class probabilities over a temporal sliding window.
    Decides the single winning action using time-weighted integration and hysteresis locking.
    """

    def __init__(
        self,
        window_sec: float = 1.0,
        margin_threshold: float = 0.12,
        min_conf: float = 0.35,
    ):
        self.window_sec = window_sec
        self.margin_threshold = margin_threshold
        self.min_conf = min_conf

        # Queue storing (timestamp, {action_name: probability})
        self.history = deque()
        self.locked_action: str = "other"
        self.locked_conf: float = 0.0

    def reset(self):
        """Reset history and locked state."""
        self.history.clear()
        self.locked_action = "other"
        self.locked_conf = 0.0

    def update(
        self,
        frame_probs: Dict[str, float],
        timestamp: Optional[float] = None,
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Add new frame probabilities and return:
          (dominant_action, dominant_probability, integrated_probabilities)
        """
        now = timestamp if timestamp is not None else time.time()
        self.history.append((now, frame_probs))

        # 1. Purge items older than window_sec
        cutoff = now - self.window_sec
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

        if not self.history:
            return "other", 1.0, {"other": 1.0}

        # 2. Time-weighted probability accumulation
        # Newer frames in the window get slightly higher weight (0.6 -> 1.0)
        integrated_scores: Dict[str, float] = {k: 0.0 for k in LABELS.values()}
        total_weight = 0.0

        for t, p_dict in self.history:
            age = max(0.0, now - t)
            # Linear decay weight based on age inside window
            weight = 1.0 - (0.4 * age / max(0.01, self.window_sec))
            total_weight += weight

            for act, prob in p_dict.items():
                integrated_scores[act] = integrated_scores.get(act, 0.0) + prob * weight

        # Normalize so integrated probabilities sum to 1.0
        if total_weight > 0:
            for act in integrated_scores:
                integrated_scores[act] /= total_weight

        # 3. Find Top-1 Candidate and Scores
        sorted_acts = sorted(integrated_scores.items(), key=lambda x: x[1], reverse=True)
        top1_act, top1_score = sorted_acts[0]

        # 4. Winner-Take-All Decision with Hysteresis Locking
        # If currently locked into an active gesture (not "other"), require a margin to switch away
        if self.locked_action != "other" and top1_act != self.locked_action:
            current_locked_score = integrated_scores.get(self.locked_action, 0.0)
            # Only switch if candidate exceeds current by margin threshold
            if top1_score > current_locked_score + self.margin_threshold and top1_score >= self.min_conf:
                self.locked_action = top1_act
                self.locked_conf = top1_score
            else:
                self.locked_conf = current_locked_score
        else:
            if top1_score >= self.min_conf:
                self.locked_action = top1_act
                self.locked_conf = top1_score
            else:
                self.locked_action = "other"
                self.locked_conf = integrated_scores.get("other", top1_score)

        return self.locked_action, self.locked_conf, integrated_scores
