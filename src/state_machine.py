"""
Wash Hand State Machine Module
Tracks 7-step wash progress, timing requirements, and transition conditions
in both Sequential (Teaching) and Free modes.
"""

import time
from typing import List, Dict, Set, Optional, Tuple

STEPS_ORDER = [
    "inside",
    "outside",
    "interlace",
    "knuckles",
    "thumb",
    "fingertips",
    "wrist",
]

STEPS_ZH = {
    "inside": "內",
    "outside": "外",
    "interlace": "夾",
    "knuckles": "弓",
    "thumb": "大",
    "fingertips": "立",
    "wrist": "腕",
}


class WashHandStateMachine:
    """Manages the lifecycle and state transitions of a wash-hand session."""

    def __init__(
        self,
        mode: str = "sequence",  # "sequence" or "free"
        step_duration: float = 2.5,  # required continuous seconds per step
        error_tolerance: float = 0.4,  # grace window before resetting timer on bad frame
    ):
        self.mode = mode
        self.step_duration = step_duration
        self.error_tolerance = error_tolerance

        self.current_step_idx = 0
        self.completed_steps: Set[str] = set()
        self.step_times: Dict[str, float] = {step: 0.0 for step in STEPS_ORDER}

        self.active_step: Optional[str] = None
        self.step_timer = 0.0
        self.grace_timer = 0.0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.is_completed = False

    def reset(self):
        """Reset state machine for a new wash session."""
        self.current_step_idx = 0
        self.completed_steps.clear()
        self.step_times = {step: 0.0 for step in STEPS_ORDER}
        self.active_step = None
        self.step_timer = 0.0
        self.grace_timer = 0.0
        self.start_time = None
        self.end_time = None
        self.is_completed = False

    def update(self, detected_label: str, dt: float) -> Tuple[bool, Optional[str]]:
        """
        Update state machine with the latest detected action label and delta time.
        Returns (is_step_just_completed, completed_step_name).
        """
        if self.is_completed:
            return False, None

        if self.start_time is None and detected_label in STEPS_ORDER:
            self.start_time = time.time()

        target_step = self._get_target_step()
        just_completed = False
        completed_step_name = None

        if self.mode == "sequence":
            is_valid_action = (detected_label == target_step)
            active_target = target_step
        else:  # "free" mode
            is_valid_action = (detected_label in STEPS_ORDER and detected_label not in self.completed_steps)
            active_target = detected_label if is_valid_action else self.active_step

        if is_valid_action and active_target is not None:
            if self.active_step != active_target:
                self.active_step = active_target
                self.step_timer = 0.0
                self.grace_timer = 0.0

            self.step_timer += dt
            self.step_times[active_target] += dt
            self.grace_timer = 0.0

            if self.step_timer >= self.step_duration:
                self.completed_steps.add(active_target)
                just_completed = True
                completed_step_name = active_target
                self.step_timer = 0.0
                self.active_step = None

                if self.mode == "sequence":
                    self.current_step_idx += 1
                    if self.current_step_idx >= len(STEPS_ORDER):
                        self._finish_session()
                else:
                    if len(self.completed_steps) >= len(STEPS_ORDER):
                        self._finish_session()
        else:
            # Tolerant decay when action temporarily flickers
            if self.active_step is not None and self.step_timer > 0:
                self.grace_timer += dt
                if self.grace_timer > self.error_tolerance:
                    self.step_timer = max(0.0, self.step_timer - dt * 2.0)
                    if self.step_timer == 0.0:
                        self.active_step = None

        return just_completed, completed_step_name

    def _get_target_step(self) -> Optional[str]:
        if self.mode == "sequence":
            if self.current_step_idx < len(STEPS_ORDER):
                return STEPS_ORDER[self.current_step_idx]
            return None
        return None

    def _finish_session(self):
        self.is_completed = True
        self.end_time = time.time()

    def get_progress_summary(self) -> Dict:
        """Return structured summary of current wash session status."""
        total_time = 0.0
        if self.start_time is not None:
            end = self.end_time if self.end_time is not None else time.time()
            total_time = max(0.0, end - self.start_time)

        current_step_progress = min(1.0, self.step_timer / self.step_duration) if self.step_duration > 0 else 0.0

        return {
            "mode": self.mode,
            "target_step": self._get_target_step(),
            "active_step": self.active_step,
            "current_step_progress": current_step_progress,
            "step_timer": self.step_timer,
            "step_duration": self.step_duration,
            "completed_steps": list(self.completed_steps),
            "completed_count": len(self.completed_steps),
            "total_steps": len(STEPS_ORDER),
            "is_completed": self.is_completed,
            "total_time": total_time,
            "step_times": self.step_times,
        }
