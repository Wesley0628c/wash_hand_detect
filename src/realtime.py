"""
Real-time Wash Hand Detection & Feedback Pipeline
Combines Camera, MediaPipe Hand Tracking, Feature Extraction,
Classifiers (Rule-based & LSTM), Temporal Smoothing, State Machine, and UI.
"""

import os
import time
import argparse
from collections import deque, Counter
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np

from src.camera import Camera
from src.hand_detector import HandDetector
from src.features import extract_hand_features, feature_dict_to_vector
from src.rule_classifier import WashHandRuleClassifier, LABELS, NAME_TO_LABEL
from src.state_machine import WashHandStateMachine
from src.ui import WashHandHUD


class RealtimeWashHandDetector:
    """Main Real-time Application Engine."""

    def __init__(
        self,
        mode: str = "rule",
        model_path: str = "models/wash_hand_lstm.keras",
        guide_mode: str = "sequence",
        step_duration: float = 2.5,
        confidence_thresh: float = 0.75,
        smoothing_window: int = 10,
        source: Any = 0,
    ):
        self.mode = mode
        self.confidence_thresh = confidence_thresh
        self.camera = Camera(source=source)
        self.detector = HandDetector(max_num_hands=2)
        self.rule_classifier = WashHandRuleClassifier()
        self.state_machine = WashHandStateMachine(mode=guide_mode, step_duration=step_duration)
        self.hud = WashHandHUD()

        self.seq_len = 30
        self.feature_history = deque(maxlen=self.seq_len)
        self.pred_history = deque(maxlen=smoothing_window)

        # Load LSTM model if requested and exists
        self.lstm_model = None
        if self.mode == "lstm":
            if os.path.exists(model_path):
                import tensorflow as tf
                try:
                    self.lstm_model = tf.keras.models.load_model(model_path)
                    print(f"[+] Loaded LSTM model from: {model_path}")
                except Exception as e:
                    print(f"[!] Warning: Failed to load LSTM model ({e}). Falling back to rule-based.")
                    self.mode = "rule"
            else:
                print(f"[!] Warning: Model file '{model_path}' not found. Defaulting to rule-based.")
                self.mode = "rule"

        self.prev_left = None
        self.prev_right = None
        self.prev_timestamp = time.time()

    def run(self):
        if not self.camera.open():
            print(f"[Error] Could not open video source: {self.camera.source}")
            return

        print("\n=======================================================")
        print(" Wash Hand Real-time Detector Started!")
        print(" Controls:")
        print("   [Q] Quit application")
        print("   [R] Reset wash session")
        print("   [M] Switch mode (Sequence / Free)")
        print("   [C] Switch classifier (Rule-based / LSTM)")
        print("=======================================================\n")

        while True:
            ret, frame = self.camera.read()
            if not ret:
                break

            now = time.time()
            dt = max(0.001, min(0.1, now - self.prev_timestamp))
            self.prev_timestamp = now

            # Flip for mirror selfie perspective (if camera)
            if isinstance(self.camera.source, int):
                frame = cv2.flip(frame, 1)

            # 1. MediaPipe Hand Detection
            left_hand, right_hand, results = self.detector.process(frame)
            canvas = self.detector.draw_hands(frame, results)

            # 2. Feature Extraction
            features = extract_hand_features(
                left_hand, right_hand, self.prev_left, self.prev_right
            )
            feat_vec = feature_dict_to_vector(features)
            self.prev_left = left_hand
            self.prev_right = right_hand

            self.feature_history.append(feat_vec)

            # 3. Action Prediction
            raw_label, confidence, feedback_msg = self._classify(features)

            # 4. Temporal Smoothing
            self.pred_history.append(raw_label)
            smoothed_label = self._smooth_prediction(raw_label)

            # 5. State Machine Update
            just_completed, completed_step = self.state_machine.update(smoothed_label, dt)
            if just_completed and completed_step:
                print(f"[🎉] Step Completed: {completed_step}")

            # 6. UI HUD Rendering
            progress = self.state_machine.get_progress_summary()
            mode_display = f"Mode: {self.state_machine.mode.capitalize()} | {self.mode.upper()}"
            final_frame = self.hud.draw_hud(
                canvas,
                detected_label=smoothed_label,
                confidence=confidence,
                feedback_msg=feedback_msg,
                progress_summary=progress,
                fps=self.camera.fps,
                mode_str=mode_display,
            )

            cv2.imshow("Wash Hand Detect — MediaPipe 7-Steps", final_frame)

            # Key Handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                self.state_machine.reset()
                self.pred_history.clear()
                print("[*] Session reset.")
            elif key == ord("m"):
                new_mode = "free" if self.state_machine.mode == "sequence" else "sequence"
                self.state_machine.mode = new_mode
                self.state_machine.reset()
                print(f"[*] Switched Guide Mode to: {new_mode}")
            elif key == ord("c"):
                if self.lstm_model is not None:
                    self.mode = "lstm" if self.mode == "rule" else "rule"
                    print(f"[*] Switched Classifier to: {self.mode}")
                else:
                    print("[!] LSTM model is not loaded. Train a model first via `python -m src.train`.")

        self.camera.release()
        self.detector.close()
        cv2.destroyAllWindows()

    def _classify(self, features: Dict[str, Any]) -> Tuple[str, float, str]:
        # Always evaluate rule classifier for guidance feedback
        rule_label, rule_conf, feedback = self.rule_classifier.predict(features)

        if self.mode == "lstm" and self.lstm_model is not None and len(self.feature_history) == self.seq_len:
            seq_input = np.expand_dims(np.array(self.feature_history, dtype=np.float32), axis=0)
            probs = self.lstm_model.predict(seq_input, verbose=0)[0]
            max_idx = int(np.argmax(probs))
            conf = float(probs[max_idx])
            lstm_label = LABELS.get(max_idx, "other")

            if conf < self.confidence_thresh:
                return "other", conf, feedback
            return lstm_label, conf, feedback

        return rule_label, rule_conf, feedback

    def _smooth_prediction(self, current_label: str) -> str:
        if len(self.pred_history) < 4:
            return current_label

        counter = Counter(self.pred_history)
        most_common, count = counter.most_common(1)[0]

        # Majority rule: at least 60% agreement in window
        if count >= len(self.pred_history) * 0.6:
            return most_common
        return current_label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Wash Hand Detection Application")
    parser.add_argument("--mode", type=str, default="rule", choices=["rule", "lstm"], help="Classifier mode")
    parser.add_argument("--model-path", type=str, default="models/wash_hand_lstm.keras", help="Path to LSTM model")
    parser.add_argument("--guide-mode", type=str, default="sequence", choices=["sequence", "free"], help="Guide flow mode")
    parser.add_argument("--step-duration", type=float, default=2.5, help="Seconds required per step")
    parser.add_argument("--confidence-thresh", type=float, default=0.75, help="Confidence threshold for LSTM")
    parser.add_argument("--source", default=0, help="Camera index, video path, or 'synthetic'")
    args = parser.parse_args()

    # If source is digit string, convert to int
    source_val = int(args.source) if str(args.source).isdigit() else args.source

    app = RealtimeWashHandDetector(
        mode=args.mode,
        model_path=args.model_path,
        guide_mode=args.guide_mode,
        step_duration=args.step_duration,
        confidence_thresh=args.confidence_thresh,
        source=source_val,
    )
    app.run()
