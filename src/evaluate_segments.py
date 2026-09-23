"""
Ground Truth Segment-Level Evaluation Module
Evaluates Wash Hand Detect system against ground-truth time segments,
computing Frame Accuracy, Segment Precision, Recall, F1-score, and Confusion Matrix.
"""

import os
import argparse
from typing import Dict, List, Tuple, Any
import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.hand_detector import HandDetector
from src.features import extract_hand_features
from src.rule_classifier import WashHandRuleClassifier, LABELS, NAME_TO_LABEL, LABEL_SHORT_ZH
from src.accumulator import TemporalProbabilityAccumulator

# Ground Truth Segment Definitions
BENCHMARK_GROUND_TRUTH = {
    "wash_7steps_yt.mp4": [
        (0.0, 4.0, "other"),
        (4.0, 10.0, "inside"),
        (10.0, 16.0, "outside"),
        (16.0, 21.0, "interlace"),
        (21.0, 27.0, "knuckles"),
        (27.0, 33.0, "thumb"),
        (33.0, 39.0, "fingertips"),
        (39.0, 46.0, "wrist"),
        (46.0, 51.0, "other"),
    ],
    "wash_7steps_yt2.mp4": [
        (0.0, 18.0, "other"),
        (19.0, 23.0, "inside"),
        (24.0, 27.0, "outside"),
        (28.0, 31.0, "interlace"),
        (32.0, 36.0, "knuckles"),
        (37.0, 41.0, "thumb"),
        (42.0, 46.0, "fingertips"),
        (47.0, 51.0, "wrist"),
        (52.0, 93.0, "other"),
    ],
}


def get_ground_truth_label(time_sec: float, segments: List[Tuple[float, float, str]]) -> str:
    for start_t, end_t, label in segments:
        if start_t <= time_sec < end_t:
            return label
    return "other"


def evaluate_video(
    video_path: str,
    window_sec: float = 0.5,
    margin_threshold: float = 0.15,
) -> Dict[str, Any]:
    video_name = os.path.basename(video_path)
    if video_name not in BENCHMARK_GROUND_TRUTH:
        print(f"[!] Warning: No ground truth registered for {video_name}, fallback to generic.")
        segments = [(0.0, 999.0, "other")]
    else:
        segments = BENCHMARK_GROUND_TRUTH[video_name]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detector = HandDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    classifier = WashHandRuleClassifier()
    accumulator = TemporalProbabilityAccumulator(window_sec=window_sec, margin_threshold=margin_threshold)

    y_true = []
    y_pred = []
    prev_left = None
    prev_right = None

    frame_idx = 0
    print(f"\n=======================================================")
    print(f"📊 正在評估影片: {video_path}")
    print(f"   總幀數: {total_frames} | FPS: {fps:.1f} | 窗口: {window_sec}s")
    print(f"=======================================================")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        gt_label = get_ground_truth_label(timestamp, segments)

        left_hand, right_hand, _ = detector.process(frame)
        features = extract_hand_features(
            left_hand, right_hand, prev_left_hand=prev_left, prev_right_hand=prev_right
        )
        probs = classifier.predict_probabilities(features)
        pred_label, _, _ = accumulator.update(probs, timestamp=timestamp)

        prev_left = left_hand
        prev_right = right_hand

        y_true.append(gt_label)
        y_pred.append(pred_label)
        frame_idx += 1

    cap.release()
    detector.close()

    # Calculate metrics
    target_names = [LABELS[i] for i in range(8)]
    report = classification_report(y_true, y_pred, labels=target_names, zero_division=0, output_dict=True)
    report_text = classification_report(y_true, y_pred, labels=target_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=target_names)

    # Segment Accuracy calculation (for active washing segments)
    active_segments = [s for s in segments if s[2] != "other"]
    segment_accuracies = {}
    for s_start, s_end, s_label in active_segments:
        start_idx = int(s_start * fps)
        end_idx = min(len(y_true), int(s_end * fps))
        if end_idx > start_idx:
            sub_true = y_true[start_idx:end_idx]
            sub_pred = y_pred[start_idx:end_idx]
            correct = sum(1 for t, p in zip(sub_true, sub_pred) if p == s_label)
            acc = correct / len(sub_true)
            segment_accuracies[s_label] = acc

    print("\n📈 【分類準確率與 F1-Score 報告】")
    print(report_text)

    print("\n🎯 【七步各動作時間區間辨識率 (Segment Recall)】")
    for act, acc in segment_accuracies.items():
        zh_name = LABEL_SHORT_ZH.get(act, act)
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"   - {zh_name:2s} ({act:10s}): [{bar}] {acc*100:5.1f}%")

    print("\n🧩 【混淆矩陣 (Confusion Matrix)】")
    header = "       " + " ".join([f"{LABEL_SHORT_ZH[l]:>4s}" for l in target_names])
    print(header)
    for idx, row in enumerate(cm):
        row_str = " ".join([f"{val:4d}" for val in row])
        print(f"{LABEL_SHORT_ZH[target_names[idx]]:>4s}: {row_str}")

    return {
        "report": report,
        "segment_accuracies": segment_accuracies,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 7-step hand wash detector on ground truth video.")
    parser.add_argument("--video", type=str, default="data/raw/wash_7steps_yt.mp4", help="Video path")
    parser.add_argument("--window-sec", type=float, default=0.5, help="Temporal window size in seconds")
    parser.add_argument("--margin", type=float, default=0.15, help="Hysteresis margin")
    args = parser.parse_args()

    evaluate_video(args.video, window_sec=args.window_sec, margin_threshold=args.margin)
