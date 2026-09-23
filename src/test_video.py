"""
Offline / Batch Video Evaluation Script
Runs the Wash Hand Detection Pipeline on a video file with 1.0s Windowed Maximum-Probability Integration.
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np
from collections import Counter

from src.hand_detector import HandDetector
from src.features import extract_hand_features, feature_dict_to_vector
from src.ml_classifier import WashHandMLClassifier
from src.rule_classifier import WashHandRuleClassifier, LABEL_NAMES_ZH, LABELS, FEEDBACK_ZH
from src.accumulator import TemporalProbabilityAccumulator
from src.state_machine import WashHandStateMachine
from src.ui import WashHandHUD


def evaluate_video(
    video_path: str,
    output_annotated_path: str = None,
    model_type: str = "hybrid",
    guide_mode: str = "free",
    step_duration: float = 1.0,
    window_sec: float = 0.5,
    max_frames: int = 1500,
    roi_mode: str = "auto",
    show_window: bool = False,
):
    if not os.path.exists(video_path):
        print(f"[!] Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Could not open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    use_right_roi = (roi_mode == "right") or (roi_mode == "auto" and width > height * 1.3)

    print(f"\n=======================================================")
    print(f"🎬 測試影片: {video_path}")
    print(f"   解析度: {width}x{height} | FPS: {fps:.1f} | 總幀數: {total_frames}")
    print(f"   模型類型: {model_type.upper()} | 決策窗口: {window_sec} 秒")
    print(f"   ROI 模式: {roi_mode} ({'啟用右側子母畫面裁切' if use_right_roi else '全畫面模式'})")
    print(f"=======================================================")

    detector = HandDetector(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    ml_classifier = WashHandMLClassifier(model_path="models/wash_hand_xgb.joblib", hybrid_with_rules=(model_type == "hybrid"))
    rule_classifier = WashHandRuleClassifier()
    accumulator = TemporalProbabilityAccumulator(window_sec=window_sec, margin_threshold=0.12)
    state_machine = WashHandStateMachine(mode=guide_mode, step_duration=step_duration)
    state_machine.start()

    writer = None
    if output_annotated_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_annotated_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_annotated_path, fourcc, fps, (width, height))

    prev_left, prev_right = None, None
    detected_frames = 0
    predictions = Counter()
    start_time = time.time()
    frame_idx = 0

    hud = WashHandHUD()

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames > 0 and frame_idx >= max_frames):
            break

        frame_idx += 1
        dt = 1.0 / fps
        simulated_time = frame_idx * dt

        canvas = frame.copy()

        if use_right_roi:
            xmin = int(width * 0.42)
            roi = frame[:, xmin:].copy()
            left_hand, right_hand, results = detector.process(roi)
            # Draw skeleton directly on ROI, then composite back into canvas
            roi_annotated = detector.draw_hands(roi, results)
            canvas[:, xmin:] = roi_annotated
            cv2.rectangle(canvas, (xmin, 0), (width, height), (0, 255, 0), 2)
            cv2.putText(canvas, "[ROI Area]", (xmin + 15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            left_hand, right_hand, results = detector.process(frame)
            canvas = detector.draw_hands(canvas, results)

        if left_hand is not None or right_hand is not None:
            detected_frames += 1
            features = extract_hand_features(left_hand, right_hand, prev_left, prev_right)
            if model_type in ["hybrid", "ml"]:
                frame_probs = ml_classifier.predict_probabilities(features)
            else:
                frame_probs = rule_classifier.predict_probabilities(features)
        else:
            frame_probs = {k: 0.01 for k in LABELS.values()}
            frame_probs["other"] = 0.93

        # Windowed winner-take-all probability integration
        label, conf, integrated_probs = accumulator.update(frame_probs, timestamp=simulated_time)
        state_machine.update(label, dt=dt)
        predictions[label] += 1

        prev_left, prev_right = left_hand, right_hand

        feedback_msg = FEEDBACK_ZH.get(label, "動作調整中")
        hands_status = {"left": left_hand is not None, "right": right_hand is not None}
        annotated = hud.draw_hud(
            frame=canvas,
            detected_label=label,
            confidence=conf,
            feedback_msg=feedback_msg,
            fps=fps,
            mode_str=f"{model_type.upper()} | Win: {window_sec}s",
            hands_status=hands_status,
        )

        if writer:
            writer.write(annotated)

        if show_window:
            cv2.imshow("Video Test - Wash Hand Detect", annotated)
            if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF == ord("q"):
                break

        if frame_idx % 60 == 0 or frame_idx == total_frames:
            target_max = min(total_frames, max_frames) if max_frames > 0 else total_frames
            print(f"   進度: {frame_idx}/{target_max} 幀 ({(frame_idx/target_max)*100:.1f}%)")

    cap.release()
    if writer:
        writer.release()
        print(f"[+] 已儲存標註影片至: {output_annotated_path}")

    if show_window:
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    summary = state_machine.get_progress_summary()

    print(f"\n📊 【評估成果統計】")
    print(f"   總處理幀數: {frame_idx} 幀 (耗時: {elapsed:.2f}s, 平均速度: {frame_idx/max(0.001, elapsed):.1f} FPS)")
    print(f"   雙手偵測率: {detected_frames}/{frame_idx} ({(detected_frames/max(1,frame_idx))*100:.1f}%)")
    print(f"\n🖐️ 【時序積分決策後的動作分布】")
    for act, cnt in predictions.most_common():
        zh = LABEL_NAMES_ZH.get(act, act)
        print(f"   - {zh} ({act}): {cnt} 幀 ({(cnt/max(1,frame_idx))*100:.1f}%)")

    print(f"\n🏆 【七步洗手完成狀態】")
    for step_name in ["inside", "outside", "interlace", "knuckles", "thumb", "fingertips", "wrist"]:
        time_spent = summary["step_times"].get(step_name, 0.0)
        is_done = step_name in summary["completed_steps"]
        status = f"✅ 已達標 ({time_spent:.1f}s)" if is_done else f"⏳ 累積時間: {time_spent:.1f}s / {step_duration}s"
        zh = LABEL_NAMES_ZH.get(step_name, step_name)
        print(f"   - {zh}: {status}")

    print(f"   完成數: {summary['completed_count']}/{summary['total_steps']}")
    print(f"   總體狀態: {'🎉 全部步驟完成！' if summary['is_completed'] else '進行中'}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate video with Wash Hand Detector")
    parser.add_argument("--video", type=str, default="data/raw/wash_7steps_yt.mp4", help="Path to input video")
    parser.add_argument("--output", type=str, default=None, help="Optional output annotated video path")
    parser.add_argument("--model-type", type=str, default="hybrid", choices=["hybrid", "ml", "rule"])
    parser.add_argument("--guide-mode", type=str, default="free", choices=["free", "sequence"])
    parser.add_argument("--step-duration", type=float, default=1.0)
    parser.add_argument("--window-sec", type=float, default=0.5)
    parser.add_argument("--roi", type=str, default="auto", choices=["auto", "right", "none"])
    parser.add_argument("--max-frames", type=int, default=1500)
    parser.add_argument("--show", action="store_true", help="Display visual playback window while testing")
    args = parser.parse_args()

    evaluate_video(
        video_path=args.video,
        output_annotated_path=args.output,
        model_type=args.model_type,
        guide_mode=args.guide_mode,
        step_duration=args.step_duration,
        window_sec=args.window_sec,
        max_frames=args.max_frames,
        roi_mode=args.roi,
        show_window=args.show,
    )
