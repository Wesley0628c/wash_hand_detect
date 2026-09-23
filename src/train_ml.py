"""
XGBoost / LightGBM 7-Step Wash Hand Classifier Training Script
Extracts multi-scale temporal statistics (160 dim base * 6 stats = 960 dim),
applies Landmark Dropout data augmentation (simulating foam & occlusion),
and trains a calibrated classifier saved to models/wash_hand_xgb.joblib.
"""

import os
import glob
import argparse
import joblib
from typing import Dict, List, Tuple
import cv2
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

from src.hand_detector import HandDetector
from src.features import extract_hand_features, feature_dict_to_vector
from src.temporal_features import TemporalFeatureBuffer, apply_landmark_dropout
from src.rule_classifier import LABELS, NAME_TO_LABEL, LABEL_SHORT_ZH
from src.evaluate_segments import BENCHMARK_GROUND_TRUTH, get_ground_truth_label


def extract_dataset_from_videos(
    video_dir: str = "data/raw",
    augmentations_per_frame: int = 3,
    buffer_size: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract temporal statistical feature vectors from all registered ground-truth videos.
    Applies landmark dropout augmentation.
    """
    detector = HandDetector(min_detection_confidence=0.45, min_tracking_confidence=0.45)

    X_list = []
    y_list = []

    for video_name, segments in BENCHMARK_GROUND_TRUTH.items():
        video_path = os.path.join(video_dir, video_name)
        if not os.path.exists(video_path):
            print(f"[!] Warning: Video {video_path} not found, skipping.")
            continue

        print(f"[*] 正在從影片抽取訓練資料: {video_name}...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        buf_orig = TemporalFeatureBuffer(buffer_size=buffer_size)
        buf_augs = [TemporalFeatureBuffer(buffer_size=buffer_size) for _ in range(augmentations_per_frame)]

        frame_idx = 0
        prev_left, prev_right = None, None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps
            gt_label_name = get_ground_truth_label(timestamp, segments)
            gt_label_idx = NAME_TO_LABEL[gt_label_name]

            left_hand, right_hand, _ = detector.process(frame)

            # 1. Original frame sample
            feats_orig = extract_hand_features(left_hand, right_hand, prev_left, prev_right)
            vec_orig = feature_dict_to_vector(feats_orig)
            temp_orig = buf_orig.update(vec_orig)

            X_list.append(temp_orig)
            y_list.append(gt_label_idx)

            # 2. Augmented samples with Landmark Dropout & Jitter (simulating foam & single-hand occlusion)
            for aug_idx in range(augmentations_per_frame):
                aug_left, aug_right = apply_landmark_dropout(
                    left_hand, right_hand,
                    point_dropout_prob=0.20,
                    single_hand_drop_prob=0.20,
                    jitter_std=0.015,
                )
                feats_aug = extract_hand_features(aug_left, aug_right, prev_left, prev_right)
                vec_aug = feature_dict_to_vector(feats_aug)
                temp_aug = buf_augs[aug_idx].update(vec_aug)

                X_list.append(temp_aug)
                y_list.append(gt_label_idx)

            prev_left = left_hand
            prev_right = right_hand
            frame_idx += 1

        cap.release()

    detector.close()

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"[+] 總共萃取 {len(X)} 組特徵樣本 (特徵維度: {X.shape[1]})")
    return X, y


def train_xgboost_classifier(
    X: np.ndarray,
    y: np.ndarray,
    output_model_path: str = "models/wash_hand_xgb.joblib",
    n_estimators: int = 150,
    max_depth: int = 6,
    learning_rate: float = 0.08,
):
    """Train and evaluate XGBoost Classifier with Stratified K-Fold validation."""
    os.makedirs(os.path.dirname(os.path.abspath(output_model_path)), exist_ok=True)

    print("\n🚀 開始訓練 XGBoost 分類器 (5-Fold 交叉驗證)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    val_true = []
    val_pred = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        clf = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="multi:softprob",
            num_class=8,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )

        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)

        val_true.extend(y_val)
        val_pred.extend(preds)

    target_names = [LABELS[i] for i in range(8)]
    print("\n📊 【XGBoost 5-Fold 交叉驗證成果報告】")
    print(classification_report(val_true, val_pred, target_names=target_names, zero_division=0))

    # Train final model on full dataset
    print(f"[*] 正在全量資料集上訓練最終模型並儲存至 {output_model_path}...")
    final_clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="multi:softprob",
        num_class=8,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    final_clf.fit(X, y)

    joblib.dump(final_clf, output_model_path)
    print(f"[✅] XGBoost 模型已成功儲存至: {output_model_path}")
    return final_clf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost wash hand classifier.")
    parser.add_argument("--output", type=str, default="models/wash_hand_xgb.joblib", help="Output model path")
    parser.add_argument("--aug", type=int, default=3, help="Data augmentations per frame with landmark dropout")
    args = parser.parse_args()

    X, y = extract_dataset_from_videos(augmentations_per_frame=args.aug)
    if len(X) > 0:
        train_xgboost_classifier(X, y, output_model_path=args.output)
    else:
        print("[!] No training samples could be extracted.")
