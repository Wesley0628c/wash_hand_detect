"""
Unit & Integration Test Suite for Wash Hand Detect Pipeline
Tests features, rule classifier, state machine, LSTM model, and UI rendering.
"""

import pytest
import numpy as np
import cv2

from src.features import (
    normalize_hand_landmarks,
    calculate_palm_center,
    calculate_palm_normal,
    calculate_finger_bending_angles,
    calculate_fingertip_center,
    extract_hand_features,
    feature_dict_to_vector,
)
from src.rule_classifier import WashHandRuleClassifier, LABELS
from src.state_machine import WashHandStateMachine, STEPS_ORDER
from src.train import build_lstm_model, generate_synthetic_dataset
from src.ui import WashHandHUD


def _create_dummy_hand(offset=(0.0, 0.0, 0.0), scale=1.0) -> np.ndarray:
    """Create a synthetic 21x3 hand landmark array."""
    hand = np.zeros((21, 3), dtype=np.float32)
    # Wrist
    hand[0] = [0.0, 0.0, 0.0]
    # Thumb 1-4
    for i in range(1, 5):
        hand[i] = [-0.2 * i * scale, 0.1 * i * scale, 0.0]
    # Index 5-8
    for i in range(5, 9):
        hand[i] = [-0.1 * scale, 0.2 * (i - 4) * scale, 0.0]
    # Middle 9-12 (Middle MCP is at 9)
    for i in range(9, 13):
        hand[i] = [0.0, 0.25 * (i - 8) * scale, 0.0]
    # Ring 13-16
    for i in range(13, 17):
        hand[i] = [0.1 * scale, 0.2 * (i - 12) * scale, 0.0]
    # Pinky 17-20
    for i in range(17, 21):
        hand[i] = [0.2 * scale, 0.15 * (i - 16) * scale, 0.0]

    return hand + np.array(offset, dtype=np.float32)


def test_landmark_normalization():
    raw_hand = _create_dummy_hand(offset=(100.0, 200.0, 50.0), scale=2.0)
    norm_hand, scale = normalize_hand_landmarks(raw_hand)

    # Wrist at point 0 must be (0, 0, 0)
    np.testing.assert_allclose(norm_hand[0], [0.0, 0.0, 0.0], atol=1e-5)
    # Scale should be non-zero
    assert scale > 0.0
    assert norm_hand.shape == (21, 3)


def test_palm_center_and_normal():
    hand = _create_dummy_hand()
    center = calculate_palm_center(hand)
    assert center.shape == (3,)

    normal = calculate_palm_normal(hand, is_left=True)
    assert normal.shape == (3,)
    # Unit vector magnitude should be 1.0
    norm_val = np.linalg.norm(normal)
    assert np.isclose(norm_val, 1.0, atol=1e-4) or norm_val == 0.0


def test_feature_vector_dimension():
    left = _create_dummy_hand(offset=(-0.1, 0, 0))
    right = _create_dummy_hand(offset=(0.1, 0, 0))

    features = extract_hand_features(left, right)
    assert features["both_hands_detected"] is True

    vec = feature_dict_to_vector(features)
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert len(vec) == 157  # Expected 157-dim vector


def test_rule_classifier():
    classifier = WashHandRuleClassifier()

    # Case 1: No hands
    feats_empty = extract_hand_features(None, None)
    label, conf, msg = classifier.predict(feats_empty)
    assert label == "other"

    # Case 2: Palms facing (Inside)
    left = _create_dummy_hand(offset=(-0.05, 0, 0))
    right = _create_dummy_hand(offset=(0.05, 0, 0))
    feats_inside = extract_hand_features(left, right)
    label, conf, msg = classifier.predict(feats_inside)
    assert label in LABELS.values()
    assert conf > 0.5
    assert len(msg) > 0


def test_state_machine_sequential_flow():
    sm = WashHandStateMachine(mode="sequence", step_duration=1.0)
    assert sm.mode == "sequence"
    assert sm.current_step_idx == 0
    assert sm.is_completed is False

    # Simulate progressing through all 7 steps
    dt = 0.2
    for step in STEPS_ORDER:
        assert sm._get_target_step() == step
        # Accumulate time until step completes
        for _ in range(6):  # 6 * 0.2 = 1.2s >= 1.0s
            just_completed, completed_name = sm.update(step, dt)
            if just_completed:
                assert completed_name == step
                break

    assert sm.is_completed is True
    summary = sm.get_progress_summary()
    assert summary["completed_count"] == 7
    assert summary["is_completed"] is True


def test_state_machine_free_mode():
    sm = WashHandStateMachine(mode="free", step_duration=0.5)

    # Complete steps in reverse order
    dt = 0.1
    for step in reversed(STEPS_ORDER):
        for _ in range(6):
            just_completed, completed_name = sm.update(step, dt)
            if just_completed:
                assert completed_name == step
                break

    assert sm.is_completed is True
    assert len(sm.completed_steps) == 7


def test_lstm_model_building_and_synthetic_training():
    model = build_lstm_model(seq_len=30, feature_dim=157, num_classes=8)
    assert model.input_shape == (None, 30, 157)
    assert model.output_shape == (None, 8)

    X_syn, y_syn, groups = generate_synthetic_dataset(num_persons=3, clips_per_action=2)
    assert X_syn.shape == (3 * 8 * 2, 30, 157)
    assert len(y_syn) == len(X_syn)

    # Test forward pass
    preds = model.predict(X_syn[:4], verbose=0)
    assert preds.shape == (4, 8)
    assert np.allclose(preds.sum(axis=1), 1.0, atol=1e-4)


def test_ui_hud_rendering():
    hud = WashHandHUD()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    sm = WashHandStateMachine()
    summary = sm.get_progress_summary()

    output = hud.draw_hud(
        frame=frame,
        detected_label="inside",
        confidence=0.92,
        feedback_msg="姿勢正確：掌心對掌心搓洗",
        progress_summary=summary,
        fps=30.0,
        mode_str="Test Mode",
    )

    assert output.shape == (720, 1280, 3)
    assert output.dtype == np.uint8
