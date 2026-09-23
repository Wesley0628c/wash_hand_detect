"""
Hand Landmark Feature Extraction & Normalization Module
Implements geometric, angular, spatial relationship, and temporal velocity features
as outlined in IMPLEMENTATION.md.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np

# Landmark Indices according to MediaPipe Hand Model
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

PALM_IDS = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
TIP_IDS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
TIP_WITHOUT_THUMB = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance between two 3D points."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calculate angle in degrees at vertex b formed by (a, b, c).
    ba = a - b, bc = c - b
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc + 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def normalize_hand_landmarks(landmarks: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Normalize 21x3 landmarks:
    1. Translate Wrist (point 0) to origin (0, 0, 0).
    2. Scale relative coordinates by the distance between Wrist(0) and Middle MCP(9).
    Returns (normalized_21x3, scale).
    """
    if landmarks is None or len(landmarks) != 21:
        return np.zeros((21, 3), dtype=np.float32), 1.0

    wrist = landmarks[0]
    translated = landmarks - wrist

    scale = euclidean_distance(landmarks[0], landmarks[MIDDLE_MCP])
    if scale < 1e-5:
        scale = 1.0

    normalized = translated / scale
    return normalized.astype(np.float32), float(scale)


def calculate_palm_center(hand_landmarks: np.ndarray) -> np.ndarray:
    """Calculate the 3D center of the palm using palm anchor landmarks."""
    if hand_landmarks is None or len(hand_landmarks) < 21:
        return np.zeros(3, dtype=np.float32)
    points = np.array([hand_landmarks[i] for i in PALM_IDS], dtype=np.float32)
    return points.mean(axis=0)


def calculate_palm_normal(hand_landmarks: np.ndarray, is_left: bool = True) -> np.ndarray:
    """
    Estimate normal vector of the palm surface using Wrist, Index MCP, and Pinky MCP.
    Returns unit vector (3,).
    """
    if hand_landmarks is None or len(hand_landmarks) < 21:
        return np.zeros(3, dtype=np.float32)

    wrist = np.array(hand_landmarks[WRIST], dtype=np.float32)
    index = np.array(hand_landmarks[INDEX_MCP], dtype=np.float32)
    pinky = np.array(hand_landmarks[PINKY_MCP], dtype=np.float32)

    v1 = index - wrist
    v2 = pinky - wrist

    # For Left Hand, cross(v1, v2) points outward from palm;
    # For Right Hand, cross(v2, v1) points outward from palm.
    if is_left:
        normal = np.cross(v1, v2)
    else:
        normal = np.cross(v2, v1)

    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        return np.zeros(3, dtype=np.float32)
    return (normal / norm).astype(np.float32)


def calculate_fingertip_center(hand_landmarks: np.ndarray, include_thumb: bool = False) -> np.ndarray:
    """Calculate mean position of fingertips (for 'fingertips' gesture detection)."""
    if hand_landmarks is None or len(hand_landmarks) < 21:
        return np.zeros(3, dtype=np.float32)

    target_ids = TIP_IDS if include_thumb else TIP_WITHOUT_THUMB
    points = np.array([hand_landmarks[i] for i in target_ids], dtype=np.float32)
    return points.mean(axis=0)


def calculate_finger_bending_angles(hand_landmarks: np.ndarray) -> List[float]:
    """
    Calculate the joint bending angles (in degrees) for all 5 fingers.
    Returns list of 5 angles (thumb, index, middle, ring, pinky).
    """
    if hand_landmarks is None or len(hand_landmarks) < 21:
        return [0.0] * 5

    # Finger joint triplets (MCP, PIP, DIP)
    finger_triplets = [
        (THUMB_CMC, THUMB_MCP, THUMB_IP),
        (INDEX_MCP, INDEX_PIP, INDEX_DIP),
        (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP),
        (RING_MCP, RING_PIP, RING_DIP),
        (PINKY_MCP, PINKY_PIP, PINKY_DIP),
    ]
    return [
        calculate_angle(hand_landmarks[p1], hand_landmarks[p2], hand_landmarks[p3])
        for p1, p2, p3 in finger_triplets
    ]


def extract_hand_features(
    left_hand: Optional[np.ndarray],
    right_hand: Optional[np.ndarray],
    prev_left_hand: Optional[np.ndarray] = None,
    prev_right_hand: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Extract comprehensive structured feature dictionary from raw (21, 3) left & right landmarks.
    Handles presence/absence of hands gracefully.
    """
    has_left = bool(left_hand is not None and np.any(left_hand != 0))
    has_right = bool(right_hand is not None and np.any(right_hand != 0))

    # 1. Coordinate Normalization
    left_norm, left_scale = normalize_hand_landmarks(left_hand) if has_left else (np.zeros((21, 3), dtype=np.float32), 1.0)
    right_norm, right_scale = normalize_hand_landmarks(right_hand) if has_right else (np.zeros((21, 3), dtype=np.float32), 1.0)

    # 2. Palm Centers & Normals
    left_palm_center = calculate_palm_center(left_hand) if has_left else np.zeros(3, dtype=np.float32)
    right_palm_center = calculate_palm_center(right_hand) if has_right else np.zeros(3, dtype=np.float32)

    left_normal = calculate_palm_normal(left_hand, is_left=True) if has_left else np.zeros(3, dtype=np.float32)
    right_normal = calculate_palm_normal(right_hand, is_left=False) if has_right else np.zeros(3, dtype=np.float32)

    # Palm normal dot product:
    # If normals point towards each other (palms facing): dot is negative
    # If palms face the same direction (e.g. one palm on back of other): dot is positive
    palm_normal_dot = float(np.dot(left_normal, right_normal)) if (has_left and has_right) else 0.0

    # 3. Finger Bending Angles
    left_angles = calculate_finger_bending_angles(left_hand) if has_left else [0.0] * 5
    right_angles = calculate_finger_bending_angles(right_hand) if has_right else [0.0] * 5

    # 4. Fingertip Centers
    left_tip_center = calculate_fingertip_center(left_hand, include_thumb=False) if has_left else np.zeros(3, dtype=np.float32)
    right_tip_center = calculate_fingertip_center(right_hand, include_thumb=False) if has_right else np.zeros(3, dtype=np.float32)

    # 5. Inter-Hand Distance Metrics
    inter_hand_features: Dict[str, float] = {}
    if has_left and has_right:
        # Reference scale to normalize inter-hand distances
        avg_scale = max((left_scale + right_scale) / 2.0, 1e-4)

        wrist_dist = euclidean_distance(left_hand[WRIST], right_hand[WRIST]) / avg_scale
        palm_center_dist = euclidean_distance(left_palm_center, right_palm_center) / avg_scale

        # Tip to opposite palm center distances
        left_tips_to_right_palm = euclidean_distance(left_tip_center, right_palm_center) / avg_scale
        right_tips_to_left_palm = euclidean_distance(right_tip_center, left_palm_center) / avg_scale

        # Palm to opposite wrist distances
        left_palm_to_right_wrist = euclidean_distance(left_palm_center, right_hand[WRIST]) / avg_scale
        right_palm_to_left_wrist = euclidean_distance(right_palm_center, left_hand[WRIST]) / avg_scale
        min_p_to_w = min(left_palm_to_right_wrist, right_palm_to_left_wrist)
        wrist_ratio = min_p_to_w / (palm_center_dist + 1e-6)

        # Knuckle (PIP joints) to opposite palm center distances (for Knuckles / 弓)
        pip_ids = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
        left_knuckles_to_right_palm = float(np.mean([
            euclidean_distance(left_hand[pid], right_palm_center) for pid in pip_ids
        ])) / avg_scale
        right_knuckles_to_left_palm = float(np.mean([
            euclidean_distance(right_hand[pid], left_palm_center) for pid in pip_ids
        ])) / avg_scale
        min_knuckles_to_palm = min(left_knuckles_to_right_palm, right_knuckles_to_left_palm)

        # Palm to opposite thumb distances
        left_palm_to_right_thumb = euclidean_distance(left_palm_center, right_hand[THUMB_TIP]) / avg_scale
        right_palm_to_left_thumb = euclidean_distance(right_palm_center, left_hand[THUMB_TIP]) / avg_scale

        # Fingertip pair distances
        tip_distances = [
            euclidean_distance(left_hand[tid], right_hand[tid]) / avg_scale for tid in TIP_IDS
        ]

        # Symmetric Finger overlap / interlace metric (for Interlace / 夾)
        left_to_right_interlace = float(np.mean([
            euclidean_distance(left_hand[INDEX_TIP], right_hand[INDEX_MCP]),
            euclidean_distance(left_hand[MIDDLE_TIP], right_hand[MIDDLE_MCP]),
            euclidean_distance(left_hand[RING_TIP], right_hand[RING_MCP]),
            euclidean_distance(left_hand[PINKY_TIP], right_hand[PINKY_MCP]),
        ])) / avg_scale

        right_to_left_interlace = float(np.mean([
            euclidean_distance(right_hand[INDEX_TIP], left_hand[INDEX_MCP]),
            euclidean_distance(right_hand[MIDDLE_TIP], left_hand[MIDDLE_MCP]),
            euclidean_distance(right_hand[RING_TIP], left_hand[RING_MCP]),
            euclidean_distance(right_hand[PINKY_TIP], left_hand[PINKY_MCP]),
        ])) / avg_scale

        interlace_depth = min(left_to_right_interlace, right_to_left_interlace)

        # Fingertip spread (cluster compactness for Fingertips / 立)
        left_spread = float(np.mean([
            euclidean_distance(left_hand[INDEX_TIP], left_hand[MIDDLE_TIP]),
            euclidean_distance(left_hand[MIDDLE_TIP], left_hand[RING_TIP]),
            euclidean_distance(left_hand[RING_TIP], left_hand[PINKY_TIP]),
        ])) / avg_scale
        right_spread = float(np.mean([
            euclidean_distance(right_hand[INDEX_TIP], right_hand[MIDDLE_TIP]),
            euclidean_distance(right_hand[MIDDLE_TIP], right_hand[RING_TIP]),
            euclidean_distance(right_hand[RING_TIP], right_hand[PINKY_TIP]),
        ])) / avg_scale
        min_fingertip_spread = min(left_spread, right_spread)

        inter_hand_features = {
            "wrist_dist": wrist_dist,
            "palm_center_dist": palm_center_dist,
            "left_tips_to_right_palm": left_tips_to_right_palm,
            "right_tips_to_left_palm": right_tips_to_left_palm,
            "min_tips_to_palm": min(left_tips_to_right_palm, right_tips_to_left_palm),
            "left_palm_to_right_wrist": left_palm_to_right_wrist,
            "right_palm_to_left_wrist": right_palm_to_left_wrist,
            "min_palm_to_wrist": min_p_to_w,
            "wrist_ratio": wrist_ratio,
            "left_knuckles_to_right_palm": left_knuckles_to_right_palm,
            "right_knuckles_to_left_palm": right_knuckles_to_left_palm,
            "min_knuckles_to_palm": min_knuckles_to_palm,
            "left_palm_to_right_thumb": left_palm_to_right_thumb,
            "right_palm_to_left_thumb": right_palm_to_left_thumb,
            "min_palm_to_thumb": min(left_palm_to_right_thumb, right_palm_to_left_thumb),
            "mean_tip_dist": float(np.mean(tip_distances)),
            "interlace_depth": float(interlace_depth),
            "min_fingertip_spread": min_fingertip_spread,
        }
    else:
        inter_hand_features = {
            "wrist_dist": 99.0,
            "palm_center_dist": 99.0,
            "left_tips_to_right_palm": 99.0,
            "right_tips_to_left_palm": 99.0,
            "min_tips_to_palm": 99.0,
            "left_palm_to_right_wrist": 99.0,
            "right_palm_to_left_wrist": 99.0,
            "min_palm_to_wrist": 99.0,
            "wrist_ratio": 99.0,
            "left_knuckles_to_right_palm": 99.0,
            "right_knuckles_to_left_palm": 99.0,
            "min_knuckles_to_palm": 99.0,
            "left_palm_to_right_thumb": 99.0,
            "right_palm_to_left_thumb": 99.0,
            "min_palm_to_thumb": 99.0,
            "mean_tip_dist": 99.0,
            "interlace_depth": 99.0,
            "min_fingertip_spread": 99.0,
        }

    # 6. Velocity / Motion Features
    left_vel = 0.0
    right_vel = 0.0
    if has_left and prev_left_hand is not None and np.any(prev_left_hand != 0):
        left_vel = float(np.mean(np.linalg.norm(left_norm - normalize_hand_landmarks(prev_left_hand)[0], axis=1)))
    if has_right and prev_right_hand is not None and np.any(prev_right_hand != 0):
        right_vel = float(np.mean(np.linalg.norm(right_norm - normalize_hand_landmarks(prev_right_hand)[0], axis=1)))
    avg_velocity = (left_vel + right_vel) / 2.0 if (has_left and has_right) else (left_vel or right_vel)

    # 7. Single Hand / Merged Cluster Morphology (for when hands overlap under soap/occlusion)
    active_hand = left_norm if has_left else (right_norm if has_right else None)
    active_angles = left_angles if has_left else (right_angles if has_right else [0.0] * 5)
    active_spread = 0.0
    if active_hand is not None:
        active_spread = float(np.mean([np.linalg.norm(active_hand[TIP_IDS[i]] - active_hand[TIP_IDS[i+1]]) for i in range(4)]))

    return {
        "has_left": has_left,
        "has_right": has_right,
        "both_hands_detected": has_left and has_right,
        "left_norm": left_norm,
        "right_norm": right_norm,
        "left_palm_center": left_palm_center,
        "right_palm_center": right_palm_center,
        "left_normal": left_normal,
        "right_normal": right_normal,
        "palm_normal_dot": palm_normal_dot,
        "left_angles": left_angles,
        "right_angles": right_angles,
        "active_angles": active_angles,
        "active_spread": active_spread,
        "inter_hand": inter_hand_features,
        "velocity": avg_velocity,
    }


def feature_dict_to_vector(features: Dict[str, Any]) -> np.ndarray:
    """
    Flattens the structured feature dictionary into a 1D NumPy vector for ML / LSTM models.
    Dimension:
      Left normalized landmarks: 21 * 3 = 63
      Right normalized landmarks: 21 * 3 = 63
      Left Palm Normal: 3
      Right Palm Normal: 3
      Palm Normal Dot: 1
      Left Angles: 5
      Right Angles: 5
      Inter-hand geometric metrics: 16
      Velocity: 1
    Total Feature Vector Dimension: 63 + 63 + 3 + 3 + 1 + 5 + 5 + 16 + 1 = 160
    """
    left_flat = features["left_norm"].flatten() if features.get("left_norm") is not None else np.zeros(63, dtype=np.float32)
    right_flat = features["right_norm"].flatten() if features.get("right_norm") is not None else np.zeros(63, dtype=np.float32)

    vec = [
        left_flat,
        right_flat,
        features.get("left_normal", np.zeros(3, dtype=np.float32)),
        features.get("right_normal", np.zeros(3, dtype=np.float32)),
        np.array([features.get("palm_normal_dot", 0.0)], dtype=np.float32),
        np.array(features.get("left_angles", [0.0] * 5), dtype=np.float32),
        np.array(features.get("right_angles", [0.0] * 5), dtype=np.float32),
        np.array([
            features["inter_hand"].get("wrist_dist", 99.0),
            features["inter_hand"].get("palm_center_dist", 99.0),
            features["inter_hand"].get("left_tips_to_right_palm", 99.0),
            features["inter_hand"].get("right_tips_to_left_palm", 99.0),
            features["inter_hand"].get("min_tips_to_palm", 99.0),
            features["inter_hand"].get("left_palm_to_right_wrist", 99.0),
            features["inter_hand"].get("right_palm_to_left_wrist", 99.0),
            features["inter_hand"].get("min_palm_to_wrist", 99.0),
            features["inter_hand"].get("wrist_ratio", 99.0),
            features["inter_hand"].get("min_knuckles_to_palm", 99.0),
            features["inter_hand"].get("left_palm_to_right_thumb", 99.0),
            features["inter_hand"].get("right_palm_to_left_thumb", 99.0),
            features["inter_hand"].get("min_palm_to_thumb", 99.0),
            features["inter_hand"].get("mean_tip_dist", 99.0),
            features["inter_hand"].get("interlace_depth", 99.0),
            features["inter_hand"].get("min_fingertip_spread", 99.0),
        ], dtype=np.float32),
        np.array([features.get("velocity", 0.0)], dtype=np.float32),
    ]

    return np.concatenate(vec).astype(np.float32)
