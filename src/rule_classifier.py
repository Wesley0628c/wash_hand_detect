"""
Rule-Based Wash Hand Classifier Module
Implements domain-rule gesture classification for the 7 steps + 1 other class,
along with multi-class probability scoring and real-time posture feedback generation.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np

# Label Constants
LABELS = {
    0: "other",
    1: "inside",
    2: "outside",
    3: "interlace",
    4: "knuckles",
    5: "thumb",
    6: "fingertips",
    7: "wrist",
}

LABEL_NAMES_ZH = {
    "other": "其他 (尚未開始 / 非標準動作)",
    "inside": "內 (掌心對掌心搓洗)",
    "outside": "外 (掌心搓洗手背)",
    "interlace": "夾 (十指交錯搓洗)",
    "knuckles": "弓 (指背搓洗掌心)",
    "thumb": "大 (旋轉搓洗大拇指)",
    "fingertips": "立 (指尖搓洗掌心)",
    "wrist": "腕 (旋轉搓洗手腕)",
}

LABEL_SHORT_ZH = {
    "other": "其他",
    "inside": "內",
    "outside": "外",
    "interlace": "夾",
    "knuckles": "弓",
    "thumb": "大",
    "fingertips": "立",
    "wrist": "腕",
}

FEEDBACK_ZH = {
    "other": "未偵測到雙手或姿勢非洗手動作",
    "inside": "姿勢正確：掌心對掌心搓洗",
    "outside": "姿勢正確：掌心搓洗手背",
    "interlace": "姿勢正確：十指交錯搓洗",
    "knuckles": "姿勢正確：指背搓洗掌心",
    "thumb": "姿勢正確：旋轉搓洗大拇指",
    "fingertips": "姿勢正確：指尖搓洗掌心",
    "wrist": "姿勢正確：正在旋轉搓洗手腕",
}

NAME_TO_LABEL = {v: k for k, v in LABELS.items()}


class WashHandRuleClassifier:
    """Rule-based evaluator for 7-step hand washing gestures with multi-class probability distribution."""

    def __init__(self, min_motion_velocity: float = 0.005):
        self.min_motion_velocity = min_motion_velocity

    def predict_probabilities(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate continuous probability distribution across all 8 classes.
        Returns a dictionary mapping class names to probabilities summing to 1.0.
        """
        has_left = features.get("has_left", False)
        has_right = features.get("has_right", False)
        both_hands = features.get("both_hands_detected", False)

        if not (has_left or has_right):
            probs = {k: 0.01 for k in LABELS.values()}
            probs["other"] = 0.93
            return probs

        # Logit evidence scores
        scores = {k: 0.1 for k in LABELS.values()}

        # 1. Dual-Hand Geometric Evidence (when 2 hands are tracked)
        if both_hands:
            inter = features.get("inter_hand", {})
            palm_dot = float(features.get("palm_normal_dot", 0.0))
            palm_dist = float(inter.get("palm_center_dist", 99.0))
            min_p_to_w = float(inter.get("min_palm_to_wrist", 99.0))
            min_t_to_p = float(inter.get("min_tips_to_palm", 99.0))
            min_p_to_th = float(inter.get("min_palm_to_thumb", 99.0))
            interlace_depth = float(inter.get("interlace_depth", 99.0))

            left_angles = features.get("left_angles", [0.0]*5)
            right_angles = features.get("right_angles", [0.0]*5)
            mean_curl = float(np.mean(left_angles[1:]) + np.mean(right_angles[1:])) / 2.0

            # 1. 腕 (Wrist): palm is close to the other hand's wrist
            if min_p_to_w < 1.3:
                wrist_bonus = 4.5 * max(0.0, 1.0 - min_p_to_w / 1.3)
                if min_p_to_w <= min_t_to_p + 0.1:
                    wrist_bonus += 1.5
                scores["wrist"] += wrist_bonus

            # 2. 立 (Fingertips): fingertips close to palm, but palm NOT at wrist
            if min_t_to_p < 1.1 and min_p_to_w > 0.6:
                scores["fingertips"] += 3.5 * max(0.0, 1.0 - min_t_to_p / 1.1)

            # 3. 大 (Thumb): palm grasping thumb
            if min_p_to_th < 1.2:
                scores["thumb"] += 3.2 * max(0.0, 1.0 - min_p_to_th / 1.2)

            # 4. 弓 (Knuckles): fingers curled/hooked
            if mean_curl < 166.0 and palm_dist < 1.8:
                scores["knuckles"] += 3.8 * max(0.0, (166.0 - mean_curl) / 35.0)

            # 5. 夾 (Interlace): finger bases interlaced
            if interlace_depth < 1.2 and palm_dist < 1.6:
                scores["interlace"] += 3.2 * max(0.0, 1.0 - interlace_depth / 1.2)

            # 6. 外 (Outside): one hand on top of another
            if palm_dist < 1.6 and mean_curl > 140.0 and min_p_to_w > 0.6:
                scores["outside"] += 2.8

            # 7. 內 (Inside): opposing palms ONLY when fingers are flat/extended
            if palm_dot < -0.2 and palm_dist < 1.4 and min_p_to_w > 0.8 and min_t_to_p > 0.6 and mean_curl > 165.0:
                scores["inside"] += 2.8

        # 2. Single-Hand / Merged Cluster Morphology Evidence (for foam/occlusion)
        active_angles = features.get("active_angles", [0.0]*5)
        active_spread = float(features.get("active_spread", 0.0))
        mean_4_angle = float(np.mean(active_angles[1:])) if len(active_angles) >= 5 else 180.0
        thumb_angle = float(active_angles[0]) if len(active_angles) >= 1 else 180.0

        if not both_hands:
            if mean_4_angle < 135.0 and thumb_angle > 135.0:
                scores["thumb"] += 3.0
            elif active_spread < 0.30 or (mean_4_angle < 125.0 and active_spread < 0.35):
                scores["fingertips"] += 3.0
            elif mean_4_angle < 158.0:
                scores["knuckles"] += 3.2
            elif active_spread > 0.38 and mean_4_angle < 170.0:
                scores["interlace"] += 2.8
            elif thumb_angle < 140.0 and mean_4_angle > 145.0:
                scores["outside"] += 2.7
            elif mean_4_angle > 162.0:
                scores["inside"] += 2.3

        # Softmax normalization
        exp_scores = np.exp(np.array(list(scores.values()), dtype=np.float32))
        sum_exp = float(np.sum(exp_scores))
        norm_probs = exp_scores / max(1e-6, sum_exp)

        return {act: float(prob) for act, prob in zip(scores.keys(), norm_probs)}

    def predict(self, features: Dict[str, Any]) -> Tuple[str, float, str]:
        """
        Evaluate extracted features and return:
          (label_name, confidence, feedback_message)
        """
        probs = self.predict_probabilities(features)
        top_action = max(probs, key=probs.get)
        top_conf = probs[top_action]
        feedback = FEEDBACK_ZH.get(top_action, "動作調整中，請依步驟搓洗")
        return top_action, top_conf, feedback
