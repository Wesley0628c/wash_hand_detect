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

        # 1. Dual-Hand Geometric Evidence with Negative Evidence & Mutual Exclusion
        if both_hands:
            inter = features.get("inter_hand", {})
            palm_dot = float(features.get("palm_normal_dot", 0.0))
            palm_dist = float(inter.get("palm_center_dist", 99.0))
            min_p_to_w = float(inter.get("min_palm_to_wrist", 99.0))
            wrist_ratio = float(inter.get("wrist_ratio", 99.0))
            min_t_to_p = float(inter.get("min_tips_to_palm", 99.0))
            min_knuckles_to_p = float(inter.get("min_knuckles_to_palm", 99.0))
            min_p_to_th = float(inter.get("min_palm_to_thumb", 99.0))
            interlace_depth = float(inter.get("interlace_depth", 99.0))
            fingertip_spread = float(inter.get("min_fingertip_spread", 99.0))

            left_angles = features.get("left_angles", [0.0]*5)
            right_angles = features.get("right_angles", [0.0]*5)
            mean_curl = float(np.mean(left_angles[1:]) + np.mean(right_angles[1:])) / 2.0

            # 1. 腕 (Wrist): Grasping opposite wrist
            if min_p_to_w < 1.15 and wrist_ratio < 0.80:
                wrist_bonus = 4.5 * max(0.0, 1.0 - wrist_ratio / 0.80)
                scores["wrist"] += wrist_bonus
            # Negative evidence for Wrist: hands centered on palms/dorsum
            if wrist_ratio > 0.88 or palm_dist < 0.55:
                scores["wrist"] -= 3.0

            # 2. 內 (Inside): Direct palm-to-palm facing, flat fingers
            if palm_dist < 1.4 and mean_curl > 148.0 and min_p_to_w > 0.65 and palm_dot < 0.15:
                # Direct palm-to-palm alignment
                scores["inside"] += 3.8
            if mean_curl < 135.0 or wrist_ratio < 0.65:
                scores["inside"] -= 2.5

            # 3. 外 (Outside): Palm on back of opposite hand, fingers extended
            if palm_dist < 1.6 and mean_curl > 148.0 and wrist_ratio > 0.72 and min_p_to_w > 0.60:
                scores["outside"] += 3.5
            # Negative evidence for Outside: curled fingers or wrist grasp
            if mean_curl < 138.0:
                scores["outside"] -= 3.5
            if wrist_ratio < 0.65:
                scores["outside"] -= 3.0

            # 4. 弓 (Knuckles): PIP/DIP knuckles rubbing palm, fingers curled (< 152 deg)
            if mean_curl < 152.0 and min_knuckles_to_p < 1.15 and palm_dist < 1.65:
                curl_strength = max(0.0, (152.0 - mean_curl) / 30.0)
                scores["knuckles"] += 4.2 * (0.5 + 0.5 * curl_strength)
            # Negative evidence for Knuckles: straight fingers
            if mean_curl > 162.0:
                scores["knuckles"] -= 4.0
            if wrist_ratio < 0.65:
                scores["knuckles"] -= 2.5

            # 5. 夾 (Interlace): Symmetric finger interleaving
            if interlace_depth < 1.15 and palm_dist < 1.55 and mean_curl > 135.0:
                scores["interlace"] += 3.8 * max(0.0, 1.0 - interlace_depth / 1.15)
            if interlace_depth > 1.35:
                scores["interlace"] -= 2.0

            # 6. 大 (Thumb): Grasping opposite thumb
            if min_p_to_th < 1.10 and wrist_ratio > 0.68 and min_p_to_w > 0.60:
                scores["thumb"] += 3.8 * max(0.0, 1.0 - min_p_to_th / 1.10)
            if min_p_to_th > 1.40:
                scores["thumb"] -= 2.5

            # 7. 立 (Fingertips): Bundled fingertips scrubbing palm
            if min_t_to_p < 1.05 and fingertip_spread < 0.42 and min_p_to_w > 0.60:
                scores["fingertips"] += 4.2 * max(0.0, 1.0 - min_t_to_p / 1.05)
            # Negative evidence for Fingertips: spread fingers or near wrist
            if fingertip_spread > 0.48 or min_p_to_w < 0.55:
                scores["fingertips"] -= 3.5

        # 2. Single-Hand / Merged Cluster Morphology Evidence (for foam/occlusion fallback)
        active_angles = features.get("active_angles", [0.0]*5)
        active_spread = float(features.get("active_spread", 0.0))
        mean_4_angle = float(np.mean(active_angles[1:])) if len(active_angles) >= 5 else 180.0
        thumb_angle = float(active_angles[0]) if len(active_angles) >= 1 else 180.0

        if not both_hands:
            if mean_4_angle < 135.0 and thumb_angle > 135.0:
                scores["thumb"] += 3.0
            elif active_spread < 0.30 or (mean_4_angle < 125.0 and active_spread < 0.35):
                scores["fingertips"] += 3.0
            elif mean_4_angle < 152.0:
                scores["knuckles"] += 3.2
            elif active_spread > 0.38 and mean_4_angle < 170.0:
                scores["interlace"] += 2.8
            elif thumb_angle < 140.0 and mean_4_angle > 145.0:
                scores["outside"] += 2.7
            elif mean_4_angle > 162.0:
                scores["inside"] += 2.3

        # Softmax normalization with stability clipping
        score_arr = np.array(list(scores.values()), dtype=np.float32)
        score_arr = np.clip(score_arr, -10.0, 15.0)
        exp_scores = np.exp(score_arr - np.max(score_arr))
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
