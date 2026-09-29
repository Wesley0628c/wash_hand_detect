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
        scores = {k: 0.0 for k in LABELS.values()}

        # 1. Dual-Hand Geometric Evidence
        if both_hands:
            inter = features.get("inter_hand", {})
            palm_dot = float(features.get("palm_normal_dot", 0.0))
            palm_dist = float(inter.get("palm_center_dist", 99.0))
            min_p_to_w = float(inter.get("min_palm_to_wrist", 99.0))
            wrist_ratio = float(inter.get("wrist_ratio", 99.0))
            min_t_to_p = float(inter.get("min_tips_to_palm", 99.0))
            min_knuckles_to_p = float(inter.get("min_knuckles_to_palm", 99.0))
            min_p_to_th = float(inter.get("min_palm_to_thumb", 99.0))
            min_web_to_th = float(inter.get("min_web_to_thumb", 99.0))
            interlace_depth = float(inter.get("interlace_depth", 99.0))
            fingertip_spread = float(inter.get("min_fingertip_spread", 99.0))
            tip_dist = float(inter.get("mean_tip_dist", 99.0))

            left_angles = features.get("left_angles", [0.0]*5)
            right_angles = features.get("right_angles", [0.0]*5)
            left_curl = float(np.mean(left_angles[1:])) if has_left else 180.0
            right_curl = float(np.mean(right_angles[1:])) if has_right else 180.0
            min_curl = min(left_curl, right_curl)
            max_curl = max(left_curl, right_curl)
            curl_diff = abs(left_curl - right_curl)

            # [Level 1] 腕 (Wrist): Grasping opposite wrist
            if wrist_ratio < 0.65 and (min_p_to_w < 1.0 or palm_dist > 0.85):
                scores["wrist"] = 10.0

            # [Level 2] 立 (Fingertips): Bundled fingertips upright into opposite palm
            elif palm_dist > 1.05 and min_t_to_p < 1.15 and min_curl > 115.0 and (palm_dist > min_t_to_p + 0.15 or tip_dist > 1.10):
                scores["fingertips"] = 10.0

            # [Level 3] 大 (Thumb): One hand curled around opposite thumb
            elif (min_curl < 115.0 or curl_diff > 25.0) and (min_p_to_th < 1.05 or min_web_to_th < 1.10) and min(min_p_to_th, min_web_to_th) <= min_knuckles_to_p + 0.15:
                scores["thumb"] = 10.0

            # [Level 4] 弓 (Knuckles): Curled fist PIP knuckles rubbing palm
            elif (min_curl < 135.0 or curl_diff > 20.0) and min_knuckles_to_p < 0.95 and palm_dist < 1.50:
                scores["knuckles"] = 10.0

            # [Level 5] 外 (Outside): Palm on back of opposite hand (dorsum)
            elif (palm_dot > -0.25 or (interlace_depth > 0.88 and fingertip_spread < 0.25)) and max_curl > 125.0 and palm_dist < 1.60 and interlace_depth >= 0.80:
                scores["outside"] = 10.0

            # [Level 6] 夾 (Interlace): Bilateral finger interleaving
            elif palm_dist < 1.50 and min_curl > 120.0 and ((interlace_depth < 1.25 and (tip_dist > 0.35 or interlace_depth < 1.08)) or (tip_dist > 0.50 and interlace_depth < 1.30)):
                scores["interlace"] = 10.0

            # [Level 7] 內 (Inside): Palm-to-palm flat rubbing
            elif palm_dist < 1.50:
                scores["inside"] = 10.0
            else:
                scores["other"] = 10.0

        # 2. Single-Hand / Merged Cluster Morphology Evidence (for foam/occlusion fallback)
        else:
            active_angles = features.get("active_angles", [0.0]*5)
            active_spread = float(features.get("active_spread", 0.0))
            mean_4_angle = float(np.mean(active_angles[1:])) if len(active_angles) >= 5 else 180.0
            thumb_angle = float(active_angles[0]) if len(active_angles) >= 1 else 180.0

            if mean_4_angle < 135.0 and thumb_angle > 135.0:
                scores["thumb"] = 10.0
            elif active_spread < 0.30 or (mean_4_angle < 125.0 and active_spread < 0.35):
                scores["fingertips"] = 10.0
            elif mean_4_angle < 152.0:
                scores["knuckles"] = 10.0
            elif active_spread > 0.38 and mean_4_angle < 170.0:
                scores["interlace"] = 10.0
            elif thumb_angle < 140.0 and mean_4_angle > 145.0:
                scores["outside"] = 10.0
            elif mean_4_angle > 162.0:
                scores["inside"] = 10.0
            else:
                scores["other"] = 10.0

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
