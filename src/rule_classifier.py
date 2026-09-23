"""
Rule-Based Wash Hand Classifier Module
Implements domain-rule gesture classification for the 7 steps + 1 other class,
along with real-time posture feedback generation.
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

NAME_TO_LABEL = {v: k for k, v in LABELS.items()}


class WashHandRuleClassifier:
    """Rule-based evaluator for 7-step hand washing gestures."""

    def __init__(self, min_motion_velocity: float = 0.005):
        self.min_motion_velocity = min_motion_velocity

    def predict(self, features: Dict[str, Any]) -> Tuple[str, float, str]:
        """
        Evaluate extracted features and return:
          (label_name, confidence, feedback_message)
        """
        if not features.get("both_hands_detected", False):
            if features.get("has_left", False) or features.get("has_right", False):
                return "other", 0.9, "請將雙手皆放入鏡頭畫面中"
            return "other", 1.0, "未偵測到雙手，請伸出雙手"

        inter = features["inter_hand"]
        wrist_dist = inter["wrist_dist"]
        palm_center_dist = inter["palm_center_dist"]
        palm_dot = features["palm_normal_dot"]
        min_tips_to_palm = inter["min_tips_to_palm"]
        min_palm_to_wrist = inter["min_palm_to_wrist"]
        min_palm_to_thumb = inter["min_palm_to_thumb"]
        interlace_depth = inter["interlace_depth"]

        left_angles = features["left_angles"]
        right_angles = features["right_angles"]
        avg_angle = (np.mean(left_angles[1:]) + np.mean(right_angles[1:])) / 2.0  # 4 fingers
        min_hand_angle = min(np.mean(left_angles[1:]), np.mean(right_angles[1:]))

        # Check if hands are too far apart
        if wrist_dist > 2.8 and palm_center_dist > 2.5:
            return "other", 0.85, "雙手距離太遠，請將雙手靠近搓洗"

        # 1. 腕 (Wrist): One palm center wraps/touches opposite wrist
        if self._is_wrist(features):
            return "wrist", 0.90, "姿勢正確：正在旋轉搓洗手腕"

        # 2. 立 (Fingertips): Fingertips cluster pressed against opposite palm
        if self._is_fingertips(features):
            return "fingertips", 0.90, "姿勢正確：指尖搓洗掌心"

        # 3. 大 (Thumb): One palm wrapping opposite thumb
        if self._is_thumb(features):
            return "thumb", 0.88, "姿勢正確：旋轉搓洗大拇指"

        # 4. 弓 (Knuckles): Fingers clearly bent/curled, knuckles against opposite palm
        if self._is_knuckles(features, min_hand_angle):
            return "knuckles", 0.86, "姿勢正確：指背搓洗掌心"

        # 5. 夾 (Interlace): Fingers interwoven, deep overlap, palms facing
        if self._is_interlace(features):
            return "interlace", 0.88, "姿勢正確：十指交錯搓洗"

        # 6. 外 (Outside): One palm touching back of opposite hand (normals point same direction)
        if self._is_outside(features):
            return "outside", 0.87, "姿勢正確：掌心搓洗手背"

        # 7. 內 (Inside): Palms facing each other and rubbing
        if self._is_inside(features):
            return "inside", 0.89, "姿勢正確：掌心對掌心搓洗"

        return "other", 0.75, "動作調整中，請依步驟搓洗"

    def _is_wrist(self, features: Dict[str, Any]) -> bool:
        inter = features["inter_hand"]
        # One palm is very close to opposite wrist, but palm centers are not touching
        return (
            inter["min_palm_to_wrist"] < 0.75
            and inter["palm_center_dist"] > 0.6
        )

    def _is_fingertips(self, features: Dict[str, Any]) -> bool:
        inter = features["inter_hand"]
        # Fingertips of one hand touch opposite palm, while that hand's palm center is further
        return (
            inter["min_tips_to_palm"] < 0.65
            and inter["palm_center_dist"] > 0.5
        )

    def _is_thumb(self, features: Dict[str, Any]) -> bool:
        inter = features["inter_hand"]
        # One palm close to opposite thumb tip, wrist distance not too close
        return (
            inter["min_palm_to_thumb"] < 0.70
            and inter["min_tips_to_palm"] > 0.45
            and inter["min_palm_to_wrist"] > 0.65
        )

    def _is_knuckles(self, features: Dict[str, Any], min_angle: float) -> bool:
        inter = features["inter_hand"]
        # Bent fingers (< 130 deg) and close palm distance
        return (
            min_angle < 135.0
            and inter["palm_center_dist"] < 1.1
            and inter["min_palm_to_wrist"] > 0.5
        )

    def _is_interlace(self, features: Dict[str, Any]) -> bool:
        inter = features["inter_hand"]
        palm_dot = features["palm_normal_dot"]
        # Deep overlap between fingers, palms facing each other
        return (
            inter["interlace_depth"] < 0.85
            and inter["palm_center_dist"] < 0.95
            and palm_dot < 0.2
        )

    def _is_outside(self, features: Dict[str, Any]) -> bool:
        inter = features["inter_hand"]
        palm_dot = features["palm_normal_dot"]
        # Normals pointing in similar direction (palm on dorsal surface)
        return (
            palm_dot > 0.05
            and inter["palm_center_dist"] < 1.3
        )

    def _is_inside(self, features: Dict[str, Any]) -> bool:
        inter = features["inter_hand"]
        palm_dot = features["palm_normal_dot"]
        # Palms facing each other, palms close
        return (
            palm_dot < 0.1
            and inter["palm_center_dist"] < 1.3
            and inter["wrist_dist"] < 1.8
        )
