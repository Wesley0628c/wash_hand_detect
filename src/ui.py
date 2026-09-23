"""
UI & Visual HUD Rendering Module
Renders modern, semi-transparent overlays, Chinese typography, and real-time wash step trackers.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, Optional, List

from src.rule_classifier import LABEL_SHORT_ZH, LABEL_NAMES_ZH
from src.state_machine import STEPS_ORDER, STEPS_ZH

# Find macOS default Chinese font or fallback
FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]

AVAILABLE_FONT_PATH = None
for path in FONT_CANDIDATES:
    if os.path.exists(path):
        AVAILABLE_FONT_PATH = path
        break


def get_font(size: int = 20) -> ImageFont.FreeTypeFont:
    if AVAILABLE_FONT_PATH:
        try:
            return ImageFont.truetype(AVAILABLE_FONT_PATH, size)
        except Exception:
            pass
    return ImageFont.load_default()


class WashHandHUD:
    """Renders sleek, modern, real-time action recognition UI without distracting checklist sidebars."""

    def __init__(self):
        self.font_sm = get_font(16)
        self.font_md = get_font(22)
        self.font_lg = get_font(30)
        self.font_xl = get_font(38)

    def draw_hud(
        self,
        frame: np.ndarray,
        detected_label: str,
        confidence: float,
        feedback_msg: str,
        progress_summary: Optional[Dict[str, Any]] = None,
        fps: float = 0.0,
        mode_str: str = "HYBRID",
        hands_status: Optional[Dict[str, bool]] = None,
    ) -> np.ndarray:
        """Render clean, instant real-time HUD onto the frame."""
        h, w, _ = frame.shape
        overlay = frame.copy()

        # 1. Top Header Bar (Semi-transparent dark glass)
        cv2.rectangle(overlay, (0, 0), (w, 54), (18, 22, 28), -1)

        # 2. Bottom Main Action Card (Sleek Glassmorphic Card)
        banner_h = 80
        banner_y = h - banner_h - 18
        banner_x = 20
        banner_w = w - 40
        cv2.rectangle(
            overlay,
            (banner_x, banner_y),
            (banner_x + banner_w, banner_y + banner_h),
            (18, 22, 32),
            -1,
        )
        # Highlight card border based on detected label
        border_color = (40, 200, 100) if detected_label in STEPS_ORDER else (70, 80, 95)
        cv2.rectangle(
            overlay,
            (banner_x, banner_y),
            (banner_x + banner_w, banner_y + banner_h),
            border_color,
            2,
        )

        # Blend semi-transparent cards with camera image
        alpha = 0.85
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Convert to PIL Image for crisp Chinese typography
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # Draw Header Info
        draw.text((25, 14), "七步洗手即時辨識系統 (Wash Hand Detect)", fill=(255, 255, 255), font=self.font_md)

        # Hand Tracking Indicator Dots in Header
        if hands_status is not None:
            left_ok = hands_status.get("left", False)
            right_ok = hands_status.get("right", False)
            left_color = (0, 255, 150) if left_ok else (130, 140, 150)
            right_color = (255, 160, 50) if right_ok else (130, 140, 150)
            status_text = f"左手 [{'●' if left_ok else '○'}]  右手 [{'●' if right_ok else '○'}]"
            draw.text((w // 2 - 80, 16), status_text, fill=(220, 230, 240), font=self.font_sm)

        draw.text((w - 240, 16), f"FPS: {fps:.1f}  |  {mode_str}", fill=(180, 205, 230), font=self.font_sm)

        # 3. Top-Center 7-Step Quick Badges Bar (Visual reference)
        badges_y = 66
        badge_w, badge_h = 58, 30
        total_badges_w = len(STEPS_ORDER) * (badge_w + 10)
        start_bx = max(20, (w - total_badges_w) // 2)

        for i, step in enumerate(STEPS_ORDER):
            bx = start_bx + i * (badge_w + 10)
            is_active = (detected_label == step)
            step_zh = STEPS_ZH.get(step, step)

            # Badge background
            if is_active:
                bg_col = (255, 180, 30)
                txt_col = (20, 20, 20)
                outline_col = (255, 220, 100)
            else:
                bg_col = (30, 35, 45)
                txt_col = (160, 175, 190)
                outline_col = (60, 70, 85)

            draw.rectangle([bx, badges_y, bx + badge_w, badges_y + badge_h], fill=bg_col, outline=outline_col, width=1)
            draw.text((bx + 10, badges_y + 4), f"{step_zh} ({step[0].upper()})", fill=txt_col, font=self.font_sm)

        # 4. Bottom Main Action Card Content
        curr_zh = LABEL_NAMES_ZH.get(detected_label, LABEL_SHORT_ZH.get(detected_label, detected_label))
        action_color = (100, 255, 160) if detected_label in STEPS_ORDER else (200, 210, 225)
        draw.text((banner_x + 22, banner_y + 12), f"當前動作：【 {curr_zh} 】", fill=action_color, font=self.font_lg)

        # Confidence Bar
        conf_pct = int(max(0.0, min(1.0, confidence)) * 100)
        draw.text((banner_x + 420, banner_y + 16), f"信心度：{conf_pct}%", fill=(100, 220, 255), font=self.font_md)

        # Mini confidence progress bar
        cbar_x = banner_x + 560
        cbar_y = banner_y + 22
        cbar_w = min(180, w - cbar_x - 30)
        if cbar_w > 50:
            draw.rectangle([cbar_x, cbar_y, cbar_x + cbar_w, cbar_y + 12], fill=(45, 55, 70))
            fill_w = int(cbar_w * (conf_pct / 100.0))
            bar_color = (0, 220, 140) if conf_pct >= 60 else (255, 180, 50)
            draw.rectangle([cbar_x, cbar_y, cbar_x + fill_w, cbar_y + 12], fill=bar_color)

        # Real-time Feedback Hint
        draw.text((banner_x + 24, banner_y + 48), f"指導提示：{feedback_msg}", fill=(255, 215, 120), font=self.font_sm)

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
