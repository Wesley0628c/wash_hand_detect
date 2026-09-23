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
    """Renders state-of-the-art UI overlay on top of OpenCV camera frames."""

    def __init__(self):
        self.font_sm = get_font(16)
        self.font_md = get_font(22)
        self.font_lg = get_font(30)
        self.font_xl = get_font(40)

    def draw_hud(
        self,
        frame: np.ndarray,
        detected_label: str,
        confidence: float,
        feedback_msg: str,
        progress_summary: Dict[str, Any],
        fps: float,
        mode_str: str = "Rule-based",
    ) -> np.ndarray:
        """Render complete HUD onto the frame."""
        h, w, _ = frame.shape
        overlay = frame.copy()

        # 1. Top Header Bar
        cv2.rectangle(overlay, (0, 0), (w, 50), (18, 22, 28), -1)

        # 2. Side Checklist Panel
        # In wide split-screen (e.g. teaching video with text on left, video on right),
        # place checklist on the left (x=20) so it doesn't block the washing hand video on the right.
        panel_w = 260
        if w > h * 1.3:
            panel_x = 20
        else:
            panel_x = w - panel_w - 20
        panel_y = 60
        panel_h = 360
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (22, 26, 35),
            -1,
        )
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (55, 65, 80),
            1,
        )

        # 3. Bottom Feedback Banner
        banner_h = 70
        banner_y = h - banner_h - 15
        banner_x = 20
        banner_w = w - 40
        cv2.rectangle(
            overlay,
            (banner_x, banner_y),
            (banner_x + banner_w, banner_y + banner_h),
            (18, 22, 30),
            -1,
        )
        cv2.rectangle(
            overlay,
            (banner_x, banner_y),
            (banner_x + banner_w, banner_y + banner_h),
            (45, 140, 240),
            2,
        )

        # Blend semi-transparent cards with camera image
        alpha = 0.85
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Convert to PIL Image for high quality Unicode/Chinese text rendering
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # Draw Header
        draw.text((25, 12), "七步洗手即時辨識系統 (Wash Hand Detect)", fill=(255, 255, 255), font=self.font_md)
        draw.text((w - 220, 14), f"FPS: {fps:.1f}  |  {mode_str}", fill=(180, 200, 220), font=self.font_sm)

        # Draw Checklist in Panel
        draw.text((panel_x + 15, panel_y + 12), "七步洗手進度", fill=(255, 220, 100), font=self.font_md)
        completed_set = set(progress_summary.get("completed_steps", []))
        target_step = progress_summary.get("target_step")
        active_step = progress_summary.get("active_step")
        step_progress = progress_summary.get("current_step_progress", 0.0)

        step_y = panel_y + 46
        for step in STEPS_ORDER:
            step_zh = STEPS_ZH[step]
            is_done = step in completed_set
            is_curr = (step == target_step) or (progress_summary["mode"] == "free" and step == active_step)

            if is_done:
                icon = "[OK]"
                text = f"{icon} {step_zh} ({step})"
                color = (100, 240, 120)
            elif is_curr:
                icon = "[->]"
                text = f"{icon} {step_zh} ({step})"
                color = (255, 200, 60)
            else:
                icon = "[  ]"
                text = f"{icon} {step_zh} ({step})"
                color = (160, 170, 185)

            draw.text((panel_x + 15, step_y), text, fill=color, font=self.font_sm)

            # Draw mini progress bar for active step
            if is_curr and not is_done:
                bar_x = panel_x + 140
                bar_y = step_y + 4
                bar_w_max = 100
                draw.rectangle([bar_x, bar_y, bar_x + bar_w_max, bar_y + 10], fill=(60, 60, 60))
                draw.rectangle([bar_x, bar_y, bar_x + int(bar_w_max * step_progress), bar_y + 10], fill=(255, 180, 0))

            step_y += 36

        # Draw Total Wash Time
        total_time = progress_summary.get("total_time", 0.0)
        completed_count = progress_summary.get("completed_count", 0)
        draw.text(
            (panel_x + 15, panel_y + panel_h - 32),
            f"進度: {completed_count}/7 | 總時間: {total_time:.1f}s",
            fill=(220, 230, 245),
            font=self.font_sm,
        )

        # Draw Current Detected Gesture & Feedback Banner
        curr_zh = LABEL_SHORT_ZH.get(detected_label, detected_label)
        draw.text((banner_x + 20, banner_y + 10), f"當前動作：{curr_zh} ({detected_label})", fill=(255, 255, 255), font=self.font_lg)
        draw.text((banner_x + 360, banner_y + 15), f"信心度：{confidence*100:.0f}%", fill=(100, 220, 255), font=self.font_md)
        draw.text((banner_x + 20, banner_y + 42), f"指導提示：{feedback_msg}", fill=(240, 200, 100), font=self.font_sm)

        # Completion Victory Screen (Bottom right toast or banner)
        if progress_summary.get("is_completed", False):
            toast_w, toast_h = 460, 60
            toast_x = w - toast_w - 20
            toast_y = 60
            draw.rectangle([toast_x, toast_y, toast_x + toast_w, toast_y + toast_h], fill=(20, 45, 30), outline=(0, 255, 150), width=2)
            draw.text((toast_x + 15, toast_y + 10), "★ 洗手完成！恭喜達成七步洗手！", fill=(100, 255, 150), font=self.font_md)
            draw.text((toast_x + 15, toast_y + 35), f"總洗手時間：{total_time:.1f} 秒 | 動作確實清潔！", fill=(220, 240, 230), font=self.font_sm)

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
