import cv2
import numpy as np
from typing import List, Tuple


def obstacle_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Generic: treat darker/colored regions as obstacles; tune live
    mask_dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, 70))
    mask_sat = cv2.inRange(hsv, (0, 80, 40), (180, 255, 255))  # saturated colors
    mask = cv2.bitwise_or(mask_dark, mask_sat)
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def sample_headings(mask: np.ndarray, num: int = 31, fov_deg: float = 120) -> List[Tuple[float, float]]:
    # Returns list of (angle_deg, obstacle_cost [0..1]).
    h, w = mask.shape[:2]
    cx = w // 2
    band = mask[h // 3 : (h // 3) * 2, :]
    angles = np.linspace(-fov_deg / 2, fov_deg / 2, num)
    costs: List[Tuple[float, float]] = []
    for a in angles:
        rad = np.deg2rad(a)
        col = int(cx + np.tan(rad) * (0.3 * w))
        col = int(np.clip(col, 0, w - 1))
        col_vals = band[:, col]
        cost = float(np.mean(col_vals > 0))
        costs.append((float(a), cost))
    return costs


def annotate_debug(img: np.ndarray, mask: np.ndarray, text: str) -> np.ndarray:
    m3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(img, 0.8, m3, 0.2, 0)
    cv2.putText(overlay, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return overlay


