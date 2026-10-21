"""Draws holds, routes (green / yellow / red), hold IDs and a legend on a photo.
Routes sharing holds are separated with a small perpendicular line offset."""
from __future__ import annotations

import math
from typing import List, Optional

import cv2
import numpy as np

from .schema import DIFFICULTY_COLORS, DIFFICULTY_LABELS


def _bgr(hex_color: str):
    c = hex_color.lstrip("#")
    return (int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16))


def render_overlay(image: np.ndarray, holds: List[dict], routes: List[dict],
                   selected: Optional[str] = None, show_ids: bool = True) -> np.ndarray:
    img = image.copy()
    h, w = img.shape[:2]
    P = lambda x, y: (int(round(x * w)), int(round(y * h)))
    unit = max(1.0, w / 800.0)
    for hd in holds:
        if hd.get("status") == "deleted":
            continue
        p = P(hd["x"], hd["y"])
        cv2.circle(img, p, int(4 * unit), (255, 255, 255), 1, cv2.LINE_AA)
        if show_ids:
            cv2.putText(img, str(hd["id"]), (p[0] + int(5 * unit), p[1] - int(5 * unit)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35 * unit, (255, 255, 255), 1, cv2.LINE_AA)
    n = len(routes)
    for idx, r in enumerate(routes):
        if not r.get("valid", True):
            continue
        dim = selected is not None and r["id"] != selected
        col = _bgr(r["color"])
        off = (idx - (n - 1) / 2.0) * 2.5 * unit
        pts = [P(p["x"], p["y"]) for p in r["points"]]
        shifted = []
        for k, q in enumerate(pts):
            a, b = pts[max(k - 1, 0)], pts[min(k + 1, len(pts) - 1)]
            dx, dy = b[0] - a[0], b[1] - a[1]
            ln = math.hypot(dx, dy) or 1.0
            shifted.append((int(q[0] - dy / ln * off), int(q[1] + dx / ln * off)))
        layer = img.copy()
        thick = int((2 if dim else 4) * unit)
        for a, b in zip(shifted, shifted[1:]):
            cv2.line(layer, a, b, col, thick, cv2.LINE_AA)
        for k, q in enumerate(pts):
            role = r["points"][k]["role"]
            rad = int((9 if role != "hold" else 6) * unit * (0.7 if dim else 1.0))
            cv2.circle(layer, q, rad, col, -1, cv2.LINE_AA)
            cv2.circle(layer, q, rad, (255, 255, 255), 1, cv2.LINE_AA)
        a = 0.3 if dim else 0.95
        img = cv2.addWeighted(layer, a, img, 1 - a, 0)
        if selected == r["id"]:
            for k, q in enumerate(pts):
                cv2.putText(img, f"{k + 1}", (q[0] + int(8 * unit), q[1] + int(14 * unit)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * unit, (255, 255, 255), 2, cv2.LINE_AA)
    # legend
    lx, ly, lh = int(10 * unit), h - int(78 * unit), int(18 * unit)
    cv2.rectangle(img, (lx - 6, ly - 18), (lx + int(330 * unit), h - 6), (0, 0, 0), -1)
    for i, key in enumerate(("easy", "medium", "hard")):
        y = ly + i * lh
        cv2.circle(img, (lx + 6, y), int(6 * unit), _bgr(DIFFICULTY_COLORS[key]), -1)
        cv2.putText(img, f"{DIFFICULTY_LABELS[key]} (estimated)", (lx + 20, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * unit, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "Estimated difficulty - not an official grade", (lx, ly + 3 * lh),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38 * unit, (200, 200, 200), 1, cv2.LINE_AA)
    return img
