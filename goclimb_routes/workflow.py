"""Known vs unknown wall workflow. No database needed: saved routes are JSON files."""
from __future__ import annotations

import json
import os
from typing import List, Optional

import cv2

from .detect import ClassicalDetector, GymColorDetector, Hold, HoldDetector, holds_from_dicts, validate_holds
from .planner import plan_routes, setter_routes

NOTE = ("Difficulty is estimated from image geometry only; it is not an official grade "
        "and says nothing about safety. Hold detection is heuristic unless a trained "
        "detector is used.")


def build_route_set(wall_name: str, image_w: int, image_h: int, holds: List[Hold],
                    detector_name: str, alternatives: int = 1, mode: str = "outdoor") -> dict:
    aspect = image_h / float(image_w)
    clean, w0 = validate_holds(holds)
    if mode == "gym":
        routes, w1 = setter_routes(clean, aspect)
        if not routes:          # no usable colour groups: fall back to generated routes
            r2, w2 = plan_routes(clean, aspect, alternatives=alternatives)
            routes, w1 = r2, w1 + ["fell back to generated routes"] + w2
    else:
        routes, w1 = plan_routes(clean, aspect, alternatives=alternatives)
    return {
        "schema": 1, "wall": wall_name, "aspect": round(aspect, 5),
        "coordinate_system": "normalized image: origin top-left, x right, y down, 0..1",
        "detector": detector_name, "note": NOTE,
        "holds": [{"id": h.id, "x": h.x, "y": h.y, "confidence": h.confidence, "type": h.type,
                   "color": h.color, "group": h.group}
                  for h in clean],
        "routes": routes, "warnings": w0 + w1,
    }


def saved_path(store_dir: str, wall_name: str) -> str:
    return os.path.join(store_dir, f"{wall_name}.routes.json")


def get_routes(wall_name: str, store_dir: str, image_path: Optional[str] = None,
               holds_override: Optional[list] = None, detector: Optional[HoldDetector] = None,
               save: bool = False, alternatives: int = 1, mode: str = "outdoor") -> dict:
    """KNOWN wall (saved JSON exists) -> load it.
    UNKNOWN wall -> detect (or use supplied holds) and generate; save only if save=True."""
    path = saved_path(store_dir, wall_name)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["source"] = "saved"
        return data

    if not image_path:
        raise ValueError(f"unknown wall '{wall_name}' and no image supplied")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"cannot read image: {image_path}")
    h, w = img.shape[:2]
    if holds_override is not None:
        holds, name = holds_from_dicts(holds_override), "external"
    else:
        det = detector or (GymColorDetector() if mode == "gym" else ClassicalDetector())
        holds, name = det.detect(img), type(det).__name__
    data = build_route_set(wall_name, w, h, holds, name, alternatives, mode)
    data["source"] = "generated"
    if save:
        os.makedirs(store_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    return data
