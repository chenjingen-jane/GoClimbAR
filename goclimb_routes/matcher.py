"""Wall matching: is this scan a wall we already saved?

ORB features + ratio test + RANSAC homography, then geometric sanity checks. Works on grayscale
(CLAHE evens out lighting) and tolerates different scale, lighting and moderate perspective.
Matching only IDENTIFIES a saved wall. It says nothing about live camera pose: tracking is Vuforia's job.

Levels: 'confident' (safe to reuse the record automatically), 'possible' (ask the user),
'none'. Thresholds are deliberately conservative: similar rock textures must not merge walls.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

MAX_SIDE = 800
N_FEATURES = 1500
RATIO = 0.75
MIN_GOOD = 12
RANSAC_THRESH = 0.015            # in normalized image units
CONFIDENT = (35, 0.45)           # (min inliers, min inlier ratio)
POSSIBLE = (15, 0.25)
MIN_SPREAD = 0.25                # inliers must cover this fraction of the scan width and height


@dataclass
class Features:
    pts: np.ndarray              # (N, 2) normalized x, y
    des: np.ndarray              # (N, 32) uint8


@dataclass
class MatchResult:
    wall_id: str = ""
    name: str = ""
    venue: str = ""
    inliers: int = 0
    ratio: float = 0.0
    level: str = "none"
    H: Optional[np.ndarray] = field(default=None, repr=False)   # maps STORED-photo coords -> SCAN-photo coords (normalized)


def compute_features(img: np.ndarray) -> Optional[Features]:
    if img is None or img.size == 0:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    k = MAX_SIDE / float(max(h, w))
    if k < 1:
        gray = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    gh, gw = gray.shape[:2]
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    kp, des = cv2.ORB_create(nfeatures=N_FEATURES).detectAndCompute(gray, None)
    if des is None or len(kp) < MIN_GOOD:
        return None
    pts = np.array([(p.pt[0] / gw, p.pt[1] / gh) for p in kp], np.float32)
    return Features(pts, des)


def classify(inliers: int, ratio: float) -> str:
    if inliers >= CONFIDENT[0] and ratio >= CONFIDENT[1]:
        return "confident"
    if inliers >= POSSIBLE[0] and ratio >= POSSIBLE[1]:
        return "possible"
    return "none"


def apply_h(H: np.ndarray, x: float, y: float):
    v = H @ np.array([x, y, 1.0])
    if abs(v[2]) < 1e-9:
        return float("nan"), float("nan")
    return float(v[0] / v[2]), float(v[1] / v[2])


def _homography_ok(H: np.ndarray) -> bool:
    quad = [apply_h(H, x, y) for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))]
    if any(not np.isfinite(c).all() for c in quad):
        return False
    pts = np.array(quad, np.float32)
    if not cv2.isContourConvex(pts):
        return False
    area = abs(cv2.contourArea(pts))
    return 0.04 <= area <= 25.0


def match_features(q: Features, r: Features) -> MatchResult:
    """q = scan, r = stored reference."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = bf.knnMatch(q.des, r.des, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
    if len(good) < MIN_GOOD:
        return MatchResult()
    dst = q.pts[[m.queryIdx for m in good]]
    src = r.pts[[m.trainIdx for m in good]]
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH)
    if H is None or mask is None or not _homography_ok(H):
        return MatchResult()
    keep = mask.ravel().astype(bool)
    inliers = int(keep.sum())
    pts = dst[keep]
    if inliers == 0 or (np.ptp(pts[:, 0]) < MIN_SPREAD or np.ptp(pts[:, 1]) < MIN_SPREAD):
        return MatchResult()
    ratio = inliers / float(len(good))
    return MatchResult(inliers=inliers, ratio=round(ratio, 3), level=classify(inliers, ratio), H=H)


def _ref_features(store, wall_id: str) -> Optional[Features]:
    path = store.photo_path(wall_id)
    if path is None:
        return None
    cache = os.path.join(os.path.dirname(path), "features.npz")
    mtime = os.path.getmtime(path)
    try:
        if os.path.isfile(cache):
            with np.load(cache) as z:
                if float(z["mtime"]) == mtime:
                    return Features(z["pts"], z["des"])
    except Exception:
        pass
    feats = compute_features(cv2.imread(path))
    if feats is not None:
        try:
            np.savez(cache, pts=feats.pts, des=feats.des, mtime=mtime)
        except OSError:
            pass
    return feats


def find_matches(store, scan_img: np.ndarray, limit: int = 200, top: int = 5) -> List[MatchResult]:
    """Compare a scan with saved walls. Returns the best candidates (level != 'none'), best first."""
    q = compute_features(scan_img)
    if q is None:
        return []
    out: List[MatchResult] = []
    for w in store.search_walls(limit=limit):
        r = _ref_features(store, w["wall_id"])
        if r is None:
            continue
        m = match_features(q, r)
        if m.level != "none":
            m.wall_id, m.name, m.venue = w["wall_id"], w["name"], w["venue"]
            out.append(m)
    out.sort(key=lambda m: (m.level == "confident", m.inliers), reverse=True)
    return out[:top]
