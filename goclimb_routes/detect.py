"""Hold detection. All coordinates are normalized image coords:
origin top-left, x to the right, y DOWN, both in [0, 1].

IMPORTANT LIMITATION: ClassicalDetector is a heuristic (no training needed).
It finds contrast features on the rock (dark pockets, ledges with a shadow
underneath). It does NOT know what a real hold is. For gym walls with
artificial holds, replace it with a trained detector (e.g. YOLOv8) that
implements the same `detect(bgr) -> list[Hold]` interface.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Protocol

import cv2
import numpy as np


@dataclass
class Hold:
    id: int
    x: float
    y: float
    confidence: float
    type: str = "feature"
    source: str = "classical_cv"
    color: str = ""          # "#RRGGBB" (gym mode: colour of the hold)
    group: str = ""          # colour name (gym mode: one group = one setter route)


class HoldDetector(Protocol):
    def detect(self, bgr: np.ndarray) -> List[Hold]: ...


def rock_mask(bgr: np.ndarray) -> np.ndarray:
    """Rough wall region: remove sky (blue) and grass (green), keep the largest blob."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, _ = cv2.split(hsv)
    sky = (h >= 90) & (h <= 135) & (s >= 40)
    grass = (h >= 23) & (h < 90) & (s >= 70)
    mask = ((~(sky | grass)).astype(np.uint8)) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((labels == largest) * 255).astype(np.uint8)


def _norm(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = a[mask > 0]
    if vals.size == 0:
        return np.zeros_like(a)
    hi = float(np.percentile(vals, 99))
    return np.clip(a / hi, 0, 1) if hi > 1e-6 else np.zeros_like(a)


class ClassicalDetector:
    def __init__(self, max_holds: int = 150, min_dist_frac: float = 0.03,
                 min_score: float = 0.30, work_width: int = 800):
        self.max_holds = max_holds
        self.min_dist_frac = min_dist_frac
        self.min_score = min_score
        self.work_width = work_width

    def detect(self, bgr: np.ndarray) -> List[Hold]:
        if bgr is None or bgr.size == 0:
            return []
        h0, w0 = bgr.shape[:2]
        scale = self.work_width / float(w0) if w0 > self.work_width else 1.0
        img = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bgr
        h, w = img.shape[:2]

        mask = rock_mask(img)
        mask = cv2.erode(mask, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (int(0.03 * w) | 1, int(0.03 * w) | 1)))
        if cv2.countNonZero(mask) < 0.05 * w * h:
            return []   # no wall region found

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

        # Pockets / dark cavities: black-hat.
        k = (int(0.04 * w) | 1)
        pocket = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)

        # Ledges: brightness drops going DOWN the image (lit top, shadow below).
        g = cv2.GaussianBlur(gray.astype(np.float32) / 255.0, (0, 0), max(1.5, 0.004 * w))
        d = max(2, int(0.012 * w))
        ledge = np.zeros_like(g)
        ledge[:-d] = np.clip(g[:-d] - g[d:], 0, None)
        ledge = cv2.GaussianBlur(ledge, (0, 0), max(2.0, 0.006 * w))

        p = _norm(pocket, mask)
        l = _norm(ledge, mask)
        score = cv2.GaussianBlur(0.5 * p + 0.5 * l, (0, 0), max(1.5, 0.004 * w))
        score[mask == 0] = 0

        r = max(2, int(self.min_dist_frac * w))
        dil = cv2.dilate(score, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        ys, xs = np.where((score >= dil) & (score >= self.min_score))
        order = np.argsort(-score[ys, xs])[: self.max_holds]

        holds: List[Hold] = []
        for i in order:
            y, x = int(ys[i]), int(xs[i])
            pv, lv = float(p[y, x]), float(l[y, x])
            t = "pocket" if pv > 1.3 * lv else ("ledge" if lv > 1.3 * pv else "feature")
            holds.append(Hold(id=len(holds), x=round(x / w, 5), y=round(y / h, 5),
                              confidence=round(float(min(1.0, score[y, x])), 4), type=t))
        return holds


def validate_holds(holds: List[Hold]):
    """Drop invalid / duplicate holds. Returns (clean_holds, warnings)."""
    warnings, clean = [], []
    for hd in holds:
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (hd.x, hd.y)):
            warnings.append(f"hold {hd.id}: non-finite coordinates dropped")
            continue
        if not (0.0 <= hd.x <= 1.0 and 0.0 <= hd.y <= 1.0):
            warnings.append(f"hold {hd.id}: outside image bounds dropped")
            continue
        if any(math.hypot(hd.x - c.x, hd.y - c.y) < 0.008 for c in clean):
            continue
        clean.append(hd)
    for i, hd in enumerate(clean):
        hd.id = i
    return clean, warnings


def holds_from_dicts(items) -> List[Hold]:
    """Accept holds from a file / external detector: [{x, y, confidence?, type?}, ...]."""
    return [Hold(id=i, x=float(d["x"]), y=float(d["y"]),
                 confidence=float(d.get("confidence", 1.0)),
                 type=str(d.get("type", "feature")),
                 source=str(d.get("source", "external")))
            for i, d in enumerate(items)]


# --------------------------------------------------------------------------- indoor / gym

def _hue_name(h: float) -> str:
    """OpenCV hue 0..179 -> colour name."""
    if h < 8 or h >= 170: return "red"
    if h < 22: return "orange"
    if h < 35: return "yellow"
    if h < 85: return "green"
    if h < 100: return "cyan"
    if h < 130: return "blue"
    if h < 150: return "purple"
    return "pink"


class GymColorDetector:
    """Indoor walls: artificial holds are saturated colours on a plainer wall.
    Finds saturated blobs, filters by size/shape, groups them by colour.
    Same colour = same route (gym convention), so group names are meaningful.

    Limits: white / black / grey holds are NOT found (no saturation); a wall
    painted in a saturated colour needs min_sat raised; two routes of the same
    colour end up in one group; volumes (big shapes) are ignored.
    """
    def __init__(self, min_sat: int = 90, min_val: int = 70, min_area_frac: float = 0.0002,
                 max_area_frac: float = 0.03, min_solidity: float = 0.6,
                 max_elongation: float = 4.0, work_width: int = 800):
        self.min_sat, self.min_val = min_sat, min_val
        self.min_area_frac, self.max_area_frac = min_area_frac, max_area_frac
        self.min_solidity, self.max_elongation = min_solidity, max_elongation
        self.work_width = work_width

    def detect(self, bgr: np.ndarray) -> List[Hold]:
        if bgr is None or bgr.size == 0:
            return []
        w0 = bgr.shape[1]
        sc = self.work_width / float(w0) if w0 > self.work_width else 1.0
        img = cv2.resize(bgr, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1 else bgr
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(cv2.GaussianBlur(img, (3, 3), 0), cv2.COLOR_BGR2HSV)
        mask = ((hsv[..., 1] >= self.min_sat) & (hsv[..., 2] >= self.min_val)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        holds: List[Hold] = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (self.min_area_frac * w * h <= area <= self.max_area_frac * w * h):
                continue
            hull_a = cv2.contourArea(cv2.convexHull(c))
            solidity = area / hull_a if hull_a > 0 else 0
            (_, _), (rw, rh), _ = cv2.minAreaRect(c)
            elong = max(rw, rh) / max(1.0, min(rw, rh))
            if solidity < self.min_solidity or elong > self.max_elongation:
                continue
            m = cv2.moments(c)
            if m["m00"] <= 0:
                continue
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            blob = np.zeros((h, w), np.uint8)
            cv2.drawContours(blob, [c], -1, 255, -1)
            hue = float(np.degrees(np.arctan2(
                np.mean(np.sin(np.radians(hsv[..., 0][blob > 0] * 2.0))),
                np.mean(np.cos(np.radians(hsv[..., 0][blob > 0] * 2.0))))) % 360) / 2.0
            b, g, r = [int(v) for v in cv2.mean(img, mask=blob)[:3]]
            holds.append(Hold(id=len(holds), x=round(cx / w, 5), y=round(cy / h, 5),
                              confidence=round(float(min(1.0, solidity)), 4), type="hold",
                              source="gym_color", color=f"#{r:02X}{g:02X}{b:02X}",
                              group=_hue_name(hue)))
        return holds


# --------------------------------------------------------------------------- indoor (colour-independent)

class IndoorDetector:
    """Indoor walls WITHOUT using hold colour (holds may match the wall or each other).

    Uses luminance only:
      * multi-scale difference-of-Gaussians (blob / protrusion contrast),
      * local edge density (holds have contours, flat panels do not),
      * shadow-under-ledge response (lit top, dark underside).
    Small fine-scale-only spots (bolts, screw holes, chalk dots) are down-weighted.

    HONEST LIMITS: it returns CANDIDATES with a confidence score, not verified holds.
    Textured walls, strong shadows, tape and wall marks create false candidates; small,
    flat or occluded holds are missed. It must be reviewed with the manual editor.
    Measure it on real photos (store.detection_report) before trusting it.
    """
    def __init__(self, max_holds: int = 120, min_dist_frac: float = 0.025, min_score: float = 0.35,
                 work_width: int = 900, margin_frac: float = 0.02, region_contrast: float = 0.10):
        self.max_holds, self.min_dist_frac, self.min_score = max_holds, min_dist_frac, min_score
        self.work_width, self.margin_frac = work_width, margin_frac
        self.region_contrast = region_contrast      # luminance difference to the local background (0..1)

    @staticmethod
    def _two_tone_mask(g: np.ndarray):
        """If the picture has two clearly different brightness groups (e.g. dark holds on a light wall, or
        the reverse), return the mask of the smaller group (the holds). Local comparison fails when holds
        are larger than the neighbourhood. Returns None for ordinary low-contrast pictures."""
        g8 = cv2.GaussianBlur((g * 255).astype(np.uint8), (0, 0), 1.5)
        t, _ = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark = g8 <= t
        if dark.all() or not dark.any():
            return None
        if (g8[~dark].mean() - g8[dark].mean()) / 255.0 < 0.25:
            return None                                     # groups too similar: not a clean two-tone picture
        minority = dark if dark.mean() < 0.5 else ~dark
        if not (0.03 <= minority.mean() <= 0.45):
            return None
        return minority.astype(np.uint8) * 255

    def _regions(self, g: np.ndarray):
        """Whole-hold candidates: shapes that clearly differ in brightness from the wall around them.
        One point per shape (its thickest part), so a big hold is not reported as many edge points.
        Holds that match the wall brightness are NOT found here; the feature peaks below cover those."""
        h, w = g.shape
        diff = np.abs(g - cv2.GaussianBlur(g, (0, 0), 0.06 * w))
        mask = self._two_tone_mask(g)                       # clear light-wall / dark-holds picture?
        if mask is None:                                    # otherwise: brighter or darker than the surroundings
            mask = (diff > self.region_contrast).astype(np.uint8) * 255
        ell = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ell(max(3, int(0.008 * w) | 1)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ell(max(3, int(0.005 * w) | 1)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (0.0008 <= area / float(w * h) <= 0.15):
                continue
            hull = cv2.contourArea(cv2.convexHull(c))
            if hull <= 0 or area / hull < 0.5:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw > 0.7 * w or bh > 0.7 * h:
                continue
            blob = np.zeros((bh + 2, bw + 2), np.uint8)
            cv2.drawContours(blob, [c - np.array([[x - 1, y - 1]])], -1, 255, -1)
            dist = cv2.distanceTransform(blob, cv2.DIST_L2, 3)
            py, px = np.unravel_index(int(np.argmax(dist)), dist.shape)      # thickest point, always inside the shape
            cx, cy = x - 1 + px, y - 1 + py
            contrast = float(diff[y:y + bh, x:x + bw][blob[1:-1, 1:-1] > 0].mean())
            touches = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
            conf = min(1.0, contrast / 0.30) * (0.7 if touches else 1.0)
            out.append((cx, cy, conf, (x, y, bw, bh)))
        return out

    def detect(self, bgr: np.ndarray) -> List[Hold]:
        if bgr is None or bgr.size == 0:
            return []
        w0 = bgr.shape[1]
        sc = self.work_width / float(w0) if w0 > self.work_width else 1.0
        img = cv2.resize(bgr, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1 else bgr
        h, w = img.shape[:2]
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        def dog(s):
            return np.abs(cv2.GaussianBlur(g, (0, 0), s) - cv2.GaussianBlur(g, (0, 0), s * 1.6))

        d_fine, d_mid, d_big = dog(0.006 * w), dog(0.012 * w), dog(0.022 * w)
        mid = np.maximum(d_mid, d_big)
        edges = cv2.Canny((g * 255).astype(np.uint8), 40, 120)
        dens = cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (0, 0), 0.012 * w)
        gs = cv2.GaussianBlur(g, (0, 0), 0.004 * w)
        d = max(2, int(0.012 * w))
        ledge = np.zeros_like(g)
        ledge[:-d] = np.clip(gs[:-d] - gs[d:], 0, None)
        ledge = cv2.GaussianBlur(ledge, (0, 0), 0.008 * w)

        def norm(a, floor):      # absolute floor so flat, noise-only walls stay near zero
            return np.clip(a / max(float(np.percentile(a, 99.5)), floor), 0, 1)

        n_mid, n_dens, n_led, n_fine = norm(mid, 0.04), norm(dens, 0.15), norm(ledge, 0.04), norm(d_fine, 0.04)
        score = 0.40 * n_mid + 0.35 * n_dens + 0.25 * n_led
        score[(n_fine > 0.5) & (n_mid < 0.5 * n_fine)] *= 0.3          # bolt / screw-hole like
        score = cv2.GaussianBlur(score, (0, 0), max(1.5, 0.004 * w))
        m = int(self.margin_frac * w)
        score[:m] = score[-m:] = 0
        score[:, :m] = score[:, -m:] = 0

        r = max(2, int(self.min_dist_frac * w))
        dil = cv2.dilate(score, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        ys, xs = np.where((score >= dil) & (score >= self.min_score))
        order = np.argsort(-score[ys, xs])[: self.max_holds]
        regions = self._regions(g)
        pad = int(0.012 * w)
        inside = lambda x, y: any(bx - pad <= x <= bx + bw + pad and by - pad <= y <= by + bh + pad
                                  for _, _, _, (bx, by, bw, bh) in regions)
        cands = [(cx, cy, conf, "indoor_region") for cx, cy, conf, _ in regions]
        for i in order:                                    # feature peaks, but not on outlines of whole holds
            y, x = int(ys[i]), int(xs[i])
            if not inside(x, y):
                cands.append((x, y, float(min(1.0, score[y, x])), "indoor_cv"))
        cands.sort(key=lambda t: -t[2])
        holds: List[Hold] = []
        for x, y, conf, src in cands[: self.max_holds]:
            holds.append(Hold(id=len(holds), x=round(x / w, 5), y=round(y / h, 5), confidence=round(conf, 4),
                              type="candidate" if conf >= 0.5 else "uncertain", source=src))
        return holds
