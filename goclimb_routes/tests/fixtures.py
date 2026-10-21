import numpy as np
import cv2
from goclimb_routes.detect import Hold


def synthetic_indoor_wall(seed=2, n=45, size=(600, 800)):
    """SYNTHETIC wall: holds only a few grey levels from the wall colour, with shadows, plus bolts.
    Passing on this proves nothing about real gyms; it only checks the code path."""
    rs = np.random.RandomState(seed)
    h, w = size
    img = np.full((h, w, 3), 175, np.uint8)
    img = (img.astype(np.int16) + rs.normal(0, 3, img.shape)).clip(0, 255).astype(np.uint8)
    pts = []
    for _ in range(n):
        cx, cy, r = int(rs.uniform(60, w - 60)), int(rs.uniform(60, h - 60)), int(rs.uniform(13, 24))
        if any(np.hypot(cx - a, cy - b) < 60 for a, b in pts):
            continue
        pts.append((cx, cy))
        cv2.ellipse(img, (cx + 4, cy + 7), (r, int(r * .85)), 0, 0, 360, (140, 140, 140), -1)
        cv2.ellipse(img, (cx, cy), (r, int(r * .85)), 0, 0, 360, (182, 182, 182), -1)
        cv2.ellipse(img, (cx - r // 4, cy - r // 3), (r // 2, r // 3), 0, 0, 360, (190, 190, 190), -1)
    for _ in range(40):
        cv2.circle(img, (int(rs.uniform(10, w - 10)), int(rs.uniform(10, h - 10))), 3, (120, 120, 120), -1)
    return cv2.GaussianBlur(img, (0, 0), 1.0), pts


def png_bytes(img):
    return cv2.imencode(".png", img)[1].tobytes()


def grid(cols=7, rows=12, conf=0.8):
    rs = np.random.RandomState(3)
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append(Hold(id=f"H{len(out) + 1}", x=float(0.08 + c * 0.14 + rs.uniform(-.01, .01)),
                            y=float(0.05 + r * (0.90 / (rows - 1)) + rs.uniform(-.01, .01)), confidence=conf))
    return out
