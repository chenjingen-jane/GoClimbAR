import unittest

import numpy as np

from goclimb_routes.detect import IndoorDetector
from .fixtures import synthetic_indoor_wall


class IndoorDetectorTests(unittest.TestCase):
    def test_synthetic_same_colour_holds_recall_and_precision(self):
        img, gt = synthetic_indoor_wall()
        h, w = img.shape[:2]
        det = IndoorDetector().detect(img)
        used, tp = set(), 0
        for gx, gy in gt:
            for j, d in enumerate(det):
                if j not in used and np.hypot(d.x * w - gx, d.y * h - gy) < 0.04 * w:
                    used.add(j); tp += 1; break
        print(f"\n[synthetic only] gt={len(gt)} det={len(det)} tp={tp}")
        self.assertGreaterEqual(tp / len(gt), 0.7)
        self.assertGreaterEqual(tp / max(1, len(det)), 0.7)

    def test_outputs_normalized_with_confidence(self):
        img, _ = synthetic_indoor_wall()
        for d in IndoorDetector().detect(img):
            self.assertTrue(0 <= d.x <= 1 and 0 <= d.y <= 1 and 0 <= d.confidence <= 1)
            self.assertIn(d.type, ("candidate", "uncertain"))

    def test_big_dark_holds_on_light_wall_get_one_marker_each(self):
        """Regression from a real camera view: large holds were reported as many points along their outline."""
        import cv2
        img = np.full((700, 500, 3), 215, np.uint8)
        holds = [((130, 150), (90, 110)), ((370, 200), (70, 90)), ((150, 430), (100, 80)),
                 ((380, 470), (60, 100)), ((250, 620), (80, 45))]
        for (cx, cy), (ax, ay) in holds:
            cv2.ellipse(img, (cx, cy), (ax, ay), 0, 0, 360, (70, 70, 70), -1)
        img = cv2.GaussianBlur(img, (0, 0), 1.2)
        found = IndoorDetector().detect(img)
        for (cx, cy), (ax, ay) in holds:
            inside = [d for d in found if abs(d.x * 500 - cx) < ax and abs(d.y * 700 - cy) < ay]
            self.assertEqual(len(inside), 1, f"hold at {(cx, cy)} got {len(inside)} markers")
            self.assertAlmostEqual(inside[0].x * 500, cx, delta=0.35 * ax)           # near its centre, not its rim
            self.assertAlmostEqual(inside[0].y * 700, cy, delta=0.35 * ay)

    def test_flat_wall_and_bad_input(self):
        self.assertEqual(IndoorDetector().detect(np.full((400, 600, 3), 170, np.uint8)), [])
        self.assertEqual(IndoorDetector().detect(None), [])

    def test_does_not_use_colour(self):
        img, _ = synthetic_indoor_wall()
        tinted = img.copy()
        tinted[..., 0] = np.clip(tinted[..., 0].astype(int) + 40, 0, 255)       # change hue, keep structure
        a = IndoorDetector().detect(img)
        b = IndoorDetector().detect(tinted)
        self.assertLess(abs(len(a) - len(b)), 0.35 * max(1, len(a)))


if __name__ == "__main__":
    unittest.main()
