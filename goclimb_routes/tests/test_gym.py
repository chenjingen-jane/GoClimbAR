import os
import tempfile
import unittest

import cv2
import numpy as np

from goclimb_routes.detect import GymColorDetector
from goclimb_routes.workflow import get_routes

COLORS = {"red": (40, 40, 220), "green": (60, 200, 60), "blue": (220, 90, 40)}   # BGR


def gym_wall(path=None):
    """Light grey wall, 3 colours x 8 holds climbing upward, plus things that must be ignored."""
    img = np.full((600, 800, 3), 200, np.uint8)
    cols = {"red": 150, "green": 400, "blue": 650}
    rs = np.random.RandomState(5)
    for name, x0 in cols.items():
        for k in range(8):
            cx = int(x0 + rs.uniform(-30, 30)); cy = int(540 - k * 62 + rs.uniform(-8, 8))
            cv2.circle(img, (cx, cy), 14, COLORS[name], -1)
    cv2.rectangle(img, (0, 0), (800, 40), (200, 120, 30), -1)          # huge saturated banner: ignored
    cv2.line(img, (20, 323), (780, 323), (30, 200, 230), 4)             # thin tape line: ignored
    if path:
        cv2.imwrite(path, img)
    return img


class GymTests(unittest.TestCase):
    def test_detects_all_holds_grouped_by_colour(self):
        holds = GymColorDetector().detect(gym_wall())
        self.assertEqual(len(holds), 24)
        counts = {}
        for h in holds:
            counts[h.group] = counts.get(h.group, 0) + 1
            self.assertTrue(0 <= h.x <= 1 and 0 <= h.y <= 1)
        self.assertEqual(counts, {"red": 8, "green": 8, "blue": 8})

    def test_setter_routes_follow_colour_bottom_to_top(self):
        tmp = tempfile.mkdtemp()
        p = os.path.join(tmp, "g.jpg"); gym_wall(p)
        data = get_routes("gym-1", os.path.join(tmp, "s"), p, mode="gym")
        self.assertEqual(len(data["routes"]), 3)
        for r in data["routes"]:
            ys = [pt["y"] for pt in r["points"]]
            self.assertEqual(ys, sorted(ys, reverse=True))              # strictly lowest -> highest
            self.assertEqual(r["points"][0]["role"], "start")
            self.assertEqual(r["points"][-1]["role"], "finish")
            self.assertEqual(len(r["points"]), 8)
            self.assertEqual(r["difficulty"], "setter")

    def test_plain_wall_returns_nothing(self):
        plain = np.full((300, 400, 3), 190, np.uint8)
        self.assertEqual(GymColorDetector().detect(plain), [])
        self.assertEqual(GymColorDetector().detect(None), [])

    def test_too_few_holds_per_colour_is_skipped_with_warning(self):
        img = np.full((400, 400, 3), 200, np.uint8)
        for i in range(3):
            cv2.circle(img, (100 + i * 80, 300 - i * 70), 14, COLORS["red"], -1)
        tmp = tempfile.mkdtemp(); p = os.path.join(tmp, "g.jpg"); cv2.imwrite(p, img)
        data = get_routes("few", os.path.join(tmp, "s"), p, mode="gym")
        self.assertTrue(any("only 3 holds" in w for w in data["warnings"]))


if __name__ == "__main__":
    unittest.main()
