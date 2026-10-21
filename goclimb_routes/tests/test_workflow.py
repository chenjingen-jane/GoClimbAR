import json
import os
import tempfile
import unittest

import cv2
import numpy as np

from goclimb_routes.detect import ClassicalDetector, holds_from_dicts
from goclimb_routes.workflow import get_routes, saved_path
from .test_planner import grid_holds


def synthetic_wall(path):
    """Blue sky strip, green grass strip, brown rock with dark pockets."""
    img = np.zeros((400, 600, 3), np.uint8)
    img[:] = (60, 90, 120)                       # brown-ish rock (BGR)
    img[:60] = (235, 160, 60)                    # sky
    img[340:] = (50, 160, 60)                    # grass
    rs = np.random.RandomState(1)
    for _ in range(40):
        cv2.circle(img, (int(rs.uniform(40, 560)), int(rs.uniform(90, 320))), 7, (15, 25, 35), -1)
    cv2.imwrite(path, img)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.img = os.path.join(self.tmp, "wall.jpg")
        synthetic_wall(self.img)
        self.store = os.path.join(self.tmp, "store")
        self.holds = [{"x": h.x, "y": h.y, "confidence": h.confidence} for h in grid_holds()]

    def test_unknown_wall_generates_without_saved_record(self):
        data = get_routes("never-seen", self.store, self.img, holds_override=self.holds)
        self.assertEqual(data["source"], "generated")
        self.assertGreater(len(data["routes"]), 0)
        self.assertFalse(os.path.exists(saved_path(self.store, "never-seen")))   # not saved w/o approval

    def test_save_then_known_wall_loads_saved(self):
        get_routes("wall-a", self.store, self.img, holds_override=self.holds, save=True)
        self.assertTrue(os.path.exists(saved_path(self.store, "wall-a")))
        data = get_routes("wall-a", self.store)          # no image needed any more
        self.assertEqual(data["source"], "saved")

    def test_unknown_wall_without_image_raises(self):
        with self.assertRaises(ValueError):
            get_routes("nope", self.store)

    def test_unreadable_image_raises(self):
        with self.assertRaises(ValueError):
            get_routes("nope", self.store, os.path.join(self.tmp, "missing.jpg"))

    def test_detector_stays_inside_rock_and_in_bounds(self):
        img = cv2.imread(self.img)
        for h in ClassicalDetector().detect(img):
            self.assertTrue(0 <= h.x <= 1 and 0 <= h.y <= 1)
            self.assertTrue(0.15 <= h.y <= 0.85, "must not place holds in sky or grass")

    def test_detector_returns_empty_on_blank_or_none(self):
        self.assertEqual(ClassicalDetector().detect(None), [])
        sky = np.full((300, 400, 3), (235, 160, 60), np.uint8)
        self.assertEqual(ClassicalDetector().detect(sky), [])

    def test_json_is_unity_friendly(self):
        data = get_routes("w", self.store, self.img, holds_override=self.holds)
        json.dumps(data)
        for r in data["routes"]:
            self.assertTrue(r["color"].startswith("#"))


if __name__ == "__main__":
    unittest.main()
