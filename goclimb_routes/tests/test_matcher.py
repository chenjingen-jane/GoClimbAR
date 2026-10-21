import tempfile
import unittest

import cv2
import numpy as np

from goclimb_routes.matcher import (MatchResult, apply_h, classify, compute_features, find_matches,
                                    match_features)
from goclimb_routes.store import Store
from .fixtures import png_bytes, synthetic_indoor_wall


def persp(img, dst_corners):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, np.float32(dst_corners))
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT), M


class MatcherTests(unittest.TestCase):
    def setUp(self):
        self.img, _ = synthetic_indoor_wall(seed=2)
        self.other, _ = synthetic_indoor_wall(seed=7)
        self.ref = compute_features(self.img)

    def level(self, scan):
        q = compute_features(scan)
        return match_features(q, self.ref).level if q else "no-features"

    def test_same_image_and_grayscale(self):
        self.assertEqual(self.level(self.img), "confident")
        gray = cv2.cvtColor(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        self.assertEqual(self.level(gray), "confident")

    def test_different_scale_lighting_compression(self):
        small = cv2.resize(self.img, None, fx=0.7, fy=0.7)
        dark = np.clip(self.img.astype(np.float32) * 0.75, 0, 255).astype(np.uint8)
        jpg = cv2.imdecode(cv2.imencode(".jpg", self.img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1], 1)
        for name, scan in (("scale", small), ("lighting", dark), ("jpeg", jpg)):
            self.assertEqual(self.level(scan), "confident", name)
        # All three stacked on this noise-textured SYNTHETIC wall may drop to 'possible' (the app then asks the
        # user). It must never be rejected outright. (The real crag photo stays 'confident' with 640 inliers.)
        stacked = cv2.imdecode(cv2.imencode(".jpg", np.clip(small.astype(np.float32) * 0.75, 0, 255).astype(np.uint8),
                                            [cv2.IMWRITE_JPEG_QUALITY, 40])[1], 1)
        self.assertIn(self.level(stacked), ("possible", "confident"))

    def test_perspective_mapping_is_accurate(self):
        h, w = self.img.shape[:2]
        corners = [[30, 20], [w - 15, 5], [w - 35, h - 25], [10, h - 5]]
        warped, M = persp(self.img, corners)
        m = match_features(compute_features(warped), self.ref)
        self.assertEqual(m.level, "confident")
        N = np.diag([1.0 / w, 1.0 / h, 1.0])
        expected = N @ M @ np.linalg.inv(N)
        for x, y in [(0.3, 0.3), (0.5, 0.5), (0.7, 0.8), (0.2, 0.7)]:
            ex = apply_h(expected, x, y)
            got = apply_h(m.H, x, y)
            self.assertLess(max(abs(ex[0] - got[0]), abs(ex[1] - got[1])), 0.02, (x, y, ex, got))

    def test_different_wall_and_mirror_do_not_match(self):
        self.assertEqual(self.level(self.other), "none")
        self.assertNotEqual(self.level(cv2.flip(self.img, 1)), "confident")    # mirrored image is NOT the same view

    def test_blank_and_tiny_images(self):
        self.assertIsNone(compute_features(None))
        self.assertIsNone(compute_features(np.full((300, 400, 3), 128, np.uint8)))

    def test_thresholds(self):
        self.assertEqual(classify(60, 0.8), "confident")
        self.assertEqual(classify(20, 0.3), "possible")
        self.assertEqual(classify(60, 0.2), "none")
        self.assertEqual(classify(10, 0.9), "none")

    def test_find_matches_picks_the_right_saved_wall(self):
        st = Store(tempfile.mkdtemp())
        a = st.create_wall("A", "Gym", "indoor", image_bytes=png_bytes(self.img))
        st.create_wall("B", "Gym", "indoor", image_bytes=png_bytes(self.other))
        res = find_matches(st, cv2.resize(self.img, None, fx=0.8, fy=0.8))
        self.assertEqual([m.wall_id for m in res], [a])
        self.assertEqual(res[0].level, "confident")
        self.assertEqual(find_matches(st, np.full((300, 400, 3), 128, np.uint8)), [])
        self.assertTrue(res[0].venue == "Gym" and res[0].name == "A")


if __name__ == "__main__":
    unittest.main()
