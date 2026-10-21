import math
import unittest

import numpy as np

from goclimb_routes.detect import Hold, validate_holds
from goclimb_routes.planner import MIN_HOLDS_PER_ROUTE, PROFILES, plan_routes

ASPECT = 0.664


def grid_holds(cols=7, rows=12):
    """Deterministic jittered grid (test fixture only; the app never uses randomness)."""
    rs = np.random.RandomState(3)
    out = []
    for r in range(rows):
        for c in range(cols):
            x = 0.08 + c * (0.84 / (cols - 1)) + rs.uniform(-0.01, 0.01)
            y = 0.05 + r * (0.90 / (rows - 1)) + rs.uniform(-0.01, 0.01)
            out.append(Hold(id=len(out), x=float(x), y=float(y), confidence=0.8))
    return out


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.routes, self.warnings = plan_routes(grid_holds(), ASPECT, alternatives=2)

    def test_routes_exist_for_all_difficulties(self):
        self.assertEqual({r["difficulty"] for r in self.routes}, {"easy", "medium", "hard"})

    def test_connectivity_upward_and_reach_limit(self):
        for r in self.routes:
            prof = PROFILES[r["difficulty"]]
            pts = r["points"]
            self.assertGreaterEqual(len(pts), MIN_HOLDS_PER_ROUTE)
            for a, b in zip(pts, pts[1:]):
                dy = (a["y"] - b["y"]) * ASPECT      # image y is down, so higher = smaller y
                d = math.hypot(b["x"] - a["x"], dy)
                self.assertGreater(dy, 0, "every move must go upward")
                self.assertLessEqual(d, prof.max_reach + 1e-9)

    def test_start_low_finish_high_and_roles(self):
        for r in self.routes:
            self.assertEqual(r["points"][0]["role"], "start")
            self.assertEqual(r["points"][-1]["role"], "finish")
            self.assertGreater(r["points"][0]["y"], 0.6)    # lower part of image
            self.assertLess(r["points"][-1]["y"], 0.3)      # upper part

    def test_points_are_actual_holds_and_in_bounds(self):
        holds = grid_holds()
        known = {(round(h.x, 6), round(h.y, 6)) for h in holds}
        for r in self.routes:
            for p in r["points"]:
                self.assertTrue(0.0 <= p["x"] <= 1.0 and 0.0 <= p["y"] <= 1.0)
                self.assertIn((round(p["x"], 6), round(p["y"], 6)), known)

    def test_alternatives_differ(self):
        by = {}
        for r in self.routes:
            by.setdefault(r["difficulty"], []).append(tuple(r["hold_ids"]))
        for ids in by.values():
            if len(ids) == 2:
                self.assertNotEqual(ids[0], ids[1])

    def test_insufficient_holds(self):
        routes, warnings = plan_routes(grid_holds()[:3], ASPECT)
        self.assertEqual(routes, [])
        self.assertTrue(any("insufficient" in w for w in warnings))

    def test_invalid_input_is_handled(self):
        bad = [Hold(0, float("nan"), 0.5, 1), Hold(1, 1.7, 0.2, 1), Hold(2, -0.1, 0.2, 1)]
        clean, warnings = validate_holds(bad)
        self.assertEqual(clean, [])
        self.assertEqual(len(warnings), 3)
        routes, _ = plan_routes(bad, ASPECT)
        self.assertEqual(routes, [])
        self.assertEqual(plan_routes(grid_holds(), 0)[0], [])

    def test_sparse_holds_give_warning_not_crash(self):
        sparse = [Hold(i, 0.5, 0.06 + 0.22 * i, 1.0) for i in range(5)]   # 0.146 apart in wall widths: too far for easy
        routes, warnings = plan_routes(sparse, ASPECT)
        self.assertTrue(any("no valid easy" in w for w in warnings))


if __name__ == "__main__":
    unittest.main()
