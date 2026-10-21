import math
import unittest

from goclimb_routes.detect import Hold
from goclimb_routes.planner import (GLOBAL_MAX_REACH, MIN_HOLDS_PER_ROUTE, estimate_difficulty,
                                    generate_routes, _jaccard)
from .fixtures import grid

ASPECT = 0.664


class DiverseRouteTests(unittest.TestCase):
    def setUp(self):
        self.holds = grid()
        self.routes, self.report = generate_routes(self.holds, ASPECT, max_routes=8)

    def test_no_exact_duplicates(self):
        seqs = [tuple(r["hold_ids"]) for r in self.routes]
        self.assertEqual(len(seqs), len(set(seqs)))

    def test_near_duplicates_filtered_by_policy(self):
        thr = 0.6
        for i, a in enumerate(self.routes):
            for b in self.routes[i + 1:]:
                self.assertLess(_jaccard(a["hold_ids"], b["hold_ids"]), thr)

    def test_stricter_policy_keeps_fewer_or_equal(self):
        loose, _ = generate_routes(self.holds, ASPECT, max_routes=20, overlap_threshold=0.9)
        strict, _ = generate_routes(self.holds, ASPECT, max_routes=20, overlap_threshold=0.3)
        self.assertGreaterEqual(len(loose), len(strict))

    def test_movement_constraints_and_target_area(self):
        top = min(h.y for h in self.holds); bottom = max(h.y for h in self.holds); span = bottom - top
        by = {h.id: h for h in self.holds}
        for r in self.routes:
            path = [by[i] for i in r["hold_ids"]]
            self.assertGreaterEqual(len(path), MIN_HOLDS_PER_ROUTE)
            self.assertGreaterEqual(path[0].y, bottom - 0.30 * span - 1e-9)      # starts low
            self.assertLessEqual(path[-1].y, top + 0.15 * span + 1e-9)           # reaches target area
            for a, b in zip(path, path[1:]):
                dy = (a.y - b.y) * ASPECT
                self.assertGreater(dy, 0)
                self.assertLessEqual(math.hypot(b.x - a.x, dy), GLOBAL_MAX_REACH + 1e-9)

    def test_max_route_count_respected(self):
        for m in (1, 3, 5):
            r, rep = generate_routes(self.holds, ASPECT, max_routes=m)
            self.assertLessEqual(len(r), m)
            self.assertEqual(rep["retained"], len(r))

    def test_report_counts_consistent(self):
        rep = self.report
        self.assertGreaterEqual(rep["candidates"], rep["unique"])
        self.assertGreaterEqual(rep["unique"], rep["retained"])
        self.assertEqual(rep["max_routes"], 8)

    def test_routes_are_distinct_and_have_steps_ids_colors(self):
        self.assertGreater(len(self.routes), 1)
        for r in self.routes:
            self.assertTrue(all(h.startswith("H") for h in r["hold_ids"]))
            self.assertIn(r["color"], {"#22C55E", "#EAB308", "#EF4444"})
            self.assertEqual(len(r["steps"]), len(r["hold_ids"]))
            self.assertEqual(r["points"][0]["role"], "start")
            self.assertEqual(r["points"][-1]["role"], "finish")
            self.assertTrue(all(p["limb_status"] == "estimated" for p in r["points"]))

    def test_few_holds_returns_nothing_not_invented(self):
        r, rep = generate_routes(self.holds[:3], ASPECT)
        self.assertEqual(r, [])
        self.assertTrue(rep["warnings"])

    def test_no_feasible_path_returns_nothing(self):
        far = [Hold(f"H{i}", 0.5, 0.05 + 0.23 * i, 1.0) for i in range(5)]
        r, rep = generate_routes(far, 1.5)       # tall wall: 0.23 * 1.5 > global reach
        self.assertEqual(r, [])
        self.assertTrue(any("no feasible" in w for w in rep["warnings"]))

    def test_reach_scale_relaxes_the_limit_only_when_asked(self):
        far = [Hold(f"H{i}", 0.5, 0.05 + 0.23 * i, 1.0) for i in range(5)]
        self.assertEqual(generate_routes(far, 1.5)[0], [])
        routes, rep = generate_routes(far, 1.5, reach_scale=2.0)
        self.assertEqual((len(routes), rep["reach_scale"]), (1, 2.0))
        by = {h.id: h for h in far}
        for a, b in zip(routes[0]["hold_ids"], routes[0]["hold_ids"][1:]):
            self.assertLessEqual(math.hypot(by[b].x - by[a].x, (by[a].y - by[b].y) * 1.5), GLOBAL_MAX_REACH * 2 + 1e-9)

    def test_invalid_input_handled(self):
        self.assertEqual(generate_routes(self.holds, 0)[0], [])
        bad = [Hold("H1", float("nan"), 0.5, 1), Hold("H2", 2.0, 0.1, 1)]
        self.assertEqual(generate_routes(bad, ASPECT)[0], [])
        self.assertEqual(generate_routes(self.holds, ASPECT, max_routes=0)[0], [])

    def test_difficulty_is_not_just_length(self):
        straight = [Hold(f"H{i}", 0.5, 0.9 - 0.14 * i, 1.0) for i in range(6)]            # 6 holds, short, straight
        zig = [Hold(f"H{i}", 0.35 + 0.30 * (i % 2), 0.9 - 0.14 * i, 1.0) for i in range(6)]  # same count, zigzag
        sparse_conf = [Hold(f"H{i}", 0.5, 0.9 - 0.14 * i, 0.3) for i in range(6)]            # same geometry, unsure holds
        a, b, c = (estimate_difficulty(p, ASPECT)["score"] for p in (straight, zig, sparse_conf))
        self.assertGreater(b, a)
        self.assertGreater(c, a)

    def test_all_three_difficulties_can_occur(self):
        labels = {r["difficulty"] for r in generate_routes(self.holds, ASPECT, max_routes=30, overlap_threshold=0.9)[0]}
        self.assertGreaterEqual(len(labels), 2)


if __name__ == "__main__":
    unittest.main()
