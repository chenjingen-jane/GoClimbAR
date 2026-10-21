import unittest

import numpy as np

from goclimb_routes.scan import replan_in_view, transform_doc


def _doc():
    # grid of holds over the whole photo; one saved route that starts at the very bottom
    holds = [{"id": f"H{i}", "x": 0.1 + 0.1 * (i % 9), "y": 0.05 + 0.1 * (i // 9), "confidence": 1.0,
              "type": "feature", "status": "detected"} for i in range(90)]
    pts = [{"hold_id": f"H{81 + k}", "x": 0.1 + 0.1 * k, "y": 0.95 - 0.2 * k} for k in range(5)]
    route = {"id": "R1", "difficulty": "easy", "color": "#22C55E", "points": pts, "steps": [{}] * 5,
             "hold_ids": [p["hold_id"] for p in pts], "edited": False}
    return {"holds": holds, "routes": [route], "warnings": [], "aspect": 0.5}


class ReplanTests(unittest.TestCase):
    def test_routes_replanned_when_saved_ones_leave_view(self):
        H = np.array([[1.0, 0, 0], [0, 1.0, 0.2], [0, 0, 1.0]])     # shift down: route bottom leaves view
        doc = transform_doc(_doc(), H, 1000, 500)
        self.assertEqual(doc["routes"], [])
        replan_in_view(doc, 4)
        self.assertGreater(len(doc["routes"]), 0)
        ids = {h["id"] for h in doc["holds"]}
        for r in doc["routes"]:
            self.assertEqual(len(r["points"]), len(r["steps"]))
            self.assertTrue(all(p["hold_id"] in ids for p in r["points"]))   # only holds in view
        self.assertIn("not saved", doc["warnings"][-1])


if __name__ == "__main__":
    unittest.main()