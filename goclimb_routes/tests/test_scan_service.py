import base64
import tempfile
import time
import unittest

import cv2
import numpy as np

from goclimb_routes.matcher import MatchResult
from goclimb_routes.scan import ScanExpired, ScanService, transform_doc
from goclimb_routes.store import DuplicateWallError, Store, WallNotFound
from .fixtures import png_bytes, synthetic_indoor_wall


def b64(img):
    return base64.b64encode(png_bytes(img)).decode()


class ScanServiceTests(unittest.TestCase):
    def setUp(self):
        self.store = Store(tempfile.mkdtemp())
        self.svc = ScanService(self.store)
        self.img, _ = synthetic_indoor_wall(seed=2)
        self.other, _ = synthetic_indoor_wall(seed=7)

    def scan(self, img=None, **kw):
        return self.svc.scan({"image_b64": b64(self.img if img is None else img), "mode": "indoor", **kw})

    # ---------------------------------------------------------------- unknown walls
    def test_unknown_wall_gets_routes_and_saves_nothing(self):
        r = self.scan()
        self.assertEqual(r["status"], "unknown")
        self.assertEqual(r["wall_id"], "")
        d = r["doc"]
        self.assertEqual((d["scan_id"], d["coordinate_frame"], d["schema"]), (r["scan_id"], "scan_photo", 2))
        self.assertEqual((d["image_width"], d["image_height"]), (self.img.shape[1], self.img.shape[0]))
        self.assertGreater(len(d["routes"]), 0)
        self.assertEqual(self.store.search_walls(), [])             # unsaved scans never create records

    def test_repeated_scans_of_unsaved_wall_create_no_records(self):
        ids = {self.scan()["scan_id"] for _ in range(4)}
        self.assertEqual(len(ids), 4)                               # each scan has its own session id
        self.assertEqual(self.store.search_walls(), [])

    def test_save_creates_exactly_one_record_even_if_pressed_twice(self):
        r = self.scan()
        s1 = self.svc.save(r["scan_id"], {"venue": "My Gym", "name": "Wall 1"})
        s2 = self.svc.save(r["scan_id"], {"venue": "My Gym", "name": "Wall 1"})
        self.assertTrue(s1["created"]); self.assertFalse(s2["created"])
        self.assertEqual(s1["wall_id"], s2["wall_id"])
        self.assertEqual(len(self.store.search_walls()), 1)

    def test_saved_ids_match_what_was_shown_in_the_preview(self):
        r = self.scan()
        shown = {h["id"]: (h["x"], h["y"]) for h in r["doc"]["holds"]}
        wid = self.svc.save(r["scan_id"], {"venue": "V", "name": "N"})["wall_id"]
        saved = {h["id"]: (h["x"], h["y"]) for h in self.store.get_wall(wid)["holds"] if h["status"] != "deleted"}
        self.assertEqual(shown, saved)                              # same hold IDs and positions

    def test_duplicate_name_is_refused_not_merged(self):
        self.svc.save(self.scan()["scan_id"], {"venue": "V", "name": "N"})
        r = self.scan(self.other)
        with self.assertRaises(DuplicateWallError):
            self.svc.save(r["scan_id"], {"venue": "V", "name": "N"})
        self.assertEqual(len(self.store.search_walls()), 1)

    # ---------------------------------------------------------------- known walls
    def saved_wall(self):
        r = self.scan()
        return self.svc.save(r["scan_id"], {"venue": "V", "name": "Known"})["wall_id"]

    def test_rescan_reuses_record_and_never_duplicates(self):
        wid = self.saved_wall()
        route_ids = [x["id"] for x in self.store.get_wall(wid)["routes"]]
        for _ in range(5):
            r = self.scan()
            self.assertEqual((r["status"], r["wall_id"]), ("known", wid))
            self.assertEqual(len(r["doc"]["routes"]), len(route_ids))      # no duplicated routes
        self.assertEqual(len(self.store.search_walls()), 1)
        self.assertEqual([x["id"] for x in self.store.get_wall(wid)["routes"]], route_ids)  # store untouched

    def test_known_scan_from_new_angle_is_mapped_into_the_scan_frame(self):
        wid = self.saved_wall()
        h, w = self.img.shape[:2]
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[40, 25], [w - 20, 8], [w - 45, h - 30], [12, h - 6]])
        M = cv2.getPerspectiveTransform(src, dst)
        scan_img = cv2.warpPerspective(self.img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        r = self.scan(scan_img)
        self.assertEqual((r["status"], r["wall_id"]), ("known", wid))
        stored = {x["id"]: x for x in self.store.get_wall(wid)["holds"]}
        checked = 0
        for hd in r["doc"]["holds"]:
            sx, sy = stored[hd["id"]]["x"] * w, stored[hd["id"]]["y"] * h
            v = M @ np.array([sx, sy, 1.0])
            ex, ey = v[0] / v[2] / w, v[1] / v[2] / h
            self.assertLess(max(abs(hd["x"] - ex), abs(hd["y"] - ey)), 0.02)
            checked += 1
        self.assertGreater(checked, 10)
        for rt in r["doc"]["routes"]:
            for p in rt["points"]:
                self.assertTrue(0 <= p["x"] <= 1 and 0 <= p["y"] <= 1)

    def test_regenerate_known_wall_changes_routes_not_records(self):
        wid = self.saved_wall()
        r = self.scan()
        out = self.svc.regenerate(r["scan_id"], max_routes=2)
        self.assertLessEqual(len(out["doc"]["routes"]), 2)
        self.assertEqual(out["doc"]["wall_id"], wid)
        self.assertEqual(len(self.store.search_walls()), 1)

    def test_switching_walls_keeps_data_separate(self):
        a = self.saved_wall()
        b = self.svc.save(self.scan(self.other)["scan_id"], {"venue": "V", "name": "Other"})["wall_id"]
        ra, rb = self.scan(self.img), self.scan(self.other)
        self.assertEqual((ra["wall_id"], rb["wall_id"]), (a, b))
        self.assertNotEqual(a, b)
        self.assertNotEqual({h["id"]: (h["x"], h["y"]) for h in ra["doc"]["holds"]},
                            {h["id"]: (h["x"], h["y"]) for h in rb["doc"]["holds"]})

    def test_transform_drops_routes_that_leave_the_view(self):
        wid = self.saved_wall()
        doc = self.store.export_doc(wid)
        H = np.array([[1, 0, 0.9], [0, 1, 0], [0, 0, 1]], float)   # shifts everything right by 90%
        out = transform_doc(doc, H, 800, 600)
        self.assertEqual(out["routes"], [])
        self.assertTrue(any("leaves the camera view" in w for w in out["warnings"]))

    # ---------------------------------------------------------------- uncertain matches
    def test_uncertain_match_asks_user_and_never_auto_merges(self):
        wid = self.saved_wall()
        fake = lambda store, img: [MatchResult(wall_id=wid, name="Known", venue="V", inliers=20, ratio=0.3, level="possible")]
        svc = ScanService(self.store, matcher=fake)
        r = svc.scan({"image_b64": b64(self.other), "mode": "indoor"})
        self.assertEqual(r["status"], "uncertain")
        self.assertEqual([c["wall_id"] for c in r["candidates"]], [wid])
        self.assertEqual(r["wall_id"], "")
        self.assertGreater(len(r["doc"]["routes"]), 0)
        self.assertEqual(len(self.store.search_walls()), 1)                 # nothing merged or created
        out = svc.save(r["scan_id"], {"use_existing_wall_id": wid})         # user says: same wall
        self.assertEqual((out["wall_id"], out["created"]), (wid, False))
        self.assertEqual(len(self.store.search_walls()), 1)
        r2 = svc.scan({"image_b64": b64(self.other), "mode": "indoor"})      # user says: different wall
        new = svc.save(r2["scan_id"], {"venue": "V", "name": "Really new"})
        self.assertTrue(new["created"]); self.assertEqual(len(self.store.search_walls()), 2)
        with self.assertRaises(WallNotFound):
            svc.save(svc.scan({"image_b64": b64(self.other), "mode": "indoor"})["scan_id"], {"use_existing_wall_id": "nope"})

    # ---------------------------------------------------------------- sessions and input
    def test_stale_scan_cannot_be_saved_or_regenerated(self):
        svc = ScanService(self.store, max_sessions=2)
        first = svc.scan({"image_b64": b64(self.img), "mode": "indoor"})["scan_id"]
        for _ in range(3):
            svc.scan({"image_b64": b64(self.other), "mode": "indoor"})
        with self.assertRaises(ScanExpired):
            svc.save(first, {})
        with self.assertRaises(ScanExpired):
            svc.regenerate(first)
        self.assertEqual(self.store.search_walls(), [])

    def test_ttl_expiry(self):
        svc = ScanService(self.store, ttl_seconds=0)
        sid = svc.scan({"image_b64": b64(self.img), "mode": "indoor"})["scan_id"]
        time.sleep(0.01)
        with self.assertRaises(ScanExpired):
            svc.save(sid, {})

    def test_bad_input_raises_value_error(self):
        for body in ({}, {"image_b64": "@@@"}, {"image_b64": base64.b64encode(b"junk").decode()},
                     {"image_b64": b64(self.img), "mode": "moon"}):
            with self.assertRaises(ValueError):
                self.svc.scan(body)
        flat = np.full((400, 600, 3), 170, np.uint8)
        r = self.svc.scan({"image_b64": b64(flat), "mode": "indoor"})        # nothing detectable
        self.assertEqual(r["doc"]["routes"], [])
        self.assertTrue(r["doc"]["warnings"] is not None)


if __name__ == "__main__":
    unittest.main()
