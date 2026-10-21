import json
import os
import shutil
import tempfile
import unittest

from goclimb_routes.store import DuplicateWallError, Store, WallNotFound
from .fixtures import png_bytes, synthetic_indoor_wall


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.img, _ = synthetic_indoor_wall()
        self.png = png_bytes(self.img)
        self.s = Store(self.root)

    def new(self, name="Wall A", venue="Example Gym", mode="indoor", **kw):
        return self.s.create_wall(name, venue, mode, image_bytes=self.png, **kw)

    def test_save_and_survive_restart(self):
        wid = self.new(vuforia_target="t1")
        s2 = Store(self.root)                                     # "restart"
        w = s2.get_wall(wid)
        self.assertEqual(w["wall_id"], wid)
        self.assertEqual(w["photo_status"], "ok")
        self.assertTrue(os.path.isfile(s2.photo_path(wid)))
        self.assertGreater(len(w["holds"]), 5)
        with open(s2.photo_path(wid), "rb") as f:
            self.assertEqual(f.read(), self.png)                   # original bytes preserved

    def test_relative_storage_reference_and_moved_root(self):
        wid = self.new()
        self.assertFalse(os.path.isabs(self.s.get_wall(wid)["photo_ref"]))
        moved = tempfile.mkdtemp()
        shutil.rmtree(moved)
        shutil.copytree(self.root, moved)
        self.assertEqual(Store(moved).get_wall(wid)["photo_status"], "ok")

    def test_multiple_walls_same_venue_and_both_modes(self):
        a = self.new("Wall A"); b = self.new("Wall B"); c = self.new("Crag 1", "Example Gym", "outdoor")
        self.assertEqual(len({a, b, c}), 3)
        names = {w["name"] for w in self.s.search_walls(venue="example gym")}
        self.assertEqual(names, {"Wall A", "Wall B", "Crag 1"})
        self.assertEqual([w["name"] for w in self.s.search_walls(venue="Example", mode="outdoor")], ["Crag 1"])

    def test_search_by_name_location_and_no_match(self):
        self.new("Wall A", lat=1.3521, lon=103.8198)
        self.new("Wall B", venue="Other Gym", lat=51.5, lon=-0.12)
        self.assertEqual([w["name"] for w in self.s.search_walls(name="wall b")], ["Wall B"])
        near = self.s.search_walls(near=(1.35, 103.82, 10))
        self.assertEqual([w["name"] for w in near], ["Wall A"])
        self.assertEqual(self.s.search_walls(venue="nonexistent"), [])
        self.assertTrue(self.s.search_walls(venue="gym")[0]["photo_available"])

    def test_gps_is_optional(self):
        wid = self.new()
        self.assertIsNone(self.s.get_wall(wid)["lat"])

    def test_duplicate_prevented_not_merged(self):
        self.new("Wall A", "Example Gym")
        with self.assertRaises(DuplicateWallError):
            self.new("wall a", "example gym")
        self.assertEqual(len(self.s.search_walls()), 1)
        self.new("Wall A", "Example Gym", allow_duplicate=True)      # explicit user choice
        self.assertEqual(len(self.s.search_walls()), 2)

    def test_update_keeps_wall_id_and_no_duplicate_record(self):
        wid = self.new()
        self.s.update_wall(wid, name="Wall A (renamed)", lat=1.0, lon=2.0)
        w = self.s.get_wall(wid)
        self.assertEqual(w["name"], "Wall A (renamed)")
        self.assertEqual(len(self.s.search_walls()), 1)
        other = self.new("Wall B")
        with self.assertRaises(DuplicateWallError):
            self.s.update_wall(other, name="Wall A (renamed)")

    def test_corrections_persist_and_ids_never_reused(self):
        wid = self.new()
        before = {h["id"]: h for h in self.s.get_wall(wid)["holds"]}
        first = sorted(before, key=lambda i: int(i[1:]))[0]
        second = sorted(before, key=lambda i: int(i[1:]))[1]
        self.s.move_hold(wid, first, 0.123, 0.456)
        self.s.delete_hold(wid, second)
        new_id = self.s.add_hold(wid, 0.5, 0.5)
        s2 = Store(self.root)
        h = {x["id"]: x for x in s2.get_wall(wid)["holds"]}
        self.assertEqual(h[first]["status"], "moved"); self.assertAlmostEqual(h[first]["x"], 0.123)
        self.assertEqual(h[second]["status"], "deleted")
        self.assertEqual(h[new_id]["status"], "manual_added")
        self.assertNotIn(new_id, before)
        self.assertNotIn(second, [x.id for x in s2.active_holds(wid)])
        self.s.delete_hold(wid, new_id)                           # manual hold removed entirely
        self.assertEqual(self.s.add_hold(wid, 0.4, 0.4), f"H{int(new_id[1:]) + 1}")

    def test_redetect_preserves_corrections_and_adds_only(self):
        wid = self.new()
        n = len(self.s.get_wall(wid)["holds"])
        self.s.delete_hold(wid, "H1")
        self.s.move_hold(wid, "H2", 0.9, 0.9)
        added = self.s.redetect(wid)
        w = self.s.get_wall(wid)
        st = {h["id"]: h["status"] for h in w["holds"]}
        self.assertEqual(st["H1"], "deleted")                      # not resurrected
        self.assertEqual(st["H2"], "moved")
        self.assertEqual(len(w["holds"]), n + added)

    def test_invalid_hold_coordinates_rejected(self):
        wid = self.new()
        for xy in ((1.5, 0.5), (-0.1, 0.5), (float("nan"), 0.5)):
            with self.assertRaises(ValueError):
                self.s.add_hold(wid, *xy)

    def test_regenerate_routes_from_saved_holds_and_keep_edited(self):
        wid = self.new()
        rep = self.s.generate_routes_for(wid, max_routes=5)
        w = self.s.get_wall(wid)
        self.assertEqual(rep["retained"], len(w["routes"]))
        self.assertLessEqual(len(w["routes"]), 5)
        self.assertTrue(os.path.isfile(os.path.join(self.root, "walls", wid, "overlay.png")))   # separate overlay
        with open(self.s.photo_path(wid), "rb") as f:
            self.assertEqual(f.read(), self.png)                                               # original untouched
        if w["routes"]:
            r0 = w["routes"][0]
            self.s.set_route_sequence(wid, r0["id"], r0["hold_ids"][:])
            self.s.generate_routes_for(wid, max_routes=3)
            ids = [r["id"] for r in self.s.get_wall(wid)["routes"]]
            self.assertIn(r0["id"], ids)                                                       # edited route survives

    def test_sparse_holds_relax_reach_with_a_visible_warning_and_never_invent(self):
        import numpy as np
        from .fixtures import png_bytes
        flat = png_bytes(np.full((600, 400, 3), 170, np.uint8))                # portrait photo, nothing detectable
        wid = self.s.create_wall("Sparse", "Gym", "indoor", image_bytes=flat)
        self.assertEqual(self.s.generate_routes_for(wid)["retained"], 0)       # no holds: nothing invented
        for i in range(5):
            self.s.add_hold(wid, 0.5, 0.06 + 0.22 * i)                          # far apart for a portrait wall
        rep = self.s.generate_routes_for(wid)
        self.assertEqual(rep["retained"], 1)
        self.assertTrue(any("relaxed" in w for w in rep["warnings"]))
        far = [self.s.add_hold(wid, 0.9, 0.5 + 0.01 * i) for i in range(3)]    # unrelated stragglers change nothing
        self.assertGreaterEqual(self.s.generate_routes_for(wid)["retained"], 1)

    def test_limb_assignment_confirmed_vs_estimated_persists(self):
        wid = self.new()
        self.s.generate_routes_for(wid)
        r = self.s.get_wall(wid)["routes"][0]
        hid = r["hold_ids"][1]
        self.s.set_limb(wid, r["id"], hid, "left_foot")
        p = {x["hold_id"]: x for x in Store(self.root).get_wall(wid)["routes"][0]["points"]}
        self.assertEqual((p[hid]["limb"], p[hid]["limb_status"]), ("left_foot", "confirmed"))
        self.assertEqual(p[r["hold_ids"][0]]["limb_status"], "estimated")
        with self.assertRaises(ValueError):
            self.s.set_limb(wid, r["id"], hid, "elbow")
        self.s.set_limb(wid, r["id"], hid, None)
        p = {x["hold_id"]: x for x in self.s.get_wall(wid)["routes"][0]["points"]}
        self.assertEqual(p[hid]["limb_status"], "estimated")

    def test_route_using_deleted_hold_becomes_invalid_not_exported(self):
        wid = self.new()
        self.s.generate_routes_for(wid)
        r = self.s.get_wall(wid)["routes"][0]
        self.s.delete_hold(wid, r["hold_ids"][2])
        r2 = [x for x in self.s.get_wall(wid)["routes"] if x["id"] == r["id"]][0]
        self.assertFalse(r2["valid"])
        out = tempfile.mkdtemp()
        doc = load(self.s.export_unity(wid, out))
        self.assertNotIn(r["id"], [x["id"] for x in doc["routes"]])
        self.assertTrue(doc["warnings"])

    def test_delete_policy(self):
        wid = self.new()
        with self.assertRaises(ValueError):
            self.s.delete_wall(wid)                                # needs confirm
        self.s.delete_wall(wid, confirm=True)
        self.assertFalse(os.path.exists(os.path.join(self.root, "walls", wid)))
        with self.assertRaises(WallNotFound):
            self.s.get_wall(wid)
        with self.assertRaises(WallNotFound):
            self.s.delete_wall(wid, confirm=True)

    def test_missing_photo_and_corrupt_route_handled(self):
        wid = self.new()
        self.s.generate_routes_for(wid)
        with self.s._conn() as c:
            c.execute("INSERT INTO routes(wall_id,route_id,data) VALUES(?,?,?)", (wid, "R99", "{not json"))
        w = self.s.get_wall(wid)
        self.assertIn(False, [r["valid"] for r in w["routes"]])
        os.remove(self.s.photo_path(wid))
        w = Store(self.root).get_wall(wid)
        self.assertEqual(w["photo_status"], "missing")
        self.assertFalse(self.s.search_walls()[0]["photo_available"])
        with self.assertRaises(FileNotFoundError):
            self.s.redetect(wid)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            self.s.create_wall("", "V", "indoor", image_bytes=self.png)
        with self.assertRaises(ValueError):
            self.s.create_wall("A", "V", "space", image_bytes=self.png)
        with self.assertRaises(ValueError):
            self.s.create_wall("A", "V", "indoor", image_bytes=b"not an image")
        with self.assertRaises(ValueError):
            self.s.create_wall("A", "V", "indoor", image_bytes=self.png, lat=5.0)
        self.assertEqual(self.s.search_walls(), [])                # nothing half-created
        self.assertEqual(os.listdir(os.path.join(self.root, "walls")), [])

    def test_export_keeps_walls_separate_and_maps_targets(self):
        a = self.new("Wall A", vuforia_target="tgtA"); b = self.new("Wall B", vuforia_target="tgtB")
        for w in (a, b):
            self.s.generate_routes_for(w)
        out = tempfile.mkdtemp()
        pa, pb = self.s.export_unity(a, out), self.s.export_unity(b, out)
        da, db = load(pa), load(pb)
        self.assertEqual((da["wall_id"], db["wall_id"]), (a, b))
        self.assertNotEqual(pa, pb)
        idx = {t["target"]: t["wall_id"] for t in load(os.path.join(out, "index.json"))["targets"]}
        self.assertEqual(idx, {"tgtA": a, "tgtB": b})
        self.s.export_unity(a, out)                                # re-export: no duplicate mapping
        self.assertEqual(len(load(os.path.join(out, "index.json"))["targets"]), 2)
        for d in (da, db):
            self.assertEqual(d["schema"], 2)
            for r in d["routes"]:
                self.assertTrue(all(0 <= p["x"] <= 1 and 0 <= p["y"] <= 1 for p in r["points"]))

    def test_detection_report_from_corrections(self):
        wid = self.new()
        n = len(self.s.get_wall(wid)["holds"])
        self.s.delete_hold(wid, "H1"); self.s.add_hold(wid, 0.05, 0.05)
        rep = self.s.detection_report(wid)
        self.assertEqual((rep["auto_detected"], rep["false_detections"], rep["missed_added_by_user"]), (n, 1, 1))
        self.assertAlmostEqual(rep["precision"], round((n - 1) / n, 3))


if __name__ == "__main__":
    unittest.main()
