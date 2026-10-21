"""API 4: phone uploads (with holds detected offline), discarding a scan, admin route editing, review queue,
admin token, and upgrading an old database."""
import base64
import json
import os
import sqlite3
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from goclimb_routes.scan import ScanExpired, ScanService
from goclimb_routes.server import make_server
from goclimb_routes.store import Store
from .fixtures import png_bytes, synthetic_indoor_wall

PNG = png_bytes(synthetic_indoor_wall()[0])
B64 = base64.b64encode(PNG).decode()
PHONE_HOLDS = [{"x": 0.1 + 0.15 * (i % 6), "y": 0.08 + 0.085 * (i // 6), "confidence": 0.8, "type": "candidate"}
               for i in range(66)]


class StoreApi4Tests(unittest.TestCase):
    def setUp(self):
        self.st = Store(tempfile.mkdtemp())

    def test_upload_with_phone_holds_keeps_them_and_flags_review(self):
        wid = self.st.create_wall("W", "Gym", "indoor", image_bytes=PNG, holds=PHONE_HOLDS, source="app")
        holds = self.st.active_holds(wid)
        self.assertEqual(len(holds), len(PHONE_HOLDS))                      # detector NOT run
        self.assertAlmostEqual(holds[0].x, PHONE_HOLDS[0]["x"])
        self.assertEqual([h.id for h in holds[:3]], ["H1", "H2", "H3"])
        w = self.st.get_wall(wid)
        self.assertEqual((w["source"], w["reviewed"]), ("app", 0))
        self.assertEqual([r["wall_id"] for r in self.st.search_walls(needs_review=True)], [wid])
        self.st.set_reviewed(wid, True)
        self.assertEqual(self.st.search_walls(needs_review=True), [])

    def test_bad_phone_holds_rejected_and_nothing_written(self):
        for bad in ([{"x": 2, "y": 0.5}], [{"y": 0.5}], "nope", [{"x": 0.5, "y": 0.5, "confidence": 7}]):
            with self.assertRaises(ValueError):
                self.st.create_wall("W", "Gym", "indoor", image_bytes=PNG, holds=bad, source="app")
        self.assertEqual(self.st.search_walls(), [])
        self.assertEqual(os.listdir(os.path.join(self.st.root, "walls")), [])

    def test_web_walls_are_reviewed_and_search_matches_venue_or_name(self):
        wid = self.st.create_wall("Slab", "Boulder Hub", "indoor", image_bytes=PNG)
        self.assertEqual(self.st.get_wall(wid)["reviewed"], 1)
        self.assertEqual(len(self.st.search_walls(text="slab")), 1)
        self.assertEqual(len(self.st.search_walls(text="hub")), 1)
        self.assertEqual(len(self.st.search_walls(text="zzz")), 0)

    def test_manual_route_survives_regeneration_and_can_be_deleted(self):
        wid = self.st.create_wall("W", "Gym", "indoor", image_bytes=PNG, holds=PHONE_HOLDS, source="app")
        rid = self.st.add_route(wid, ["H61", "H50", "H38", "H26", "H14", "H2"])
        self.st.generate_routes_for(wid, 3)
        routes = {r["id"]: r for r in self.st.get_wall(wid)["routes"]}
        self.assertIn(rid, routes)
        self.assertTrue(routes[rid]["edited"])
        self.assertEqual(routes[rid]["hold_ids"], ["H61", "H50", "H38", "H26", "H14", "H2"])
        self.assertIn(rid, [r["id"] for r in self.st.export_doc(wid)["routes"]])
        with self.assertRaises(ValueError):
            self.st.add_route(wid, ["H1"])
        with self.assertRaises(ValueError):
            self.st.add_route(wid, ["H1", "H999"])
        self.st.delete_route(wid, rid)
        self.assertNotIn(rid, [r["id"] for r in self.st.get_wall(wid)["routes"]])
        with self.assertRaises(KeyError):
            self.st.delete_route(wid, rid)

    def test_old_database_is_upgraded(self):
        root = tempfile.mkdtemp()
        os.makedirs(os.path.join(root, "walls"))
        c = sqlite3.connect(os.path.join(root, "goclimb.sqlite3"))
        c.executescript("""CREATE TABLE walls(wall_id TEXT PRIMARY KEY, name TEXT NOT NULL, venue TEXT NOT NULL,
          mode TEXT NOT NULL, photo_ref TEXT, image_w INTEGER, image_h INTEGER, lat REAL, lon REAL,
          vuforia_target TEXT, created_at TEXT, updated_at TEXT, next_hold_no INTEGER NOT NULL DEFAULT 1,
          next_route_no INTEGER NOT NULL DEFAULT 1, algo_version TEXT, schema_version INTEGER);
          INSERT INTO walls(wall_id,name,venue,mode) VALUES('00000000-0000-0000-0000-000000000001','Old','V','indoor');""")
        c.commit(); c.close()
        st = Store(root)
        w = st.search_walls()[0]
        self.assertEqual((w["source"], w["reviewed"]), ("web", True))


class ScanDiscardTests(unittest.TestCase):
    def test_discard_removes_photo_and_session(self):
        st = Store(tempfile.mkdtemp())
        svc = ScanService(st)
        r = svc.scan({"image_b64": B64, "mode": "indoor"})
        folder = os.path.join(svc.scan_root, r["scan_id"])
        self.assertTrue(os.path.isdir(folder))
        self.assertEqual(svc.discard(r["scan_id"]), {"discarded": True})
        self.assertFalse(os.path.exists(folder))
        with self.assertRaises(ScanExpired):
            svc.save(r["scan_id"], {"venue": "V", "name": "N"})
        self.assertEqual(svc.discard(r["scan_id"]), {"discarded": False})
        self.assertEqual(st.search_walls(), [])                              # nothing was saved

    def test_saved_scan_is_marked_as_app_upload(self):
        st = Store(tempfile.mkdtemp())
        svc = ScanService(st)
        r = svc.scan({"image_b64": B64, "mode": "indoor"})
        wid = svc.save(r["scan_id"], {"venue": "V", "name": "N"})["wall_id"]
        self.assertEqual(st.get_wall(wid)["source"], "app")


class ServerApi4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = make_server(tempfile.mkdtemp(), 0, tempfile.mkdtemp(), admin_token="s3cret")
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown(); cls.srv.server_close()

    def call(self, method, path, body=None, token=None):
        h = {"Content-Type": "application/json"}
        if token:
            h["X-Admin-Token"] = token
        req = urllib.request.Request(self.base + path, method=method, headers=h,
                                     data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_version(self):
        st, j = self.call("GET", "/api/version")
        self.assertEqual((st, j["api_version"], j["admin_token_required"]), (200, 4, True))

    def test_app_upload_is_public_and_returns_routes(self):
        st, j = self.call("POST", "/api/walls", {"name": "Up", "venue": "Phone", "mode": "indoor", "image_b64": B64,
                                                  "holds": PHONE_HOLDS, "source": "app", "max_routes": 4})
        self.assertEqual(st, 201)
        self.assertEqual(j["doc"]["wall_id"], j["wall_id"])
        self.assertEqual(len(j["doc"]["holds"]), len(PHONE_HOLDS))
        self.assertTrue(j["doc"]["routes"])
        st, lst = self.call("GET", "/api/walls?review=1&q=phone")
        self.assertEqual([w["wall_id"] for w in lst], [j["wall_id"]])

    def test_editing_needs_the_admin_token(self):
        st, j = self.call("POST", "/api/walls", {"name": "Adm", "venue": "Phone", "mode": "indoor", "image_b64": B64,
                                                  "holds": PHONE_HOLDS, "source": "app"})
        wid = j["wall_id"]
        self.assertEqual(self.call("POST", f"/api/walls/{wid}/holds", {"x": .5, "y": .5})[0], 401)
        self.assertEqual(self.call("POST", f"/api/walls/{wid}/holds", {"x": .5, "y": .5}, token="wrong")[0], 401)
        self.assertEqual(self.call("DELETE", f"/api/walls/{wid}?confirm=1")[0], 401)
        self.assertEqual(self.call("GET", "/api/admin/check")[0], 401)
        self.assertEqual(self.call("GET", "/api/admin/check", token="s3cret")[0], 200)
        st, j = self.call("POST", f"/api/walls/{wid}/routes", {"hold_ids": ["H61", "H49", "H37", "H25", "H13"]}, token="s3cret")
        self.assertEqual(st, 201)
        rid = j["route_id"]
        self.assertEqual(self.call("POST", f"/api/walls/{wid}/routes/{rid}/sequence",
                                   {"hold_ids": ["H61", "H50", "H37"]}, token="s3cret")[0], 200)
        self.assertEqual(self.call("DELETE", f"/api/walls/{wid}/routes/{rid}", token="s3cret")[0], 200)
        self.assertEqual(self.call("POST", f"/api/walls/{wid}/review", {"reviewed": True}, token="s3cret")[0], 200)
        self.assertTrue(self.call("GET", f"/api/walls/{wid}")[1]["reviewed"])
        self.assertEqual(self.call("GET", f"/api/walls/{wid}/unity")[0], 200)                # reading stays public

    def test_discard_scan_over_http(self):
        st, r = self.call("POST", "/api/scan", {"image_b64": B64, "mode": "indoor"})
        self.assertEqual(st, 200)
        self.assertEqual(self.call("DELETE", f"/api/scans/{r['scan_id']}"), (200, {"discarded": True}))
        self.assertEqual(self.call("POST", f"/api/scans/{r['scan_id']}/save", {"venue": "V", "name": "N"})[0], 410)


if __name__ == "__main__":
    unittest.main()
