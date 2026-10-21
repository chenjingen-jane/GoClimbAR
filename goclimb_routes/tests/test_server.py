import base64
import json
import os
import re
import shutil
import subprocess
import threading
import tempfile
import unittest
import urllib.error
import urllib.request

from goclimb_routes.server import make_server
from .fixtures import png_bytes, synthetic_indoor_wall


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.unity = tempfile.mkdtemp(), tempfile.mkdtemp()
        cls.srv = make_server(cls.data, 0, cls.unity)
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.png = png_bytes(synthetic_indoor_wall()[0])

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown(); cls.srv.server_close()

    def call(self, method, path, body=None, raw=False):
        req = urllib.request.Request(self.base + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                data = r.read()
                return r.status, (data if raw else json.loads(data))
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_full_workflow(self):
        body = {"name": "Wall S", "venue": "Server Gym", "mode": "indoor", "vuforia_target": "srv-t",
                "image_b64": base64.b64encode(self.png).decode()}
        st, j = self.call("POST", "/api/walls", body); self.assertEqual(st, 201)
        wid = j["wall_id"]
        st, j = self.call("POST", "/api/walls", body); self.assertEqual(st, 409)           # duplicate guard
        st, j = self.call("GET", "/api/walls?venue=server"); self.assertEqual([w["wall_id"] for w in j], [wid])
        st, w = self.call("GET", f"/api/walls/{wid}"); self.assertEqual(st, 200)
        self.assertIn("detection_report", w)
        st, img = self.call("GET", f"/photo/{wid}", raw=True); self.assertEqual(img, self.png)
        st, j = self.call("POST", f"/api/walls/{wid}/holds", {"x": 0.5, "y": 0.5}); self.assertEqual(st, 201)
        hid = j["hold_id"]
        self.assertEqual(self.call("PUT", f"/api/walls/{wid}/holds/{hid}", {"x": 0.6, "y": 0.6})[0], 200)
        self.assertEqual(self.call("POST", f"/api/walls/{wid}/holds", {"x": 7, "y": 0.5})[0], 400)
        self.assertEqual(self.call("DELETE", f"/api/walls/{wid}/holds/{hid}")[0], 200)
        st, rep = self.call("POST", f"/api/walls/{wid}/routes/generate", {"max_routes": 4}); self.assertEqual(st, 200)
        self.assertLessEqual(rep["retained"], 4)
        st, ov = self.call("GET", f"/overlay/{wid}", raw=True); self.assertEqual(ov[:4], b"\x89PNG")
        st, j = self.call("POST", f"/api/walls/{wid}/export", {}); self.assertEqual(st, 200)
        self.assertIn(wid, j["path"])
        self.assertEqual(self.call("DELETE", f"/api/walls/{wid}")[0], 400)                 # no confirm
        self.assertEqual(self.call("DELETE", f"/api/walls/{wid}?confirm=1")[0], 200)
        self.assertEqual(self.call("GET", f"/api/walls/{wid}")[0], 404)

    def test_errors_and_page(self):
        self.assertEqual(self.call("GET", "/api/walls/00000000-0000-0000-0000-000000000000")[0], 404)
        self.assertEqual(self.call("POST", "/api/walls", {"name": "x"})[0], 400)
        self.assertEqual(self.call("GET", "/nope")[0], 404)
        st, page = self.call("GET", "/", raw=True)
        self.assertEqual(st, 200); self.assertIn(b"GoClimb admin editor", page)

    def test_scan_flow_unknown_save_known_over_http(self):
        """What Unity does: scan -> routes shown BEFORE saving -> save -> rescan reuses the record."""
        import cv2, numpy as np
        b64 = base64.b64encode(self.png).decode()
        n_before = len(self.call("GET", "/api/walls")[1])
        st, r = self.call("POST", "/api/scan", {"image_b64": b64, "mode": "indoor"})
        self.assertEqual((st, r["status"], r["wall_id"]), (200, "unknown", ""))
        self.assertEqual(r["doc"]["scan_id"], r["scan_id"])
        self.assertGreater(len(r["doc"]["routes"]), 0)
        self.assertEqual(len(self.call("GET", "/api/walls")[1]), n_before)           # nothing saved yet
        # coordinates belong to the uploaded photo
        for h in r["doc"]["holds"]:
            self.assertTrue(0 <= h["x"] <= 1 and 0 <= h["y"] <= 1)
        # save once; pressing save again must not create a second record
        st, s1 = self.call("POST", f"/api/scans/{r['scan_id']}/save", {"venue": "Scan Gym", "name": "Wall X"})
        st2, s2 = self.call("POST", f"/api/scans/{r['scan_id']}/save", {"venue": "Scan Gym", "name": "Wall X"})
        self.assertEqual((st, s1["created"], s2["created"], s1["wall_id"]), (200, True, False, s2["wall_id"]))
        wid = s1["wall_id"]
        self.assertEqual(len(self.call("GET", "/api/walls")[1]), n_before + 1)
        # scanning the same wall again: known, same wall_id, no new record
        for _ in range(3):
            st, k = self.call("POST", "/api/scan", {"image_b64": b64, "mode": "indoor"})
            self.assertEqual((k["status"], k["wall_id"]), ("known", wid))
        self.assertEqual(len(self.call("GET", "/api/walls")[1]), n_before + 1)
        # regenerate: changes routes, not records
        st, g = self.call("POST", f"/api/scans/{k['scan_id']}/regenerate", {"max_routes": 2})
        self.assertEqual(st, 200); self.assertLessEqual(len(g["doc"]["routes"]), 2)
        self.assertEqual(len(self.call("GET", "/api/walls")[1]), n_before + 1)
        # stored-frame document for "load saved wall"
        st, d = self.call("GET", f"/api/walls/{wid}/unity")
        self.assertEqual((st, d["wall_id"], d["coordinate_frame"]), (200, wid, "stored_photo"))
        self.call("DELETE", f"/api/walls/{wid}?confirm=1")

    def test_version_endpoint_and_scan_reports_version(self):
        from goclimb_routes.schema import API_VERSION
        st, v = self.call("GET", "/api/version")
        self.assertEqual((st, v["api_version"]), (200, API_VERSION))
        st, r = self.call("POST", "/api/scan", {"image_b64": base64.b64encode(self.png).decode(), "mode": "indoor"})
        self.assertEqual(r["api_version"], API_VERSION)

    def test_scan_errors_expired_and_duplicate_name(self):
        import cv2, numpy as np
        self.assertEqual(self.call("POST", "/api/scan", {"image_b64": "!!!not base64!!!"})[0], 400)
        self.assertEqual(self.call("POST", "/api/scan", {"image_b64": base64.b64encode(b"not an image").decode()})[0], 400)
        self.assertEqual(self.call("POST", "/api/scan", {})[0], 400)
        self.assertEqual(self.call("POST", "/api/scan", {"image_b64": base64.b64encode(self.png).decode(), "mode": "moon"})[0], 400)
        self.assertEqual(self.call("POST", "/api/scans/" + "0" * 32 + "/save", {})[0], 410)         # expired / unknown scan
        self.assertEqual(self.call("POST", "/api/scans/" + "0" * 32 + "/regenerate", {})[0], 410)
        flat = cv2.imencode(".png", np.full((400, 600, 3), 170, np.uint8))[1].tobytes()
        st, r = self.call("POST", "/api/scan", {"image_b64": base64.b64encode(flat).decode(), "mode": "indoor"})
        self.assertEqual((st, r["doc"]["routes"]), (200, []))                                      # nothing invented
        other = png_bytes(synthetic_indoor_wall(seed=9)[0])
        a = self.call("POST", "/api/scan", {"image_b64": base64.b64encode(other).decode(), "mode": "indoor"})[1]
        w1 = self.call("POST", f"/api/scans/{a['scan_id']}/save", {"venue": "Dup Gym", "name": "Same"})[1]["wall_id"]
        b = self.call("POST", "/api/scan", {"image_b64": base64.b64encode(png_bytes(synthetic_indoor_wall(seed=11)[0])).decode(), "mode": "indoor"})[1]
        st, e = self.call("POST", f"/api/scans/{b['scan_id']}/save", {"venue": "Dup Gym", "name": "Same"})
        self.assertEqual((st, e["error"]), (409, "duplicate"))
        self.call("DELETE", f"/api/walls/{w1}?confirm=1")

    @unittest.skipUnless(shutil.which("node"), "Node.js not installed; cannot syntax-check the page script")
    def test_page_javascript_has_no_syntax_errors(self):
        """Regression: a duplicate `const r` in render() once broke the whole editor page."""
        st, page = self.call("GET", "/", raw=True)
        js = re.search(r"<script>(.*)</script>", page.decode("utf-8"), re.S).group(1)
        path = os.path.join(tempfile.mkdtemp(), "page.js")
        with open(path, "w", encoding="utf-8") as f:
            f.write(js)
        r = subprocess.run(["node", "--check", path], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
