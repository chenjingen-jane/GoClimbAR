"""GoClimb server + admin web editor (stdlib only). Run:
    python -m goclimb_routes.server --data goclimb_data --port 8765 [--host 0.0.0.0] [--admin-token SECRET]
Then open http://127.0.0.1:8765 . Prototype: plain HTTP, trusted network only.

Who may call what:
  * the phone app (no token): /api/version, /api/scan, /api/scans/<id>/save|regenerate|DELETE,
    GET /api/walls..., GET /photo/..., POST /api/walls (upload a wall, e.g. one scanned offline)
  * the admin web editor: everything that CHANGES a saved wall (holds, routes, review, delete, export).
    With --admin-token these calls need the header  X-Admin-Token: <token>. Without it they are open
    (fine on your own PC, not on a shared network).
"""
from __future__ import annotations

import argparse
import base64
import hmac
import json
import os
import re
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .scan import ScanExpired, ScanService
from .schema import API_VERSION
from .store import DuplicateWallError, Store, WallNotFound

WEB = os.path.join(os.path.dirname(__file__), "web")
ID = r"([0-9a-fA-F-]{36})"
SCAN = r"([0-9a-f]{32})"


class Forbidden(Exception):
    pass


def make_handler(store: Store, unity_dir: str, service: ScanService, admin_token: str = ""):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            if not isinstance(body, (bytes, bytearray)):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n > 40_000_000:
                raise ValueError("request too large")
            return json.loads(self.rfile.read(n) or b"{}")

        def _admin(self):
            """Editing a SAVED wall is an admin action."""
            if admin_token and not hmac.compare_digest(self.headers.get("X-Admin-Token", ""), admin_token):
                raise Forbidden()

        def _dispatch(self, method):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            p = u.path
            try:
                # ---------------------------------------------------------------- public (app + browser)
                if method == "GET" and p in ("/", "/index.html"):
                    with open(os.path.join(WEB, "index.html"), "rb") as f:
                        return self._send(200, f.read(), "text/html; charset=utf-8")
                if method == "GET" and p == "/api/version":
                    return self._send(200, {"api_version": API_VERSION, "name": "goclimb",
                                            "admin_token_required": bool(admin_token)})
                if method == "GET" and p == "/api/admin/check":
                    self._admin()
                    return self._send(200, {"ok": True})
                if method == "GET" and p == "/api/walls":
                    near = None
                    if "lat" in q and "lon" in q:
                        near = (float(q["lat"]), float(q["lon"]), float(q.get("radius", 25)))
                    return self._send(200, store.search_walls(q.get("venue"), q.get("name"), q.get("mode"), near,
                                                              needs_review=q.get("review") == "1", text=q.get("q")))
                if method == "POST" and p == "/api/walls":
                    # Upload a wall. The app sends source="app" (+ the holds it detected offline, if any).
                    b = self._json()
                    try:
                        wid = store.create_wall(b.get("name", ""), b.get("venue", ""), b.get("mode", ""),
                                                image_bytes=base64.b64decode(b.get("image_b64", "")),
                                                lat=b.get("lat"), lon=b.get("lon"),
                                                vuforia_target=b.get("vuforia_target") or None,
                                                allow_duplicate=bool(b.get("allow_duplicate")),
                                                holds=b.get("holds"), source=b.get("source") or "web")
                    except DuplicateWallError as e:
                        return self._send(409, {"error": "duplicate", "existing": e.existing})
                    rep = store.generate_routes_for(wid, int(b.get("max_routes", 6)))
                    doc = store.export_doc(wid)
                    doc["warnings"] = doc["warnings"] + rep.get("warnings", [])
                    return self._send(201, {"wall_id": wid, "created": True, "doc": doc})
                if method == "POST" and p == "/api/scan":
                    return self._send(200, service.scan(self._json()))
                m = re.fullmatch(r"/api/scans/" + SCAN, p)
                if m and method == "DELETE":                       # user chose NOT to upload
                    return self._send(200, service.discard(m.group(1)))
                m = re.fullmatch(r"/api/scans/" + SCAN + r"/(save|regenerate)", p)
                if m and method == "POST":
                    sid, what = m.groups()
                    b = self._json()
                    if what == "save":
                        try:
                            return self._send(200, service.save(sid, b))
                        except DuplicateWallError as e:
                            return self._send(409, {"error": "duplicate", "existing": e.existing})
                    return self._send(200, service.regenerate(sid, int(b.get("max_routes", 6)),
                                                              bool(b.get("refresh_only"))))
                m = re.fullmatch(r"/api/walls/" + ID + "/unity", p)
                if m and method == "GET":
                    return self._send(200, store.export_doc(m.group(1)))
                m = re.fullmatch(r"/photo/" + ID, p)
                if m and method == "GET":
                    path = store.photo_path(m.group(1))
                    if path is None:
                        return self._send(404, {"error": "photo missing"})
                    with open(path, "rb") as f:
                        return self._send(200, f.read(), "image/png" if path.endswith("png") else "image/jpeg")
                m = re.fullmatch(r"/overlay/" + ID, p)
                if m and method == "GET":
                    store.save_overlay(m.group(1))
                    with open(os.path.join(store.root, "walls", m.group(1), "overlay.png"), "rb") as f:
                        return self._send(200, f.read(), "image/png")
                m = re.fullmatch(r"/api/walls/" + ID, p)
                if m and method == "GET":
                    w = store.get_wall(m.group(1))
                    w["detection_report"] = store.detection_report(m.group(1))
                    return self._send(200, w)

                # ---------------------------------------------------------------- admin (edits saved walls)
                if m and method in ("PUT", "DELETE"):
                    self._admin()
                    wid = m.group(1)
                    if method == "PUT":
                        b = self._json()
                        store.update_wall(wid, allow_duplicate=bool(b.pop("allow_duplicate", False)), **b)
                        return self._send(200, {"ok": True})
                    store.delete_wall(wid, confirm=q.get("confirm") == "1")
                    return self._send(200, {"ok": True})
                m = re.fullmatch(r"/api/walls/" + ID + r"/(holds|redetect|routes/generate|routes|export|review)", p)
                if m and method == "POST":
                    self._admin()
                    wid, what = m.groups()
                    b = self._json()
                    if what == "holds":
                        return self._send(201, {"hold_id": store.add_hold(wid, float(b["x"]), float(b["y"]))})
                    if what == "redetect":
                        return self._send(200, {"added": store.redetect(wid)})
                    if what == "routes/generate":
                        return self._send(200, store.generate_routes_for(
                            wid, int(b.get("max_routes", 6)), float(b.get("overlap_threshold", 0.6))))
                    if what == "routes":
                        return self._send(201, {"route_id": store.add_route(wid, list(b.get("hold_ids") or []))})
                    if what == "review":
                        store.set_reviewed(wid, bool(b.get("reviewed", True)))
                        return self._send(200, {"ok": True})
                    return self._send(200, {"path": store.export_unity(wid, unity_dir)})
                m = re.fullmatch(r"/api/walls/" + ID + r"/holds/(H\d+)", p)
                if m and method in ("PUT", "DELETE"):
                    self._admin()
                    wid, hid = m.groups()
                    if method == "PUT":
                        b = self._json()
                        store.move_hold(wid, hid, float(b["x"]), float(b["y"]))
                    else:
                        store.delete_hold(wid, hid)
                    return self._send(200, {"ok": True})
                m = re.fullmatch(r"/api/walls/" + ID + r"/routes/(R\d+)", p)
                if m and method == "DELETE":
                    self._admin()
                    store.delete_route(*m.groups())
                    return self._send(200, {"ok": True})
                m = re.fullmatch(r"/api/walls/" + ID + r"/routes/(R\d+)/(limb|sequence)", p)
                if m and method == "POST":
                    self._admin()
                    wid, rid, what = m.groups()
                    b = self._json()
                    if what == "limb":
                        store.set_limb(wid, rid, b["hold_id"], b.get("limb") or None)
                    else:
                        store.set_route_sequence(wid, rid, list(b["hold_ids"]))
                    return self._send(200, {"ok": True})
                return self._send(404, {"error": "not found"})
            except Forbidden:
                return self._send(401, {"error": "admin token required"})
            except ScanExpired as e:
                return self._send(410, {"error": str(e)})
            except WallNotFound:
                return self._send(404, {"error": "wall not found"})
            except KeyError as e:
                return self._send(404, {"error": f"not found: {e}"})
            except (ValueError, TypeError, FileNotFoundError) as e:
                return self._send(400, {"error": str(e)})
            except Exception as e:                      # never drop the connection silently
                return self._send(500, {"error": f"internal error: {type(e).__name__}: {e}"})

        def do_GET(self): self._dispatch("GET")
        def do_POST(self): self._dispatch("POST")
        def do_PUT(self): self._dispatch("PUT")
        def do_DELETE(self): self._dispatch("DELETE")
    return H


def make_server(data_dir: str, port: int = 8765, unity_dir: str = "unity_export", host: str = "127.0.0.1",
                service: ScanService = None, admin_token: str = ""):
    store = Store(data_dir)
    return ThreadingHTTPServer((host, port), make_handler(store, unity_dir, service or ScanService(store), admin_token))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="goclimb_data")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1",
                    help="use 0.0.0.0 so a phone on the same Wi-Fi can reach it (trusted network only)")
    ap.add_argument("--unity", default="unity_export", help="folder for exported route JSON (point at Assets/StreamingAssets/Routes)")
    ap.add_argument("--admin-token", default=os.environ.get("GOCLIMB_ADMIN_TOKEN", ""),
                    help="password the web editor must send to change saved walls (recommended with --host 0.0.0.0)")
    a = ap.parse_args()
    srv = make_server(a.data, a.port, a.unity, a.host, admin_token=a.admin_token)
    print(f"GoClimb editor: http://127.0.0.1:{a.port}   data={os.path.abspath(a.data)}   api_version={API_VERSION}"
          f"   admin token: {'ON' if a.admin_token else 'off'}")
    if a.host != "127.0.0.1":
        try:
            print(f"Phone/Unity address: http://{socket.gethostbyname(socket.gethostname())}:{a.port}")
        except OSError:
            pass
        if not a.admin_token:
            print("WARNING: anyone on this network can edit walls. Add --admin-token <password>.")
    srv.serve_forever()
