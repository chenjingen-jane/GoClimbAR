"""Runtime scan service (no HTTP here, so it is easy to test).

Flow for one camera scan:
  1. Unity uploads a photo. We try to MATCH it against saved walls (matcher.py).
  2. confident match  -> status 'known'   : reuse that wall_id, NO new record. Saved holds/routes are
                                            warped into the scan photo's frame so they line up with it.
     possible match   -> status 'uncertain': routes are generated for the scan; the user decides
     no match         -> status 'unknown'  : routes are generated for the scan
  3. NOTHING is saved until save() is called. Unknown/uncertain scans live in a temporary session
     (expires) so scanning the same wall repeatedly can never create duplicate records.
All coordinates in the returned doc are normalized to the SCAN photo (coordinate_frame='scan_photo').
"""
from __future__ import annotations

import base64
import copy
import os
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime
from typing import Dict, Optional

import cv2
import numpy as np

from .detect import Hold
from .matcher import MatchResult, apply_h, find_matches
from .planner import generate_routes
from .schema import API_VERSION, MODES
from .store import Store, WallNotFound


class ScanExpired(ValueError):
    pass


class Session:
    def __init__(self, scan_id, raw, mode, size, max_routes):
        self.scan_id, self.raw, self.mode, self.size, self.max_routes = scan_id, raw, mode, size, max_routes
        self.created = time.time()
        self.status = "unknown"
        self.wall_id = ""                 # known wall (reused record)
        self.H = None                     # stored photo -> scan photo
        self.preview: Optional[Store] = None
        self.preview_wall_id = ""
        self.saved_wall_id = ""           # set by save()
        self.candidates = []


def transform_doc(doc: dict, H: np.ndarray, width: int, height: int) -> dict:
    """Warp a stored-photo document into the scan photo's frame. Holds/routes that fall outside the
    scan photo are dropped (never drawn at wrong positions)."""
    out = copy.deepcopy(doc)
    warnings = list(out.get("warnings", []))

    def tp(x, y):
        X, Y = apply_h(H, x, y)
        if not (np.isfinite(X) and np.isfinite(Y)) or not (-0.02 <= X <= 1.02 and -0.02 <= Y <= 1.02):
            return None
        return min(1.0, max(0.0, X)), min(1.0, max(0.0, Y))

    holds, kept_ids = [], set()
    for h in out["holds"]:
        p = tp(h["x"], h["y"])
        if p:
            h["x"], h["y"] = round(p[0], 5), round(p[1], 5)
            holds.append(h); kept_ids.add(h["id"])
    routes = []
    for r in out["routes"]:
        pts = [tp(p["x"], p["y"]) for p in r["points"]]
        if any(p is None for p in pts):
            warnings.append(f"{r['id']}: leaves the camera view, not shown")
            continue
        for p, q in zip(r["points"], pts):
            p["x"], p["y"] = round(q[0], 5), round(q[1], 5)
        routes.append(r)
    out.update({"holds": holds, "routes": routes, "warnings": warnings, "aspect": round(height / float(width), 5),
                "image_width": width, "image_height": height})
    return out


def replan_in_view(doc: dict, max_routes: int) -> None:
    """Known wall, but every saved route runs out of the current camera view (the saved photo was
    framed differently). Plan routes on the saved holds that ARE in view, so the user still sees
    routes. Nothing is written to the store: the saved routes stay as they are."""
    holds = [Hold(id=h["id"], x=h["x"], y=h["y"], confidence=h.get("confidence") or 1.0,
                  type=h.get("type") or "") for h in doc["holds"]]
    aspect = doc["aspect"]
    routes, report = generate_routes(holds, aspect, max_routes)
    if not routes and len(holds) >= 5:
        for scale in (1.5, 2.0):
            routes, report = generate_routes(holds, aspect, max_routes, reach_scale=scale)
            if routes:
                report["warnings"].append(f"holds are far apart: reach limit relaxed to {scale}x")
                break
    for r in routes:
        r["edited"] = False
    doc["routes"] = routes
    doc["warnings"] = list(doc.get("warnings", [])) + list(report.get("warnings", []))
    if routes:
        doc["warnings"].append("the saved routes leave the camera view; showing routes planned on the "
                               "holds in view (not saved)")


class ScanService:
    def __init__(self, store: Store, ttl_seconds: int = 3600, max_sessions: int = 30, matcher=None,
                 scan_root: Optional[str] = None):
        self.store = store
        self.ttl, self.max_sessions = ttl_seconds, max_sessions
        self.matcher = matcher or find_matches
        self.scan_root = scan_root or os.path.join(store.root, "scans")
        shutil.rmtree(self.scan_root, ignore_errors=True)           # leftovers from a previous run
        os.makedirs(self.scan_root, exist_ok=True)
        self._sessions: "OrderedDict[str, Session]" = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ session bookkeeping
    def _drop(self, sid):
        self._sessions.pop(sid, None)
        shutil.rmtree(os.path.join(self.scan_root, sid), ignore_errors=True)

    def _purge(self):
        now = time.time()
        for sid in [k for k, s in self._sessions.items() if now - s.created > self.ttl]:
            self._drop(sid)
        while len(self._sessions) > self.max_sessions:
            self._drop(next(iter(self._sessions)))

    def _get(self, sid) -> Session:
        with self._lock:
            self._purge()
            s = self._sessions.get(sid)
        if s is None:
            raise ScanExpired("this scan has expired or does not exist; scan the wall again")
        return s

    # ------------------------------------------------------------------ scan
    def scan(self, body: dict) -> dict:
        mode = body.get("mode") or "outdoor"
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        try:
            raw = base64.b64decode(body.get("image_b64", ""), validate=True)
        except Exception:
            raise ValueError("image_b64 is not valid base64")
        if not raw:
            raise ValueError("a photo is required")
        try:
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        except cv2.error:
            img = None
        if img is None:
            raise ValueError("the photo could not be decoded as an image")
        max_routes = max(1, min(30, int(body.get("max_routes", 6))))
        h, w = img.shape[:2]
        sid = uuid.uuid4().hex
        sess = Session(sid, raw, mode, (w, h), max_routes)

        matches = self.matcher(self.store, img)
        best: Optional[MatchResult] = matches[0] if matches else None
        if best and best.level == "confident":
            sess.status, sess.wall_id, sess.H = "known", best.wall_id, best.H
            doc = self._known_doc(sess, regenerate=False)
        else:
            sess.status = "uncertain" if best else "unknown"
            sess.candidates = [m for m in matches if m.level == "possible"]
            tmp = Store(os.path.join(self.scan_root, sid))
            sess.preview = tmp
            sess.preview_wall_id = tmp.create_wall("scan", "scan", mode, image_bytes=raw)
            tmp.generate_routes_for(sess.preview_wall_id, max_routes)
            doc = self._preview_doc(sess)
        with self._lock:
            self._sessions[sid] = sess
            self._purge()
        return self._response(sess, doc, best)

    def _response(self, sess: Session, doc: dict, best: Optional[MatchResult]) -> dict:
        m = best or MatchResult()
        return {"api_version": API_VERSION, "scan_id": sess.scan_id, "status": sess.status, "wall_id": sess.wall_id or sess.saved_wall_id,
                "vuforia_target": "rt-" + sess.scan_id[:8], "holds_detected": len(doc["holds"]),
                "match": {"wall_id": m.wall_id, "level": m.level, "inliers": m.inliers, "ratio": float(m.ratio)},
                "candidates": [{"wall_id": c.wall_id, "name": c.name, "venue": c.venue,
                                "inliers": c.inliers, "ratio": float(c.ratio)} for c in sess.candidates],
                "doc": doc}

    def _stamp(self, sess: Session, doc: dict, wall_id: str) -> dict:
        doc["scan_id"], doc["wall_id"] = sess.scan_id, wall_id
        doc["vuforia_target"] = "rt-" + sess.scan_id[:8]
        doc["coordinate_frame"] = "scan_photo"
        doc["image_width"], doc["image_height"] = sess.size
        return doc

    def _known_doc(self, sess: Session, regenerate: bool, max_routes: int = 6) -> dict:
        if regenerate:
            self.store.generate_routes_for(sess.wall_id, max_routes)
        doc = transform_doc(self.store.export_doc(sess.wall_id), sess.H, *sess.size)
        if not doc["routes"]:
            replan_in_view(doc, max_routes if regenerate else sess.max_routes)
        return self._stamp(sess, doc, sess.wall_id)

    def _preview_doc(self, sess: Session) -> dict:
        doc = sess.preview.export_doc(sess.preview_wall_id)
        doc["name"], doc["venue"] = "(not saved)", ""
        return self._stamp(sess, doc, sess.saved_wall_id)

    # ------------------------------------------------------------------ save / regenerate
    def save(self, scan_id: str, body: dict) -> dict:
        """Persist a scan. Known walls and already-saved scans never create a second record."""
        sess = self._get(scan_id)
        if sess.status == "known" or sess.saved_wall_id:
            return {"wall_id": sess.wall_id or sess.saved_wall_id, "created": False}
        existing = (body.get("use_existing_wall_id") or "").strip()
        if existing:
            self.store.get_wall(existing)                      # raises WallNotFound if wrong
            sess.saved_wall_id = existing
            return {"wall_id": existing, "created": False}
        venue = (body.get("venue") or "").strip() or "Scanned walls"
        name = (body.get("name") or "").strip() or "Scan " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        wid = self.store.create_wall(name, venue, sess.mode, image_bytes=sess.raw, source="app",
                                     allow_duplicate=bool(body.get("allow_duplicate")))   # may raise DuplicateWallError
        self.store.generate_routes_for(wid, sess.max_routes)
        sess.saved_wall_id = wid
        doc = self._stamp(sess, self.store.export_doc(wid), wid)
        return {"wall_id": wid, "created": True, "doc": doc}

    def discard(self, scan_id: str) -> dict:
        """The user chose NOT to upload: delete the temporary photo and session right away (instead of
        waiting for the expiry). Saved walls are never touched. Unknown/expired ids are not an error."""
        with self._lock:
            existed = scan_id in self._sessions
            self._drop(scan_id)
        return {"discarded": existed}

    def regenerate(self, scan_id: str, max_routes: int = 6, refresh_only: bool = False) -> dict:
        """New routes (or just reload saved data) for this scan. Never creates a wall record."""
        sess = self._get(scan_id)
        max_routes = max(1, min(30, int(max_routes)))
        if sess.status == "known":
            doc = self._known_doc(sess, regenerate=not refresh_only, max_routes=max_routes)
        elif sess.saved_wall_id:
            if not refresh_only:
                self.store.generate_routes_for(sess.saved_wall_id, max_routes)
            doc = self._stamp(sess, self.store.export_doc(sess.saved_wall_id), sess.saved_wall_id)
        else:
            if not refresh_only:
                sess.preview.generate_routes_for(sess.preview_wall_id, max_routes)
            doc = self._preview_doc(sess)
        return {"scan_id": scan_id, "doc": doc}