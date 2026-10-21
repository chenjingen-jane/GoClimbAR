"""Persistent wall storage: SQLite (stdlib) for records, plain files for photos.

Layout under the data root (configurable via Store(root) or env GOCLIMB_DATA):
    goclimb.sqlite3
    walls/<wall_id>/original.jpg|png   original photo (never overwritten)
    walls/<wall_id>/overlay.png        generated route picture (separate file)
Only RELATIVE references (photo_ref) are stored, so the data root can be moved.

Deletion policy: delete_wall(confirm=True) HARD-deletes the wall row, its holds, its routes
and its photo folder. Exported Unity JSON files live outside the data root and are not touched.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import cv2
import numpy as np

from .detect import ClassicalDetector, Hold, IndoorDetector
from .planner import describe_route, generate_routes
from .schema import (ALGO_VERSION, COORD_SYSTEM, DISCLAIMER, LIMBS, MODES, SCHEMA_VERSION,
                     hold_label)
from .viz import render_overlay

SQL = """
CREATE TABLE IF NOT EXISTS walls(
  wall_id TEXT PRIMARY KEY, name TEXT NOT NULL, venue TEXT NOT NULL,
  mode TEXT NOT NULL CHECK(mode IN ('indoor','outdoor')),
  photo_ref TEXT, image_w INTEGER, image_h INTEGER, lat REAL, lon REAL,
  vuforia_target TEXT, created_at TEXT, updated_at TEXT,
  next_hold_no INTEGER NOT NULL DEFAULT 1, next_route_no INTEGER NOT NULL DEFAULT 1,
  algo_version TEXT, schema_version INTEGER);
CREATE TABLE IF NOT EXISTS holds(
  wall_id TEXT NOT NULL, hold_id TEXT NOT NULL, x REAL, y REAL, orig_x REAL, orig_y REAL,
  confidence REAL, type TEXT,
  status TEXT NOT NULL CHECK(status IN ('detected','manual_added','moved','deleted')),
  PRIMARY KEY(wall_id, hold_id), FOREIGN KEY(wall_id) REFERENCES walls(wall_id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS routes(
  wall_id TEXT NOT NULL, route_id TEXT NOT NULL, data TEXT NOT NULL, created_at TEXT,
  PRIMARY KEY(wall_id, route_id), FOREIGN KEY(wall_id) REFERENCES walls(wall_id) ON DELETE CASCADE);
CREATE INDEX IF NOT EXISTS idx_walls_venue ON walls(venue COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_walls_name ON walls(name COLLATE NOCASE);
"""


class DuplicateWallError(Exception):
    def __init__(self, existing):
        super().__init__("a wall with this venue and name already exists")
        self.existing = existing


class WallNotFound(KeyError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def _in01(*vals) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) and 0.0 <= v <= 1.0 for v in vals)


class Store:
    def __init__(self, root: Optional[str] = None):
        self.root = os.path.abspath(root or os.environ.get("GOCLIMB_DATA", "goclimb_data"))
        os.makedirs(os.path.join(self.root, "walls"), exist_ok=True)
        self.db_path = os.path.join(self.root, "goclimb.sqlite3")
        with self._conn() as c:
            c.executescript(SQL)
            cols = {r["name"] for r in c.execute("PRAGMA table_info(walls)").fetchall()}
            if "source" not in cols:          # databases created before API 4
                c.execute("ALTER TABLE walls ADD COLUMN source TEXT NOT NULL DEFAULT 'web'")
            if "reviewed" not in cols:
                c.execute("ALTER TABLE walls ADD COLUMN reviewed INTEGER NOT NULL DEFAULT 1")

    @contextlib.contextmanager
    def _conn(self):
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

    # ------------------------------------------------------------------ walls

    def find_similar(self, venue: str, name: str) -> List[dict]:
        """Same venue (case-insensitive) and a name that matches or contains the given name."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT wall_id,name,venue,mode FROM walls WHERE lower(venue)=lower(?) "
                "AND (lower(name)=lower(?) OR lower(name) LIKE ?)",
                (venue.strip(), name.strip(), f"%{name.strip().lower()}%")).fetchall()
        return [dict(r) for r in rows]

    def create_wall(self, name: str, venue: str, mode: str, image_bytes: Optional[bytes] = None,
                    image_path: Optional[str] = None, lat: Optional[float] = None,
                    lon: Optional[float] = None, vuforia_target: Optional[str] = None,
                    allow_duplicate: bool = False, auto_detect: bool = True,
                    holds: Optional[list] = None, source: str = "web") -> str:
        """holds: optional [{x, y, confidence?, type?}] already detected on the phone (offline scan). They are
        stored instead of running the server detector, so the saved wall shows what the user saw.
        source: 'web' (admin editor) or 'app' (uploaded from the phone; flagged for admin review)."""
        name, venue = (name or "").strip(), (venue or "").strip()
        if not name or not venue:
            raise ValueError("wall name and venue name are required")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if (lat is None) != (lon is None):
            raise ValueError("give both lat and lon, or neither")
        if lat is not None and not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("invalid GPS coordinates")
        if source not in ("web", "app"):
            raise ValueError("source must be 'web' or 'app'")
        clean_holds = None
        if holds is not None:
            if not isinstance(holds, list) or len(holds) > 500:
                raise ValueError("holds must be a list of at most 500 items")
            clean_holds = []
            for h in holds:
                try:
                    x, y = float(h["x"]), float(h["y"])
                    conf = float(h.get("confidence", 1.0) or 1.0)
                except (KeyError, TypeError, ValueError):
                    raise ValueError("each hold needs numeric x and y")
                if not _in01(x, y) or not (math.isfinite(conf) and 0 <= conf <= 1):
                    raise ValueError("hold coordinates and confidence must be within 0..1")
                clean_holds.append((x, y, conf, str(h.get("type") or "device")[:20]))
        if not allow_duplicate:
            same = [r for r in self.find_similar(venue, name) if r["name"].lower() == name.lower()]
            if same:
                raise DuplicateWallError(same)
        if image_bytes is None:
            if not image_path:
                raise ValueError("a photo is required")
            with open(image_path, "rb") as f:
                image_bytes = f.read()
        if not image_bytes:
            raise ValueError("a photo is required")
        try:
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        except cv2.error:
            img = None
        if img is None:
            raise ValueError("the photo could not be decoded as an image")
        ih, iw = img.shape[:2]
        if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            ext = "png"
        elif image_bytes[:2] == b"\xff\xd8":
            ext = "jpg"
        else:
            ext, image_bytes = "png", cv2.imencode(".png", img)[1].tobytes()

        wall_id = str(uuid.uuid4())
        rel = f"walls/{wall_id}/original.{ext}"
        folder = os.path.join(self.root, "walls", wall_id)
        os.makedirs(folder)
        try:
            with open(os.path.join(self.root, rel), "wb") as f:
                f.write(image_bytes)
            ts = _now()
            with self._conn() as c:
                c.execute("INSERT INTO walls(wall_id,name,venue,mode,photo_ref,image_w,image_h,lat,lon,"
                          "vuforia_target,created_at,updated_at,algo_version,schema_version,source,reviewed) "
                          "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                          (wall_id, name, venue, mode, rel, iw, ih, lat, lon, vuforia_target,
                           ts, ts, ALGO_VERSION, SCHEMA_VERSION, source, 0 if source == "app" else 1))
                for x, y, conf, typ in clean_holds or []:
                    hid = self._next_hold_id(c, wall_id)
                    c.execute("INSERT INTO holds(wall_id,hold_id,x,y,orig_x,orig_y,confidence,type,status) "
                              "VALUES(?,?,?,?,?,?,?,?,'detected')", (wall_id, hid, x, y, x, y, conf, typ))
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        if auto_detect and clean_holds is None:
            self.redetect(wall_id)
        return wall_id

    def set_reviewed(self, wall_id: str, reviewed: bool = True) -> None:
        """Admin marks an app upload as checked (holds/routes corrected where needed)."""
        with self._conn() as c:
            self._wall_row(c, wall_id)
            c.execute("UPDATE walls SET reviewed=?, updated_at=? WHERE wall_id=?", (1 if reviewed else 0, _now(), wall_id))

    def _wall_row(self, c, wall_id):
        row = c.execute("SELECT * FROM walls WHERE wall_id=?", (wall_id,)).fetchone()
        if row is None:
            raise WallNotFound(wall_id)
        return row

    def photo_path(self, wall_id: str) -> Optional[str]:
        with self._conn() as c:
            ref = self._wall_row(c, wall_id)["photo_ref"]
        if not ref:
            return None
        p = os.path.abspath(os.path.join(self.root, ref))
        if not p.startswith(self.root) or not os.path.isfile(p):
            return None
        return p

    def search_walls(self, venue: Optional[str] = None, name: Optional[str] = None,
                     mode: Optional[str] = None, near: Optional[tuple] = None, limit: int = 50,
                     needs_review: bool = False, text: Optional[str] = None) -> List[dict]:
        """near = (lat, lon, radius_km). GPS is optional; text search works without it.
        text matches venue OR name (one search box in the app)."""
        q, args = "SELECT * FROM walls WHERE 1=1", []
        if text:
            q += " AND (lower(venue) LIKE ? OR lower(name) LIKE ?)"
            args += [f"%{text.strip().lower()}%"] * 2
        if needs_review:
            q += " AND reviewed=0"
        if venue:
            q += " AND lower(venue) LIKE ?"; args.append(f"%{venue.strip().lower()}%")
        if name:
            q += " AND lower(name) LIKE ?"; args.append(f"%{name.strip().lower()}%")
        if mode:
            q += " AND mode=?"; args.append(mode)
        q += " ORDER BY venue COLLATE NOCASE, name COLLATE NOCASE"
        with self._conn() as c:
            rows = [dict(r) for r in c.execute(q, args).fetchall()]
        if near:
            lat, lon, rad = near
            rows = [r for r in rows if r["lat"] is not None
                    and _haversine_km(lat, lon, r["lat"], r["lon"]) <= rad]
        out = []
        for r in rows[:limit]:
            out.append({k: r[k] for k in ("wall_id", "name", "venue", "mode", "lat", "lon",
                                          "vuforia_target", "updated_at", "source")}
                       | {"reviewed": bool(r["reviewed"])}
                       | {"photo_ref": r["photo_ref"],
                          "photo_available": self.photo_path(r["wall_id"]) is not None})
        return out

    def update_wall(self, wall_id: str, allow_duplicate: bool = False, **fields) -> None:
        allowed = {"name", "venue", "mode", "lat", "lon", "vuforia_target"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"cannot update fields: {sorted(bad)}")
        with self._conn() as c:
            row = self._wall_row(c, wall_id)
            new = {**dict(row), **fields}
            if new["mode"] not in MODES:
                raise ValueError(f"mode must be one of {MODES}")
            if not str(new["name"]).strip() or not str(new["venue"]).strip():
                raise ValueError("wall name and venue name are required")
            if not allow_duplicate and ({"name", "venue"} & set(fields)):
                dup = c.execute("SELECT wall_id FROM walls WHERE lower(venue)=lower(?) AND lower(name)=lower(?) "
                                "AND wall_id<>?", (new["venue"].strip(), new["name"].strip(), wall_id)).fetchone()
                if dup:
                    raise DuplicateWallError([dict(dup)])
            sets = ", ".join(f"{k}=?" for k in fields)
            if sets:
                c.execute(f"UPDATE walls SET {sets}, updated_at=? WHERE wall_id=?",
                          (*fields.values(), _now(), wall_id))

    def delete_wall(self, wall_id: str, confirm: bool = False) -> None:
        if not confirm:
            raise ValueError("deleting a wall removes its photo, holds and routes; pass confirm=True")
        with self._conn() as c:
            self._wall_row(c, wall_id)
            c.execute("DELETE FROM walls WHERE wall_id=?", (wall_id,))
        shutil.rmtree(os.path.join(self.root, "walls", wall_id), ignore_errors=True)

    # ------------------------------------------------------------------ holds

    def _touch(self, c, wall_id):
        c.execute("UPDATE walls SET updated_at=? WHERE wall_id=?", (_now(), wall_id))

    def _next_hold_id(self, c, wall_id) -> str:
        n = c.execute("SELECT next_hold_no FROM walls WHERE wall_id=?", (wall_id,)).fetchone()[0]
        c.execute("UPDATE walls SET next_hold_no=? WHERE wall_id=?", (n + 1, wall_id))
        return hold_label(n)          # IDs are never reused, even after deletion

    def redetect(self, wall_id: str) -> int:
        """Run the mode's detector. ADDS new candidates only; never removes, moves or renumbers existing
        holds, and skips candidates near any existing/moved/deleted hold. Returns number added."""
        path = self.photo_path(wall_id)
        if path is None:
            raise FileNotFoundError("photo file is missing or unreadable")
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError("photo file is missing or unreadable")
        with self._conn() as c:
            mode = self._wall_row(c, wall_id)["mode"]
        found = (IndoorDetector() if mode == "indoor" else ClassicalDetector()).detect(img)
        added = 0
        with self._conn() as c:
            existing = c.execute("SELECT x,y,orig_x,orig_y FROM holds WHERE wall_id=?", (wall_id,)).fetchall()
            near = [(r["x"], r["y"]) for r in existing] + [(r["orig_x"], r["orig_y"]) for r in existing
                                                           if r["orig_x"] is not None]
            for hd in found:
                if not _in01(hd.x, hd.y) or any(math.hypot(hd.x - a, hd.y - b) < 0.02 for a, b in near):
                    continue
                hid = self._next_hold_id(c, wall_id)
                c.execute("INSERT INTO holds(wall_id,hold_id,x,y,orig_x,orig_y,confidence,type,status) "
                          "VALUES(?,?,?,?,?,?,?,?,'detected')",
                          (wall_id, hid, hd.x, hd.y, hd.x, hd.y, hd.confidence, hd.type))
                near.append((hd.x, hd.y))
                added += 1
            self._touch(c, wall_id)
        return added

    def add_hold(self, wall_id: str, x: float, y: float, confidence: float = 1.0) -> str:
        if not _in01(x, y):
            raise ValueError("x and y must be within 0..1")
        with self._conn() as c:
            self._wall_row(c, wall_id)
            hid = self._next_hold_id(c, wall_id)
            c.execute("INSERT INTO holds(wall_id,hold_id,x,y,orig_x,orig_y,confidence,type,status) "
                      "VALUES(?,?,?,?,NULL,NULL,?, 'manual','manual_added')",
                      (wall_id, hid, x, y, confidence))
            self._touch(c, wall_id)
        return hid

    def move_hold(self, wall_id: str, hold_id: str, x: float, y: float) -> None:
        if not _in01(x, y):
            raise ValueError("x and y must be within 0..1")
        with self._conn() as c:
            row = c.execute("SELECT status FROM holds WHERE wall_id=? AND hold_id=?", (wall_id, hold_id)).fetchone()
            if row is None or row["status"] == "deleted":
                raise KeyError(hold_id)
            status = "manual_added" if row["status"] == "manual_added" else "moved"
            c.execute("UPDATE holds SET x=?, y=?, status=? WHERE wall_id=? AND hold_id=?",
                      (x, y, status, wall_id, hold_id))
            self._touch(c, wall_id)

    def delete_hold(self, wall_id: str, hold_id: str) -> None:
        """Detected holds become tombstones (status 'deleted', kept for the evaluation report and
        so re-detection does not bring them back). Manual holds are simply removed."""
        with self._conn() as c:
            row = c.execute("SELECT status FROM holds WHERE wall_id=? AND hold_id=?", (wall_id, hold_id)).fetchone()
            if row is None:
                raise KeyError(hold_id)
            if row["status"] == "manual_added":
                c.execute("DELETE FROM holds WHERE wall_id=? AND hold_id=?", (wall_id, hold_id))
            else:
                c.execute("UPDATE holds SET status='deleted' WHERE wall_id=? AND hold_id=?", (wall_id, hold_id))
            self._touch(c, wall_id)

    def _all_holds(self, c, wall_id) -> List[dict]:
        rows = c.execute("SELECT * FROM holds WHERE wall_id=?", (wall_id,)).fetchall()
        rows = [dict(r) for r in rows]
        rows.sort(key=lambda r: int(r["hold_id"][1:]) if r["hold_id"][1:].isdigit() else 0)
        return rows

    def active_holds(self, wall_id: str) -> List[Hold]:
        with self._conn() as c:
            self._wall_row(c, wall_id)
            rows = self._all_holds(c, wall_id)
        return [Hold(id=r["hold_id"], x=r["x"], y=r["y"], confidence=r["confidence"] or 1.0, type=r["type"] or "")
                for r in rows if r["status"] != "deleted"]

    def detection_report(self, wall_id: str) -> dict:
        """Quality measured from the user's own corrections (needs manual review to mean anything)."""
        with self._conn() as c:
            rows = self._all_holds(c, wall_id)
        det = [r for r in rows if r["status"] in ("detected", "moved", "deleted")]
        false_det = sum(1 for r in det if r["status"] == "deleted")
        moved = sum(1 for r in det if r["status"] == "moved")
        missed = sum(1 for r in rows if r["status"] == "manual_added")
        correct = len(det) - false_det
        return {"auto_detected": len(det), "false_detections": false_det, "moved_by_user": moved,
                "missed_added_by_user": missed,
                "precision": round(correct / len(det), 3) if det else None,
                "recall": round(correct / (correct + missed), 3) if (correct + missed) else None,
                "note": "only meaningful if the user reviewed the whole wall"}

    # ------------------------------------------------------------------ routes

    def generate_routes_for(self, wall_id: str, max_routes: int = 6, overlap_threshold: float = 0.6,
                            start_frac: float = 0.30, target_frac: float = 0.15) -> dict:
        """(Re)generate routes from the saved, user-corrected holds. User-edited routes are kept."""
        with self._conn() as c:
            row = self._wall_row(c, wall_id)
            aspect = row["image_h"] / float(row["image_w"])
        holds = self.active_holds(wall_id)
        routes, report = generate_routes(holds, aspect, max_routes, overlap_threshold, start_frac, target_frac)
        if not routes and len(holds) >= 5:
            # Reach is measured in wall widths and assumes a wall several metres wide. On a close-up with few,
            # large holds every gap looks too long, so retry with a relaxed limit and SAY so.
            for scale in (1.5, 2.0):
                r2, rep2 = generate_routes(holds, aspect, max_routes, overlap_threshold, start_frac, target_frac,
                                           reach_scale=scale)
                if r2:
                    routes, report = r2, rep2
                    report["warnings"].append(f"holds are far apart: reach limit relaxed to {scale}x; "
                                              "difficulty is relative to this photo")
                    break
        with self._conn() as c:
            for r in c.execute("SELECT route_id,data FROM routes WHERE wall_id=?", (wall_id,)).fetchall():
                try:
                    if not json.loads(r["data"]).get("edited"):
                        c.execute("DELETE FROM routes WHERE wall_id=? AND route_id=?", (wall_id, r["route_id"]))
                except ValueError:
                    c.execute("DELETE FROM routes WHERE wall_id=? AND route_id=?", (wall_id, r["route_id"]))
            for r in routes:
                n = c.execute("SELECT next_route_no FROM walls WHERE wall_id=?", (wall_id,)).fetchone()[0]
                c.execute("UPDATE walls SET next_route_no=? WHERE wall_id=?", (n + 1, wall_id))
                rid = f"R{n}"
                c.execute("INSERT INTO routes(wall_id,route_id,data,created_at) VALUES(?,?,?,?)",
                          (wall_id, rid, json.dumps({"hold_ids": r["hold_ids"], "limbs": {}, "edited": False,
                                                     "source": "generated"}), _now()))
            self._touch(c, wall_id)
        try:
            self.save_overlay(wall_id)
        except FileNotFoundError:
            report["warnings"].append("photo missing: overlay not saved")
        return report

    def _materialize(self, c, wall_id, aspect, holds_by_id) -> List[dict]:
        out, rows = [], c.execute("SELECT route_id,data FROM routes WHERE wall_id=?", (wall_id,)).fetchall()
        rows = sorted(rows, key=lambda r: int(r["route_id"][1:]))
        for r in rows:
            try:
                data = json.loads(r["data"])
                ids = data["hold_ids"]
            except (ValueError, KeyError, TypeError):
                out.append({"id": r["route_id"], "valid": False, "invalid_reason": "corrupt route record",
                            "points": [], "steps": [], "hold_ids": [], "difficulty": "medium",
                            "color": "#999999", "edited": False})
                continue
            path = [holds_by_id[i] for i in ids if i in holds_by_id]
            if len(path) != len(ids) or len(path) < 2:
                out.append({"id": r["route_id"], "valid": False, "invalid_reason": "route uses a deleted hold",
                            "points": [], "steps": [], "hold_ids": ids, "difficulty": "medium",
                            "color": "#999999", "edited": bool(data.get("edited"))})
                continue
            d = describe_route(path, aspect, data.get("limbs"))
            d.update({"id": r["route_id"], "valid": True, "edited": bool(data.get("edited")),
                      "source": data.get("source", "generated")})
            out.append(d)
        return out

    def set_limb(self, wall_id: str, route_id: str, hold_id: str, limb: Optional[str]) -> None:
        """limb=None removes the user's choice (back to the automatic estimate)."""
        if limb is not None and limb not in LIMBS:
            raise ValueError(f"limb must be one of {LIMBS}")
        with self._conn() as c:
            row = c.execute("SELECT data FROM routes WHERE wall_id=? AND route_id=?", (wall_id, route_id)).fetchone()
            if row is None:
                raise KeyError(route_id)
            data = json.loads(row["data"])
            if hold_id not in data["hold_ids"]:
                raise KeyError(hold_id)
            if limb is None:
                data.setdefault("limbs", {}).pop(hold_id, None)
            else:
                data.setdefault("limbs", {})[hold_id] = {"limb": limb, "status": "confirmed"}
            c.execute("UPDATE routes SET data=? WHERE wall_id=? AND route_id=?", (json.dumps(data), wall_id, route_id))
            self._touch(c, wall_id)

    def set_route_sequence(self, wall_id: str, route_id: str, hold_ids: List[str]) -> None:
        active = {h.id for h in self.active_holds(wall_id)}
        if len(hold_ids) < 2 or len(set(hold_ids)) != len(hold_ids) or any(h not in active for h in hold_ids):
            raise ValueError("sequence needs 2+ distinct, existing, non-deleted holds")
        with self._conn() as c:
            row = c.execute("SELECT data FROM routes WHERE wall_id=? AND route_id=?", (wall_id, route_id)).fetchone()
            if row is None:
                raise KeyError(route_id)
            data = json.loads(row["data"])
            data["hold_ids"] = list(hold_ids)
            data["limbs"] = {k: v for k, v in data.get("limbs", {}).items() if k in hold_ids}
            data["edited"], data["source"] = True, "edited"
            c.execute("UPDATE routes SET data=? WHERE wall_id=? AND route_id=?", (json.dumps(data), wall_id, route_id))
            self._touch(c, wall_id)

    def add_route(self, wall_id: str, hold_ids: List[str]) -> str:
        """Admin draws a route by hand (holds in climbing order). Kept when routes are regenerated."""
        active = {h.id for h in self.active_holds(wall_id)}
        if len(hold_ids) < 2 or len(set(hold_ids)) != len(hold_ids) or any(h not in active for h in hold_ids):
            raise ValueError("a route needs 2+ distinct, existing, non-deleted holds")
        with self._conn() as c:
            n = c.execute("SELECT next_route_no FROM walls WHERE wall_id=?", (wall_id,)).fetchone()[0]
            c.execute("UPDATE walls SET next_route_no=? WHERE wall_id=?", (n + 1, wall_id))
            rid = f"R{n}"
            c.execute("INSERT INTO routes(wall_id,route_id,data,created_at) VALUES(?,?,?,?)",
                      (wall_id, rid, json.dumps({"hold_ids": list(hold_ids), "limbs": {}, "edited": True,
                                                 "source": "manual"}), _now()))
            self._touch(c, wall_id)
        return rid

    def delete_route(self, wall_id: str, route_id: str) -> None:
        with self._conn() as c:
            self._wall_row(c, wall_id)
            if c.execute("DELETE FROM routes WHERE wall_id=? AND route_id=?", (wall_id, route_id)).rowcount == 0:
                raise KeyError(route_id)
            self._touch(c, wall_id)

    # ------------------------------------------------------------------ read / export / overlay

    def get_wall(self, wall_id: str) -> dict:
        with self._conn() as c:
            row = dict(self._wall_row(c, wall_id))
            holds = self._all_holds(c, wall_id)
            aspect = row["image_h"] / float(row["image_w"])
            by_id = {h["hold_id"]: Hold(id=h["hold_id"], x=h["x"], y=h["y"], confidence=h["confidence"] or 1.0)
                     for h in holds if h["status"] != "deleted"}
            routes = self._materialize(c, wall_id, aspect, by_id)
        row["aspect"] = round(aspect, 5)
        row["photo_status"] = "ok" if self.photo_path(wall_id) else "missing"
        row["holds"] = [{"id": h["hold_id"], "x": h["x"], "y": h["y"], "confidence": h["confidence"],
                         "type": h["type"], "status": h["status"]} for h in holds]
        row["routes"] = routes
        return row

    def save_overlay(self, wall_id: str) -> str:
        path = self.photo_path(wall_id)
        img = cv2.imread(path) if path else None
        if img is None:
            raise FileNotFoundError("photo file is missing or unreadable")
        wall = self.get_wall(wall_id)
        out = render_overlay(img, wall["holds"], wall["routes"])
        out_path = os.path.join(self.root, "walls", wall_id, "overlay.png")
        cv2.imwrite(out_path, out)
        return out_path

    def export_doc(self, wall_id: str) -> dict:
        """The schema-2 document Unity reads (also served over HTTP). Only valid routes are included."""
        wall = self.get_wall(wall_id)
        warnings = [f"{r['id']}: {r['invalid_reason']} (not exported)" for r in wall["routes"] if not r["valid"]]
        routes = [r for r in wall["routes"] if r["valid"]]
        return {"schema": SCHEMA_VERSION, "wall_id": wall_id, "scan_id": "", "name": wall["name"],
                "venue": wall["venue"], "mode": wall["mode"], "vuforia_target": wall["vuforia_target"] or "",
                "coordinate_frame": "stored_photo", "image_width": wall["image_w"], "image_height": wall["image_h"],
                "aspect": wall["aspect"], "coordinate_system": COORD_SYSTEM, "algo_version": ALGO_VERSION,
                "disclaimer": DISCLAIMER, "updated_at": wall["updated_at"],
                "holds": [h for h in wall["holds"] if h["status"] != "deleted"],
                "routes": [{k: r[k] for k in ("id", "difficulty", "color", "estimated_score", "feasibility",
                                               "hold_ids", "points", "steps", "metrics", "edited")}
                           for r in routes],
                "warnings": warnings}

    def export_unity(self, wall_id: str, out_dir: str) -> str:
        """Writes <out_dir>/<wall_id>.routes.json and merges target -> wall_id into <out_dir>/index.json."""
        doc = self.export_doc(wall_id)
        wall = {"vuforia_target": doc["vuforia_target"] or None}
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{wall_id}.routes.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        idx_path = os.path.join(out_dir, "index.json")
        idx = {"targets": []}
        if os.path.isfile(idx_path):
            try:
                with open(idx_path, encoding="utf-8") as f:
                    idx = json.load(f)
            except ValueError:
                pass
        idx["targets"] = [t for t in idx.get("targets", []) if t.get("wall_id") != wall_id
                          and t.get("target") != wall["vuforia_target"]]
        if wall["vuforia_target"]:
            idx["targets"].append({"target": wall["vuforia_target"], "wall_id": wall_id})
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(idx, f, indent=2)
        return path
