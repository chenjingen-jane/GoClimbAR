"""Checks that the C# classes in RouteSetData.cs match the JSON the server really sends.
Unity's JsonUtility silently ignores mismatched names (you would just see empty data), so this matters.
Looks for the C# file in ../unity/Scripts next to the goclimb_routes folder, or in $GOCLIMB_UNITY_SCRIPTS."""
import base64
import os
import re
import tempfile
import unittest

from goclimb_routes.scan import ScanService
from goclimb_routes.store import Store
from .fixtures import png_bytes, synthetic_indoor_wall


def find_cs():
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.environ.get("GOCLIMB_UNITY_SCRIPTS", ""),
             os.path.normpath(os.path.join(here, "..", "..", "Assets", "Script")),     # Unity project layout
             os.path.normpath(os.path.join(here, "..", "..", "unity", "Scripts"))]
    for d in cands:
        p = os.path.join(d, "RouteSetData.cs") if d else ""
        if p and os.path.isfile(p):
            return p
    return None


def parse_classes(path):
    with open(path, encoding="utf-8") as f:
        src = f.read()
    out = {}
    for m in re.finditer(r"public class (\w+)\s*\{(.*?)\n\}", src, re.S):
        out[m.group(1)] = re.findall(r"public\s+([\w<>]+)\s+(\w+)\s*;", m.group(2))
    return out


PRIMS = {"string": str, "int": int, "float": (int, float), "bool": bool}


def check(classes, cls, obj, where, errors):
    for typ, name in classes[cls]:
        if name not in obj:
            errors.append(f"{where}.{name} missing in JSON (C# class {cls})")
            continue
        val = obj[name]
        if typ in PRIMS:
            if not isinstance(val, PRIMS[typ]) and not (typ == "string" and val is None):
                errors.append(f"{where}.{name}: C# {typ} but JSON {type(val).__name__}")
        elif typ.startswith("List<"):
            inner = typ[5:-1]
            if not isinstance(val, list):
                errors.append(f"{where}.{name}: C# {typ} but JSON {type(val).__name__}")
            elif inner in classes:
                for i, item in enumerate(val[:3]):
                    check(classes, inner, item, f"{where}.{name}[{i}]", errors)
            elif inner in PRIMS and any(not isinstance(v, PRIMS[inner]) for v in val):
                errors.append(f"{where}.{name}: list items are not {inner}")
        elif typ in classes:
            check(classes, typ, val, f"{where}.{name}", errors)


class UnityContractTests(unittest.TestCase):
    def test_csharp_classes_match_server_json(self):
        path = find_cs()
        if path is None:
            self.skipTest("RouteSetData.cs not found (set GOCLIMB_UNITY_SCRIPTS to its folder)")
        classes = parse_classes(path)
        st = Store(tempfile.mkdtemp())
        svc = ScanService(st)
        b64 = base64.b64encode(png_bytes(synthetic_indoor_wall()[0])).decode()
        unknown = svc.scan({"image_b64": b64, "mode": "indoor"})
        self.assertEqual(unknown["status"], "unknown")
        self.assertTrue(unknown["doc"]["routes"], "fixture should produce routes so they can be compared")
        saved = svc.save(unknown["scan_id"], {"venue": "Contract", "name": "W"})
        known = svc.scan({"image_b64": b64, "mode": "indoor"})
        self.assertEqual(known["status"], "known")
        errors = []
        check(classes, "GCScanResponse", unknown, "scan(unknown)", errors)
        check(classes, "GCScanResponse", known, "scan(known)", errors)
        check(classes, "GCSaveResponse", saved, "save", errors)
        check(classes, "GCVersion", {"api_version": __import__("goclimb_routes.schema", fromlist=["x"]).API_VERSION}, "version", errors)
        check(classes, "GCDocResponse", svc.regenerate(known["scan_id"], 3), "regenerate", errors)
        check(classes, "GCWallList", {"walls": st.search_walls()}, "list", errors)
        check(classes, "GCRouteSet", st.export_doc(saved["wall_id"]), "stored doc", errors)
        # API 4: the reply to an upload from the phone (POST /api/walls) and the request it sends
        up = st.create_wall("Up", "Contract", "indoor", image_bytes=png_bytes(synthetic_indoor_wall()[0]),
                            holds=[{"x": .2, "y": .3, "confidence": .9, "type": "candidate"}], source="app")
        check(classes, "GCUploadResponse", {"wall_id": up, "created": True, "doc": st.export_doc(up)}, "upload", errors)
        self.assertIn(("List<GCHoldIn>", "holds"), classes["GCUploadRequest"])
        self.assertEqual({n for _, n in classes["GCHoldIn"]}, {"x", "y", "confidence", "type"})
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
