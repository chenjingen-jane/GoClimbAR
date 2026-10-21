"""Usage (from the folder that contains goclimb_routes/):
  python -m goclimb_routes.cli wall.jpg --wall dairy-farm-quarry-rgb --store routes --save --preview preview.png
  python -m goclimb_routes.cli wall.jpg --wall my-wall --holds holds.json    # holds from a trained detector / manual
"""
import argparse
import json

import cv2

from .workflow import get_routes


def draw_preview(image_path, data, out):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    P = lambda x, y: (int(x * w), int(y * h))
    for hd in data["holds"]:
        cv2.circle(img, P(hd["x"], hd["y"]), 4, (255, 255, 255), 1)
    for r in data["routes"]:
        c = r["color"].lstrip("#")
        bgr = (int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16))
        pts = [P(p["x"], p["y"]) for p in r["points"]]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, bgr, 2, cv2.LINE_AA)
        for p, q in zip(r["points"], pts):
            cv2.circle(img, q, 8 if p["role"] != "hold" else 5, bgr, -1)
            if p["role"] != "hold":
                cv2.putText(img, p["role"][0].upper(), (q[0] + 8, q[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--wall", required=True)
    ap.add_argument("--store", default="routes")
    ap.add_argument("--holds", help="JSON list of {x,y,confidence?,type?} (normalized, y down)")
    ap.add_argument("--alternatives", type=int, default=1)
    ap.add_argument("--mode", choices=["outdoor", "gym"], default="outdoor",
                    help="gym = indoor coloured holds; each colour becomes one route")
    ap.add_argument("--save", action="store_true", help="persist result (approval step)")
    ap.add_argument("--preview")
    a = ap.parse_args()
    holds = json.load(open(a.holds)) if a.holds else None
    data = get_routes(a.wall, a.store, a.image, holds, save=a.save, alternatives=a.alternatives, mode=a.mode)
    print(f"source={data['source']} holds={len(data['holds'])} routes={len(data['routes'])}")
    for r in data["routes"]:
        print(f"  {r['id']}: {r['metrics']}")
    for w in data["warnings"]:
        print("  WARNING:", w)
    if a.preview:
        draw_preview(a.image, data, a.preview)


if __name__ == "__main__":
    main()
