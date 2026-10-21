"""Command line for the wall store.
  python -m goclimb_routes.wall_cli --data goclimb_data add photo.jpg --name "Wall A" --venue "My Gym" --mode indoor
  python -m goclimb_routes.wall_cli --data goclimb_data search --venue gym
  python -m goclimb_routes.wall_cli --data goclimb_data generate <wall_id> --max-routes 6
  python -m goclimb_routes.wall_cli --data goclimb_data export <wall_id> --out path/to/Assets/StreamingAssets/Routes
  python -m goclimb_routes.wall_cli --data goclimb_data report <wall_id>
  python -m goclimb_routes.wall_cli --data goclimb_data delete <wall_id> --yes
"""
import argparse
import json

from .store import DuplicateWallError, Store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="goclimb_data")
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("add"); a.add_argument("photo"); a.add_argument("--name", required=True)
    a.add_argument("--venue", required=True); a.add_argument("--mode", required=True, choices=["indoor", "outdoor"])
    a.add_argument("--target"); a.add_argument("--lat", type=float); a.add_argument("--lon", type=float)
    a.add_argument("--allow-duplicate", action="store_true")
    s = sp.add_parser("search"); s.add_argument("--venue"); s.add_argument("--name"); s.add_argument("--mode")
    for n in ("show", "report", "export", "generate", "delete"):
        p = sp.add_parser(n); p.add_argument("wall_id")
        if n == "export": p.add_argument("--out", default="unity_export")
        if n == "generate": p.add_argument("--max-routes", type=int, default=6)
        if n == "delete": p.add_argument("--yes", action="store_true")
    r = ap.parse_args()
    st = Store(r.data)
    if r.cmd == "add":
        try:
            wid = st.create_wall(r.name, r.venue, r.mode, image_path=r.photo, lat=r.lat, lon=r.lon,
                                 vuforia_target=r.target, allow_duplicate=r.allow_duplicate)
        except DuplicateWallError as e:
            raise SystemExit(f"Already exists: {e.existing}. Use --allow-duplicate to create another.")
        print("wall_id:", wid, "holds:", len(st.active_holds(wid)))
    elif r.cmd == "search":
        for w in st.search_walls(r.venue, r.name, r.mode):
            print(w["wall_id"], "|", w["venue"], "|", w["name"], "|", w["mode"])
    elif r.cmd == "show":
        w = st.get_wall(r.wall_id)
        print(w["name"], w["venue"], w["mode"], "holds:", len(w["holds"]))
        for rt in w["routes"]:
            print(" ", rt["id"], rt["difficulty"], " → ".join(rt["hold_ids"]))
    elif r.cmd == "report":
        print(json.dumps(st.detection_report(r.wall_id), indent=2))
    elif r.cmd == "export":
        print("wrote", st.export_unity(r.wall_id, r.out))
    elif r.cmd == "generate":
        print(json.dumps(st.generate_routes_for(r.wall_id, r.max_routes), indent=2))
    elif r.cmd == "delete":
        st.delete_wall(r.wall_id, confirm=r.yes)
        print("deleted")


if __name__ == "__main__":
    main()
