"""Graph-based route planner. Holds are nodes; plausible reaches are edges.

Internally uses 'up-y' coordinates (y increases upward) and measures distance in
wall-width units: d = hypot(dx, dy * aspect). Distances are image-space proxies,
NOT meters. Difficulty is an ESTIMATE from geometry only, never an official grade.
Edges only go upward, so every route is y-monotone and cannot cross itself.
"""
from __future__ import annotations

import dataclasses
import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .detect import Hold, validate_holds

MIN_HOLDS_PER_ROUTE = 5


@dataclass(frozen=True)
class Profile:
    max_reach: float   # longest allowed move (wall widths)
    min_reach: float   # ignore moves shorter than this (holds clumped together)
    min_rise: float    # minimum upward progress per move
    p: float           # distance exponent: >1 favours many short moves, <1 few long moves
    c0: float          # constant cost per move
    lateral: float     # weight of sideways movement
    color: str


PROFILES: Dict[str, Profile] = {
    "easy":   Profile(0.14, 0.03, 0.012, 2.0, 0.004, 1.5, "#22C55E"),
    "medium": Profile(0.19, 0.03, 0.012, 1.2, 0.010, 1.0, "#F59E0B"),
    "hard":   Profile(0.26, 0.03, 0.010, 0.7, 0.030, 0.6, "#EF4444"),
}


def _geom(a: Hold, b: Hold, aspect: float) -> Tuple[float, float, float]:
    dx = b.x - a.x
    dy = (a.y - b.y) * aspect          # positive = b is higher (image y is down)
    return dx, dy, math.hypot(dx, dy)


def _edge_cost(a: Hold, b: Hold, prof: Profile, aspect: float) -> Optional[float]:
    dx, dy, d = _geom(a, b, aspect)
    if dy < prof.min_rise or d > prof.max_reach or d < prof.min_reach:
        return None
    quality = 0.5 + 0.5 * min(a.confidence, b.confidence)
    return (prof.c0 + d ** prof.p + prof.lateral * abs(dx) * 0.5) / quality


def _shortest_path(holds: List[Hold], prof: Profile, aspect: float,
                   starts: List[int], ends: set, node_penalty: Dict[int, float]) -> Optional[List[int]]:
    """Dijkstra over states (hold, min(hold_count, MIN_HOLDS_PER_ROUTE)) so the
    minimum-length constraint does not break optimality."""
    cap = MIN_HOLDS_PER_ROUTE
    dist: Dict[Tuple[int, int], float] = {(i, 1): node_penalty.get(i, 0.0) for i in starts}
    prev: Dict[Tuple[int, int], Tuple[int, int]] = {}
    heap = [(c, s) for s, c in dist.items()]
    heapq.heapify(heap)
    goal = None
    while heap:
        c, state = heapq.heappop(heap)
        if c > dist.get(state, math.inf):
            continue
        i, k = state
        if i in ends and k >= cap:
            goal = state
            break
        for j, hj in enumerate(holds):
            if j == i:
                continue
            w = _edge_cost(holds[i], hj, prof, aspect)
            if w is None:
                continue
            ns = (j, min(k + 1, cap))
            nc = c + w + node_penalty.get(j, 0.0)
            if nc < dist.get(ns, math.inf):
                dist[ns] = nc
                prev[ns] = state
                heapq.heappush(heap, (nc, ns))
    if goal is None:
        return None
    path = [goal[0]]
    while goal in prev:
        goal = prev[goal]
        path.append(goal[0])
    return path[::-1]


def route_metrics(path: List[Hold], aspect: float) -> dict:
    reaches, lat, rise = [], 0.0, 0.0
    for a, b in zip(path, path[1:]):
        dx, dy, d = _geom(a, b, aspect)
        reaches.append(d)
        lat += abs(dx)
        rise += dy
    mean_r = sum(reaches) / len(reaches)
    lat_ratio = lat / rise if rise > 1e-9 else 0.0
    score = (0.5 * min(1, mean_r / 0.22) + 0.3 * min(1, lat_ratio)
             + 0.2 * max(0.0, min(1, 1 - len(path) / 12)))
    label = "easy" if score < 0.33 else ("medium" if score < 0.55 else "hard")
    return {"hold_count": len(path), "mean_reach": round(mean_r, 4),
            "max_reach": round(max(reaches), 4), "lateral_ratio": round(lat_ratio, 4),
            "estimated_score": round(score, 3), "estimated_label": label}


def plan_routes(holds: List[Hold], aspect: float, alternatives: int = 1,
                difficulties=("easy", "medium", "hard")) -> Tuple[List[dict], List[str]]:
    """Returns (routes, warnings). Never raises on bad input."""
    warnings: List[str] = []
    holds, w = validate_holds(list(holds))
    warnings += w
    if aspect <= 0 or not math.isfinite(aspect):
        return [], warnings + ["invalid aspect ratio"]
    if len(holds) < MIN_HOLDS_PER_ROUTE:
        return [], warnings + [f"insufficient holds ({len(holds)}); need at least {MIN_HOLDS_PER_ROUTE}"]

    ys = [h.y for h in holds]
    top, bottom = min(ys), max(ys)          # image y: smaller = higher
    span = bottom - top
    if span < 0.2:
        return [], warnings + ["holds do not span enough of the wall height"]
    start_ids = [i for i, h in enumerate(holds) if h.y >= bottom - 0.30 * span]
    end_ids = {i for i, h in enumerate(holds) if h.y <= top + 0.15 * span}

    routes: List[dict] = []
    used_by_all: Dict[int, float] = {}
    for diff in difficulties:
        prof = PROFILES[diff]
        penalty: Dict[int, float] = dict(used_by_all)   # keep difficulties distinct
        found_for_diff = 0
        for alt in range(alternatives):
            path_ids = _shortest_path(holds, prof, aspect, start_ids, end_ids, penalty)
            if path_ids is None:
                if alt == 0:
                    warnings.append(f"no valid {diff} route (holds too sparse for reach limit)")
                break
            for i in path_ids:
                penalty[i] = penalty.get(i, 0.0) + 0.15
                used_by_all[i] = used_by_all.get(i, 0.0) + 0.05
            path = [holds[i] for i in path_ids]
            pts = [{"x": h.x, "y": h.y,
                    "role": "start" if k == 0 else ("finish" if k == len(path) - 1 else "hold")}
                   for k, h in enumerate(path)]
            routes.append({"id": f"{diff}-{alt + 1}", "difficulty": diff, "color": prof.color,
                           "hold_ids": path_ids, "points": pts,
                           "metrics": route_metrics(path, aspect)})
            found_for_diff += 1
    return routes, warnings


def setter_routes(holds: List[Hold], aspect: float) -> Tuple[List[dict], List[str]]:
    """Gym mode: every colour group with enough holds is one setter route, ordered bottom to top.
    No path search - the setter already chose these holds."""
    warnings: List[str] = []
    holds, w = validate_holds(list(holds))
    warnings += w
    if aspect <= 0 or not math.isfinite(aspect):
        return [], warnings + ["invalid aspect ratio"]
    groups: Dict[str, List[Hold]] = {}
    for h in holds:
        if h.group:
            groups.setdefault(h.group, []).append(h)
    routes: List[dict] = []
    for name, hs in sorted(groups.items()):
        if len(hs) < MIN_HOLDS_PER_ROUTE:
            warnings.append(f"colour '{name}': only {len(hs)} holds, skipped (need {MIN_HOLDS_PER_ROUTE})")
            continue
        hs = sorted(hs, key=lambda q: -q.y)                 # image y down: largest y = lowest hold
        _, _, biggest = max((_geom(a, b, aspect) for a, b in zip(hs, hs[1:])), key=lambda t: t[2])
        if biggest > 0.35:
            warnings.append(f"colour '{name}': gap of {biggest:.2f} wall widths - may be two routes of the same colour")
        pts = [{"x": q.x, "y": q.y, "role": "start" if k == 0 else ("finish" if k == len(hs) - 1 else "hold")}
               for k, q in enumerate(hs)]
        mid = sorted(h.color for h in hs)[len(hs) // 2]
        routes.append({"id": f"setter-{name}", "difficulty": "setter", "color": mid or "#FFFFFF",
                       "hold_ids": [q.id for q in hs], "points": pts,
                       "metrics": route_metrics(hs, aspect)})
    if not routes:
        warnings.append("no colour group has enough holds")
    return routes, warnings


# =========================================================================== schema 2
# Diverse multi-route generation shared by indoor and outdoor modes.
from .schema import DIFFICULTY_COLORS  # noqa: E402

GLOBAL_MAX_REACH = 0.26     # no route may contain a move longer than this (wall widths)


def _clip(v: float) -> float:
    return max(0.0, min(1.0, v))


def estimate_difficulty(path: List[Hold], aspect: float) -> dict:
    """Heuristic difficulty from several geometric factors (NOT only length / hold count)."""
    reaches, dxs, dys = [], [], []
    for a, b in zip(path, path[1:]):
        dx, dy, d = _geom(a, b, aspect)
        reaches.append(d); dxs.append(dx); dys.append(dy)
    moves = max(1, len(reaches))
    mean_r = sum(reaches) / moves
    max_r = max(reaches) if reaches else 0.0
    rise = sum(dys)
    lat_ratio = sum(abs(v) for v in dxs) / rise if rise > 1e-9 else 0.0
    signs = [1 if v > 0.01 else (-1 if v < -0.01 else 0) for v in dxs]
    signs = [s for s in signs if s != 0]
    dir_changes = sum(1 for p, q in zip(signs, signs[1:]) if p != q) / max(1, len(signs) - 1) if len(signs) > 1 else 0.0
    mean_conf = sum(h.confidence for h in path) / len(path)
    score = (0.30 * _clip(max_r / GLOBAL_MAX_REACH) + 0.25 * _clip(mean_r / 0.20)
             + 0.20 * _clip(lat_ratio / 1.0) + 0.10 * dir_changes
             + 0.10 * (1 - mean_conf) + 0.05 * _clip(1 - len(path) / 12))
    label = "easy" if score < 0.40 else ("medium" if score < 0.55 else "hard")
    feasibility = _clip(1 - 0.5 * _clip(max_r / GLOBAL_MAX_REACH) - 0.5 * (1 - mean_conf))
    return {"hold_count": len(path), "mean_reach": round(mean_r, 4), "max_reach": round(max_r, 4),
            "lateral_ratio": round(lat_ratio, 4), "direction_changes": round(dir_changes, 3),
            "mean_confidence": round(mean_conf, 3), "score": round(score, 3),
            "label": label, "feasibility": round(feasibility, 3)}


def build_steps(path: List[Hold], aspect: float) -> List[dict]:
    steps = [{"index": 0, "from": None, "to": str(path[0].id), "text": f"Start on {path[0].id}."}]
    for k, (a, b) in enumerate(zip(path, path[1:]), start=1):
        dx, dy, d = _geom(a, b, aspect)
        side = "straight up" if abs(dx) < 0.02 else ("up and to the left" if dx < 0 else "up and to the right")
        reach = "short" if d < 0.09 else ("medium" if d < 0.16 else "long")
        verb = "Finish on" if k == len(path) - 1 else "Move to"
        steps.append({"index": k, "from": str(a.id), "to": str(b.id),
                      "text": f"{verb} {b.id} ({side}, {reach} reach, estimated)."})
    return steps


def suggest_limbs(path: List[Hold]) -> Dict[str, str]:
    """WEAK estimate: hands only, by left/right position relative to the neighbouring hold.
    A single photo cannot reliably tell which limb a climber uses; users should confirm."""
    out: Dict[str, str] = {}
    for i, h in enumerate(path):
        nb = path[i + 1] if i + 1 < len(path) else path[i - 1]
        if h.x < nb.x - 0.01:
            out[str(h.id)] = "left_hand"
        elif h.x > nb.x + 0.01:
            out[str(h.id)] = "right_hand"
        else:
            out[str(h.id)] = "left_hand" if i % 2 == 0 else "right_hand"
    return out


def describe_route(path: List[Hold], aspect: float, limbs: Optional[dict] = None) -> dict:
    """Everything the UI / Unity needs about one route. `limbs` = {hold_id: {"limb", "status"}}."""
    limbs = limbs or {}
    est = suggest_limbs(path)
    m = estimate_difficulty(path, aspect)
    pts = []
    for k, h in enumerate(path):
        hid = str(h.id)
        if hid in limbs:
            limb, status = limbs[hid]["limb"], limbs[hid].get("status", "confirmed")
        else:
            limb, status = est.get(hid), "estimated"
        pts.append({"hold_id": hid, "x": h.x, "y": h.y,
                    "role": "start" if k == 0 else ("finish" if k == len(path) - 1 else "hold"),
                    "limb": limb or "", "limb_status": status})
    return {"difficulty": m["label"], "color": DIFFICULTY_COLORS[m["label"]],
            "estimated_score": m["score"], "feasibility": m["feasibility"], "metrics": m,
            "hold_ids": [str(h.id) for h in path], "points": pts, "steps": build_steps(path, aspect)}


def _adjacency(holds, prof: Profile, aspect: float):
    adj = [[] for _ in holds]
    for i, a in enumerate(holds):
        for j, b in enumerate(holds):
            if i != j:
                w = _edge_cost(a, b, prof, aspect)
                if w is not None:
                    adj[i].append((j, w))
    return adj


def _dijkstra_adj(adj, starts, ends, penalty, cap=MIN_HOLDS_PER_ROUTE):
    dist = {(i, 1): 0.0 for i in starts}
    prev: Dict[Tuple[int, int], Tuple[int, int]] = {}
    heap = [(0.0, s) for s in dist]
    heapq.heapify(heap)
    goal = None
    while heap:
        c, st = heapq.heappop(heap)
        if c > dist.get(st, math.inf):
            continue
        i, k = st
        if i in ends and k >= cap:
            goal = st
            break
        for j, w in adj[i]:
            ns = (j, min(k + 1, cap))
            nc = c + w + penalty.get(j, 0.0)
            if nc < dist.get(ns, math.inf):
                dist[ns] = nc
                prev[ns] = st
                heapq.heappush(heap, (nc, ns))
    if goal is None:
        return None
    path = [goal[0]]
    while goal in prev:
        goal = prev[goal]
        path.append(goal[0])
    return path[::-1]


def _jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 1.0


def generate_routes(holds: List[Hold], aspect: float, max_routes: int = 6,
                    overlap_threshold: float = 0.6, start_frac: float = 0.30,
                    target_frac: float = 0.15, max_start_nodes: int = 6, rounds: int = 3,
                    reach_scale: float = 1.0):
    """Returns (routes, report). Candidate pool = 3 cost profiles x several start holds x penalty
    rounds. Then: exact-duplicate removal, near-duplicate filter (Jaccard overlap of hold sets),
    greedy selection by feasibility + diversity, capped at max_routes.
    Never invents routes: if the hold graph has no valid path, routes is empty.
    Not exhaustive: it does NOT claim to find every possible route."""
    report = {"candidates": 0, "unique": 0, "near_duplicates_rejected": 0, "retained": 0,
              "max_routes": max_routes, "overlap_threshold": overlap_threshold, "reach_scale": reach_scale,
              "warnings": []}
    profiles = {k: dataclasses.replace(v, max_reach=v.max_reach * reach_scale) for k, v in PROFILES.items()}
    reach_limit = GLOBAL_MAX_REACH * reach_scale
    holds = [h for h in holds if all(math.isfinite(v) and 0 <= v <= 1 for v in (h.x, h.y))]
    if aspect <= 0 or not math.isfinite(aspect):
        report["warnings"].append("invalid aspect ratio"); return [], report
    if max_routes < 1:
        return [], report
    if len(holds) < MIN_HOLDS_PER_ROUTE:
        report["warnings"].append(f"insufficient holds ({len(holds)}); need at least {MIN_HOLDS_PER_ROUTE}")
        return [], report
    ys = [h.y for h in holds]
    top, bottom = min(ys), max(ys)
    span = bottom - top
    if span < 0.2:
        report["warnings"].append("holds do not span enough of the wall height"); return [], report
    start_ids = sorted((i for i, h in enumerate(holds) if h.y >= bottom - start_frac * span),
                       key=lambda i: holds[i].x)
    end_ids = {i for i, h in enumerate(holds) if h.y <= top + target_frac * span}
    if len(start_ids) > max_start_nodes:
        step = (len(start_ids) - 1) / float(max_start_nodes - 1)
        start_ids = [start_ids[int(round(k * step))] for k in range(max_start_nodes)]

    pool = []
    for pname, prof in profiles.items():
        adj = _adjacency(holds, prof, aspect)
        for s in start_ids:
            penalty: Dict[int, float] = {}
            for _ in range(rounds):
                path = _dijkstra_adj(adj, [s], end_ids, penalty)
                if path is None:
                    break
                pool.append(tuple(path))
                for i in path[1:-1]:
                    penalty[i] = penalty.get(i, 0.0) + 0.12
    report["candidates"] = len(pool)
    unique = list(dict.fromkeys(pool))
    report["unique"] = len(unique)
    if not unique:
        report["warnings"].append("no feasible route under the movement constraints")
        return [], report

    desc = {}
    for p in unique:
        path = [holds[i] for i in p]
        if max(_geom(a, b, aspect)[2] for a, b in zip(path, path[1:])) > reach_limit + 1e-9:
            continue            # safety net: reject any move beyond the global reach limit
        desc[p] = describe_route(path, aspect)
    chosen: List[tuple] = []
    remaining = list(desc)
    while remaining and len(chosen) < max_routes:
        best, best_val = None, -1.0
        for p in remaining:
            ov = max((_jaccard(p, q) for q in chosen), default=0.0)
            if ov >= overlap_threshold:
                continue
            bonus = 0.1 if desc[p]["difficulty"] not in {desc[q]["difficulty"] for q in chosen} else 0.0
            val = 0.5 * desc[p]["feasibility"] + 0.5 * (1 - ov) + bonus
            if val > best_val:
                best, best_val = p, val
        if best is None:
            break
        chosen.append(best)
        remaining.remove(best)
    report["near_duplicates_rejected"] = sum(
        1 for p in remaining if max((_jaccard(p, q) for q in chosen), default=0.0) >= overlap_threshold)
    routes = []
    for k, p in enumerate(chosen, start=1):
        d = dict(desc[p]); d["id"] = f"R{k}"
        routes.append(d)
    report["retained"] = len(routes)
    if len(routes) < max_routes:
        report["warnings"].append(f"only {len(routes)} distinct feasible route(s) found; none were invented")
    return routes, report
