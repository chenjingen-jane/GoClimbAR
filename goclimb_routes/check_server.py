"""
GoClimb server checker.

Put this file in the folder that CONTAINS goclimb_routes\ (the same folder you
run `python -m goclimb_routes.server --data goclimb_data` from), then run:

    python check_server.py                 diagnose
    python check_server.py --port 8000     only check the port Unity uses
    python check_server.py --clean         also delete stale __pycache__ folders

It answers two questions:
  1. Are the goclimb_routes files Python loads from here the NEW ones?
  2. Is the server process actually listening the NEW one, or an old one
     that was never stopped?
"""
import argparse
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import urllib.request
from datetime import datetime

PKG = "goclimb_routes"
DEFAULT_PORTS = [8000, 5000, 8080, 8765, 5005, 8888]
PROBE_PATHS = ["/health", "/api/health", "/version", "/api/version", "/status", "/"]

problems = []


def hdr(title):
    print(f"\n=== {title} ===")


def fmt(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- 1. files
def check_files(clean):
    hdr("1. Which goclimb_routes does Python load from this folder?")
    sys.path.insert(0, os.getcwd())
    spec = importlib.util.find_spec(PKG)
    if not spec or not spec.origin:
        print(f"  NOT FOUND. Run this from the folder that contains {PKG}\\")
        problems.append("Run the checker (and the server) from the folder containing goclimb_routes.")
        return None, None

    pkg_dir = os.path.dirname(os.path.abspath(spec.origin))
    print(f"  Loads from: {pkg_dir}")
    if "site-packages" in pkg_dir:
        print("  !! This is an INSTALLED copy in site-packages, not your project folder.")
        problems.append(f"An installed copy in site-packages shadows your files. Run: pip uninstall {PKG}")

    server_py = os.path.join(pkg_dir, "server.py")
    server_mtime, files_new = None, None
    for name in ("server.py", "schema.py", "scan.py"):
        p = os.path.join(pkg_dir, name)
        if not os.path.exists(p):
            print(f"  {name}: MISSING")
            continue
        txt = open(p, encoding="utf-8", errors="ignore").read()
        found = [m for m in ("api_version", "scan_id") if m in txt]
        print(f"  {name}: modified {fmt(os.path.getmtime(p))}, mentions {found or 'NEITHER api_version NOR scan_id'}")
        if name == "server.py":
            server_mtime = os.path.getmtime(p)
            files_new = "api_version" in txt

    if files_new is False:
        print("  -> server.py is the OLD version.")
        problems.append(f"The files in {pkg_dir} were never replaced. Copy the six v3 .py files INTO this folder.")
    elif files_new:
        print("  -> server.py is the NEW version.")

    # other copies lying around that you might have replaced by mistake
    others = []
    for root, dirs, files in os.walk(os.getcwd()):
        if root[len(os.getcwd()):].count(os.sep) > 4:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in (".git", "node_modules", "Library", "venv", ".venv")]
        if os.path.basename(root) == PKG and "server.py" in files:
            if os.path.abspath(root) != pkg_dir:
                others.append(root)
    if others:
        print("  Other goclimb_routes copies found (Python does NOT use these):")
        for o in others:
            print(f"    {o}")
        problems.append("You have more than one goclimb_routes folder. Make sure you replaced the one listed under 'Loads from'.")

    caches = [os.path.join(r, d) for r, ds, _ in os.walk(pkg_dir) for d in ds if d == "__pycache__"]
    if caches:
        if clean:
            for c in caches:
                shutil.rmtree(c, ignore_errors=True)
            print(f"  Deleted {len(caches)} __pycache__ folder(s).")
        else:
            print(f"  {len(caches)} __pycache__ folder(s) present (harmless usually; --clean removes them).")

    return server_mtime, files_new


# ---------------------------------------------------------------- 2. process
def pids_on_port(port):
    try:
        import psutil
        return sorted({c.pid for c in psutil.net_connections("tcp")
                       if c.laddr and c.laddr.port == port and c.status == "LISTEN" and c.pid})
    except Exception:
        pass
    try:
        if os.name == "nt":
            out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True).stdout
            pids = set()
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 5 and "LISTEN" in parts[3].upper() and parts[1].endswith(f":{port}"):
                    pids.add(int(parts[4]))
            return sorted(pids)
        out = subprocess.run(["lsof", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], capture_output=True, text=True).stdout
        return sorted({int(x) for x in out.split()})
    except Exception:
        return []


def process_info(pid):
    """Return (start_datetime or None, command line str)."""
    try:
        import psutil
        p = psutil.Process(pid)
        return datetime.fromtimestamp(p.create_time()), " ".join(p.cmdline())
    except Exception:
        pass
    try:
        if os.name == "nt":
            cmd = (f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; "
                   "$p.CreationDate.ToString('yyyy-MM-dd HH:mm:ss'); $p.CommandLine")
            out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                                 capture_output=True, text=True).stdout.strip().splitlines()
        else:
            out = subprocess.run(["ps", "-o", "lstart=,args=", "-p", str(pid)],
                                 capture_output=True, text=True).stdout.strip()
            return None, out
        start = datetime.strptime(out[0].strip(), "%Y-%m-%d %H:%M:%S") if out else None
        return start, out[1].strip() if len(out) > 1 else "?"
    except Exception:
        return None, "?"


def probe_http(port):
    for path in PROBE_PATHS:
        url = f"http://127.0.0.1:{port}{path}"
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                body = r.read(800).decode("utf-8", "ignore")
        except Exception:
            continue
        version = None
        try:
            version = json.loads(body).get("api_version")
        except Exception:
            if "api_version" in body:
                version = "present"
        return path, body.strip().replace("\n", " ")[:200], version
    return None, None, None


def check_ports(ports, server_mtime):
    hdr("2. Which server is actually answering?")
    open_ports = []
    for port in ports:
        with socket.socket() as s:
            s.settimeout(0.3)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                open_ports.append(port)
    if not open_ports:
        print(f"  Nothing listening on {ports}.")
        print("  Check the port in Unity's server URL and rerun with --port <that number>.")
        return

    for port in open_ports:
        print(f"\n  Port {port}: LISTENING")
        path, body, version = probe_http(port)
        if path:
            print(f"    GET {path} -> {body}")
            print(f"    api_version in reply: {version if version is not None else 'NOT PRESENT'}")
            if version is None:
                problems.append(f"The server on port {port} does not report api_version, so it is an OLD server.")

        pids = pids_on_port(port)
        if len(pids) > 1:
            print(f"    !! {len(pids)} processes listening on this port: {pids}")
            problems.append(f"Several servers are on port {port}. Kill them all, then start one.")
        for pid in pids:
            start, cmdline = process_info(pid)
            print(f"    PID {pid}, started {start or '?'}")
            print(f"    cmd: {cmdline}")
            if start and server_mtime and start.timestamp() < server_mtime:
                print("    !! This process started BEFORE server.py was replaced -> it is running the OLD code.")
                kill = f"taskkill /PID {pid} /F" if os.name == "nt" else f"kill {pid}"
                problems.append(f"Old server still running (PID {pid}). Stop it: {kill}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, action="append", help="port Unity connects to (repeatable)")
    ap.add_argument("--clean", action="store_true", help="delete __pycache__ folders in goclimb_routes")
    args = ap.parse_args()

    print(f"Python: {sys.executable}")
    print(f"Folder: {os.getcwd()}")
    server_mtime, _ = check_files(args.clean)
    check_ports(args.port or DEFAULT_PORTS, server_mtime)

    hdr("Verdict")
    if not problems:
        print("  No problems found. Files are new and the running server reports api_version.")
        print("  If Unity still says 'wrong format', check its server URL points to this machine and port.")
    else:
        for i, p in enumerate(dict.fromkeys(problems), 1):
            print(f"  {i}. {p}")
        print("\n  Then start fresh:  python -m goclimb_routes.server --data goclimb_data")
        print("  The first line it prints must end in api_version=3.")


if __name__ == "__main__":
    main()
