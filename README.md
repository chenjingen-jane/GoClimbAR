# GoClimb: setup and usage (schema 2, runtime scanning)

## Python (Windows, CPU only; same packages as before)
    pip install numpy opencv-python-headless
    python -m unittest discover -s goclimb_routes/tests -t .          # expect: Ran 80 tests ... OK
    python -m goclimb_routes.server --data goclimb_data               # add --host 0.0.0.0 for a phone

## Unity: scanning an unknown wall
1. Scripts in Assets/Script: WallScanner, RouteRenderer, RouteSetData, ImageTargetHandler (+ Calibration*).
2. Empty GameObject with the WallScanner component; drag your RouteRenderer object into its slot.
3. Server Url = http://127.0.0.1:8765 (same PC) or http://<PC address>:8765 (phone).
4. Android: Player Settings > Other Settings > Allow downloads over HTTP = Always allowed.
5. Play. Point at a wall, hold still, SCAN THIS WALL.
   - Routes appear at once as a PREVIEW (fixed at the scan position).
   - When Vuforia recognises the wall the overlay becomes TRACKED and follows the camera.
   - Nothing is saved until you press Save. Re-scanning a saved wall reuses its record.
The on-screen line "Overlay:" tells you which state you are in; the text below it says what went wrong.
