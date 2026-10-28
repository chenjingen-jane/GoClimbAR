using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;
using Vuforia;

/// <summary>
/// Phone-style wall scanner.
///
///   1. The user fits the wall inside the GUIDE FRAME (drag its corners) and taps the round CAMERA button.
///      Only the part inside the frame is analysed, so people, lights and background never get "holds".
///   2. ONLINE: the cropped photo goes to the Python server (/api/scan), which also recognises saved walls.
///      OFFLINE (no connection, or "Analyse on this phone only" switched on): OfflineRouteEngine finds holds
///      and plans routes on the phone itself.
///   3. Routes appear at once as a PREVIEW placed exactly over the frame at the moment of capture. A Vuforia
///      target is created from the same photo; when Vuforia recognises the wall the overlay locks onto it.
///   4. The user decides: UPLOAD TO GOCLIMB (venue + name; queued if offline) or DON'T UPLOAD
///      (the server's temporary copy is deleted right away). Nothing is stored before that choice.
/// Saved walls can be searched and opened; opened walls are kept on the phone so they also work offline.
/// Prototype: no login for climbers, plain HTTP, trusted network only.
/// </summary>
public class WallScanner : MonoBehaviour
{
    /// <summary>True while a scan or loaded wall owns the overlay; registered targets stay quiet.</summary>
    public static bool SessionActive { get; private set; }

    public enum LostBehaviour { Hide, Freeze }
    private enum OverlayState { None, Preview, Tracked, Lost }
    private enum Panel { None, Saved, Settings, Upload }

    [Header("Server")]
    [Tooltip("Same PC: http://127.0.0.1:8765. Phone: http://<PC address>:8765 (can also be changed in the app's settings)")]
    public string serverUrl = "http://127.0.0.1:8765";
    public int timeoutSeconds = 30;
    [Tooltip("How long to wait for the server before computing the routes on the phone instead.")]
    public int connectTimeoutSeconds = 4;

    [Header("Scan")]
    public RouteRenderer routeRenderer;
    public string mode = "indoor";               // "indoor" or "outdoor"
    public int maxRoutes = 6;
    [Tooltip("Vuforia accepts at most 2048 px per side for runtime image targets.")]
    public int maxPhotoDimension = 1600;
    [Tooltip("Arbitrary unit. RouteRenderer expects a target 1 unit wide.")]
    public float targetWidth = 1f;
    public bool showUI = true;
    [Tooltip("Initial guide frame (fractions of the screen; x, y from the top-left, width, height).")]
    public Rect defaultFrame = new Rect(0.07f, 0.16f, 0.86f, 0.60f);

    [Header("Tracking / overlay")]
    public float previewDistance = 1f;
    public float lockTimeoutSeconds = 10f;
    public LostBehaviour onTrackingLost = LostBehaviour.Hide;

    private const int RequiredApiVersion = 3;
    private const float BarHeight = 104f;          // bottom bar, GUI units

    // ---- state
    private string status = "";
    private float statusUntil;
    private bool busy;
    private string busyText = "";
    private bool hideUIForCapture;
    private int requestSerial;
    private OverlayState state = OverlayState.None;
    private Panel panel = Panel.None;

    private string sessionKind = "";              // "scan", "loaded" or ""
    private bool sessionOffline;
    private string scanId = "";
    private string scanStatus = "";               // unknown / known / uncertain
    private string sessionWallId = "";
    private string decision = "";                 // "", "uploaded", "queued", "private"
    private GCRouteSet currentSet;
    private List<GCCandidate> candidates = new List<GCCandidate>();
    private byte[] sessionJpg;

    private ImageTargetBehaviour runtimeTarget;
    private Texture2D targetPhoto;
    private Transform anchor;
    private Coroutine lockWatch;

    private Rect frame;                           // normalized, GUI orientation (origin top-left)
    private int dragCorner = -1;

    private string venue = "";
    private string wallName = "";
    private string searchText = "";
    private string serverField = "";
    private bool phoneOnly;
    private List<GCWallSummary> found = new List<GCWallSummary>();
    private List<GCRouteSet> cached = new List<GCRouteSet>();
    private Vector2 scroll;
    private bool uploadingPending;
    private string authToken = "";          // GoClimb login, handed over by the website's "Open in AR camera" link

    // ------------------------------------------------------------ lifecycle

    private void Awake()
    {
        if (routeRenderer == null) routeRenderer = GetComponent<RouteRenderer>();
        if (routeRenderer == null) routeRenderer = FindObjectOfType<RouteRenderer>();
        if (routeRenderer == null) Debug.LogError("[WallScanner] No RouteRenderer found in the scene. Routes cannot be drawn.");

        serverUrl = PlayerPrefs.GetString("gc_server", serverUrl);
        authToken = PlayerPrefs.GetString("gc_token", "");
        phoneOnly = PlayerPrefs.GetInt("gc_phone_only", 0) == 1;
        mode = PlayerPrefs.GetString("gc_mode", mode);
        venue = PlayerPrefs.GetString("gc_venue", "");
        serverField = serverUrl;
        frame = defaultFrame;
    }

    private void Start()
    {
        StartCoroutine(UploadPending(false));       // send uploads queued while offline, silently
    }

    private void OnDestroy()
    {
        EndSession(null);
    }

    private void Toast(string msg, float seconds = 6f)
    {
        status = msg;
        statusUntil = Time.unscaledTime + seconds;
        Debug.Log("[WallScanner] " + msg);
    }

    // ------------------------------------------------------------ guide frame (drag the corners)

    private bool FrameEditable { get { return showUI && !busy && sessionKind == "" && panel == Panel.None; } }

    /// <summary>
    /// Corner dragging with IMGUI events (works with touch, and with both the old and the new Input System;
    /// this project uses the new one only, where UnityEngine.Input would throw).
    /// </summary>
    private void HandleFrameDrag(float W, float H)
    {
        Event ev = Event.current;
        if (!FrameEditable) { dragCorner = -1; return; }
        Vector2 g = ev.mousePosition;                           // already in GUI units (GUI.matrix applied)
        if (ev.type == EventType.MouseDown)
        {
            dragCorner = -1;
            float best = 44f;                                   // touch radius in GUI units
            for (int i = 0; i < 4; i++)
            {
                float d = Vector2.Distance(Corner(i, W, H), g);
                if (d < best) { best = d; dragCorner = i; }
            }
            if (dragCorner >= 0) ev.Use();
        }
        else if (ev.type == EventType.MouseDrag && dragCorner >= 0)
        {
            float nx = Mathf.Clamp(g.x / W, 0.02f, 0.98f);
            float maxY = 1f - (BarHeight + 30f) / H;
            float ny = Mathf.Clamp(g.y / H, 0.02f, maxY);
            float x0 = frame.xMin, y0 = frame.yMin, x1 = frame.xMax, y1 = frame.yMax;
            const float min = 0.2f;
            if (dragCorner == 0 || dragCorner == 3) x0 = Mathf.Min(nx, x1 - min); else x1 = Mathf.Max(nx, x0 + min);
            if (dragCorner == 0 || dragCorner == 1) y0 = Mathf.Min(ny, y1 - min); else y1 = Mathf.Max(ny, y0 + min);
            frame = Rect.MinMaxRect(x0, y0, x1, y1);
            ev.Use();
        }
        else if (ev.type == EventType.MouseUp && dragCorner >= 0)
        {
            dragCorner = -1;
            ev.Use();
        }
    }

    /// <summary>0 top-left, 1 top-right, 2 bottom-right, 3 bottom-left, in GUI units.</summary>
    private Vector2 Corner(int i, float W, float H)
    {
        float x = (i == 0 || i == 3) ? frame.xMin : frame.xMax;
        float y = (i == 0 || i == 1) ? frame.yMin : frame.yMax;
        return new Vector2(x * W, y * H);
    }

    private void DrawFrame(float W, float H)
    {
        Rect r = new Rect(frame.x * W, frame.y * H, frame.width * W, frame.height * H);
        Color shade = new Color(0f, 0f, 0f, 0.35f);              // darken what will NOT be scanned
        GCUi.Rect(new Rect(0, 0, W, r.yMin), shade);
        GCUi.Rect(new Rect(0, r.yMax, W, H - r.yMax), shade);
        GCUi.Rect(new Rect(0, r.yMin, r.xMin, r.height), shade);
        GCUi.Rect(new Rect(r.xMax, r.yMin, W - r.xMax, r.height), shade);
        Color c = Color.white;
        float L = Mathf.Min(34f, Mathf.Min(r.width, r.height) * 0.25f), t = 3f;
        GCUi.Rect(new Rect(r.xMin, r.yMin, L, t), c); GCUi.Rect(new Rect(r.xMin, r.yMin, t, L), c);
        GCUi.Rect(new Rect(r.xMax - L, r.yMin, L, t), c); GCUi.Rect(new Rect(r.xMax - t, r.yMin, t, L), c);
        GCUi.Rect(new Rect(r.xMin, r.yMax - t, L, t), c); GCUi.Rect(new Rect(r.xMin, r.yMax - L, t, L), c);
        GCUi.Rect(new Rect(r.xMax - L, r.yMax - t, L, t), c); GCUi.Rect(new Rect(r.xMax - t, r.yMax - L, t, L), c);
        GCUi.Rect(new Rect(r.xMin, r.yMin, r.width, 1f), new Color(1f, 1f, 1f, 0.35f));
        GCUi.Rect(new Rect(r.xMin, r.yMax - 1f, r.width, 1f), new Color(1f, 1f, 1f, 0.35f));
        GCUi.Rect(new Rect(r.xMin, r.yMin, 1f, r.height), new Color(1f, 1f, 1f, 0.35f));
        GCUi.Rect(new Rect(r.xMax - 1f, r.yMin, 1f, r.height), new Color(1f, 1f, 1f, 0.35f));
        for (int i = 0; i < 4; i++) GCUi.Dot(Corner(i, W, H), dragCorner == i ? 22f : 14f, Color.white);
        var lbl = new Rect(r.xMin, r.yMin - 24f, r.width, 22f);
        if (lbl.y < 2f) lbl.y = r.yMin + 6f;
        GUI.Label(lbl, "Fit the wall inside the frame · drag the corners", GCUi.Small);
    }

    // ------------------------------------------------------------ UI

    private void OnGUI()
    {
        if (!showUI || hideUIForCapture) return;
        GCUi.Begin();
        float W = GCUi.W, H = GCUi.H;

        HandleFrameDrag(W, H);
        if (FrameEditable) DrawFrame(W, H);

        // settings (gear) - top right
        if (GUI.Button(new Rect(W - 88, 8, 80, 36), "Settings", GCUi.Pill)) panel = panel == Panel.Settings ? Panel.None : Panel.Settings;

        // status toast above the bottom area
        float sheetH = SheetHeight();
        float toastY = H - BarHeight - sheetH - 34f;
        if (busy || Time.unscaledTime < statusUntil)
        {
            string text = busy ? busyText : status;
            var sz = GCUi.Label.CalcHeight(new GUIContent(text), Mathf.Min(W - 24, 520));
            var tr = new Rect((W - Mathf.Min(W - 24, 520)) / 2 - 8, toastY - sz + 24, Mathf.Min(W - 24, 520) + 16, sz + 10);
            GUI.Box(tr, GUIContent.none, GCUi.Panel);
            GUI.Label(new Rect(tr.x + 8, tr.y + 5, tr.width - 16, sz), text, GCUi.Label);
        }

        DrawBottomBar(W, H);
        if (sessionKind != "") DrawSessionSheet(W, H, sheetH);

        if (panel == Panel.Saved) DrawSavedPanel(W, H);
        else if (panel == Panel.Settings) DrawSettingsPanel(W, H);
        else if (panel == Panel.Upload) DrawUploadPanel(W, H);
    }

    private float SheetHeight()
    {
        if (sessionKind == "") return 0f;
        return 96f;
    }

    private void DrawBottomBar(float W, float H)
    {
        float y = H - BarHeight;
        GCUi.Rect(new Rect(0, y, W, BarHeight), new Color(0f, 0f, 0f, 0.35f));
        float cx = W / 2f, cy = y + BarHeight / 2f;

        // left: indoor / outdoor
        GUI.enabled = !busy;
        float bw = Mathf.Min(78f, (cx - 60f) / 2f - 6f);
        if (GUI.Button(new Rect(10, cy - 18, bw, 36), "Indoor", mode == "indoor" ? GCUi.PillOn : GCUi.Pill)) SetMode("indoor");
        if (GUI.Button(new Rect(14 + bw, cy - 18, bw, 36), "Outdoor", mode == "outdoor" ? GCUi.PillOn : GCUi.Pill)) SetMode("outdoor");

        // centre: camera shutter
        float r = 34f;
        var shutter = new Rect(cx - r, cy - r, 2 * r, 2 * r);
        var old = GUI.color;
        GUI.color = Color.white;
        GUI.DrawTexture(shutter, GCUi.Ring);
        GUI.color = busy ? new Color(1f, 1f, 1f, 0.35f) : Color.white;
        GUI.DrawTexture(new Rect(cx - r + 8, cy - r + 8, 2 * r - 16, 2 * r - 16), GCUi.Circle);
        GUI.color = old;
        if (busy) GUI.Label(new Rect(cx - 30, cy - 10, 60, 20), "<b>…</b>", new GUIStyle(GCUi.Title) { alignment = TextAnchor.MiddleCenter });
        if (GUI.Button(shutter, GUIContent.none, GUIStyle.none) && !busy) StartCoroutine(ScanRoutine());

        // right: saved walls
        if (GUI.Button(new Rect(W - 10 - Mathf.Min(120f, cx - 60f), cy - 18, Mathf.Min(120f, cx - 60f), 36), "Saved walls", GCUi.Pill))
        {
            panel = panel == Panel.Saved ? Panel.None : Panel.Saved;
            if (panel == Panel.Saved) { cached = LocalWallStore.CachedWalls(); if (found.Count == 0) StartCoroutine(SearchRoutine()); }
        }
        GUI.enabled = true;
    }

    private void SetMode(string m)
    {
        mode = m;
        PlayerPrefs.SetString("gc_mode", m);
    }

    /// <summary>Result card above the bottom bar: what was found + the upload decision.</summary>
    private void DrawSessionSheet(float W, float H, float sheetH)
    {
        float pw = Mathf.Min(W - 12, 560), x = (W - pw) / 2, y = H - BarHeight - sheetH - 4;
        GUI.Box(new Rect(x, y, pw, sheetH), GUIContent.none, GCUi.Panel);
        int n = currentSet != null && currentSet.routes != null ? currentSet.routes.Count : 0;
        string title;
        if (sessionKind == "loaded") title = "<b>" + Safe(currentSet) + "</b> · " + n + " routes";
        else if (scanStatus == "known") title = "Recognised: <b>" + Safe(currentSet) + "</b> · " + n + " routes";
        else if (decision == "uploaded") title = "Uploaded to GoClimb · " + n + " routes";
        else if (decision == "queued") title = "Will upload when online · " + n + " routes";
        else if (decision == "private") title = "Not uploaded (only on this phone) · " + n + " routes";
        else title = (scanStatus == "uncertain" ? "Possibly a saved wall" : "New wall") + " · " + n + " routes";
        GUI.Label(new Rect(x + 12, y + 8, pw - 24, 22), title, GCUi.Label);
        GUI.Label(new Rect(x + 12, y + 28, pw - 24, 18), (sessionOffline ? "<b>Offline</b> · routes computed on this phone · " : "") + OverlayText(), GCUi.Small);

        float by = y + 52, bh = 36;
        bool undecided = sessionKind == "scan" && scanStatus != "known" && decision == "";
        GUI.enabled = !busy;
        if (undecided)
        {
            float bw = (pw - 24 - 8) / 2f;
            if (GUI.Button(new Rect(x + 12, by, bw, bh), "Upload to GoClimb", GCUi.Primary)) { panel = Panel.Upload; }
            if (GUI.Button(new Rect(x + 20 + bw, by, bw, bh), "Don't upload", GCUi.Button)) DontUpload();
        }
        else
        {
            bool canRegen = sessionKind == "scan" && !sessionOffline;
            float bw = canRegen ? (pw - 24 - 8) / 2f : pw - 24;
            if (canRegen && GUI.Button(new Rect(x + 12, by, bw, bh), "New routes", GCUi.Button)) StartCoroutine(RegenerateRoutine());
            GUI.enabled = true;
            if (GUI.Button(new Rect(x + 12 + (canRegen ? bw + 8 : 0), by, bw, bh), sessionKind == "loaded" ? "Close wall" : "Done · scan another", GCUi.Button))
                EndSession("Ready. Fit the next wall in the frame.");
        }
        GUI.enabled = true;
    }

    private static string Safe(GCRouteSet s)
    {
        if (s == null) return "";
        return string.IsNullOrEmpty(s.venue) ? s.name : s.venue + " / " + s.name;
    }

    private string OverlayText()
    {
        switch (state)
        {
            case OverlayState.Preview: return "Preview from the photo · hold the phone still until it locks onto the wall";
            case OverlayState.Tracked: return "Locked onto the wall";
            case OverlayState.Lost: return "Wall lost · point the camera back at it";
            default: return sessionKind == "loaded" ? "Point the camera at this wall" : "";
        }
    }

    private Rect PanelRect(float W, float H, float wantH)
    {
        float pw = Mathf.Min(W - 16, 520);
        float ph = Mathf.Min(wantH, H - BarHeight - 70);
        return new Rect((W - pw) / 2, 56, pw, ph);
    }

    private void DrawUploadPanel(float W, float H)
    {
        Rect r = PanelRect(W, H, 340);
        GUILayout.BeginArea(r, GCUi.Panel);
        scroll = GUILayout.BeginScrollView(scroll);
        GUILayout.Label("Upload this wall to GoClimb", GCUi.Title);
        GUILayout.Label("The photo and its holds are shared so other climbers can use this wall. An admin may correct holds and routes.", GCUi.Small);
        GUILayout.Label("Venue (gym or crag)", GCUi.Small);
        venue = GUILayout.TextField(venue, 60, GCUi.Field, GUILayout.Height(36));
        GUILayout.Label("Wall name", GCUi.Small);
        wallName = GUILayout.TextField(wallName, 60, GCUi.Field, GUILayout.Height(36));
        foreach (var c in candidates)
            if (GUILayout.Button("Same wall as: " + c.venue + " / " + c.name, GCUi.Button, GUILayout.Height(36)))
                StartCoroutine(UploadRoutine(c.wall_id));
        GUILayout.Space(6);
        GUILayout.BeginHorizontal();
        GUI.enabled = !busy;
        if (GUILayout.Button("Upload", GCUi.Primary, GUILayout.Height(40))) StartCoroutine(UploadRoutine(null));
        GUI.enabled = true;
        if (GUILayout.Button("Cancel", GCUi.Button, GUILayout.Height(40))) panel = Panel.None;
        GUILayout.EndHorizontal();
        GUILayout.EndScrollView();
        GUILayout.EndArea();
    }

    private void DrawSavedPanel(float W, float H)
    {
        Rect r = PanelRect(W, H, 520);
        GUILayout.BeginArea(r, GCUi.Panel);
        GUILayout.BeginHorizontal();
        GUILayout.Label("Saved walls", GCUi.Title);
        if (GUILayout.Button("Close", GCUi.Button, GUILayout.Width(80), GUILayout.Height(32))) panel = Panel.None;
        GUILayout.EndHorizontal();
        GUILayout.BeginHorizontal();
        searchText = GUILayout.TextField(searchText, 60, GCUi.Field, GUILayout.Height(36));
        GUI.enabled = !busy;
        if (GUILayout.Button("Search", GCUi.Button, GUILayout.Width(90), GUILayout.Height(36))) StartCoroutine(SearchRoutine());
        GUI.enabled = true;
        GUILayout.EndHorizontal();
        scroll = GUILayout.BeginScrollView(scroll);
        GUI.enabled = !busy;
        foreach (var f in found)
            if (GUILayout.Button(f.venue + " / " + f.name + "  (" + f.mode + ")", GCUi.Button, GUILayout.Height(38)))
                StartCoroutine(LoadRoutine(f.wall_id));
        if (cached.Count > 0)
        {
            GUILayout.Space(8);
            GUILayout.Label("On this phone (work offline)", GCUi.Small);
            foreach (var d in cached)
                if (GUILayout.Button(Safe(d) + "  (" + (d.routes != null ? d.routes.Count : 0) + " routes)", GCUi.Button, GUILayout.Height(38)))
                    LoadCached(d);
        }
        GUI.enabled = true;
        GUILayout.EndScrollView();
        GUILayout.EndArea();
    }

    private void DrawSettingsPanel(float W, float H)
    {
        Rect r = PanelRect(W, H, 380);
        GUILayout.BeginArea(r, GCUi.Panel);
        scroll = GUILayout.BeginScrollView(scroll);
        GUILayout.BeginHorizontal();
        GUILayout.Label("Settings", GCUi.Title);
        if (GUILayout.Button("Close", GCUi.Button, GUILayout.Width(80), GUILayout.Height(32))) panel = Panel.None;
        GUILayout.EndHorizontal();
        GUILayout.Label("GoClimb server address", GCUi.Small);
        GUILayout.BeginHorizontal();
        serverField = GUILayout.TextField(serverField, 120, GCUi.Field, GUILayout.Height(36));
        if (GUILayout.Button("Save", GCUi.Button, GUILayout.Width(70), GUILayout.Height(36)))
        {
            serverUrl = serverField.Trim().TrimEnd('/');
            PlayerPrefs.SetString("gc_server", serverUrl);
            Toast("Server address saved.");
        }
        GUILayout.EndHorizontal();
        bool po = GUILayout.Toggle(phoneOnly, "  Analyse on this phone only (the photo leaves the phone only if you upload)");
        if (po != phoneOnly) { phoneOnly = po; PlayerPrefs.SetInt("gc_phone_only", po ? 1 : 0); }
        int pending = LocalWallStore.PendingFiles().Count;
        GUILayout.Space(6);
        GUILayout.Label(pending == 0 ? "No uploads waiting." : pending + " upload(s) waiting for a connection.", GCUi.Small);
        if (pending > 0 && !uploadingPending && GUILayout.Button("Upload now", GCUi.Button, GUILayout.Height(36))) StartCoroutine(UploadPending(true));
        if (GUILayout.Button("Reset scan frame", GCUi.Button, GUILayout.Height(36))) frame = defaultFrame;
        GUILayout.EndScrollView();
        GUILayout.EndArea();
    }

    // ------------------------------------------------------------ scan

    private IEnumerator ScanRoutine()
    {
        busy = true;
        panel = Panel.None;
        busyText = "Capturing...";
        EndSession(null);
        SessionActive = true;
        int serial = ++requestSerial;
        decision = "";
        hideUIForCapture = true;
        yield return null;                                  // one frame without UI or routes
        yield return new WaitForEndOfFrame();

        Camera cam = GetCamera();
        Vector3 capPos = cam != null ? cam.transform.position : Vector3.zero;
        Quaternion capRot = cam != null ? cam.transform.rotation : Quaternion.identity;
        Matrix4x4 proj = cam != null ? cam.projectionMatrix : Matrix4x4.identity;
        Texture2D shot = ScreenCapture.CaptureScreenshotAsTexture();
        hideUIForCapture = false;
        if (shot == null) { Fail("Screen capture failed."); yield break; }

        Rect vp = new Rect(frame.xMin, 1f - frame.yMax, frame.width, frame.height);  // viewport: origin bottom-left
        Texture2D photo = CropScaleRgb24(shot, vp, maxPhotoDimension);
        Destroy(shot);
        byte[] jpg = photo.EncodeToJPG(88);
        sessionJpg = jpg;

        GCRouteSet doc = null;
        GCScanResponse online = null;
        string offlineReason = phoneOnly ? "phone-only mode" : "";

        if (!phoneOnly)
        {
            busyText = "Analysing the wall...";
            var req = new GCScanRequest { image_b64 = Convert.ToBase64String(jpg), mode = mode, max_routes = maxRoutes };
            string body = null, err = null;
            bool netFail = false;
            yield return Send("POST", "/api/scan", JsonUtility.ToJson(req), (b, e, nf) => { body = b; err = e; netFail = nf; }, timeoutSeconds, connectTimeoutSeconds);
            if (serial != requestSerial) { Destroy(photo); busy = false; yield break; }
            if (err == null)
            {
                online = ParseScanReply(body, out offlineReason);
            }
            else offlineReason = netFail ? "no connection to the server" : err;
        }

        if (online != null)
        {
            doc = online.doc;
            sessionOffline = false;
            scanId = online.scan_id;
            scanStatus = online.status;
            sessionWallId = online.status == "known" ? online.wall_id : "";
            candidates = online.candidates ?? new List<GCCandidate>();
            StartCoroutine(UploadPending(false));
        }
        else
        {
            busyText = "Finding holds on this phone...";
            byte[] rgb = ToRgbTopDown(photo);
            int pw = photo.width, ph = photo.height;
            string m = mode;
            int mr = maxRoutes;
            Task<GCRouteSet> task = Task.Run(() => OfflineRouteEngine.Run(rgb, pw, ph, m, mr));
            while (!task.IsCompleted) yield return null;
            if (serial != requestSerial) { Destroy(photo); busy = false; yield break; }
            if (task.IsFaulted || task.Result == null)
            {
                Debug.LogException(task.Exception != null ? (Exception)task.Exception : new Exception("offline engine returned nothing"));
                Destroy(photo);
                Fail("Could not analyse the photo on the phone.");
                yield break;
            }
            doc = task.Result;
            sessionOffline = true;
            scanId = "offline-" + Guid.NewGuid().ToString("N").Substring(0, 8);
            doc.scan_id = scanId;
            scanStatus = "unknown";
            sessionWallId = "";
            candidates = new List<GCCandidate>();
            Debug.Log("[WallScanner] Offline analysis (" + offlineReason + ").");
        }

        sessionKind = "scan";
        SessionActive = true;
        currentSet = doc;
        targetPhoto = photo;
        int n = doc.routes != null ? doc.routes.Count : 0;
        int holds = doc.holds != null ? doc.holds.Count : 0;
        string msg = holds + " holds, " + n + " routes" + (sessionOffline ? " (computed on this phone: " + offlineReason + ")." : ".");
        if (n == 0) msg += " " + FirstWarning(doc) + " Try fitting the frame more tightly around the wall.";
        Toast(msg, 8f);

        ShowPreview(capPos, capRot, proj, vp);
        CreateRuntimeTarget(photo, "rt-" + scanId.Replace("offline-", "").Substring(0, 8), true);
        busy = false;
    }

    /// <summary>Returns the parsed reply, or null with the reason (then the phone computes the routes).</summary>
    private GCScanResponse ParseScanReply(string body, out string problem)
    {
        problem = "";
        GCScanResponse resp = null;
        try { resp = JsonUtility.FromJson<GCScanResponse>(body); }
        catch (Exception e) { problem = "unreadable server reply (" + e.Message + ")"; return null; }
        if (resp == null || resp.doc == null) { problem = "server reply without route data"; return null; }
        if (resp.api_version < RequiredApiVersion) { problem = "the server is an old version"; return null; }
        if (resp.doc.schema != 2 || string.IsNullOrEmpty(resp.scan_id)) { problem = "server reply has the wrong format"; return null; }
        return resp;
    }

    // ------------------------------------------------------------ preview (fixed at the capture pose)

    /// <summary>
    /// Places the overlay plane exactly where the guide frame was on screen at the moment of capture, at
    /// previewDistance in front of the camera. Uses the full projection matrix, including the off-centre terms
    /// (m02, m12) that Vuforia's camera has.
    /// </summary>
    private void ShowPreview(Vector3 capPos, Quaternion capRot, Matrix4x4 proj, Rect vp)
    {
        if (routeRenderer == null || currentSet == null) return;
        if (Mathf.Abs(proj.m00) < 1e-4f || Mathf.Abs(proj.m11) < 1e-4f) { Toast("No preview: camera projection unavailable."); return; }
        float d = previewDistance;
        Vector3 center = capPos + capRot * ViewPoint(proj, vp.center.x, vp.center.y, d);
        Vector3 left = capPos + capRot * ViewPoint(proj, vp.xMin, vp.center.y, d);
        Vector3 right = capPos + capRot * ViewPoint(proj, vp.xMax, vp.center.y, d);
        Vector3 bottom = capPos + capRot * ViewPoint(proj, vp.center.x, vp.yMin, d);
        Vector3 top = capPos + capRot * ViewPoint(proj, vp.center.x, vp.yMax, d);
        float width = Vector3.Distance(left, right);
        float height = Vector3.Distance(bottom, top);
        if (currentSet.aspect > 0f && width > 1e-5f && Mathf.Abs(height / width - currentSet.aspect) > 0.05f * currentSet.aspect)
            Debug.LogWarning("[WallScanner] Photo aspect " + currentSet.aspect + " differs from the frame's aspect " + (height / width) + "; overlay may be slightly stretched.");

        Vector3 fwd = capRot * Vector3.forward, up = (top - bottom).normalized;
        Quaternion rot = routeRenderer.targetPlane == RouteRenderer.TargetPlane.XZ_YAxisOut
            ? Quaternion.LookRotation(up, -fwd)              // local Y = towards the camera, local Z = screen up
            : Quaternion.LookRotation(fwd, up);
        PlaceAnchor(center, rot, width);
        routeRenderer.selectedRouteId = "";
        routeRenderer.currentStep = 0;
        routeRenderer.DrawRouteSet(anchor, currentSet);
        state = OverlayState.Preview;
    }

    /// <summary>Camera-space point at depth d for a viewport position (u, v in 0..1, origin bottom-left).</summary>
    private static Vector3 ViewPoint(Matrix4x4 P, float u, float v, float d)
    {
        float ndcX = 2f * u - 1f, ndcY = 2f * v - 1f;
        float x = d * (ndcX + P.m02) / P.m00;
        float y = d * (ndcY + P.m12) / P.m11;
        return new Vector3(x, y, d);
    }

    private void PlaceAnchor(Vector3 pos, Quaternion rot, float width)
    {
        if (anchor == null) anchor = new GameObject("RoutePreviewAnchor").transform;
        anchor.SetParent(null);
        anchor.SetPositionAndRotation(pos, rot);
        anchor.localScale = Vector3.one * width;
    }

    private void DestroyAnchor()
    {
        if (anchor != null) Destroy(anchor.gameObject);
        anchor = null;
    }

    // ------------------------------------------------------------ runtime Vuforia target (works offline)

    private void CreateRuntimeTarget(Texture2D photo, string targetName, bool watchForLock)
    {
        try
        {
            var factory = VuforiaBehaviour.Instance != null ? VuforiaBehaviour.Instance.ObserverFactory : null;
            if (factory == null) { Debug.LogWarning("[WallScanner] Vuforia not ready: preview only."); return; }
            runtimeTarget = factory.CreateImageTarget(photo, targetWidth, targetName);
            if (runtimeTarget == null) { Debug.LogWarning("[WallScanner] Vuforia could not create a target (too few features?)."); return; }
            runtimeTarget.OnTargetStatusChanged += OnTargetStatusChanged;
            if (watchForLock) lockWatch = StartCoroutine(LockWatch(scanId));
        }
        catch (Exception e)
        {
            runtimeTarget = null;
            Debug.LogWarning("[WallScanner] Could not create the Vuforia target: " + e);
        }
    }

    private void OnTargetStatusChanged(ObserverBehaviour behaviour, TargetStatus st)
    {
        if (routeRenderer == null || currentSet == null) return;
        bool tracked = st.Status == Status.TRACKED || st.Status == Status.EXTENDED_TRACKED;
        if (tracked)
        {
            if (state == OverlayState.Tracked) return;
            DestroyAnchor();
            routeRenderer.DrawRouteSet(behaviour.transform, currentSet);
            state = OverlayState.Tracked;
        }
        else if (st.Status == Status.NO_POSE && state == OverlayState.Tracked)
        {
            if (onTrackingLost == LostBehaviour.Freeze)
            {
                PlaceAnchor(behaviour.transform.position, behaviour.transform.rotation, targetWidth);
                routeRenderer.DrawRouteSet(anchor, currentSet);
            }
            else routeRenderer.ClearRoute();
            state = OverlayState.Lost;
        }
    }

    private IEnumerator LockWatch(string forScan)
    {
        yield return new WaitForSeconds(lockTimeoutSeconds);
        if (forScan == scanId && state == OverlayState.Preview)
            Toast("Not locked onto the wall yet: the routes stay where the photo was taken. Hold still, add light, or scan again.", 8f);
    }

    private void DestroyCurrentTarget()
    {
        if (runtimeTarget == null) return;
        runtimeTarget.OnTargetStatusChanged -= OnTargetStatusChanged;
        Destroy(runtimeTarget.gameObject);
        runtimeTarget = null;
    }

    // ------------------------------------------------------------ the upload decision

    private void DontUpload()
    {
        decision = "private";
        if (!sessionOffline && !string.IsNullOrEmpty(scanId))
            StartCoroutine(Send("DELETE", "/api/scans/" + scanId, null, (b, e, nf) => { }, 10, connectTimeoutSeconds));
        Toast("Not uploaded. The routes stay on screen; nothing is kept on the server.");
    }

    private IEnumerator UploadRoutine(string existingWallId)
    {
        busy = true;
        busyText = "Uploading...";
        string forScan = scanId;
        venue = venue.Trim();
        wallName = wallName.Trim();
        if (venue.Length > 0) PlayerPrefs.SetString("gc_venue", venue);

        // 1) online scan: the server still has the photo -> save it there
        if (!sessionOffline)
        {
            var req = new GCSaveRequest { venue = venue, name = wallName, use_existing_wall_id = existingWallId ?? "", allow_duplicate = false };
            string body = null, err = null; bool nf = false;
            yield return Send("POST", "/api/scans/" + forScan + "/save", JsonUtility.ToJson(req), (b, e, n) => { body = b; err = e; nf = n; }, timeoutSeconds, connectTimeoutSeconds);
            if (forScan != scanId) { busy = false; yield break; }
            if (err == null)
            {
                GCSaveResponse r = null;
                try { r = JsonUtility.FromJson<GCSaveResponse>(body); } catch (Exception) { }
                if (r != null && !string.IsNullOrEmpty(r.wall_id))
                {
                    Uploaded(r.wall_id, r.created ? "Uploaded as a new wall." : "Linked to the existing wall.");
                    yield break;
                }
            }
            else if (err == "duplicate") { Fail("A wall with this venue and name already exists. Change the name."); yield break; }
            // expired scan or connection lost: fall through to a full upload (photo + holds)
        }

        // 2) full upload (offline scans, or the server forgot the scan)
        GCUploadRequest up = BuildUpload();
        string ubody = null, uerr = null; bool unf = false;
        yield return Send("POST", "/api/walls", JsonUtility.ToJson(up), (b, e, n) => { ubody = b; uerr = e; unf = n; }, timeoutSeconds, connectTimeoutSeconds);
        if (forScan != scanId) { busy = false; yield break; }
        if (uerr == null)
        {
            GCUploadResponse r = null;
            try { r = JsonUtility.FromJson<GCUploadResponse>(ubody); } catch (Exception) { }
            if (r != null && !string.IsNullOrEmpty(r.wall_id)) { Uploaded(r.wall_id, "Uploaded to GoClimb."); yield break; }
            Fail("The server's reply to the upload could not be read.");
            yield break;
        }
        if (uerr == "duplicate") { Fail("A wall with this venue and name already exists. Change the name."); yield break; }
        if (unf)
        {
            LocalWallStore.QueueUpload(up);
            decision = "queued";
            panel = Panel.None;
            busy = false;
            Toast("No connection. The wall is saved on this phone and will upload automatically next time you are online.", 8f);
            yield break;
        }
        Fail("Upload failed: " + uerr);
    }

    private void Uploaded(string wallId, string message)
    {
        sessionWallId = wallId;
        if (currentSet != null) currentSet.wall_id = wallId;
        decision = "uploaded";
        panel = Panel.None;
        busy = false;
        Toast(message);
    }

    private GCUploadRequest BuildUpload()
    {
        var up = new GCUploadRequest();
        up.name = wallName;
        up.venue = venue.Length > 0 ? venue : "Scanned walls";
        if (string.IsNullOrEmpty(up.name)) up.name = "Scan " + DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
        up.mode = mode;
        up.image_b64 = sessionJpg != null ? Convert.ToBase64String(sessionJpg) : "";
        up.source = "app";
        up.max_routes = maxRoutes;
        up.holds = new List<GCHoldIn>();
        if (currentSet != null && currentSet.holds != null)
            foreach (var h in currentSet.holds)
                up.holds.Add(new GCHoldIn { x = h.x, y = h.y, confidence = h.confidence, type = h.type });
        return up;
    }

    /// <summary>Send uploads that were queued while offline. Quiet unless the user pressed "Upload now".</summary>
    private IEnumerator UploadPending(bool loud)
    {
        if (uploadingPending) yield break;
        if (string.IsNullOrEmpty(authToken))
        {
            if (loud && LocalWallStore.PendingFiles().Count > 0)
                Toast("Uploads are waiting. Open the AR camera from the GoClimb website once to sign in.");
            yield break;                              // keep them queued until the user is signed in
        }
        uploadingPending = true;
        int sent = 0, failed = 0;
        foreach (string path in LocalWallStore.PendingFiles())
        {
            GCUploadRequest up = LocalWallStore.ReadPending(path);
            if (up == null) { LocalWallStore.RemovePending(path); continue; }
            string err = null; bool nf = false;
            yield return Send("POST", "/api/walls", JsonUtility.ToJson(up), (b, e, n) => { err = e; nf = n; }, timeoutSeconds, connectTimeoutSeconds);
            if (err == null) { LocalWallStore.RemovePending(path); sent++; }
            else if (err == "duplicate")
            {
                up.name = up.name + " (" + DateTime.Now.ToString("MMdd-HHmm") + ")";      // keep it, under a new name
                LocalWallStore.RemovePending(path);
                LocalWallStore.QueueUpload(up);
                failed++;
            }
            else { failed++; if (nf) break; }                                           // still offline: try later
        }
        uploadingPending = false;
        if (sent > 0 || loud)
            Toast(sent + " waiting upload(s) sent" + (failed > 0 ? ", " + failed + " still waiting." : "."));
    }

    // ------------------------------------------------------------ regenerate (online scans only)

    private IEnumerator RegenerateRoutine()
    {
        busy = true;
        busyText = "Generating new routes...";
        string forScan = scanId;
        var req = new GCRegenerateRequest { max_routes = maxRoutes, refresh_only = false };
        string body = null, err = null;
        yield return Send("POST", "/api/scans/" + forScan + "/regenerate", JsonUtility.ToJson(req), (b, e, n) => { body = b; err = e; }, timeoutSeconds, connectTimeoutSeconds);
        if (forScan != scanId) { busy = false; yield break; }
        if (err != null) { Fail(err); yield break; }
        GCDocResponse r = null;
        try { r = JsonUtility.FromJson<GCDocResponse>(body); } catch (Exception) { }
        if (r == null || r.doc == null || r.doc.scan_id != forScan) { Fail("Bad server reply."); yield break; }
        currentSet = r.doc;
        if (routeRenderer != null)
        {
            routeRenderer.selectedRouteId = "";
            routeRenderer.currentStep = 0;
            if (state == OverlayState.Tracked && runtimeTarget != null) routeRenderer.DrawRouteSet(runtimeTarget.transform, currentSet);
            else if (anchor != null) routeRenderer.DrawRouteSet(anchor, currentSet);
        }
        busy = false;
        Toast((currentSet.routes != null ? currentSet.routes.Count : 0) + " routes.");
    }

    // ------------------------------------------------------------ saved walls (online search + offline cache)

    private IEnumerator SearchRoutine()
    {
        busy = true;
        busyText = "Searching...";
        string body = null, err = null; bool nf = false;
        yield return Send("GET", "/api/walls?q=" + UnityWebRequest.EscapeURL(searchText), null, (b, e, n) => { body = b; err = e; nf = n; }, 15, connectTimeoutSeconds);
        busy = false;
        cached = LocalWallStore.CachedWalls();
        if (err != null) { Toast(nf ? "Offline: showing the walls saved on this phone." : "Search failed: " + err); found.Clear(); yield break; }
        try
        {
            var list = JsonUtility.FromJson<GCWallList>("{\"walls\":" + body + "}");
            found = list != null && list.walls != null ? list.walls : new List<GCWallSummary>();
        }
        catch (Exception) { found = new List<GCWallSummary>(); }
        if (found.Count == 0) Toast("No matching wall on the server.");
    }

    private IEnumerator LoadRoutine(string wallId)
    {
        busy = true;
        busyText = "Loading wall...";
        EndSession(null);
        SessionActive = true;
        int serial = ++requestSerial;
        string body = null, err = null;
        yield return Send("GET", "/api/walls/" + wallId + "/unity", null, (b, e, n) => { body = b; err = e; }, timeoutSeconds, connectTimeoutSeconds);
        if (serial != requestSerial) { busy = false; yield break; }
        if (err != null) { Fail("Load failed: " + err); yield break; }
        GCRouteSet doc = null;
        try { doc = JsonUtility.FromJson<GCRouteSet>(body); } catch (Exception) { }
        if (doc == null || doc.schema != 2 || doc.wall_id != wallId) { Fail("Wall data has the wrong format."); yield break; }

        byte[] jpg = null;
        using (var www = UnityWebRequest.Get(serverUrl.TrimEnd('/') + "/photo/" + wallId))
        {
            AddAuth(www);
            www.timeout = timeoutSeconds;
            yield return www.SendWebRequest();
            if (serial != requestSerial) { busy = false; yield break; }
            if (www.result != UnityWebRequest.Result.Success) { Fail("Photo missing or unreadable: " + www.error); yield break; }
            jpg = www.downloadHandler.data;
        }
        LocalWallStore.CacheWall(doc, jpg);            // available offline from now on
        StartLoadedSession(doc, jpg);
        busy = false;
    }

    // ------------------------------------------------------------ deep links (website "Open in AR camera")

    /// <summary>goclimb://wall/&lt;id&gt;?server=... : open that wall straight in the AR camera.</summary>
    public void OpenWallFromLink(string wallId, string server)
    {
        UseServer(server);
        if (busy) { Toast("Busy - tap the link on the website again in a moment."); return; }
        bool offline = Application.internetReachability == NetworkReachability.NotReachable;
        if (offline || phoneOnly)
        {
            foreach (var doc in LocalWallStore.CachedWalls())
                if (doc != null && doc.wall_id == wallId) { LoadCached(doc); return; }
            if (offline) { Toast("You're offline and this wall isn't saved on the phone yet."); return; }
        }
        StartCoroutine(LoadRoutine(wallId));      // downloads, caches for offline, starts the AR session
    }

    /// <summary>goclimb://scan?server=... : open the camera ready to scan a new wall.</summary>
    public void OpenScannerFromLink(string server)
    {
        UseServer(server);
        if (busy) return;
        EndSession(null);
        panel = Panel.None;
        Toast("Frame the wall and tap the shutter to scan it.");
    }

    /// <summary>The website hands over the user's GoClimb login (a token) inside the AR link.</summary>
    public void UseAuthToken(string token)
    {
        if (string.IsNullOrEmpty(token) || token == authToken) return;
        authToken = token.Trim();
        PlayerPrefs.SetString("gc_token", authToken);
        PlayerPrefs.Save();
        Toast("Signed in from the GoClimb website.");
        StartCoroutine(UploadPending(false));         // send scans that were saved while offline
    }

    private void AddAuth(UnityWebRequest www)
    {
        if (!string.IsNullOrEmpty(authToken)) www.SetRequestHeader("Authorization", "Token " + authToken);
    }

    /// <summary>The website tells the app where it lives, so nobody has to type the server address.</summary>
    private void UseServer(string server)
    {
        if (string.IsNullOrEmpty(server)) return;
        server = server.Trim().TrimEnd('/');
        if (!(server.StartsWith("http://") || server.StartsWith("https://"))) return;
        if (server == serverUrl) return;
        serverUrl = server;
        serverField = server;
        PlayerPrefs.SetString("gc_server", serverUrl);
        Toast("Connected to " + serverUrl);
    }

    private void LoadCached(GCRouteSet doc)
    {
        EndSession(null);
        byte[] jpg = LocalWallStore.CachedPhoto(doc.wall_id);
        if (jpg == null) { Toast("The photo of this wall is missing on the phone."); return; }
        SessionActive = true;
        StartLoadedSession(doc, jpg);
    }

    private void StartLoadedSession(GCRouteSet doc, byte[] jpg)
    {
        var tex = new Texture2D(2, 2, TextureFormat.RGB24, false);
        if (!tex.LoadImage(jpg)) { Destroy(tex); Fail("The wall photo could not be read."); return; }
        Texture2D photo = ScaleRgb24(tex, maxPhotoDimension);
        Destroy(tex);
        panel = Panel.None;
        sessionKind = "loaded";
        sessionOffline = false;
        sessionWallId = doc.wall_id;
        currentSet = doc;
        targetPhoto = photo;
        CreateRuntimeTarget(photo, "rt-" + doc.wall_id.Substring(0, 8), false);
        Toast("Point the camera at " + Safe(doc) + " to see its routes.");
    }

    // ------------------------------------------------------------ session end

    private void EndSession(string message)
    {
        requestSerial++;
        if (lockWatch != null) { StopCoroutine(lockWatch); lockWatch = null; }
        DestroyCurrentTarget();
        DestroyAnchor();
        if (routeRenderer != null) routeRenderer.ClearRoute();
        if (targetPhoto != null) { Destroy(targetPhoto); targetPhoto = null; }
        currentSet = null;
        scanId = scanStatus = sessionWallId = sessionKind = decision = "";
        sessionOffline = false;
        sessionJpg = null;
        candidates = new List<GCCandidate>();
        state = OverlayState.None;
        SessionActive = false;
        if (panel == Panel.Upload) panel = Panel.None;
        if (message != null) Toast(message, 4f);
    }

    // ------------------------------------------------------------ helpers

    private Camera GetCamera()
    {
        if (VuforiaBehaviour.Instance != null)
        {
            var c = VuforiaBehaviour.Instance.GetComponent<Camera>();
            if (c != null) return c;
        }
        return Camera.main;
    }

    /// <summary>done(body, error, networkFailure). error == "duplicate" for HTTP 409.</summary>
    private IEnumerator Send(string method, string path, string json, Action<string, string, bool> done, int timeout, int connectTimeout)
    {
        string url = serverUrl.TrimEnd('/') + path;
        using (var www = new UnityWebRequest(url, method))
        {
            if (json != null)
            {
                www.uploadHandler = new UploadHandlerRaw(System.Text.Encoding.UTF8.GetBytes(json));
                www.SetRequestHeader("Content-Type", "application/json");
            }
            www.downloadHandler = new DownloadHandlerBuffer();
            AddAuth(www);
            www.timeout = timeout;
            var op = www.SendWebRequest();
            float start = Time.realtimeSinceStartup;
            while (!op.isDone)
            {
                // nothing uploaded after connectTimeout -> the server is not reachable: give up early
                if (Time.realtimeSinceStartup - start > connectTimeout && www.uploadedBytes == 0 && www.downloadedBytes == 0)
                {
                    www.Abort();
                    done(null, "no connection to the server", true);
                    yield break;
                }
                yield return null;
            }
            if (www.result == UnityWebRequest.Result.Success) { done(www.downloadHandler.text, null, false); yield break; }
            bool netFail = www.result == UnityWebRequest.Result.ConnectionError;
            string text = www.downloadHandler != null ? www.downloadHandler.text : "";
            string msg = www.error;
            if (!string.IsNullOrEmpty(text))
            {
                try { var e = JsonUtility.FromJson<GCError>(text); if (e != null && !string.IsNullOrEmpty(e.error)) msg = e.error; }
                catch (Exception) { }
            }
            done(null, netFail ? "no connection to the server" : msg, netFail);
        }
    }

    /// <summary>Crop the viewport rect (origin bottom-left, 0..1) out of the screenshot and scale it.</summary>
    private static Texture2D CropScaleRgb24(Texture2D src, Rect vp, int maxDim)
    {
        int cw = Mathf.Max(2, Mathf.RoundToInt(src.width * vp.width)), ch = Mathf.Max(2, Mathf.RoundToInt(src.height * vp.height));
        float k = Mathf.Min(1f, maxDim / (float)Mathf.Max(cw, ch));
        int nw = Mathf.Max(2, Mathf.RoundToInt(cw * k)), nh = Mathf.Max(2, Mathf.RoundToInt(ch * k));
        var rt = RenderTexture.GetTemporary(nw, nh, 0, RenderTextureFormat.ARGB32);
        Graphics.Blit(src, rt, new Vector2(vp.width, vp.height), new Vector2(vp.x, vp.y));
        var prev = RenderTexture.active;
        RenderTexture.active = rt;
        var dst = new Texture2D(nw, nh, TextureFormat.RGB24, false);
        dst.ReadPixels(new Rect(0, 0, nw, nh), 0, 0);
        dst.Apply();
        RenderTexture.active = prev;
        RenderTexture.ReleaseTemporary(rt);
        return dst;
    }

    private static Texture2D ScaleRgb24(Texture2D src, int maxDim)
    {
        return CropScaleRgb24(src, new Rect(0, 0, 1, 1), maxDim);
    }

    /// <summary>RGB bytes with row 0 = TOP of the picture (Unity textures store the bottom row first).</summary>
    private static byte[] ToRgbTopDown(Texture2D t)
    {
        Color32[] px = t.GetPixels32();
        int w = t.width, h = t.height;
        var o = new byte[w * h * 3];
        for (int y = 0; y < h; y++)
        {
            int srcRow = (h - 1 - y) * w, dstRow = y * w * 3;
            for (int x = 0; x < w; x++)
            {
                Color32 c = px[srcRow + x];
                int i = dstRow + x * 3;
                o[i] = c.r; o[i + 1] = c.g; o[i + 2] = c.b;
            }
        }
        return o;
    }

    private void Fail(string message)
    {
        hideUIForCapture = false;
        busy = false;
        if (string.IsNullOrEmpty(sessionKind)) SessionActive = false;
        Toast(message, 8f);
        Debug.LogWarning("[WallScanner] " + message);
    }

    private static string FirstWarning(GCRouteSet d)
    {
        return d.warnings != null && d.warnings.Count > 0 ? d.warnings[0] + "." : "";
    }
}
