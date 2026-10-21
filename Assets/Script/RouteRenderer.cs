using System.Collections.Generic;
using UnityEngine;
using TMPro;
using Vuforia;

/// <summary>
/// Draws routes on an Image Target or on the scan-preview plane, in the same style as the web editor:
/// straight lines between holds with a dark outline, flat round markers with a white rim, bigger start/finish.
/// Everything is painted FLAT on the wall plane (not 3D spheres), anti-aliased, and uses the Sprites/Default
/// shader, which is always included in Android builds (Unlit/Color is not and could render pink).
///  - DrawRouteSet(): routes generated from detected holds (server or offline engine). Normal path.
///  - GenerateAndDrawLocalRoutes(): decorative template routes, fallback only.
/// Coordinate chain: ImageToNorm() -> Calibrate() -> ToTarget().
/// </summary>
public class RouteRenderer : MonoBehaviour
{
    public enum TargetPlane
    {
        XZ_YAxisOut, // Vuforia Unity default: image lies in XZ, Y is the normal, image top = +Z
        XY_ZAxisOut
    }

    public enum LineStyle { Straight, Smooth }

    [Header("Calibration (optional)")]
    public CalibrationDatabase database;
    public CalibrationProfile fallbackProfile;
    [Tooltip("Generated routes come from the target image itself, so calibration is OFF by default.")]
    public bool calibrateGeneratedRoutes = false;
    public bool liveRedraw = true;

    [Header("Which generated routes to show")]
    [Tooltip("all, easy, medium or hard")]
    public string visibleDifficulty = "all";
    public bool showLabels = false;
    public bool showAllHolds = true;

    [Header("Route inspection (also controlled by the on-screen buttons)")]
    public string selectedRouteId = "";
    public int currentStep = 0;
    public bool showOnScreenUI = true;

    [Header("Target geometry")]
    public TargetPlane targetPlane = TargetPlane.XZ_YAxisOut;
    public float surfaceLift = 0.002f;

    [Header("Look (sizes are fractions of the wall/photo width)")]
    [Tooltip("Straight = same as the web editor. Smooth = gentle curve through every hold (never overshoots).")]
    public LineStyle lineStyle = LineStyle.Straight;
    public float routeLineWidth = 0.009f;
    public float outlineWidth = 0.004f;
    public float holdDotSize = 0.022f;
    public float endpointScale = 1.6f;
    public float allHoldsDotSize = 0.014f;
    [Tooltip("Sideways shift between routes that share holds, so lines do not sit exactly on top of each other.")]
    public float lineOffset = 0.004f;
    public Color outlineColor = new Color(0f, 0f, 0f, 0.55f);
    public Color footColor = new Color(1f, 0.54f, 0.24f);

    private readonly List<GameObject> spawnedObjects = new List<GameObject>();
    private GameObject routeRoot;
    private Transform lastParent;
    private GCRouteSet currentSet;
    private CalibrationProfile activeProfile;
    private bool useCalibration;
    private float aspect = 1f;
    private int lastSignature;
    private string lastVisible;
    private string lastSelected;
    private int lastStep;

    private static Mesh discMesh, ringMesh;
    private static Texture2D softLineTex;
    private readonly Dictionary<string, Material> materials = new Dictionary<string, Material>();

    // render order (transparent queue): outline < line < marker rim < marker < labels
    private const int QOutline = 3001, QLine = 3002, QRim = 3003, QDot = 3004, QTop = 3005;

    /// <summary>The Transform whose routes are currently drawn (null if none).</summary>
    public Transform CurrentParent { get { return lastParent; } }
    public GCRouteSet CurrentSet { get { return currentSet; } }

    // ------------------------------------------------------------ public API

    public void DrawRouteSet(Transform parent, GCRouteSet set)
    {
        lastParent = parent;
        currentSet = set;
        Redraw();
    }

    public void GenerateAndDrawLocalRoutes(Transform parent)
    {
        lastParent = parent;
        currentSet = null;
        Redraw();
    }

    public void ClearRoute()
    {
        ClearSpawned();
        lastParent = null;
    }

    private void ClearSpawned()
    {
        foreach (var obj in spawnedObjects) if (obj != null) Destroy(obj);
        spawnedObjects.Clear();
        if (routeRoot != null) Destroy(routeRoot);
        routeRoot = null;
    }

    private void OnDestroy()
    {
        foreach (var m in materials.Values) if (m != null) Destroy(m);
        materials.Clear();
    }

    // ------------------------------------------------------------ redraw logic

    private void Update()
    {
        if (!liveRedraw || lastParent == null) return;
        int sig = activeProfile != null ? activeProfile.Signature() : 0;
        if (sig != lastSignature || visibleDifficulty != lastVisible || selectedRouteId != lastSelected || currentStep != lastStep) Redraw();
    }

    private void Redraw()
    {
        if (lastParent == null) return;
        ClearSpawned();

        activeProfile = ResolveProfile(lastParent);
        lastSignature = activeProfile != null ? activeProfile.Signature() : 0;
        lastVisible = visibleDifficulty;
        lastSelected = selectedRouteId;
        lastStep = currentStep;
        aspect = ResolveAspect(lastParent, currentSet != null && currentSet.aspect > 0 ? currentSet.aspect : 0.75f);
        useCalibration = currentSet == null || calibrateGeneratedRoutes;

        routeRoot = new GameObject("RouteRoot");
        routeRoot.transform.SetParent(lastParent, false);

        if (currentSet != null) DrawGenerated(currentSet);
        else DrawTemplates();
    }

    // ------------------------------------------------------------ generated routes

    private void DrawGenerated(GCRouteSet set)
    {
        if (set.routes == null) return;
        bool hasSel = !string.IsNullOrEmpty(selectedRouteId) && FindRoute(selectedRouteId) != null;

        if (showAllHolds && set.holds != null)          // hollow white rings = every hold (context), like the website
            foreach (var h in set.holds)
                SpawnDisc(ImageToNorm(h.x, h.y), allHoldsDotSize, new Color(1f, 1f, 1f, hasSel ? 0.45f : 0.85f), true, QRim, "Hold_" + h.id);

        var shown = new List<GCRoute>();
        foreach (var r in set.routes)
        {
            if (r == null || r.points == null || r.points.Count < 2) continue;
            if (visibleDifficulty != "all" && r.difficulty != visibleDifficulty) continue;
            shown.Add(r);
        }
        // draw the selected route last so it is on top
        shown.Sort((a, b) => (a.id == selectedRouteId ? 1 : 0).CompareTo(b.id == selectedRouteId ? 1 : 0));

        for (int ri = 0; ri < shown.Count; ri++)
        {
            var r = shown[ri];
            bool isSel = hasSel && r.id == selectedRouteId;
            bool dim = hasSel && !isSel;

            Color col;
            if (!ColorUtility.TryParseHtmlString(r.color, out col)) col = Color.white;
            if (dim) col.a = 0.30f;

            var ctrl = new List<Vector2>();
            foreach (var p in r.points) ctrl.Add(ImageToNorm(p.x, p.y));

            List<Vector2> path = lineStyle == LineStyle.Smooth ? SampleCentripetal(ctrl, 10) : new List<Vector2>(ctrl);
            int idx = set.routes.IndexOf(r);
            float off = (idx - (set.routes.Count - 1) / 2f) * lineOffset;
            if (!isSel) OffsetPath(path, off);

            float w = routeLineWidth * (isSel ? 1.5f : (dim ? 0.6f : 1f));
            int qBase = isSel ? 10 : 0;               // selected route above the others
            if (!dim) CreateLine("Outline_" + r.id, path, outlineColor, w + 2f * outlineWidth, QOutline + qBase);
            CreateLine("Route_" + r.id, path, col, w, QLine + qBase);

            for (int i = 0; i < r.points.Count; i++)
            {
                string role = r.points[i].role;
                bool endpoint = role == "start" || role == "finish";
                float size = holdDotSize * (endpoint ? endpointScale : 1f) * (isSel ? 1.25f : (dim ? 0.7f : 1f));
                Color rim = dim ? new Color(1f, 1f, 1f, 0.3f) : Color.white;
                SpawnDisc(ctrl[i], size * 1.25f, rim, false, QRim + qBase, "Rim_" + r.id + "_" + i);
                SpawnDisc(ctrl[i], size, col, false, QDot + qBase, "Hold_" + r.id + "_" + r.points[i].hold_id);

                if (isSel)
                {
                    if (i == currentStep)       // white ring around the current step
                        SpawnDisc(ctrl[i], holdDotSize * 3.2f, Color.white, true, QTop + qBase, "Current_" + i);
                    SpawnLabel(ctrl[i], (i + 1).ToString(), Color.white, 4.2f);
                }
            }
            if (showLabels && !dim)
                SpawnLabel(ctrl[0] + new Vector2(0f, -0.04f), r.id, col, 4f);
        }
    }

    private static void OffsetPath(List<Vector2> path, float off)
    {
        if (Mathf.Abs(off) < 1e-6f || path.Count < 2) return;
        var src = new List<Vector2>(path);
        for (int i = 0; i < src.Count; i++)
        {
            Vector2 a = src[Mathf.Max(i - 1, 0)], b = src[Mathf.Min(i + 1, src.Count - 1)];
            Vector2 t = b - a;
            if (t.sqrMagnitude < 1e-12f) continue;
            t.Normalize();
            path[i] = src[i] + new Vector2(-t.y, t.x) * off;
        }
    }

    // ------------------------------------------------------------ on-screen legend / route chips / steps

    private GCRoute FindRoute(string id)
    {
        if (currentSet == null || currentSet.routes == null || string.IsNullOrEmpty(id)) return null;
        foreach (var r in currentSet.routes) if (r != null && r.id == id) return r;
        return null;
    }

    /// <summary>Bottom of the route UI in GUI units, so other UI can avoid it.</summary>
    public float UiBottom { get; private set; }

    private void OnGUI()
    {
        UiBottom = 0;
        if (!showOnScreenUI || lastParent == null || currentSet == null || currentSet.routes == null) return;
        GCUi.Begin();
        float w = GCUi.W;
        float maxW = Mathf.Min(w - 12f, 520f);

        // legend: one compact line
        float x = 8, y = 8;
        LegendDot(ref x, y, new Color(0.13f, 0.77f, 0.37f), "easy");
        LegendDot(ref x, y, new Color(0.92f, 0.70f, 0.03f), "medium");
        LegendDot(ref x, y, new Color(0.94f, 0.27f, 0.27f), "difficult");
        float rowRight = w - 96f;                    // keep clear of the scanner's Settings button (top right)
        if (x + 150f < rowRight) GUI.Label(new Rect(x, y - 1, 220, 20), "estimated, not official", GCUi.Small);
        y += 24;

        // route chips
        x = 6;
        bool all = string.IsNullOrEmpty(selectedRouteId) || FindRoute(selectedRouteId) == null;
        if (GUI.Button(new Rect(x, y, 50, 32), "All", all ? GCUi.PillOn : GCUi.Pill)) { selectedRouteId = ""; currentStep = 0; }
        x += 54;
        foreach (var r in currentSet.routes)
        {
            if (r == null || r.points == null) continue;
            if (x + 54 > Mathf.Min(rowRight, 6 + maxW)) { x = 6; y += 36; }
            Color c;
            if (!ColorUtility.TryParseHtmlString(r.color, out c)) c = Color.white;
            bool on = r.id == selectedRouteId;
            var rect = new Rect(x, y, 50, 32);
            if (GUI.Button(rect, r.id, on ? GCUi.PillOn : GCUi.Chip)) { selectedRouteId = on ? "" : r.id; currentStep = 0; }
            GCUi.Dot(new Vector2(rect.x + 9, rect.y + 16), 9, c);
            x += 54;
        }
        y += 38;

        // step panel for the selected route
        var sel = FindRoute(selectedRouteId);
        if (sel != null && sel.points != null && sel.steps != null && sel.steps.Count == sel.points.Count && sel.points.Count > 0)
        {
            currentStep = Mathf.Clamp(currentStep, 0, sel.points.Count - 1);
            var p = sel.points[currentStep];
            string limb = string.IsNullOrEmpty(p.limb) ? "" : "  ·  " + p.limb.Replace("_", " ") + (p.limb_status == "confirmed" ? "" : " (est.)");
            GUI.Box(new Rect(6, y, maxW, 92), GUIContent.none, GCUi.Panel);
            GUI.Label(new Rect(14, y + 6, maxW - 16, 22), "<b>" + sel.id + "</b>  " + sel.difficulty + " (estimated)  ·  " + sel.points.Count + " holds", GCUi.Label);
            GUI.Label(new Rect(14, y + 28, maxW - 16, 22), (currentStep + 1) + "/" + sel.points.Count + "  " + sel.steps[currentStep].text + limb, GCUi.Label);
            if (GUI.Button(new Rect(12, y + 54, 90, 32), "< Prev", GCUi.Button)) currentStep = Mathf.Max(0, currentStep - 1);
            if (GUI.Button(new Rect(108, y + 54, 90, 32), "Next >", GCUi.Button)) currentStep = Mathf.Min(sel.points.Count - 1, currentStep + 1);
            y += 98;
        }
        UiBottom = y;
    }

    private static void LegendDot(ref float x, float y, Color c, string text)
    {
        GCUi.Dot(new Vector2(x + 6, y + 9), 11, c);
        GUI.Label(new Rect(x + 14, y - 1, 70, 20), text, GCUi.Small);
        x += 20 + text.Length * 6.5f;
    }

    // ------------------------------------------------------------ template fallback

    private struct TemplateSpec { public string grade; public Color color; public float startX, amp; public int bends; }

    private void DrawTemplates()
    {
        var specs = new[]
        {
            new TemplateSpec { grade = "5A", color = new Color(0.13f, 0.77f, 0.37f), startX = -0.36f, amp = 0.05f, bends = 2 },
            new TemplateSpec { grade = "6B", color = new Color(0.96f, 0.62f, 0.04f), startX = -0.12f, amp = 0.15f, bends = 3 },
            new TemplateSpec { grade = "7A", color = new Color(0.94f, 0.27f, 0.27f), startX = 0.15f, amp = 0.30f, bends = 5 }
        };
        foreach (var s in specs)
        {
            var control = new List<Vector2> { new Vector2(s.startX, -0.4f) };
            for (int j = 1; j <= s.bends; j++)
                control.Add(new Vector2(s.startX + ((j % 2 == 1) ? 1f : -1f) * s.amp, Mathf.Lerp(-0.4f, 0.4f, j / (float)(s.bends + 1))));
            control.Add(new Vector2(s.startX, 0.4f));
            CreateLine("TemplateOutline_" + s.grade, control, outlineColor, routeLineWidth + 2f * outlineWidth, QOutline);
            CreateLine("Template_" + s.grade, control, s.color, routeLineWidth, QLine);
            foreach (var p in control)
            {
                SpawnDisc(p, holdDotSize * 1.25f, Color.white, false, QRim, "T_rim");
                SpawnDisc(p, holdDotSize, s.color, false, QDot, "T_" + s.grade);
            }
            SpawnLabel(new Vector2(s.startX, -0.46f), s.grade, s.color, 5f);
        }
    }

    // ------------------------------------------------------------ coordinates (single source)

    /// <summary>Image space (0..1, origin top-left, y down) to authoring space (-0.5..0.5, y up).</summary>
    private static Vector2 ImageToNorm(float nx, float ny)
    {
        return new Vector2(nx - 0.5f, 0.5f - ny);
    }

    private Vector2 Calibrate(Vector2 p)
    {
        var c = activeProfile;
        if (useCalibration && c != null)
        {
            if (c.flipHorizontal) p.x = -p.x;
            if (c.flipVertical) p.y = -p.y;
            p *= c.routeScale;
            p = Quaternion.Euler(0f, 0f, c.rotationAngle) * p;
            p += new Vector2(c.offsetX, c.offsetY);
        }
        p.x = Mathf.Clamp(p.x, -0.5f, 0.5f);
        p.y = Mathf.Clamp(p.y, -0.5f, 0.5f);
        return p;
    }

    private Vector3 ToTarget(Vector2 p, float lift)
    {
        Vector2 c = Calibrate(p);
        float v = c.y * aspect;   // image height = aspect * width
        return targetPlane == TargetPlane.XZ_YAxisOut ? new Vector3(c.x, lift, v) : new Vector3(c.x, v, lift);
    }

    /// <summary>Rotation that lays an XY-plane object flat on the target plane.</summary>
    private Quaternion FlatRotation()
    {
        return targetPlane == TargetPlane.XZ_YAxisOut ? Quaternion.Euler(90f, 0f, 0f) : Quaternion.identity;
    }

    private float ResolveAspect(Transform parent, float fallback)
    {
        var itb = parent.GetComponentInParent<ImageTargetBehaviour>();
        if (itb != null)
        {
            Vector2 size = itb.GetSize();
            if (size.x > 1e-4f && size.y > 1e-4f) return size.y / size.x;
        }
        if (currentSet != null && currentSet.aspect > 0f) return currentSet.aspect;
        if (activeProfile != null && activeProfile.imageAspect > 0f) return activeProfile.imageAspect;
        return fallback;
    }

    private CalibrationProfile ResolveProfile(Transform parent)
    {
        var observer = parent.GetComponentInParent<ObserverBehaviour>();
        string n = observer != null ? observer.TargetName : parent.name;
        var found = database != null ? database.Find(n) : null;
        return found != null ? found : fallbackProfile;
    }

    // ------------------------------------------------------------ geometry helpers

    /// <summary>Centripetal Catmull-Rom: passes through every hold without loops or overshoot.</summary>
    private static List<Vector2> SampleCentripetal(List<Vector2> pts, int perSeg)
    {
        var res = new List<Vector2>();
        if (pts.Count < 3) return new List<Vector2>(pts);
        for (int i = 0; i < pts.Count - 1; i++)
        {
            Vector2 p1 = pts[i], p2 = pts[i + 1];
            Vector2 p0 = i > 0 ? pts[i - 1] : p1 + (p1 - p2);
            Vector2 p3 = i + 2 < pts.Count ? pts[i + 2] : p2 + (p2 - p1);
            float t0 = 0f;
            float t1 = t0 + Mathf.Sqrt(Mathf.Max(1e-6f, (p1 - p0).magnitude));
            float t2 = t1 + Mathf.Sqrt(Mathf.Max(1e-6f, (p2 - p1).magnitude));
            float t3 = t2 + Mathf.Sqrt(Mathf.Max(1e-6f, (p3 - p2).magnitude));
            for (int k = 0; k < perSeg; k++)
            {
                float t = Mathf.Lerp(t1, t2, k / (float)perSeg);
                Vector2 a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1;
                Vector2 a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2;
                Vector2 a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3;
                Vector2 b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2;
                Vector2 b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3;
                res.Add((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2);
            }
        }
        res.Add(pts[pts.Count - 1]);
        return res;
    }

    // ------------------------------------------------------------ materials / meshes

    private Material Mat(Color c, int queue, bool soft)
    {
        string key = ColorUtility.ToHtmlStringRGBA(c) + queue + soft;
        Material m;
        if (materials.TryGetValue(key, out m) && m != null) return m;
        Shader sh = Shader.Find("Sprites/Default");
        if (sh == null) sh = Shader.Find("Unlit/Transparent");
        m = new Material(sh);
        m.color = c;
        m.renderQueue = queue;
        if (soft) m.mainTexture = SoftLineTexture();
        materials[key] = m;
        return m;
    }

    /// <summary>Alpha fades at both edges of the line width: cheap anti-aliasing for LineRenderer.</summary>
    private static Texture2D SoftLineTexture()
    {
        if (softLineTex != null) return softLineTex;
        const int n = 32;
        softLineTex = new Texture2D(2, n, TextureFormat.RGBA32, false) { wrapMode = TextureWrapMode.Clamp, filterMode = FilterMode.Bilinear };
        for (int y = 0; y < n; y++)
        {
            float v = (y + 0.5f) / n;
            float a = Mathf.Clamp01(Mathf.Min(v, 1f - v) / 0.18f);
            softLineTex.SetPixel(0, y, new Color(1f, 1f, 1f, a));
            softLineTex.SetPixel(1, y, new Color(1f, 1f, 1f, a));
        }
        softLineTex.Apply();
        return softLineTex;
    }

    private static Mesh BuildCircleMesh(float inner, int seg)
    {
        var m = new Mesh();
        var v = new List<Vector3>();
        var uv = new List<Vector2>();
        var tri = new List<int>();
        if (inner <= 0f)
        {
            v.Add(Vector3.zero); uv.Add(new Vector2(0.5f, 0.5f));
            for (int i = 0; i <= seg; i++)
            {
                float a = i * Mathf.PI * 2f / seg;
                v.Add(new Vector3(Mathf.Cos(a) * 0.5f, Mathf.Sin(a) * 0.5f, 0f)); uv.Add(new Vector2(0.5f, 0.5f));
                if (i > 0) { tri.Add(0); tri.Add(i); tri.Add(i + 1); }
            }
        }
        else
        {
            for (int i = 0; i <= seg; i++)
            {
                float a = i * Mathf.PI * 2f / seg;
                v.Add(new Vector3(Mathf.Cos(a) * 0.5f, Mathf.Sin(a) * 0.5f, 0f));
                v.Add(new Vector3(Mathf.Cos(a) * 0.5f * inner, Mathf.Sin(a) * 0.5f * inner, 0f));
                uv.Add(new Vector2(0.5f, 0.5f)); uv.Add(new Vector2(0.5f, 0.5f));
                if (i > 0)
                {
                    int o = 2 * i;
                    tri.Add(o - 2); tri.Add(o); tri.Add(o - 1);
                    tri.Add(o - 1); tri.Add(o); tri.Add(o + 1);
                }
            }
        }
        var col = new List<Color>();
        for (int i = 0; i < v.Count; i++) col.Add(Color.white);
        m.SetVertices(v);
        m.SetColors(col);
        m.SetUVs(0, uv);
        m.SetTriangles(tri, 0);
        m.RecalculateBounds();
        return m;
    }

    // ------------------------------------------------------------ spawning

    private void CreateLine(string name, List<Vector2> path, Color color, float width, int queue)
    {
        if (path.Count < 2) return;
        var go = new GameObject(name);
        go.transform.SetParent(routeRoot.transform, false);
        Quaternion flat = FlatRotation();
        go.transform.localRotation = flat;                 // local Z = wall normal -> TransformZ lays the line flat
        var line = go.AddComponent<LineRenderer>();
        line.useWorldSpace = false;
        line.alignment = LineAlignment.TransformZ;
        line.widthMultiplier = width;
        line.numCapVertices = 8;
        line.numCornerVertices = 8;
        line.textureMode = LineTextureMode.Stretch;
        line.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        line.receiveShadows = false;
        line.sharedMaterial = Mat(Color.white, queue, true);
        line.startColor = color;
        line.endColor = color;
        Quaternion inv = Quaternion.Inverse(flat);
        var pos = new Vector3[path.Count];
        for (int i = 0; i < path.Count; i++) pos[i] = inv * ToTarget(path[i], surfaceLift);
        line.positionCount = pos.Length;
        line.SetPositions(pos);
        spawnedObjects.Add(go);
    }

    private void SpawnDisc(Vector2 p, float size, Color color, bool ring, int queue, string name)
    {
        if (discMesh == null) discMesh = BuildCircleMesh(0f, 32);
        if (ringMesh == null) ringMesh = BuildCircleMesh(0.72f, 32);
        var go = new GameObject(name);
        go.transform.SetParent(routeRoot.transform, false);
        go.transform.localPosition = ToTarget(p, surfaceLift * 1.5f);
        go.transform.localRotation = FlatRotation();
        go.transform.localScale = new Vector3(size, size, size);
        go.AddComponent<MeshFilter>().sharedMesh = ring ? ringMesh : discMesh;
        var mr = go.AddComponent<MeshRenderer>();
        mr.sharedMaterial = Mat(color, queue, false);
        mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        mr.receiveShadows = false;
        spawnedObjects.Add(go);
    }

    private void SpawnLabel(Vector2 p, string text, Color color, float fontSize)
    {
        var go = new GameObject("Label_" + text);
        go.transform.SetParent(routeRoot.transform, false);
        go.transform.localPosition = ToTarget(p, surfaceLift * 3f);
        go.transform.localRotation = FlatRotation();
        go.transform.localScale = Vector3.one * 0.01f;
        var tmp = go.AddComponent<TextMeshPro>();
        tmp.text = text;
        tmp.fontSize = fontSize;
        tmp.fontStyle = FontStyles.Bold;
        tmp.color = color;
        tmp.alignment = TextAlignmentOptions.Center;
        tmp.outlineWidth = 0.25f;
        tmp.outlineColor = new Color32(0, 0, 0, 200);
        tmp.fontMaterial.renderQueue = QTop + 20;      // per-label material instance, drawn above the markers
        spawnedObjects.Add(go);
    }
}
