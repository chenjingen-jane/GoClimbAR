using UnityEngine;

/// <summary>
/// Shared look for the on-screen UI (IMGUI). Everything is laid out in "logical pixels" on a screen whose
/// SHORT side is 420 units, so buttons have the same physical size on a PC and on an Android phone.
/// Rounded, semi-transparent dark buttons; white text. No prefabs or Canvas needed.
/// </summary>
public static class GCUi
{
    public static GUIStyle Button, Primary, Danger, Pill, PillOn, Label, Small, Title, Panel, Field, Chip;
    public static Texture2D Circle, Ring, White;
    private static bool ready;
    private static int builtFor = -1;

    public static float Scale { get { return Mathf.Max(1f, Mathf.Min(Screen.width, Screen.height) / 420f); } }
    public static float W { get { return Screen.width / Scale; } }
    public static float H { get { return Screen.height / Scale; } }

    /// <summary>Call at the start of every OnGUI.</summary>
    public static void Begin()
    {
        float s = Scale;
        GUI.matrix = Matrix4x4.Scale(new Vector3(s, s, 1f));
        if (!ready || builtFor != Screen.width * 10000 + Screen.height || Button == null || Button.normal.background == null)
            Build();
    }

    /// <summary>Convert a screen point (Input.mousePosition, origin bottom-left, pixels) to GUI units (origin top-left).</summary>
    public static Vector2 ScreenToGui(Vector2 p)
    {
        return new Vector2(p.x / Scale, (Screen.height - p.y) / Scale);
    }

    private static void Build()
    {
        ready = true;
        builtFor = Screen.width * 10000 + Screen.height;
        White = Texture2D.whiteTexture;
        Circle = MakeCircle(64, 0f, Color.white);
        Ring = MakeCircle(64, 0.80f, Color.white);

        Texture2D dark = MakeRounded(new Color(0.08f, 0.08f, 0.10f, 0.78f), 12);
        Texture2D darkHi = MakeRounded(new Color(0.22f, 0.22f, 0.26f, 0.90f), 12);
        Texture2D blue = MakeRounded(new Color(0.15f, 0.39f, 0.92f, 0.95f), 12);
        Texture2D blueHi = MakeRounded(new Color(0.25f, 0.50f, 1f, 1f), 12);
        Texture2D red = MakeRounded(new Color(0.60f, 0.12f, 0.12f, 0.90f), 12);
        Texture2D white = MakeRounded(new Color(1f, 1f, 1f, 0.95f), 12);
        Texture2D field = MakeRounded(new Color(1f, 1f, 1f, 0.92f), 8);
        Texture2D panel = MakeRounded(new Color(0.06f, 0.06f, 0.08f, 0.72f), 14);

        Button = MakeButton(dark, darkHi, Color.white, 15);
        Primary = MakeButton(blue, blueHi, Color.white, 15);
        Primary.fontStyle = FontStyle.Bold;
        Danger = MakeButton(red, darkHi, Color.white, 15);
        Pill = MakeButton(dark, darkHi, new Color(1f, 1f, 1f, 0.8f), 13);
        PillOn = MakeButton(white, white, new Color(0.08f, 0.08f, 0.1f), 13);
        PillOn.fontStyle = FontStyle.Bold;
        Chip = MakeButton(dark, darkHi, Color.white, 13);
        Chip.fontStyle = FontStyle.Bold;

        Label = new GUIStyle(GUI.skin.label) { fontSize = 14, wordWrap = true, richText = true };
        Label.normal.textColor = Color.white;
        Small = new GUIStyle(Label) { fontSize = 12 };
        Small.normal.textColor = new Color(1f, 1f, 1f, 0.75f);
        Title = new GUIStyle(Label) { fontSize = 16, fontStyle = FontStyle.Bold };
        Panel = new GUIStyle(GUI.skin.box) { padding = new RectOffset(10, 10, 8, 8) };
        Panel.normal.background = panel;
        Panel.border = new RectOffset(14, 14, 14, 14);
        Field = new GUIStyle(GUI.skin.textField) { fontSize = 15, padding = new RectOffset(8, 8, 6, 6) };
        Field.normal.background = field;
        Field.focused.background = field;
        Field.hover.background = field;
        Field.border = new RectOffset(8, 8, 8, 8);
        Field.normal.textColor = Field.focused.textColor = Field.hover.textColor = new Color(0.1f, 0.1f, 0.12f);
    }

    private static GUIStyle MakeButton(Texture2D bg, Texture2D active, Color text, int size)
    {
        var s = new GUIStyle(GUI.skin.button) { fontSize = size, alignment = TextAnchor.MiddleCenter, wordWrap = false };
        s.normal.background = bg;
        s.hover.background = bg;
        s.focused.background = bg;
        s.active.background = active;
        s.onNormal.background = bg;
        s.normal.textColor = s.hover.textColor = s.active.textColor = s.focused.textColor = text;
        s.border = new RectOffset(12, 12, 12, 12);
        s.padding = new RectOffset(10, 10, 6, 6);
        s.margin = new RectOffset(3, 3, 3, 3);
        return s;
    }

    /// <summary>Rounded rectangle, used 9-sliced (GUIStyle.border = radius).</summary>
    private static Texture2D MakeRounded(Color c, int radius)
    {
        int n = radius * 2 + 4;
        var t = new Texture2D(n, n, TextureFormat.RGBA32, false) { wrapMode = TextureWrapMode.Clamp, filterMode = FilterMode.Bilinear };
        for (int y = 0; y < n; y++)
            for (int x = 0; x < n; x++)
            {
                float cx = Mathf.Clamp(x + 0.5f, radius, n - radius), cy = Mathf.Clamp(y + 0.5f, radius, n - radius);
                float d = Vector2.Distance(new Vector2(x + 0.5f, y + 0.5f), new Vector2(cx, cy));
                float a = Mathf.Clamp01(radius - d + 0.5f);
                t.SetPixel(x, y, new Color(c.r, c.g, c.b, c.a * a));
            }
        t.Apply();
        return t;
    }

    /// <summary>Anti-aliased disc (inner = 0) or ring (inner = inner radius as a fraction of the outer).</summary>
    public static Texture2D MakeCircle(int n, float inner, Color c)
    {
        var t = new Texture2D(n, n, TextureFormat.RGBA32, false) { wrapMode = TextureWrapMode.Clamp, filterMode = FilterMode.Bilinear };
        float r = n * 0.5f - 1f, ri = r * inner;
        for (int y = 0; y < n; y++)
            for (int x = 0; x < n; x++)
            {
                float d = Vector2.Distance(new Vector2(x + 0.5f, y + 0.5f), new Vector2(n * 0.5f, n * 0.5f));
                float a = Mathf.Clamp01(r - d + 0.5f);
                if (inner > 0f) a *= Mathf.Clamp01(d - ri + 0.5f);
                t.SetPixel(x, y, new Color(c.r, c.g, c.b, c.a * a));
            }
        t.Apply();
        return t;
    }

    public static void Rect(Rect r, Color c)
    {
        var old = GUI.color;
        GUI.color = c;
        GUI.DrawTexture(r, Texture2D.whiteTexture);
        GUI.color = old;
    }

    public static void Dot(Vector2 center, float size, Color c)
    {
        var old = GUI.color;
        GUI.color = c;
        GUI.DrawTexture(new Rect(center.x - size / 2, center.y - size / 2, size, size), Circle);
        GUI.color = old;
    }
}
