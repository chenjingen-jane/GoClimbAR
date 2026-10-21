using System;
using System.Collections.Generic;

/// <summary>
/// OFFLINE route generation: hold detection + route planning that runs entirely on the phone, so a wall can be
/// scanned with no internet. It is a port of the Python server (detect.py IndoorDetector / ClassicalDetector,
/// planner.py generate_routes) simplified to plain C# with no OpenCV:
///   * Gaussian blur = three box blurs, Canny = Sobel + hysteresis (no thinning),
///   * square max/min filters instead of elliptical ones, connected components instead of contours.
/// Results are close to the server's but not identical. The server stays the reference: when the user uploads,
/// the server stores these holds and an admin can correct them in the web editor.
///
/// Pure C# (no UnityEngine types) so it can run on a background thread. Input: RGB bytes, top row first.
/// Output uses the same classes as the server JSON (RouteSetData.cs), so RouteRenderer draws it unchanged.
/// </summary>
public static class OfflineRouteEngine
{
    public const string AlgoVersion = "offline-1.0";
    public const int MinHoldsPerRoute = 5;
    public const float GlobalMaxReach = 0.26f;

    // ================================================================== public entry point

    /// <param name="rgb">width*height*3 bytes, row 0 = TOP of the picture.</param>
    public static GCRouteSet Run(byte[] rgb, int width, int height, string mode, int maxRoutes, int workWidth = 480)
    {
        if (rgb == null || width < 8 || height < 8 || rgb.Length < width * height * 3)
            throw new ArgumentException("invalid image");
        List<GCHold> holds = mode == "outdoor"
            ? DetectOutdoor(rgb, width, height, workWidth)
            : DetectIndoor(rgb, width, height, workWidth);
        float aspect = height / (float)width;
        var warnings = new List<string>();
        List<GCRoute> routes = PlanWithRelax(holds, aspect, Math.Max(1, Math.Min(30, maxRoutes)), warnings);

        var doc = new GCRouteSet();
        doc.schema = 2;
        doc.wall_id = "";
        doc.scan_id = "";
        doc.name = "(not saved)";
        doc.venue = "";
        doc.mode = mode == "outdoor" ? "outdoor" : "indoor";
        doc.vuforia_target = "";
        doc.coordinate_frame = "scan_photo";
        doc.image_width = width;
        doc.image_height = height;
        doc.aspect = (float)Math.Round(aspect, 5);
        doc.coordinate_system = "normalized image: origin top-left, x right, y down, 0..1";
        doc.algo_version = AlgoVersion;
        doc.disclaimer = "Colours show ESTIMATED difficulty computed on this phone from image geometry. " +
                         "They are not official climbing grades and say nothing about safety.";
        doc.updated_at = DateTime.UtcNow.ToString("o");
        doc.holds = holds;
        doc.routes = routes;
        doc.warnings = warnings;
        return doc;
    }

    // ================================================================== image operations

    private static int Clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

    private static void ScaledSize(int w0, int h0, int workWidth, out int w, out int h)
    {
        double k = w0 > workWidth ? workWidth / (double)w0 : 1.0;
        w = Math.Max(2, (int)Math.Round(w0 * k));
        h = Math.Max(2, (int)Math.Round(h0 * k));
    }

    /// <summary>Area-average downscale to (w,h), returns gray 0..1.</summary>
    private static float[] ToGray(byte[] rgb, int w0, int h0, int w, int h)
    {
        var g = new float[w * h];
        for (int y = 0; y < h; y++)
        {
            int sy0 = (int)Math.Floor(y * (double)h0 / h);
            int sy1 = Math.Max(sy0 + 1, (int)Math.Floor((y + 1) * (double)h0 / h));
            for (int x = 0; x < w; x++)
            {
                int sx0 = (int)Math.Floor(x * (double)w0 / w);
                int sx1 = Math.Max(sx0 + 1, (int)Math.Floor((x + 1) * (double)w0 / w));
                double s = 0; int n = 0;
                for (int yy = sy0; yy < sy1; yy++)
                    for (int xx = sx0; xx < sx1; xx++)
                    {
                        int i = (yy * w0 + xx) * 3;
                        s += 0.299 * rgb[i] + 0.587 * rgb[i + 1] + 0.114 * rgb[i + 2];
                        n++;
                    }
                g[y * w + x] = (float)(s / (n * 255.0));
            }
        }
        return g;
    }

    private static byte[] ToRgbSmall(byte[] rgb, int w0, int h0, int w, int h)
    {
        var o = new byte[w * h * 3];
        for (int y = 0; y < h; y++)
        {
            int sy0 = (int)Math.Floor(y * (double)h0 / h);
            int sy1 = Math.Max(sy0 + 1, (int)Math.Floor((y + 1) * (double)h0 / h));
            for (int x = 0; x < w; x++)
            {
                int sx0 = (int)Math.Floor(x * (double)w0 / w);
                int sx1 = Math.Max(sx0 + 1, (int)Math.Floor((x + 1) * (double)w0 / w));
                int r = 0, g = 0, b = 0, n = 0;
                for (int yy = sy0; yy < sy1; yy++)
                    for (int xx = sx0; xx < sx1; xx++)
                    {
                        int i = (yy * w0 + xx) * 3;
                        r += rgb[i]; g += rgb[i + 1]; b += rgb[i + 2]; n++;
                    }
                int oi = (y * w + x) * 3;
                o[oi] = (byte)Math.Round(r / (double)n);
                o[oi + 1] = (byte)Math.Round(g / (double)n);
                o[oi + 2] = (byte)Math.Round(b / (double)n);
            }
        }
        return o;
    }

    private static void BoxH(float[] src, float[] dst, int w, int h, int r)
    {
        float inv = 1f / (2 * r + 1);
        for (int y = 0; y < h; y++)
        {
            int row = y * w;
            float acc = 0;
            for (int k = -r; k <= r; k++) acc += src[row + Clampi(k, 0, w - 1)];
            for (int x = 0; x < w; x++)
            {
                dst[row + x] = acc * inv;
                acc += src[row + Clampi(x + r + 1, 0, w - 1)] - src[row + Clampi(x - r, 0, w - 1)];
            }
        }
    }

    private static void BoxV(float[] src, float[] dst, int w, int h, int r)
    {
        float inv = 1f / (2 * r + 1);
        for (int x = 0; x < w; x++)
        {
            float acc = 0;
            for (int k = -r; k <= r; k++) acc += src[Clampi(k, 0, h - 1) * w + x];
            for (int y = 0; y < h; y++)
            {
                dst[y * w + x] = acc * inv;
                acc += src[Clampi(y + r + 1, 0, h - 1) * w + x] - src[Clampi(y - r, 0, h - 1) * w + x];
            }
        }
    }

    /// <summary>Gaussian blur approximated by three box blurs. Cost does not depend on sigma.</summary>
    private static float[] Blur(float[] src, int w, int h, double sigma)
    {
        var o = (float[])src.Clone();
        if (sigma < 0.5) return o;
        var tmp = new float[w * h];
        double wIdeal = Math.Sqrt(12 * sigma * sigma / 3 + 1);
        int wl = (int)Math.Floor(wIdeal);
        if (wl % 2 == 0) wl--;
        int wu = wl + 2;
        double mIdeal = (12 * sigma * sigma - 3 * wl * wl - 12 * wl - 9) / (-4.0 * wl - 4);
        int m = (int)Math.Round(mIdeal);
        for (int i = 0; i < 3; i++)
        {
            int r = ((i < m ? wl : wu) - 1) / 2;
            BoxH(o, tmp, w, h, r);
            BoxV(tmp, o, w, h, r);
        }
        return o;
    }

    private static float Percentile(float[] a, double q, byte[] mask)
    {
        int n = 0;
        for (int i = 0; i < a.Length; i++) if (mask == null || mask[i] != 0) n++;
        if (n == 0) return 0f;
        var v = new float[n];
        int k = 0;
        for (int i = 0; i < a.Length; i++) if (mask == null || mask[i] != 0) v[k++] = a[i];
        Array.Sort(v);
        double pos = (n - 1) * q / 100.0;
        int lo = (int)Math.Floor(pos), hi = Math.Min(n - 1, lo + 1);
        return (float)(v[lo] + (v[hi] - v[lo]) * (pos - lo));
    }

    private static float[] Normalize(float[] a, float floor, double q, byte[] mask)
    {
        float d = Math.Max(Percentile(a, q, mask), floor);
        var o = new float[a.Length];
        if (d <= 1e-6f) return o;
        for (int i = 0; i < a.Length; i++)
        {
            float v = a[i] / d;
            o[i] = v < 0f ? 0f : (v > 1f ? 1f : v);
        }
        return o;
    }

    /// <summary>Separable square max (dilate) or min filter of radius r.</summary>
    private static float[] RankFilter(float[] src, int w, int h, int r, bool dilate)
    {
        var tmp = new float[w * h];
        var o = new float[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                float best = src[y * w + x];
                int x0 = Math.Max(0, x - r), x1 = Math.Min(w - 1, x + r);
                for (int xx = x0; xx <= x1; xx++)
                {
                    float v = src[y * w + xx];
                    if (dilate ? v > best : v < best) best = v;
                }
                tmp[y * w + x] = best;
            }
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                float best = tmp[y * w + x];
                int y0 = Math.Max(0, y - r), y1 = Math.Min(h - 1, y + r);
                for (int yy = y0; yy <= y1; yy++)
                {
                    float v = tmp[yy * w + x];
                    if (dilate ? v > best : v < best) best = v;
                }
                o[y * w + x] = best;
            }
        return o;
    }

    /// <summary>Binary morphology (0/1 mask) with a disc kernel of diameter k.</summary>
    private static byte[] Morph(byte[] mask, int w, int h, int k, bool dilate)
    {
        int r = k / 2;
        var o = new byte[w * h];
        var offs = new List<int>();
        for (int dy = -r; dy <= r; dy++)
            for (int dx = -r; dx <= r; dx++)
                if (dx * dx + dy * dy <= r * r + r) { offs.Add(dx); offs.Add(dy); }
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                byte v = dilate ? (byte)0 : (byte)1;
                for (int t = 0; t < offs.Count; t += 2)
                {
                    int xx = x + offs[t], yy = y + offs[t + 1];
                    if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
                    byte m = mask[yy * w + xx];
                    if (dilate && m != 0) { v = 1; break; }
                    if (!dilate && m == 0) { v = 0; break; }
                }
                o[y * w + x] = v;
            }
        return o;
    }

    private static byte[] OpenMask(byte[] m, int w, int h, int k) { return Morph(Morph(m, w, h, k, false), w, h, k, true); }
    private static byte[] CloseMask(byte[] m, int w, int h, int k) { return Morph(Morph(m, w, h, k, true), w, h, k, false); }

    /// <summary>8-connected labelling. 0 = background, 1..n components.</summary>
    private static int[] Label(byte[] mask, int w, int h, out int n)
    {
        var labels = new int[w * h];
        var stack = new int[w * h];
        n = 0;
        for (int i = 0; i < w * h; i++)
        {
            if (mask[i] == 0 || labels[i] != 0) continue;
            n++;
            int sp = 0;
            stack[sp++] = i;
            labels[i] = n;
            while (sp > 0)
            {
                int p = stack[--sp];
                int px = p % w, py = p / w;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        int xx = px + dx, yy = py + dy;
                        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
                        int q = yy * w + xx;
                        if (mask[q] != 0 && labels[q] == 0) { labels[q] = n; stack[sp++] = q; }
                    }
            }
        }
        return labels;
    }

    private static int Otsu(byte[] v)
    {
        var hist = new double[256];
        for (int i = 0; i < v.Length; i++) hist[v[i]]++;
        double total = v.Length, sum = 0;
        for (int t = 0; t < 256; t++) sum += t * hist[t];
        double sumB = 0, wB = 0, best = 0;
        int thr = 0;
        for (int t = 0; t < 256; t++)
        {
            wB += hist[t];
            if (wB == 0) continue;
            double wF = total - wB;
            if (wF == 0) break;
            sumB += t * hist[t];
            double mB = sumB / wB, mF = (sum - sumB) / wF;
            double between = wB * wF * (mB - mF) * (mB - mF);
            if (between > best) { best = between; thr = t; }
        }
        return thr;
    }

    private static double Cross(int ox, int oy, int ax, int ay, int bx, int by)
    {
        return (double)(ax - ox) * (by - oy) - (double)(ay - oy) * (bx - ox);
    }

    /// <summary>Convex hull area of integer points (monotone chain).</summary>
    private static double HullArea(List<int> xs, List<int> ys)
    {
        int n = xs.Count;
        if (n < 3) return 0;
        var idx = new List<int>(n);
        for (int i = 0; i < n; i++) idx.Add(i);
        idx.Sort((p, q) => xs[p] != xs[q] ? xs[p].CompareTo(xs[q]) : ys[p].CompareTo(ys[q]));
        var hx = new int[2 * n];
        var hy = new int[2 * n];
        int k = 0;
        for (int t = 0; t < n; t++)
        {
            int i = idx[t];
            while (k >= 2 && Cross(hx[k - 2], hy[k - 2], hx[k - 1], hy[k - 1], xs[i], ys[i]) <= 0) k--;
            hx[k] = xs[i]; hy[k] = ys[i]; k++;
        }
        int lower = k + 1;
        for (int t = n - 2; t >= 0; t--)
        {
            int i = idx[t];
            while (k >= lower && Cross(hx[k - 2], hy[k - 2], hx[k - 1], hy[k - 1], xs[i], ys[i]) <= 0) k--;
            hx[k] = xs[i]; hy[k] = ys[i]; k++;
        }
        double twiceArea = 0;
        for (int t = 0; t < k - 1; t++) twiceArea += (double)hx[t] * hy[t + 1] - (double)hx[t + 1] * hy[t];
        return Math.Abs(twiceArea) / 2.0;
    }

    /// <summary>Sobel (L1) magnitude with Canny-like hysteresis thresholds, no edge thinning.</summary>
    private static float[] SobelEdges(float[] g, int w, int h, float lo, float hi)
    {
        var mag = new float[w * h];
        for (int y = 1; y < h - 1; y++)
            for (int x = 1; x < w - 1; x++)
            {
                int i = y * w + x;
                float gx = (g[i - w + 1] + 2 * g[i + 1] + g[i + w + 1]) - (g[i - w - 1] + 2 * g[i - 1] + g[i + w - 1]);
                float gy = (g[i + w - 1] + 2 * g[i + w] + g[i + w + 1]) - (g[i - w - 1] + 2 * g[i - w] + g[i - w + 1]);
                mag[i] = Math.Abs(gx) + Math.Abs(gy);
            }
        var edges = new float[w * h];
        var stack = new int[w * h];
        int sp = 0;
        for (int i = 0; i < w * h; i++) if (mag[i] >= hi) { edges[i] = 1; stack[sp++] = i; }
        while (sp > 0)
        {
            int p = stack[--sp];
            int px = p % w, py = p / w;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                {
                    int xx = px + dx, yy = py + dy;
                    if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
                    int q = yy * w + xx;
                    if (edges[q] == 0 && mag[q] >= lo) { edges[q] = 1; stack[sp++] = q; }
                }
        }
        return edges;
    }

    // ================================================================== detection

    private struct Cand { public float x, y, conf; public string src; }
    private struct Region { public int cx, cy, bx, by, bw, bh; public float conf; }

    private static byte[] TwoToneMask(float[] g, int w, int h)
    {
        float[] b = Blur(g, w, h, 1.5);
        var g8 = new byte[w * h];
        for (int i = 0; i < w * h; i++) g8[i] = (byte)Clampi((int)Math.Round(b[i] * 255), 0, 255);
        int t = Otsu(g8);
        long nd = 0; double sd = 0, sl = 0;
        for (int i = 0; i < w * h; i++)
        {
            if (g8[i] <= t) { nd++; sd += g8[i]; }
            else sl += g8[i];
        }
        if (nd == 0 || nd == (long)w * h) return null;
        double meanD = sd / nd, meanL = sl / ((long)w * h - nd);
        if ((meanL - meanD) / 255.0 < 0.25) return null;            // not a clean two-tone picture
        double darkFrac = nd / (double)(w * h);
        bool minorityDark = darkFrac < 0.5;
        double frac = minorityDark ? darkFrac : 1 - darkFrac;
        if (frac < 0.03 || frac > 0.45) return null;
        var m = new byte[w * h];
        for (int i = 0; i < w * h; i++) m[i] = ((g8[i] <= t) == minorityDark) ? (byte)1 : (byte)0;
        return m;
    }

    /// <summary>Whole-hold candidates: one point per shape (its thickest part).</summary>
    private static List<Region> Regions(float[] g, int w, int h, float regionContrast)
    {
        float[] bg = Blur(g, w, h, 0.06 * w);
        var diff = new float[w * h];
        for (int i = 0; i < w * h; i++) diff[i] = Math.Abs(g[i] - bg[i]);
        byte[] mask = TwoToneMask(g, w, h);
        if (mask == null)
        {
            mask = new byte[w * h];
            for (int i = 0; i < w * h; i++) mask[i] = diff[i] > regionContrast ? (byte)1 : (byte)0;
        }
        mask = CloseMask(mask, w, h, Math.Max(3, (int)(0.008 * w) | 1));
        mask = OpenMask(mask, w, h, Math.Max(3, (int)(0.005 * w) | 1));

        // fill holes: background not connected to the image border belongs to the shape around it
        var outside = new byte[w * h];
        var stack = new int[w * h];
        int sp = 0;
        for (int x = 0; x < w; x++)
        {
            int a = x, b = (h - 1) * w + x;
            if (mask[a] == 0 && outside[a] == 0) { outside[a] = 1; stack[sp++] = a; }
            if (mask[b] == 0 && outside[b] == 0) { outside[b] = 1; stack[sp++] = b; }
        }
        for (int y = 0; y < h; y++)
        {
            int a = y * w, b = y * w + w - 1;
            if (mask[a] == 0 && outside[a] == 0) { outside[a] = 1; stack[sp++] = a; }
            if (mask[b] == 0 && outside[b] == 0) { outside[b] = 1; stack[sp++] = b; }
        }
        while (sp > 0)
        {
            int p = stack[--sp];
            int px = p % w, py = p / w;
            if (px > 0) { int q = p - 1; if (mask[q] == 0 && outside[q] == 0) { outside[q] = 1; stack[sp++] = q; } }
            if (px < w - 1) { int q = p + 1; if (mask[q] == 0 && outside[q] == 0) { outside[q] = 1; stack[sp++] = q; } }
            if (py > 0) { int q = p - w; if (mask[q] == 0 && outside[q] == 0) { outside[q] = 1; stack[sp++] = q; } }
            if (py < h - 1) { int q = p + w; if (mask[q] == 0 && outside[q] == 0) { outside[q] = 1; stack[sp++] = q; } }
        }
        var filled = new byte[w * h];
        for (int i = 0; i < w * h; i++) filled[i] = (mask[i] != 0 || outside[i] == 0) ? (byte)1 : (byte)0;

        int n;
        int[] labels = Label(filled, w, h, out n);
        var area = new int[n + 1];
        var x0 = new int[n + 1]; var y0 = new int[n + 1]; var x1 = new int[n + 1]; var y1 = new int[n + 1];
        var csum = new double[n + 1];
        for (int l = 0; l <= n; l++) { x0[l] = w; y0[l] = h; x1[l] = -1; y1[l] = -1; }
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int l = labels[y * w + x];
                if (l == 0) continue;
                area[l]++;
                csum[l] += diff[y * w + x];
                if (x < x0[l]) x0[l] = x;
                if (x > x1[l]) x1[l] = x;
                if (y < y0[l]) y0[l] = y;
                if (y > y1[l]) y1[l] = y;
            }

        var regs = new List<Region>();
        var xs = new List<int>();
        var ys = new List<int>();
        for (int l = 1; l <= n; l++)
        {
            double frac = area[l] / (double)(w * h);
            if (frac < 0.0008 || frac > 0.15) continue;
            int bw = x1[l] - x0[l] + 1, bh = y1[l] - y0[l] + 1;
            if (bw > 0.7 * w || bh > 0.7 * h) continue;

            xs.Clear(); ys.Clear();
            for (int y = y0[l]; y <= y1[l]; y++)
                for (int x = x0[l]; x <= x1[l]; x++)
                {
                    if (labels[y * w + x] != l) continue;
                    bool edge = x == 0 || y == 0 || x == w - 1 || y == h - 1 ||
                                labels[y * w + x - 1] != l || labels[y * w + x + 1] != l ||
                                labels[(y - 1) * w + x] != l || labels[(y + 1) * w + x] != l;
                    if (!edge) continue;
                    xs.Add(x); ys.Add(y); xs.Add(x + 1); ys.Add(y);
                    xs.Add(x); ys.Add(y + 1); xs.Add(x + 1); ys.Add(y + 1);
                }
            double ha = HullArea(xs, ys);
            if (ha <= 0 || area[l] / ha < 0.5) continue;

            // chamfer (3-4) distance transform inside the blob -> thickest point, always inside the shape
            int W2 = bw + 2, H2 = bh + 2;
            var dt = new float[W2 * H2];
            for (int y = 0; y < H2; y++)
                for (int x = 0; x < W2; x++)
                {
                    int gx = x0[l] + x - 1, gy = y0[l] + y - 1;
                    bool inside = gx >= 0 && gy >= 0 && gx < w && gy < h && labels[gy * w + gx] == l;
                    dt[y * W2 + x] = inside ? 1e9f : 0f;
                }
            for (int y = 1; y < H2 - 1; y++)
                for (int x = 1; x < W2 - 1; x++)
                {
                    int i = y * W2 + x;
                    if (dt[i] == 0) continue;
                    float v = dt[i];
                    v = Math.Min(v, dt[i - 1] + 3);
                    v = Math.Min(v, dt[i - W2] + 3);
                    v = Math.Min(v, dt[i - W2 - 1] + 4);
                    v = Math.Min(v, dt[i - W2 + 1] + 4);
                    dt[i] = v;
                }
            for (int y = H2 - 2; y >= 1; y--)
                for (int x = W2 - 2; x >= 1; x--)
                {
                    int i = y * W2 + x;
                    if (dt[i] == 0) continue;
                    float v = dt[i];
                    v = Math.Min(v, dt[i + 1] + 3);
                    v = Math.Min(v, dt[i + W2] + 3);
                    v = Math.Min(v, dt[i + W2 + 1] + 4);
                    v = Math.Min(v, dt[i + W2 - 1] + 4);
                    dt[i] = v;
                }
            int bi = 0; float bv = -1;
            for (int i = 0; i < W2 * H2; i++) if (dt[i] > bv) { bv = dt[i]; bi = i; }
            int pxx = bi % W2, pyy = bi / W2;

            double contrast = csum[l] / area[l];
            bool touches = x0[l] <= 1 || y0[l] <= 1 || x1[l] >= w - 2 || y1[l] >= h - 2;
            var r = new Region();
            r.cx = x0[l] + pxx - 1; r.cy = y0[l] + pyy - 1;
            r.bx = x0[l]; r.by = y0[l]; r.bw = bw; r.bh = bh;
            r.conf = (float)(Math.Min(1.0, contrast / 0.30) * (touches ? 0.7 : 1.0));
            regs.Add(r);
        }
        return regs;
    }

    /// <summary>Local maxima of score (square window radius r), strongest first, at most one per window.</summary>
    private static List<int> Peaks(float[] score, int w, int h, int r, float minScore, int maxN)
    {
        float[] dil = RankFilter(score, w, h, r, true);
        var idx = new List<int>();
        for (int i = 0; i < w * h; i++) if (score[i] >= minScore && score[i] >= dil[i]) idx.Add(i);
        idx.Sort((a, b) => score[b].CompareTo(score[a]));
        var o = new List<int>();
        foreach (int i in idx)
        {
            int x = i % w, y = i / w;
            bool near = false;
            foreach (int j in o)
            {
                int xj = j % w, yj = j / w;
                if (Math.Abs(xj - x) <= r && Math.Abs(yj - y) <= r) { near = true; break; }
            }
            if (!near) o.Add(i);
            if (o.Count >= maxN) break;
        }
        return o;
    }

    private static float[] Dog(float[] g, int w, int h, double s)
    {
        float[] a = Blur(g, w, h, s), b = Blur(g, w, h, s * 1.6);
        var o = new float[w * h];
        for (int i = 0; i < w * h; i++) o[i] = Math.Abs(a[i] - b[i]);
        return o;
    }

    private static float[] LedgeMap(float[] g, int w, int h, double preSigma, double postSigma)
    {
        float[] gs = Blur(g, w, h, preSigma);
        int d = Math.Max(2, (int)(0.012 * w));
        var l0 = new float[w * h];
        for (int y = 0; y < h - d; y++)
            for (int x = 0; x < w; x++)
            {
                float v = gs[y * w + x] - gs[(y + d) * w + x];   // lit top, shadow below
                l0[y * w + x] = v > 0 ? v : 0;
            }
        return Blur(l0, w, h, postSigma);
    }

    /// <summary>Indoor walls, colour-independent (port of detect.IndoorDetector).</summary>
    public static List<GCHold> DetectIndoor(byte[] rgb, int w0, int h0, int workWidth)
    {
        int w, h;
        ScaledSize(w0, h0, workWidth, out w, out h);
        float[] g = ToGray(rgb, w0, h0, w, h);
        const float minDist = 0.025f, minScore = 0.35f, margin = 0.02f;
        const int maxHolds = 120;

        float[] dFine = Dog(g, w, h, 0.006 * w), dMid = Dog(g, w, h, 0.012 * w), dBig = Dog(g, w, h, 0.022 * w);
        var mid = new float[w * h];
        for (int i = 0; i < w * h; i++) mid[i] = Math.Max(dMid[i], dBig[i]);
        float[] edges = SobelEdges(g, w, h, 40f / 255f, 120f / 255f);
        float[] dens = Blur(edges, w, h, 0.012 * w);
        float[] ledge = LedgeMap(g, w, h, 0.004 * w, 0.008 * w);

        float[] nMid = Normalize(mid, 0.04f, 99.5, null), nDens = Normalize(dens, 0.15f, 99.5, null);
        float[] nLed = Normalize(ledge, 0.04f, 99.5, null), nFine = Normalize(dFine, 0.04f, 99.5, null);
        var score = new float[w * h];
        for (int i = 0; i < w * h; i++)
        {
            float s = 0.40f * nMid[i] + 0.35f * nDens[i] + 0.25f * nLed[i];
            if (nFine[i] > 0.5f && nMid[i] < 0.5f * nFine[i]) s *= 0.3f;     // bolt / screw-hole like
            score[i] = s;
        }
        score = Blur(score, w, h, Math.Max(1.5, 0.004 * w));
        int m = (int)(margin * w);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                if (x < m || y < m || x >= w - m || y >= h - m) score[y * w + x] = 0;

        List<int> pk = Peaks(score, w, h, Math.Max(2, (int)(minDist * w)), minScore, maxHolds);
        List<Region> regs = Regions(g, w, h, 0.10f);
        int pad = (int)(0.012 * w);
        var cands = new List<Cand>();
        foreach (var r in regs) cands.Add(new Cand { x = r.cx, y = r.cy, conf = r.conf, src = "indoor_region" });
        foreach (int i in pk)
        {
            int x = i % w, y = i / w;
            bool inside = false;
            foreach (var r in regs)
                if (x >= r.bx - pad && x <= r.bx + r.bw + pad && y >= r.by - pad && y <= r.by + r.bh + pad) { inside = true; break; }
            if (!inside) cands.Add(new Cand { x = x, y = y, conf = Math.Min(1f, score[i]), src = "indoor_cv" });
        }
        return Finish(cands, w, h, maxHolds, true);
    }

    /// <summary>Outdoor rock (port of detect.ClassicalDetector): pockets (black-hat) + ledges inside the rock area.</summary>
    public static List<GCHold> DetectOutdoor(byte[] rgb0, int w0, int h0, int workWidth)
    {
        int w, h;
        ScaledSize(w0, h0, workWidth, out w, out h);
        float[] g = ToGray(rgb0, w0, h0, w, h);
        byte[] rgb = ToRgbSmall(rgb0, w0, h0, w, h);
        double sc = w / 800.0;

        var mask = new byte[w * h];                       // rock = not sky (blue) and not grass (green)
        for (int i = 0; i < w * h; i++)
        {
            double r = rgb[3 * i] / 255.0, gg = rgb[3 * i + 1] / 255.0, b = rgb[3 * i + 2] / 255.0;
            double mx = Math.Max(r, Math.Max(gg, b)), mn = Math.Min(r, Math.Min(gg, b)), dlt = mx - mn;
            double hue = 0;
            if (dlt > 1e-6)
            {
                if (mx == r) hue = 60 * (((gg - b) / dlt) % 6);
                else if (mx == gg) hue = 60 * ((b - r) / dlt + 2);
                else hue = 60 * ((r - gg) / dlt + 4);
            }
            if (hue < 0) hue += 360;
            double H = hue / 2, S = mx > 0 ? 255 * dlt / mx : 0;
            bool sky = H >= 90 && H <= 135 && S >= 40;
            bool grass = H >= 23 && H < 90 && S >= 70;
            mask[i] = (sky || grass) ? (byte)0 : (byte)1;
        }
        mask = OpenMask(mask, w, h, Math.Max(3, (int)Math.Round(7 * sc) | 1));
        mask = CloseMask(mask, w, h, Math.Max(3, (int)Math.Round(15 * sc) | 1));
        int n;
        int[] labels = Label(mask, w, h, out n);
        if (n > 1)
        {
            var cnt = new int[n + 1];
            for (int i = 0; i < w * h; i++) cnt[labels[i]]++;
            int best = 1;
            for (int l = 2; l <= n; l++) if (cnt[l] > cnt[best]) best = l;
            for (int i = 0; i < w * h; i++) mask[i] = labels[i] == best ? (byte)1 : (byte)0;
        }
        int er = (((int)(0.03 * w)) | 1) / 2;
        var mf = new float[w * h];
        for (int i = 0; i < w * h; i++) mf[i] = mask[i];
        float[] me = RankFilter(mf, w, h, er, false);
        int nIn = 0;
        for (int i = 0; i < w * h; i++) { mask[i] = me[i] > 0.5f ? (byte)1 : (byte)0; nIn += mask[i]; }
        if (nIn < 0.05 * w * h) return new List<GCHold>();     // no rock region found

        int kr = (((int)(0.04 * w)) | 1) / 2;
        float[] closed = RankFilter(RankFilter(g, w, h, kr, true), w, h, kr, false);
        var pocket = new float[w * h];
        for (int i = 0; i < w * h; i++) pocket[i] = closed[i] - g[i];
        float[] ledge = LedgeMap(g, w, h, Math.Max(1.5, 0.004 * w), Math.Max(2.0, 0.006 * w));
        float[] p = Normalize(pocket, 1e-6f, 99, mask), lv = Normalize(ledge, 1e-6f, 99, mask);
        var score = new float[w * h];
        for (int i = 0; i < w * h; i++) score[i] = 0.5f * p[i] + 0.5f * lv[i];
        score = Blur(score, w, h, Math.Max(1.5, 0.004 * w));
        for (int i = 0; i < w * h; i++) if (mask[i] == 0) score[i] = 0;

        List<int> pk = Peaks(score, w, h, Math.Max(2, (int)(0.03 * w)), 0.30f, 150);
        var cands = new List<Cand>();
        foreach (int i in pk)
        {
            string t = p[i] > 1.3f * lv[i] ? "pocket" : (lv[i] > 1.3f * p[i] ? "ledge" : "feature");
            cands.Add(new Cand { x = i % w, y = i / w, conf = Math.Min(1f, score[i]), src = t });
        }
        return Finish(cands, w, h, 150, false);
    }

    /// <summary>Candidates -> holds: strongest first, drop near-duplicates (same rule as Store.redetect), ids H1..</summary>
    private static List<GCHold> Finish(List<Cand> cands, int w, int h, int maxN, bool indoor)
    {
        cands.Sort((a, b) => b.conf.CompareTo(a.conf));
        var o = new List<GCHold>();
        foreach (var c in cands)
        {
            if (o.Count >= maxN) break;
            float x = (float)Math.Round(c.x / w, 5), y = (float)Math.Round(c.y / h, 5);
            bool dup = false;
            foreach (var q in o)
            {
                double dx = q.x - x, dy = q.y - y;
                if (Math.Sqrt(dx * dx + dy * dy) < 0.02) { dup = true; break; }
            }
            if (dup) continue;
            var hd = new GCHold();
            hd.id = "H" + (o.Count + 1);
            hd.x = x; hd.y = y;
            hd.confidence = (float)Math.Round(c.conf, 4);
            hd.type = indoor ? (c.conf >= 0.5f ? "candidate" : "uncertain") : c.src;
            hd.status = "detected";
            o.Add(hd);
        }
        return o;
    }

    // ================================================================== planning (port of planner.generate_routes)

    private struct Profile
    {
        public string name; public float maxReach, minReach, minRise, p, c0, lateral;
        public Profile(string n, float a, float b, float c, float d, float e, float f)
        { name = n; maxReach = a; minReach = b; minRise = c; p = d; c0 = e; lateral = f; }
    }

    private static readonly Profile[] Profiles =
    {
        new Profile("easy", 0.14f, 0.03f, 0.012f, 2.0f, 0.004f, 1.5f),
        new Profile("medium", 0.19f, 0.03f, 0.012f, 1.2f, 0.010f, 1.0f),
        new Profile("hard", 0.26f, 0.03f, 0.010f, 0.7f, 0.030f, 0.6f),
    };

    public static string ColorFor(string difficulty)
    {
        return difficulty == "easy" ? "#22C55E" : (difficulty == "medium" ? "#EAB308" : "#EF4444");
    }

    private static void Geom(GCHold a, GCHold b, float aspect, out double dx, out double dy, out double d)
    {
        dx = b.x - a.x;
        dy = (a.y - b.y) * aspect;       // positive = b is higher (image y is down)
        d = Math.Sqrt(dx * dx + dy * dy);
    }

    private static double EdgeCost(GCHold a, GCHold b, Profile pr, double maxReach, float aspect)
    {
        double dx, dy, d;
        Geom(a, b, aspect, out dx, out dy, out d);
        if (dy < pr.minRise || d > maxReach || d < pr.minReach) return -1;
        double quality = 0.5 + 0.5 * Math.Min(a.confidence, b.confidence);
        return (pr.c0 + Math.Pow(d, pr.p) + pr.lateral * Math.Abs(dx) * 0.5) / quality;
    }

    /// <summary>Binary min-heap of (cost, node, k).</summary>
    private class MinHeap
    {
        private readonly List<double> c = new List<double>();
        private readonly List<int> nd = new List<int>();
        private readonly List<int> kk = new List<int>();
        public int Count { get { return c.Count; } }

        public void Push(double cost, int node, int k)
        {
            c.Add(cost); nd.Add(node); kk.Add(k);
            int i = c.Count - 1;
            while (i > 0)
            {
                int p = (i - 1) / 2;
                if (c[p] <= c[i]) break;
                Swap(i, p); i = p;
            }
        }

        public void Pop(out double cost, out int node, out int k)
        {
            cost = c[0]; node = nd[0]; k = kk[0];
            int last = c.Count - 1;
            Swap(0, last);
            c.RemoveAt(last); nd.RemoveAt(last); kk.RemoveAt(last);
            int i = 0;
            while (true)
            {
                int l = 2 * i + 1, r = l + 1, m = i;
                if (l < c.Count && c[l] < c[m]) m = l;
                if (r < c.Count && c[r] < c[m]) m = r;
                if (m == i) break;
                Swap(i, m); i = m;
            }
        }

        private void Swap(int a, int b)
        {
            double t = c[a]; c[a] = c[b]; c[b] = t;
            int u = nd[a]; nd[a] = nd[b]; nd[b] = u;
            u = kk[a]; kk[a] = kk[b]; kk[b] = u;
        }
    }

    /// <summary>Dijkstra over states (hold, min(hold count, MinHoldsPerRoute)) so the minimum route length holds.</summary>
    private static List<int> Dijkstra(List<int>[] adjTo, List<double>[] adjW, int n, int start, bool[] isEnd, double[] penalty)
    {
        int cap = MinHoldsPerRoute, K = cap + 1, S = n * K;
        var dist = new double[S];
        var prev = new int[S];
        for (int i = 0; i < S; i++) { dist[i] = double.PositiveInfinity; prev[i] = -1; }
        var heap = new MinHeap();
        dist[start * K + 1] = 0;
        heap.Push(0, start, 1);
        int goal = -1;
        while (heap.Count > 0)
        {
            double c; int i, k;
            heap.Pop(out c, out i, out k);
            int st = i * K + k;
            if (c > dist[st]) continue;
            if (isEnd[i] && k >= cap) { goal = st; break; }
            for (int e = 0; e < adjTo[i].Count; e++)
            {
                int j = adjTo[i][e];
                int nk = Math.Min(k + 1, cap);
                int ns = j * K + nk;
                double nc = c + adjW[i][e] + penalty[j];
                if (nc < dist[ns]) { dist[ns] = nc; prev[ns] = st; heap.Push(nc, j, nk); }
            }
        }
        if (goal < 0) return null;
        var path = new List<int>();
        int s = goal;
        while (s >= 0) { path.Add(s / K); s = prev[s]; }
        path.Reverse();
        return path;
    }

    private static float Clip01(double v) { return (float)(v < 0 ? 0 : (v > 1 ? 1 : v)); }

    private static GCMetrics Estimate(List<GCHold> path, float aspect)
    {
        var reaches = new List<double>(); var dxs = new List<double>(); var dys = new List<double>();
        for (int i = 0; i + 1 < path.Count; i++)
        {
            double dx, dy, d;
            Geom(path[i], path[i + 1], aspect, out dx, out dy, out d);
            reaches.Add(d); dxs.Add(dx); dys.Add(dy);
        }
        int moves = Math.Max(1, reaches.Count);
        double sumR = 0, maxR = 0, rise = 0, lat = 0;
        for (int i = 0; i < reaches.Count; i++)
        {
            sumR += reaches[i];
            if (reaches[i] > maxR) maxR = reaches[i];
            rise += dys[i];
            lat += Math.Abs(dxs[i]);
        }
        double meanR = sumR / moves;
        double latRatio = rise > 1e-9 ? lat / rise : 0;
        var signs = new List<int>();
        foreach (double v in dxs) { if (v > 0.01) signs.Add(1); else if (v < -0.01) signs.Add(-1); }
        double dirChanges = 0;
        if (signs.Count > 1)
        {
            int ch = 0;
            for (int i = 0; i + 1 < signs.Count; i++) if (signs[i] != signs[i + 1]) ch++;
            dirChanges = ch / (double)Math.Max(1, signs.Count - 1);
        }
        double conf = 0;
        foreach (var hd in path) conf += hd.confidence;
        double meanConf = conf / path.Count;
        double score = 0.30 * Clip01(maxR / GlobalMaxReach) + 0.25 * Clip01(meanR / 0.20) + 0.20 * Clip01(latRatio / 1.0)
                     + 0.10 * dirChanges + 0.10 * (1 - meanConf) + 0.05 * Clip01(1 - path.Count / 12.0);
        var m = new GCMetrics();
        m.hold_count = path.Count;
        m.mean_reach = (float)Math.Round(meanR, 4);
        m.max_reach = (float)Math.Round(maxR, 4);
        m.lateral_ratio = (float)Math.Round(latRatio, 4);
        m.direction_changes = (float)Math.Round(dirChanges, 3);
        m.mean_confidence = (float)Math.Round(meanConf, 3);
        m.score = (float)Math.Round(score, 3);
        m.label = score < 0.40 ? "easy" : (score < 0.55 ? "medium" : "hard");
        m.feasibility = (float)Math.Round(Clip01(1 - 0.5 * Clip01(maxR / GlobalMaxReach) - 0.5 * (1 - meanConf)), 3);
        return m;
    }

    private static double Jaccard(List<int> a, List<int> b)
    {
        var sa = new HashSet<int>(a);
        var sb = new HashSet<int>(b);
        int inter = 0;
        foreach (int v in sa) if (sb.Contains(v)) inter++;
        int uni = sa.Count + sb.Count - inter;
        return uni == 0 ? 1.0 : inter / (double)uni;
    }

    /// <summary>Same route description the server produces (difficulty, steps, limb estimates).</summary>
    public static GCRoute Describe(string id, List<GCHold> path, float aspect)
    {
        GCMetrics m = Estimate(path, aspect);
        var r = new GCRoute();
        r.id = id;
        r.difficulty = m.label;
        r.color = ColorFor(m.label);
        r.estimated_score = m.score;
        r.feasibility = m.feasibility;
        r.metrics = m;
        r.edited = false;
        r.hold_ids = new List<string>();
        r.points = new List<GCPoint>();
        r.steps = new List<GCStep>();
        for (int k = 0; k < path.Count; k++)
        {
            GCHold hd = path[k];
            GCHold nb = k + 1 < path.Count ? path[k + 1] : path[k - 1];
            string limb = hd.x < nb.x - 0.01f ? "left_hand" : (hd.x > nb.x + 0.01f ? "right_hand" : (k % 2 == 0 ? "left_hand" : "right_hand"));
            r.hold_ids.Add(hd.id);
            var p = new GCPoint();
            p.hold_id = hd.id; p.x = hd.x; p.y = hd.y;
            p.role = k == 0 ? "start" : (k == path.Count - 1 ? "finish" : "hold");
            p.limb = limb; p.limb_status = "estimated";
            r.points.Add(p);
            var st = new GCStep();
            st.index = k; st.to = hd.id;
            if (k == 0) st.text = "Start on " + hd.id + ".";
            else
            {
                double dx, dy, d;
                Geom(path[k - 1], hd, aspect, out dx, out dy, out d);
                string side = Math.Abs(dx) < 0.02 ? "straight up" : (dx < 0 ? "up and to the left" : "up and to the right");
                string reach = d < 0.09 ? "short" : (d < 0.16 ? "medium" : "long");
                string verb = k == path.Count - 1 ? "Finish on" : "Move to";
                st.text = verb + " " + hd.id + " (" + side + ", " + reach + " reach, estimated).";
            }
            r.steps.Add(st);
        }
        return r;
    }

    public static List<GCRoute> GenerateRoutes(List<GCHold> holds, float aspect, int maxRoutes, double reachScale, List<string> warnings)
    {
        const double overlap = 0.6, startFrac = 0.30, targetFrac = 0.15;
        const int maxStart = 6, rounds = 3;
        var outRoutes = new List<GCRoute>();
        int n = holds.Count;
        if (n < MinHoldsPerRoute) { warnings.Add("insufficient holds (" + n + "); need at least " + MinHoldsPerRoute); return outRoutes; }
        float top = 1f, bottom = 0f;
        foreach (var hd in holds) { if (hd.y < top) top = hd.y; if (hd.y > bottom) bottom = hd.y; }
        float span = bottom - top;
        if (span < 0.2f) { warnings.Add("holds do not span enough of the wall height"); return outRoutes; }

        var starts = new List<int>();
        for (int i = 0; i < n; i++) if (holds[i].y >= bottom - startFrac * span) starts.Add(i);
        starts.Sort((a, b) => holds[a].x.CompareTo(holds[b].x));
        if (starts.Count > maxStart)
        {
            double step = (starts.Count - 1) / (double)(maxStart - 1);
            var sub = new List<int>();
            for (int k = 0; k < maxStart; k++) sub.Add(starts[(int)Math.Round(k * step, MidpointRounding.ToEven)]);
            starts = sub;
        }
        var isEnd = new bool[n];
        for (int i = 0; i < n; i++) isEnd[i] = holds[i].y <= top + targetFrac * span;
        double reachLimit = GlobalMaxReach * reachScale;

        var pool = new List<List<int>>();
        var seen = new HashSet<string>();
        foreach (var pr in Profiles)
        {
            double maxReach = pr.maxReach * reachScale;
            var adjTo = new List<int>[n];
            var adjW = new List<double>[n];
            for (int i = 0; i < n; i++)
            {
                adjTo[i] = new List<int>(); adjW[i] = new List<double>();
                for (int j = 0; j < n; j++)
                {
                    if (i == j) continue;
                    double c = EdgeCost(holds[i], holds[j], pr, maxReach, aspect);
                    if (c >= 0) { adjTo[i].Add(j); adjW[i].Add(c); }
                }
            }
            foreach (int s in starts)
            {
                var penalty = new double[n];
                for (int r = 0; r < rounds; r++)
                {
                    List<int> path = Dijkstra(adjTo, adjW, n, s, isEnd, penalty);
                    if (path == null) break;
                    string key = string.Join(",", path);
                    if (seen.Add(key)) pool.Add(path);
                    for (int t = 1; t + 1 < path.Count; t++) penalty[path[t]] += 0.12;
                }
            }
        }
        if (pool.Count == 0) { warnings.Add("no feasible route under the movement constraints"); return outRoutes; }

        var valid = new List<List<int>>();
        var est = new List<GCMetrics>();
        foreach (var p in pool)
        {
            double mx = 0;
            for (int t = 0; t + 1 < p.Count; t++)
            {
                double dx, dy, d;
                Geom(holds[p[t]], holds[p[t + 1]], aspect, out dx, out dy, out d);
                if (d > mx) mx = d;
            }
            if (mx > reachLimit + 1e-9) continue;          // safety net: no move beyond the global reach limit
            var hp = new List<GCHold>();
            foreach (int i in p) hp.Add(holds[i]);
            valid.Add(p);
            est.Add(Estimate(hp, aspect));
        }

        var chosen = new List<int>();
        var used = new bool[valid.Count];
        while (chosen.Count < maxRoutes)
        {
            int best = -1; double bestVal = -1;
            for (int c = 0; c < valid.Count; c++)
            {
                if (used[c]) continue;
                double ov = 0;
                foreach (int q in chosen) ov = Math.Max(ov, Jaccard(valid[c], valid[q]));
                if (ov >= overlap) continue;
                bool have = false;
                foreach (int q in chosen) if (est[q].label == est[c].label) have = true;
                double val = 0.5 * est[c].feasibility + 0.5 * (1 - ov) + (have ? 0 : 0.1);
                if (val > bestVal) { bestVal = val; best = c; }
            }
            if (best < 0) break;
            chosen.Add(best);
            used[best] = true;
        }
        for (int k = 0; k < chosen.Count; k++)
        {
            var hp = new List<GCHold>();
            foreach (int i in valid[chosen[k]]) hp.Add(holds[i]);
            outRoutes.Add(Describe("R" + (k + 1), hp, aspect));
        }
        if (outRoutes.Count < maxRoutes)
            warnings.Add("only " + outRoutes.Count + " distinct feasible route(s) found; none were invented");
        return outRoutes;
    }

    /// <summary>Like Store.generate_routes_for: on a close-up with few large holds, retry with a relaxed reach.</summary>
    public static List<GCRoute> PlanWithRelax(List<GCHold> holds, float aspect, int maxRoutes, List<string> warnings)
    {
        var w = new List<string>();
        List<GCRoute> routes = GenerateRoutes(holds, aspect, maxRoutes, 1.0, w);
        if (routes.Count == 0 && holds.Count >= MinHoldsPerRoute)
        {
            foreach (double s in new double[] { 1.5, 2.0 })
            {
                var w2 = new List<string>();
                List<GCRoute> r2 = GenerateRoutes(holds, aspect, maxRoutes, s, w2);
                if (r2.Count > 0)
                {
                    routes = r2; w = w2;
                    w.Add("holds are far apart: reach limit relaxed to " + s + "x; difficulty is relative to this photo");
                    break;
                }
            }
        }
        warnings.AddRange(w);
        return routes;
    }
}
