using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

/// <summary>
/// Files kept on the phone (Application.persistentDataPath/goclimb):
///   pending/&lt;id&gt;.json : uploads the user asked for while there was no connection; sent automatically later
///   cache/&lt;wall_id&gt;.json + .jpg : saved walls downloaded once, so they can be opened again offline
/// Nothing is written here unless the user uploads a wall or opens a saved one.
/// </summary>
public static class LocalWallStore
{
    private static string Root { get { return Path.Combine(Application.persistentDataPath, "goclimb"); } }
    private static string PendingDir { get { return Path.Combine(Root, "pending"); } }
    private static string CacheDir { get { return Path.Combine(Root, "cache"); } }

    // ------------------------------------------------------------ pending uploads

    public static void QueueUpload(GCUploadRequest req)
    {
        Directory.CreateDirectory(PendingDir);
        req.queued_at = DateTime.UtcNow.ToString("o");
        string path = Path.Combine(PendingDir, DateTime.UtcNow.ToString("yyyyMMddHHmmss") + "-" + Guid.NewGuid().ToString("N").Substring(0, 6) + ".json");
        File.WriteAllText(path, JsonUtility.ToJson(req));
    }

    public static List<string> PendingFiles()
    {
        var list = new List<string>();
        if (Directory.Exists(PendingDir)) list.AddRange(Directory.GetFiles(PendingDir, "*.json"));
        list.Sort();
        return list;
    }

    public static GCUploadRequest ReadPending(string path)
    {
        try { return JsonUtility.FromJson<GCUploadRequest>(File.ReadAllText(path)); }
        catch (Exception e) { Debug.LogWarning("[LocalWallStore] unreadable pending upload " + path + ": " + e.Message); return null; }
    }

    public static void RemovePending(string path)
    {
        try { File.Delete(path); } catch (Exception) { }
    }

    // ------------------------------------------------------------ cached saved walls

    public static void CacheWall(GCRouteSet doc, byte[] jpg)
    {
        if (doc == null || string.IsNullOrEmpty(doc.wall_id)) return;
        try
        {
            Directory.CreateDirectory(CacheDir);
            File.WriteAllText(Path.Combine(CacheDir, doc.wall_id + ".json"), JsonUtility.ToJson(doc));
            if (jpg != null) File.WriteAllBytes(Path.Combine(CacheDir, doc.wall_id + ".jpg"), jpg);
        }
        catch (Exception e) { Debug.LogWarning("[LocalWallStore] could not cache wall: " + e.Message); }
    }

    public static List<GCRouteSet> CachedWalls()
    {
        var list = new List<GCRouteSet>();
        if (!Directory.Exists(CacheDir)) return list;
        foreach (string f in Directory.GetFiles(CacheDir, "*.json"))
        {
            try
            {
                var d = JsonUtility.FromJson<GCRouteSet>(File.ReadAllText(f));
                if (d != null && d.schema == 2 && !string.IsNullOrEmpty(d.wall_id) &&
                    File.Exists(Path.Combine(CacheDir, d.wall_id + ".jpg"))) list.Add(d);
            }
            catch (Exception) { }
        }
        list.Sort((a, b) => string.Compare(a.venue + a.name, b.venue + b.name, StringComparison.OrdinalIgnoreCase));
        return list;
    }

    public static byte[] CachedPhoto(string wallId)
    {
        string p = Path.Combine(CacheDir, wallId + ".jpg");
        return File.Exists(p) ? File.ReadAllBytes(p) : null;
    }
}
