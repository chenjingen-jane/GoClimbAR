using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Vuforia;

/// <summary>
/// Lets the GoClimb website open this app straight into the Vuforia AR camera.
///
///   goclimb://wall/&lt;wall_id&gt;?server=https://site/ar&amp;token=...  -> load that wall and show its routes in AR
///   goclimb://scan?server=https://site/ar&amp;token=...                 -> open the camera ready to scan a new wall
///
/// "server" tells the app which GoClimb website sent the link (no typing in Settings);
/// "token" signs the app in as the logged-in website user (needed to upload scans).
///
/// No scene setup needed: this creates itself when the app starts.
/// Android also needs the goclimb:// intent-filter in Assets/Plugins/Android/AndroidManifest.xml.
/// </summary>
public class DeepLinkHandler : MonoBehaviour
{
    private const string Scheme = "goclimb://";
    private const float MaxWaitForCamera = 20f;

    private WallScanner scanner;
    private string lastHandled;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void Boot()
    {
        if (FindObjectOfType<DeepLinkHandler>() != null) return;
        var go = new GameObject("GoClimbDeepLinks");
        DontDestroyOnLoad(go);
        go.AddComponent<DeepLinkHandler>();
    }

    private void Awake()
    {
        Application.deepLinkActivated += OnDeepLink;          // app already open (warm start)
    }

    private void Start()
    {
        if (!string.IsNullOrEmpty(Application.absoluteURL))   // app launched by the link (cold start)
            OnDeepLink(Application.absoluteURL);
    }

    private void OnDestroy()
    {
        Application.deepLinkActivated -= OnDeepLink;
    }

    private void OnDeepLink(string url)
    {
        if (string.IsNullOrEmpty(url) || url == lastHandled) return;
        lastHandled = url;
        Debug.Log("[DeepLink] " + url);
        StartCoroutine(Handle(url));
    }

    private IEnumerator Handle(string url)
    {
        // The wall's image target can only be created once the Vuforia camera is running.
        float giveUpAt = Time.realtimeSinceStartup + MaxWaitForCamera;
        while (Time.realtimeSinceStartup < giveUpAt && !VuforiaRunning()) yield return null;
        yield return null;                                     // let WallScanner finish its own Start()

        if (scanner == null) scanner = FindObjectOfType<WallScanner>();
        if (scanner == null) { Debug.LogError("[DeepLink] No WallScanner in the scene."); yield break; }

        if (!TryParse(url, out string action, out string id, out Dictionary<string, string> query))
        {
            Debug.LogWarning("[DeepLink] Not a GoClimb link: " + url);
            yield break;
        }
        query.TryGetValue("server", out string server);
        if (query.TryGetValue("token", out string token)) scanner.UseAuthToken(token);   // sign in as the website user

        if (action == "wall" && IsWallId(id)) scanner.OpenWallFromLink(id, server);
        else if (action == "scan") scanner.OpenScannerFromLink(server);
        else Debug.LogWarning("[DeepLink] Unknown GoClimb link: " + url);
    }

    private static bool VuforiaRunning()
    {
        try { return VuforiaApplication.Instance != null && VuforiaApplication.Instance.IsRunning; }
        catch (Exception) { return false; }
    }

    /// <summary>goclimb://wall/ID?a=b  ->  action "wall", id "ID", query {a: b}</summary>
    public static bool TryParse(string url, out string action, out string id, out Dictionary<string, string> query)
    {
        action = ""; id = ""; query = new Dictionary<string, string>();
        if (string.IsNullOrEmpty(url) || !url.StartsWith(Scheme, StringComparison.OrdinalIgnoreCase)) return false;

        string rest = url.Substring(Scheme.Length);
        int hash = rest.IndexOf('#');
        if (hash >= 0) rest = rest.Substring(0, hash);
        string path = rest, qs = "";
        int q = rest.IndexOf('?');
        if (q >= 0) { path = rest.Substring(0, q); qs = rest.Substring(q + 1); }

        string[] parts = path.Trim('/').Split(new[] { '/' }, StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length == 0) return false;
        action = parts[0].ToLowerInvariant();
        if (parts.Length > 1) id = Uri.UnescapeDataString(parts[1]);

        foreach (string pair in qs.Split(new[] { '&' }, StringSplitOptions.RemoveEmptyEntries))
        {
            int eq = pair.IndexOf('=');
            string k = Uri.UnescapeDataString((eq >= 0 ? pair.Substring(0, eq) : pair).Replace('+', ' '));
            string v = eq >= 0 ? Uri.UnescapeDataString(pair.Substring(eq + 1).Replace('+', ' ')) : "";
            query[k] = v;
        }
        return true;
    }

    private static bool IsWallId(string id)
    {
        // Saved walls use a 36-character id like 8efedb1b-34ec-445c-aa8c-65d0e6e1f36f
        if (string.IsNullOrEmpty(id) || id.Length != 36) return false;
        foreach (char c in id)
            if (!(c == '-' || Uri.IsHexDigit(c))) return false;
        return true;
    }
}
