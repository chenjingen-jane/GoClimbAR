using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;
using Vuforia;

/// <summary>
/// Looks up the wall for the tracked Vuforia target via StreamingAssets/Routes/index.json,
/// loads StreamingAssets/Routes/{wall_id}.routes.json and checks it belongs to this target
/// (so one wall's routes can never be drawn on another wall). Falls back to the template routes.
/// </summary>
public class ImageTargetHandler : MonoBehaviour
{
    public string targetName = "dairy-farm-quarry-rgb";
    public RouteRenderer routeRenderer;

    private bool loading = false;
    private readonly Dictionary<string, GCRouteSet> cache = new Dictionary<string, GCRouteSet>();

    void Start()
    {
        if (routeRenderer == null)
            routeRenderer = FindObjectOfType<RouteRenderer>();

        var itb = GetComponent<ImageTargetBehaviour>();
        if (itb != null) itb.OnTargetStatusChanged += OnTargetStatusChanged;
    }

    void OnTargetStatusChanged(ObserverBehaviour behaviour, TargetStatus status)
    {
        if (status.Status == Status.TRACKED || status.Status == Status.EXTENDED_TRACKED)
        {
            // While the scanner owns the overlay (scan or loaded wall), registered targets stay quiet.
            if (WallScanner.SessionActive) return;
            if (!loading) StartCoroutine(LoadAndDraw());
        }
        else if (status.Status == Status.NO_POSE)
        {
            // Only clear if the routes on screen belong to THIS target.
            if (routeRenderer != null && routeRenderer.CurrentParent == this.transform)
                routeRenderer.ClearRoute();
        }
    }

    IEnumerator LoadAndDraw()
    {
        loading = true;
        string name = string.IsNullOrEmpty(targetName) ? gameObject.name : targetName;
        Debug.Log("Target detected: " + name);

        if (routeRenderer == null)
        {
            Debug.LogError("RouteRenderer is missing!");
            loading = false;
            yield break;
        }

        yield return new WaitForSeconds(0.2f);
        if (WallScanner.SessionActive) { loading = false; yield break; }

        // 1. target name -> wall_id
        string wallId = null;
        string indexJson = null;
        yield return ReadStreamingText("index.json", t => indexJson = t);
        if (!string.IsNullOrEmpty(indexJson))
        {
            try
            {
                var idx = JsonUtility.FromJson<GCIndex>(indexJson);
                if (idx != null && idx.targets != null)
                    foreach (var e in idx.targets)
                        if (e.target == name) { wallId = e.wall_id; break; }
            }
            catch (System.Exception ex) { Debug.LogError("index.json parse error: " + ex.Message); }
        }

        // 2. wall_id -> route file (cached), verified against the target
        GCRouteSet set = null;
        if (!string.IsNullOrEmpty(wallId) && !cache.TryGetValue(wallId, out set))
        {
            string json = null;
            yield return ReadStreamingText(wallId + ".routes.json", t => json = t);
            if (!string.IsNullOrEmpty(json))
            {
                try { set = JsonUtility.FromJson<GCRouteSet>(json); }
                catch (System.Exception ex) { Debug.LogError("Route JSON parse error: " + ex.Message); }
            }
            bool ok = set != null && set.schema == 2 && set.wall_id == wallId &&
                      (string.IsNullOrEmpty(set.vuforia_target) || set.vuforia_target == name);
            if (ok) cache[wallId] = set;
            else
            {
                if (set != null) Debug.LogError("Route file does not belong to target '" + name + "' (wall_id/target mismatch or wrong schema). Not drawing it.");
                set = null;
            }
        }

        if (set != null && set.routes != null && set.routes.Count > 0)
        {
            routeRenderer.DrawRouteSet(this.transform, set);
            Debug.Log("Drew " + set.routes.Count + " routes for wall '" + set.name + "' (" + set.wall_id + ").");
        }
        else
        {
            Debug.LogWarning("No saved routes for target '" + name + "'. Using template routes. " +
                             "Export the wall from the Python tool into StreamingAssets/Routes/.");
            routeRenderer.GenerateAndDrawLocalRoutes(this.transform);
        }
        loading = false;
    }

    IEnumerator ReadStreamingText(string fileName, System.Action<string> done)
    {
        string path = Path.Combine(Application.streamingAssetsPath, "Routes", fileName);
        string url = path.Contains("://") ? path : new System.Uri(path).AbsoluteUri;   // Windows + Android
        using (UnityWebRequest req = UnityWebRequest.Get(url))
        {
            yield return req.SendWebRequest();
            done(req.result == UnityWebRequest.Result.Success ? req.downloadHandler.text : null);
        }
    }
}
