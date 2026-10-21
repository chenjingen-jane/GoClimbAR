using System;
using System.Collections.Generic;

// JSON models for schema 2 exported by goclimb_routes (store.export_unity).
// Coordinates: normalized image space, origin top-left, x right, y DOWN, range 0..1.
// Field names must match the JSON exactly (JsonUtility). Unknown JSON fields are ignored.

[Serializable]
public class GCPoint
{
    public string hold_id;
    public float x;
    public float y;
    public string role;         // "start", "hold", "finish"
    public string limb;         // left_hand / right_hand / left_foot / right_foot / ""
    public string limb_status;  // "estimated" or "confirmed"
}

[Serializable]
public class GCStep
{
    public int index;
    public string to;
    public string text;
}

[Serializable]
public class GCMetrics
{
    public int hold_count;
    public float mean_reach;
    public float max_reach;
    public float lateral_ratio;
    public float direction_changes;
    public float mean_confidence;
    public float score;
    public string label;
    public float feasibility;
}

[Serializable]
public class GCRoute
{
    public string id;
    public string difficulty;   // easy / medium / hard  (ESTIMATED)
    public string color;        // "#RRGGBB"
    public float estimated_score;
    public float feasibility;
    public List<string> hold_ids;
    public List<GCPoint> points;
    public List<GCStep> steps;
    public GCMetrics metrics;
    public bool edited;
}

[Serializable]
public class GCHold
{
    public string id;           // "H1", "H2", ...
    public float x;
    public float y;
    public float confidence;
    public string type;
    public string status;       // detected / manual_added / moved
}

[Serializable]
public class GCRouteSet
{
    public int schema;
    public string wall_id;      // "" until the wall is saved
    public string scan_id;      // identifies one camera scan ("" for stored walls)
    public string name;
    public string venue;
    public string mode;         // indoor / outdoor
    public string vuforia_target;
    public string coordinate_frame;   // "scan_photo" or "stored_photo": which photo the x,y values belong to
    public int image_width;     // size in pixels of that photo
    public int image_height;
    public float aspect;        // image height / width
    public string coordinate_system;
    public string algo_version;
    public string disclaimer;
    public string updated_at;
    public List<GCHold> holds;
    public List<GCRoute> routes;
    public List<string> warnings;
}

[Serializable]
public class GCIndexEntry
{
    public string target;
    public string wall_id;
}

[Serializable]
public class GCIndex
{
    public List<GCIndexEntry> targets;
}

// ---- scan / load-from-server models ----

[Serializable]
public class GCScanRequest
{
    public string image_b64;
    public string mode;         // "indoor" or "outdoor"
    public int max_routes;
}

[Serializable]
public class GCMatch
{
    public string wall_id;
    public string level;        // "confident", "possible" or "none"
    public int inliers;
    public float ratio;
}

[Serializable]
public class GCCandidate
{
    public string wall_id;
    public string name;
    public string venue;
    public int inliers;
    public float ratio;
}

[Serializable]
public class GCScanResponse
{
    public int api_version;         // the server's API version (Unity requires >= 3)
    public string scan_id;          // identifies THIS scan; used to reject stale replies
    public string status;           // "unknown", "known" or "uncertain"
    public string wall_id;          // set when status == "known"
    public string vuforia_target;   // runtime target name, "rt-" + first 8 chars of scan_id
    public int holds_detected;
    public GCMatch match;
    public List<GCCandidate> candidates;
    public GCRouteSet doc;
}

[Serializable]
public class GCSaveRequest
{
    public string venue;
    public string name;
    public string use_existing_wall_id;
    public bool allow_duplicate;
}

[Serializable]
public class GCSaveResponse
{
    public string wall_id;
    public bool created;
}

[Serializable]
public class GCRegenerateRequest
{
    public int max_routes;
    public bool refresh_only;
}

[Serializable]
public class GCDocResponse
{
    public string scan_id;
    public GCRouteSet doc;
}

[Serializable]
public class GCWallSummary
{
    public string wall_id;
    public string name;
    public string venue;
    public string mode;
    public bool photo_available;
}

[Serializable]
public class GCWallList
{
    public List<GCWallSummary> walls;
}

[Serializable]
public class GCVersion
{
    public int api_version;
}

[Serializable]
public class GCError
{
    public string error;
}

// ---- API 4: upload a wall from the phone (also used for walls scanned OFFLINE) ----

[Serializable]
public class GCHoldIn
{
    public float x;
    public float y;
    public float confidence;
    public string type;
}

[Serializable]
public class GCUploadRequest
{
    public string name;
    public string venue;
    public string mode;             // indoor / outdoor
    public string image_b64;        // JPEG of the scanned (cropped) photo
    public string source = "app";   // the admin sees app uploads in a "needs review" list
    public int max_routes = 6;
    public bool allow_duplicate;
    public List<GCHoldIn> holds;    // holds found on the phone; the server stores these instead of re-detecting
    public string queued_at;        // set when the upload waits for a connection (ignored by the server)
}

[Serializable]
public class GCUploadResponse
{
    public string wall_id;
    public bool created;
    public GCRouteSet doc;
}
