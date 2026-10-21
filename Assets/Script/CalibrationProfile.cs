using UnityEngine;

/// <summary>
/// Per-photo calibration. Create one asset per Image Target:
/// Right-click in Project > Create > GoClimb > Calibration Profile.
/// Values edited during Play mode are written to the asset, so they are saved.
/// </summary>
[CreateAssetMenu(fileName = "CalibrationProfile", menuName = "GoClimb/Calibration Profile")]
public class CalibrationProfile : ScriptableObject
{
    [Tooltip("Must equal the target name in the Vuforia database, e.g. dairy-farm-quarry-rgb")]
    public string targetName = "dairy-farm-quarry-rgb";

    [Header("Orientation")]
    [Range(0f, 360f)] public float rotationAngle = 0f;
    public bool flipVertical = false;
    public bool flipHorizontal = false;

    [Header("Placement (normalized, image spans -0.5..0.5)")]
    [Range(-0.5f, 0.5f)] public float offsetX = 0f;
    [Range(-0.5f, 0.5f)] public float offsetY = 0f;
    [Range(0.5f, 1.5f)] public float routeScale = 1f;

    [Header("Photo shape")]
    [Tooltip("Photo height divided by width. dairy-farm-quarry = 0.664. Landscape 4:3 = 0.75, portrait 3:4 = 1.333")]
    public float imageAspect = 0.664f;

    /// <summary>Changes whenever any value changes; used for live redraw.</summary>
    public int Signature()
    {
        return System.HashCode.Combine(rotationAngle, flipVertical, flipHorizontal,
                                       offsetX, offsetY, routeScale, imageAspect);
    }
}
