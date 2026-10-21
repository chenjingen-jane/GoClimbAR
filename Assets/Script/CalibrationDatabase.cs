using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Holds all calibration profiles. The renderer looks up the profile
/// that matches the recognized Vuforia target name.
/// Create via: Create > GoClimb > Calibration Database.
/// </summary>
[CreateAssetMenu(fileName = "CalibrationDatabase", menuName = "GoClimb/Calibration Database")]
public class CalibrationDatabase : ScriptableObject
{
    public List<CalibrationProfile> profiles = new List<CalibrationProfile>();

    public CalibrationProfile Find(string targetName)
    {
        foreach (var p in profiles)
            if (p != null && p.targetName == targetName) return p;
        return null;
    }
}
