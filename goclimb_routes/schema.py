"""Shared constants for the GoClimb data format (schema 2)."""
SCHEMA_VERSION = 2
API_VERSION = 4          # bump when the HTTP API changes. 4: upload with phone holds, discard scan, manual routes, review
ALGO_VERSION = "2.0"
COORD_SYSTEM = "normalized image: origin top-left, x right, y down, 0..1"
DISCLAIMER = ("Colours show ESTIMATED difficulty computed from image geometry. They are not official "
              "climbing grades and say nothing about safety. Limb suggestions are estimates until confirmed.")
DIFFICULTY_COLORS = {"easy": "#22C55E", "medium": "#EAB308", "hard": "#EF4444"}
DIFFICULTY_LABELS = {"easy": "Easy", "medium": "Medium", "hard": "Difficult"}
LIMBS = ("left_hand", "right_hand", "left_foot", "right_foot")
MODES = ("indoor", "outdoor")
HOLD_STATUSES = ("detected", "manual_added", "moved", "deleted")


def hold_label(n: int) -> str:
    return f"H{n}"
