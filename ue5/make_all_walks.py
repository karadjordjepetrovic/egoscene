"""
Builds all 10 EgoScene walks from one config table and writes
Saved/egoscene_walks_manifest.json.

Run from the Unreal Editor Output Log:

    py "<UE project>\\Scripts\\make_all_walks.py"

walk_01 is left untouched (its sequence and poses json already exist from
Phase 1) -- it is only read back and recorded in the manifest. walk_02..10
are (re)built via make_walk.build_walk(), which also runs the per-walk
collision check and prints "OK" or the colliding frame pairs to the Output Log.
"""
import unreal, json, os, sys, importlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import make_walk
importlib.reload(make_walk)   # pick up edits without restarting the editor

LEVEL = "Apartment_01"

# walk_id, speed_cm_s, lighting, seed
# Presets: day >=2, overcast >=2, evening >=2, night >=2 (day/overcast appear 3x).
# Speeds vary ~80/110/140. walk_01 kept as-is (day, 110) per the task.
CONFIG = [
    ("walk_02", 80,  "overcast", 2),
    ("walk_03", 140, "evening",  3),
    ("walk_04", 80,  "night",    4),
    ("walk_05", 140, "day",      5),
    ("walk_06", 110, "overcast", 6),
    ("walk_07", 80,  "evening",  7),
    ("walk_08", 140, "night",    8),
    ("walk_09", 110, "day",      9),
    ("walk_10", 80,  "overcast", 10),
]

manifest = []

# --- walk_01: record only, never rebuilt ---
walk01_json = unreal.Paths.project_saved_dir() + "egoscene_walk_01_poses_ue.json"
if os.path.exists(walk01_json):
    with open(walk01_json, "r") as fh:
        data = json.load(fh)
    num_frames = len(data["frames"])
else:
    unreal.log_warning(f"walk_01: poses json not found at {walk01_json}; recording num_frames=0")
    num_frames = 0
manifest.append({"walk_id": "walk_01", "level": LEVEL, "lighting": "day",
                  "speed_cm_s": 110, "seed": 1, "num_frames": num_frames})

# --- walk_02 .. walk_10: build + collision-check + record ---
for walk_id, speed_cm_s, lighting, seed in CONFIG:
    num_frames, collisions = make_walk.build_walk(walk_id, speed_cm_s, lighting, seed)
    manifest.append({"walk_id": walk_id, "level": LEVEL, "lighting": lighting,
                      "speed_cm_s": speed_cm_s, "seed": seed, "num_frames": num_frames,
                      "collision_frames": collisions})

out = unreal.Paths.project_saved_dir() + "egoscene_walks_manifest.json"
with open(out, "w") as fh:
    json.dump(manifest, fh, indent=2)

unreal.log(f"Wrote manifest for {len(manifest)} walks -> {out}")
for entry in manifest:
    status = "OK" if not entry.get("collision_frames") else f"COLLISIONS: {entry['collision_frames']}"
    unreal.log(f"  {entry['walk_id']:8s} {entry['lighting']:9s} {entry['speed_cm_s']:3d}cm/s "
               f"seed={entry['seed']:<3d} frames={entry['num_frames']:4d}  {status}")
