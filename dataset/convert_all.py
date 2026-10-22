"""Convert every walk in the UE project's walks manifest into the dataset format.

Usage (from C:\\dev\\egoscene, venv active):
    python dataset/convert_all.py
    python dataset/convert_all.py walk_03 walk_04      # only some walks

Reads <UE project>/Saved/egoscene_walks_manifest.json, copies the current labels JSON
into dataset/labels/, converts each walk to data/<level>/<walk_id>/, and writes
data/manifest.json with per-walk metadata and frame counts.
"""
import os, sys, json, shutil, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_walk import convert_walk

# Path to the UE5 project. Set EGOSCENE_UE_PROJECT, or edit the default below.
UE_PROJECT = os.environ.get("EGOSCENE_UE_PROJECT",
                            os.path.join(os.path.expanduser("~"), "Documents", "Unreal Projects", "EgoScene"))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ue_saved = os.path.join(UE_PROJECT, "Saved")
manifest = json.load(open(os.path.join(ue_saved, "egoscene_walks_manifest.json")))
only = set(sys.argv[1:])

labels_path = os.path.join(REPO, "dataset", "labels", "apartment_01_labels.json")
shutil.copy(os.path.join(ue_saved, "egoscene_labels.json"), labels_path)

summary = []
t0 = time.time()
for e in manifest:
    walk_id, level = e["walk_id"], e["level"]
    if only and walk_id not in only:
        continue
    print(f"== {walk_id} ({level}, {e['lighting']}, {e['speed_cm_s']} cm/s)", flush=True)
    out_dir = os.path.join(REPO, "data", level.lower(), walk_id)
    n, unknown = convert_walk(
        render_dir=os.path.join(ue_saved, "EgoRenders", f"SEQ_{walk_id}"),
        poses_path=os.path.join(ue_saved, f"egoscene_{walk_id}_poses_ue.json"),
        labels_path=labels_path,
        out_dir=out_dir,
        extra_meta={"walk_id": walk_id, "level": level},
    )
    if n != e["num_frames"]:
        print(f"  WARNING: manifest says {e['num_frames']} frames, converted {n}")
    summary.append({**{k: e[k] for k in ("walk_id", "level", "lighting", "speed_cm_s", "seed")},
                    "num_frames": n, "path": os.path.relpath(out_dir, os.path.join(REPO, "data")).replace("\\", "/"),
                    "unknown_instance_ids": unknown})

if not only:
    json.dump(summary, open(os.path.join(REPO, "data", "manifest.json"), "w"), indent=1)
total = sum(s["num_frames"] for s in summary)
print(f"\nDone: {len(summary)} walks, {total} frames in {time.time() - t0:.0f}s")
