"""
Builds one EgoScene walking sequence from Target Points already placed in the level.

Run from the Unreal Editor Output Log (Python must be able to see the `unreal`
module, i.e. run it *inside* the editor, not as an external interpreter):

    py "<UE project>\\Scripts\\make_walk.py" walk_03 140 evening 3

Positional args: WALK_ID SPEED_CM_S LIGHTING [SEED]
    WALK_ID      e.g. walk_03 -- Target Points named "<WALK_ID>_NN" must already
                 exist in the level (place them by hand or via the MCP tools).
    SPEED_CM_S   walking speed in cm/s, e.g. 80 / 110 / 140
    LIGHTING     one of: day, overcast, evening, night. Only validated and
                 recorded here (poses json + manifest) -- it no longer keys
                 anything in the sequence. A runtime Sequencer key on the HDRI
                 backdrop and PostProcessVolume exposure looked correct in the
                 editor viewport but was silently ignored during an actual MRQ
                 render, so each preset is instead baked into its own map by
                 Scripts/make_lighting_maps.py, and make_render_queue.py picks
                 the matching map per job from the manifest's "lighting" field.
    SEED         optional int (default 0). Reproducibly varies the head-glance
                 phase (yaw/pitch/roll wobble) so different walks don't all
                 bob their head in lockstep. Does NOT touch the eye-height
                 formula, which is identical to the original walk_01 script.

Overwrites (does not append to) `/Game/EgoScene/Sequences/SEQ_<WALK_ID>` and
`Saved/egoscene_<WALK_ID>_poses_ue.json`. The built sequence contains only the
GlassesCam transform + camera cut track -- no lighting/exposure tracks.

Also importable: `build_walk(walk_id, speed_cm_s, lighting, seed=0)` is used by
make_all_walks.py to build every walk from one config table, and
LIGHTING_PRESETS / the baseline helpers are reused by make_lighting_maps.py.
"""
import unreal, json, math, sys, random

FPS = 30
EYE_CM = 160.0
CAM_LABEL = "GlassesCam"
FLOOR_Z_CM = 0.0                            # floor surface height in the level
SEQ_DIR = "/Game/EgoScene/Sequences"
BASELINE_PATH = unreal.Paths.project_saved_dir() + "egoscene_lighting_baseline.json"

# Actors that stand in for "lamps" / practical lights for the lighting presets.
# (Everything here was confirmed to exist in the level before writing this script.)
LAMP_ACTOR_LABELS = ["RectLight_Living_01", "RectLight_Living_02", "RectLight_Kitchen_01"]
HALLWAY_LIGHT_LABEL = "Hallway_Light_01"
SUN_ACTOR_LABEL = "DirectionalLight_0"
HDRI_ACTOR_LABEL = "HDRIBackdrop_C_0"       # keyed via its Skylight component + the
                                             # backdrop mesh's material (NOT the BP
                                             # "intensity" var -- see build_walk() below)
EXTRA_SKYLIGHT_LABEL = "SkyLight_0"         # a second, separate skylight also active in the level
# NOTE: the actual actor label is "PostProcessVolume_01" (trailing "01"), not "PostProcessVolume_0" --
# that mismatch is why exposure keys were silently skipped before.
EXPOSURE_VOLUME_LABEL = "PostProcessVolume_01"
LAMP_WARM_TEMPERATURE = 3200                # evening/night lamp colour temperature (K)


def _load_baseline():
    """Loads Saved/egoscene_lighting_baseline.json -- the actual current-level
    values recorded for the 'day' look. LIGHTING_PRESETS below are multipliers
    of these, not absolute guesses. Regenerate that file (see PHASE2_UE_TASKS.md
    Task 4 fix changelog) if the level's base lighting ever changes."""
    with open(BASELINE_PATH, "r") as fh:
        return json.load(fh)


def _sky_baseline(baseline, actor_label):
    return next(e for e in baseline["sky"] if e["actor_label"] == actor_label)["intensity"]


def _hdri_baseline(baseline):
    return next(e for e in baseline["sky"] if e["actor_label"] == HDRI_ACTOR_LABEL)


# Lighting presets, keyed as CONSTANT values across a whole walk's sequence.
# sun_mul / sky_mul / lamp_mul are multipliers of the baseline ("day") values in
# Saved/egoscene_lighting_baseline.json -- NOT absolute light-unit guesses, since
# the directional light (lux), RectLights (Lumens) and the HDRI backdrop (cd/m^2,
# via its Skylight component + backdrop material, not its BP "intensity" var)
# all use different, non-comparable units. sun_pitch/sun_temp aren't multiplicative
# so those stay as explicit per-preset overrides. exposure_bias is EV100
# compensation on PostProcessVolume_01 (where auto-exposure is actually disabled,
# AEM_Manual, for this project) -- all 4 values below were tuned by eye, screenshot
# by screenshot, from the middle of the living room; see the Task 4 fix changelog.
LIGHTING_PRESETS = {
    "day":      dict(sun_mul=1.0, sun_pitch=-40, sun_temp=6500, sky_mul=1.00, lamp_mul=1.0, lamp_warm=False, exposure_bias=3.0),
    "overcast": dict(sun_mul=0.05, sun_pitch=-40, sun_temp=7500, sky_mul=0.50, lamp_mul=1.0, lamp_warm=False, exposure_bias=3.3),
    "evening":  dict(sun_mul=0.1, sun_pitch=-8,  sun_temp=2800, sky_mul=0.25, lamp_mul=1.5, lamp_warm=True,  exposure_bias=4.0),
    "night":    dict(sun_mul=0.0, sun_pitch=-8,  sun_temp=2800, sky_mul=0.03, lamp_mul=1.0, lamp_warm=True,  exposure_bias=3.0),
}


def _catmull(p0, p1, p2, p3, t):
    return [0.5 * ((2 * b) + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t * t + (-a + 3 * b - 3 * c + d) * t * t * t)
            for a, b, c, d in zip(p0, p1, p2, p3)]


def _build_frames(walk_id, speed_cm_s, seed):
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = eas.get_all_level_actors()
    cam = next((a for a in actors if a.get_actor_label() == CAM_LABEL), None)
    if cam is None:
        raise RuntimeError(f"No actor labelled '{CAM_LABEL}' found in the level.")
    wps = sorted((a for a in actors if a.get_actor_label().startswith(walk_id + "_")),
                 key=lambda a: a.get_actor_label())
    if len(wps) < 2:
        raise RuntimeError(f"Need at least 2 Target Points named '{walk_id}_NN'; found {len(wps)}.")
    pts = [a.get_actor_location() for a in wps]

    P = [[p.x, p.y, p.z] for p in pts]
    P = [P[0]] + P + [P[-1]]
    dense = [_catmull(P[i - 1], P[i], P[i + 1], P[i + 2], s / 50)
             for i in range(1, len(P) - 2) for s in range(50)]

    step = speed_cm_s / FPS
    samples, carry = [dense[0]], 0.0
    for a, b in zip(dense, dense[1:]):
        seg = math.dist(a, b)
        while carry + seg >= step:
            t = (step - carry) / seg
            a = [a[k] + (b[k] - a[k]) * t for k in range(3)]
            seg = math.dist(a, b)
            carry = 0.0
            samples.append(a)
        carry += seg

    # Seed-driven phase offsets for the head-glance wobble only. The eye-height
    # formula (zc below) is untouched -- same amplitudes/terms as the original script.
    rng = random.Random(seed)
    ph_yaw1 = rng.uniform(0, 2 * math.pi)
    ph_yaw2 = rng.uniform(0, 2 * math.pi)
    ph_pitch = rng.uniform(0, 2 * math.pi)
    ph_roll = rng.uniform(0, 2 * math.pi)

    frames = []
    for i, (x, y, z) in enumerate(samples):
        t = i / FPS
        nx, ny = samples[min(i + 5, len(samples) - 1)][:2]
        heading = math.degrees(math.atan2(ny - y, nx - x)) if (nx, ny) != (x, y) else frames[-1]["rot"][2]
        yaw = heading + 18 * math.sin(2 * math.pi * 0.11 * t + ph_yaw1) + 7 * math.sin(2 * math.pi * 0.29 * t + ph_yaw2)
        pitch = -6 + 4 * math.sin(2 * math.pi * 0.17 * t + ph_pitch)
        roll = 1.0 * math.sin(2 * math.pi * 0.9 * t + ph_roll)
        zc = FLOOR_Z_CM + EYE_CM + 1.8 * math.sin(2 * math.pi * 1.8 * t)   # eye-height logic, unchanged
        r = unreal.Rotator(roll=roll, pitch=pitch, yaw=yaw)
        v = lambda vec: [vec.x, vec.y, vec.z]
        frames.append({"frame": i, "loc_cm": [x, y, zc], "rot": [roll, pitch, yaw],
                        "forward": v(r.get_forward_vector()), "right": v(r.get_right_vector()),
                        "up": v(r.get_up_vector())})
    return cam, frames


def _check_collisions(cam, frames):
    """Line-traces between consecutive camera positions. Returns a list of
    (frame_a, frame_b) pairs where the segment hits level geometry."""
    hits = []
    for f0, f1 in zip(frames, frames[1:]):
        start = unreal.Vector(*f0["loc_cm"])
        end = unreal.Vector(*f1["loc_cm"])
        # UE Python returns the HitResult on a hit, or None on no hit (not a tuple).
        hit_result = unreal.SystemLibrary.line_trace_single(
            cam, start, end,
            getattr(unreal.TraceTypeQuery, "ECC_VISIBILITY", unreal.TraceTypeQuery.TRACE_TYPE_QUERY1),
            False, [cam], unreal.DrawDebugTrace.NONE, True)
        if hit_result is not None:
            hits.append((f0["frame"], f1["frame"]))
    return hits


def _find_component(actors, actor_label, component_class):
    actor = next((a for a in actors if a.get_actor_label() == actor_label), None)
    if actor is None:
        return None, None
    comps = actor.get_components_by_class(component_class)
    return actor, (comps[0] if comps else None)


def build_walk(walk_id, speed_cm_s, lighting, seed=0):
    if lighting not in LIGHTING_PRESETS:
        raise ValueError(f"Unknown lighting preset '{lighting}'. Choose from: {list(LIGHTING_PRESETS)}")

    cam, frames = _build_frames(walk_id, speed_cm_s, seed)
    num_frames = len(frames)

    # --- collision check (done on the final resampled path, at eye height) ---
    collisions = _check_collisions(cam, frames)
    if collisions:
        unreal.log_warning(f"{walk_id}: COLLISION at {len(collisions)} frame-pair(s): {collisions}")
    else:
        unreal.log(f"{walk_id}: OK ({num_frames} frames, no collisions)")

    # --- (re)create the sequence asset ---
    seq_path = f"{SEQ_DIR}/SEQ_{walk_id}"
    if unreal.EditorAssetLibrary.does_asset_exist(seq_path):
        unreal.EditorAssetLibrary.delete_asset(seq_path)
    seq = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        f"SEQ_{walk_id}", SEQ_DIR, unreal.LevelSequence, unreal.LevelSequenceFactoryNew())
    seq.set_display_rate(unreal.FrameRate(FPS, 1))
    seq.set_tick_resolution(unreal.FrameRate(FPS, 1))
    seq.set_playback_start(0)
    seq.set_playback_end(num_frames)

    # camera transform + cut track -- the ONLY tracks in this sequence. Lighting is
    # no longer keyed here at all: a runtime Sequencer key on the HDRI backdrop and
    # PostProcessVolume exposure looked right in the editor but was silently ignored
    # during an actual MRQ render (see PHASE2_UE_TASKS.md Task 4 fix #3), so each
    # preset is now baked into its own map by Scripts/make_lighting_maps.py and
    # selected per render job by make_render_queue.py via the manifest's "lighting"
    # field instead.
    #
    # The camera is a SPAWNABLE (a copy of GlassesCam owned by the sequence), not a
    # possessable. A possessable binds to the actor's path inside Apartment_01, and
    # that binding does not resolve in the baked copies (Apartment_01_<lighting>),
    # so MRQ silently rendered from a static default camera. A spawnable is created
    # by the sequence itself in whatever map is rendering, with GlassesCam's settings.
    if not hasattr(seq, "add_spawnable_from_instance"):
        raise RuntimeError("This UE version has no LevelSequence.add_spawnable_from_instance()")
    cam_binding = seq.add_spawnable_from_instance(cam)
    cam_binding.set_display_name(CAM_LABEL)
    section = cam_binding.add_track(unreal.MovieScene3DTransformTrack).add_section()
    section.set_range(0, num_frames)
    ch = section.get_all_channels()
    for f in frames:
        n = unreal.FrameNumber(f["frame"])
        for k in range(3):
            ch[k].add_key(n, f["loc_cm"][k])
            ch[3 + k].add_key(n, f["rot"][k])
    cut = seq.add_track(unreal.MovieSceneCameraCutTrack).add_section()
    cut.set_range(0, num_frames)
    cut.set_camera_binding_id(seq.get_binding_id(cam_binding))

    unreal.EditorAssetLibrary.save_loaded_asset(seq)

    # --- poses json (same schema as before; "lighting" is recorded here and in the
    # manifest, but no longer drives anything in this sequence -- it's resolved to a
    # baked map (/Game/EgoScene/Maps/Lighting/Apartment_01_<lighting>) by
    # make_render_queue.py instead). ---
    out = unreal.Paths.project_saved_dir() + f"egoscene_{walk_id}_poses_ue.json"
    with open(out, "w") as fh:
        json.dump({"fps": FPS, "units": "cm, degrees, UE coordinates",
                    "lighting": lighting, "speed_cm_s": speed_cm_s, "seed": seed,
                    "frames": frames}, fh)
    unreal.log(f"{walk_id}: {num_frames} frames ({num_frames/FPS:.1f}s) -> {out}")

    return num_frames, collisions


if __name__ == "__main__":
    argv = sys.argv
    # When run via `py <script> a b c d`, UE's Python plugin passes the script's
    # own args starting at argv[0] == the script path.
    args = argv[1:] if argv and argv[0].lower().endswith(".py") else argv
    if len(args) < 3:
        raise SystemExit("Usage: py make_walk.py <walk_id> <speed_cm_s> <lighting> [seed]")
    walk_id = args[0]
    speed_cm_s = float(args[1])
    lighting = args[2]
    seed = int(args[3]) if len(args) > 3 else 0
    build_walk(walk_id, speed_cm_s, lighting, seed)
