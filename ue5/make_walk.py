import unreal, json, math

WALK, FPS = "walk_01", 30
SPEED_CM_S, EYE_CM = 110.0, 160.0          # slow indoor walk, eye height
CAM_LABEL = "GlassesCam"
FLOOR_Z_CM = 0.0                            # floor surface height in the level

eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
actors = eas.get_all_level_actors()
cam = next(a for a in actors if a.get_actor_label() == CAM_LABEL)
wps = sorted((a for a in actors if a.get_actor_label().startswith(WALK + "_")),
             key=lambda a: a.get_actor_label())
pts = [a.get_actor_location() for a in wps]

def catmull(p0, p1, p2, p3, t):            # smooth curve through the waypoints
    return [0.5 * ((2*b) + (-a + c)*t + (2*a - 5*b + 4*c - d)*t*t + (-a + 3*b - 3*c + d)*t*t*t)
            for a, b, c, d in zip(p0, p1, p2, p3)]

P = [[p.x, p.y, p.z] for p in pts]
P = [P[0]] + P + [P[-1]]
dense = [catmull(P[i-1], P[i], P[i+1], P[i+2], s / 50)
         for i in range(1, len(P) - 2) for s in range(50)]

# resample at constant walking speed
step = SPEED_CM_S / FPS
samples, carry = [dense[0]], 0.0
for a, b in zip(dense, dense[1:]):
    seg = math.dist(a, b)
    while carry + seg >= step:
        t = (step - carry) / seg
        a = [a[k] + (b[k] - a[k]) * t for k in range(3)]
        seg = math.dist(a, b); carry = 0.0
        samples.append(a)
    carry += seg

frames = []
for i, (x, y, z) in enumerate(samples):
    t = i / FPS
    nx, ny = samples[min(i + 5, len(samples) - 1)][:2]
    heading = math.degrees(math.atan2(ny - y, nx - x)) if (nx, ny) != (x, y) else frames[-1]["rot"][2]
    yaw = heading + 18 * math.sin(2 * math.pi * 0.11 * t) + 7 * math.sin(2 * math.pi * 0.29 * t)   # glances
    pitch = -6 + 4 * math.sin(2 * math.pi * 0.17 * t)                                               # look down/up
    roll = 1.0 * math.sin(2 * math.pi * 0.9 * t)
    zc = FLOOR_Z_CM + EYE_CM + 1.8 * math.sin(2 * math.pi * 1.8 * t)
    r = unreal.Rotator(roll=roll, pitch=pitch, yaw=yaw)
    v = lambda vec: [vec.x, vec.y, vec.z]
    frames.append({"frame": i, "loc_cm": [x, y, zc], "rot": [roll, pitch, yaw],
                   "forward": v(r.get_forward_vector()), "right": v(r.get_right_vector()),
                   "up": v(r.get_up_vector())})

# Level Sequence with one transform key per frame
seq = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
    f"SEQ_{WALK}", "/Game/EgoScene/Sequences", unreal.LevelSequence, unreal.LevelSequenceFactoryNew())
seq.set_display_rate(unreal.FrameRate(FPS, 1)); seq.set_tick_resolution(unreal.FrameRate(FPS, 1))
seq.set_playback_start(0); seq.set_playback_end(len(frames))
binding = seq.add_possessable(cam)
section = binding.add_track(unreal.MovieScene3DTransformTrack).add_section()
section.set_range(0, len(frames))
ch = section.get_all_channels()            # loc X,Y,Z, rot X(roll),Y(pitch),Z(yaw), scale X,Y,Z
for f in frames:
    n = unreal.FrameNumber(f["frame"])
    for k in range(3):
        ch[k].add_key(n, f["loc_cm"][k])
        ch[3 + k].add_key(n, f["rot"][k])
cut = seq.add_track(unreal.MovieSceneCameraCutTrack).add_section()
cut.set_range(0, len(frames))
cut.set_camera_binding_id(seq.get_binding_id(binding))
unreal.EditorAssetLibrary.save_loaded_asset(seq)

out = unreal.Paths.project_saved_dir() + f"egoscene_{WALK}_poses_ue.json"
with open(out, "w") as fh:
    json.dump({"fps": FPS, "units": "cm, degrees, UE coordinates", "frames": frames}, fh)
unreal.log(f"{WALK}: {len(frames)} frames ({len(frames)/FPS:.1f}s) -> {out}")