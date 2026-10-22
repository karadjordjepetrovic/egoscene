"""
Builds the Movie Render Queue for the EgoScene walks.

Full build -- one job per walk in the manifest (all 10, including walk_01),
run from the Unreal Editor Output Log:

    py "<UE project>\\Scripts\\make_render_queue.py"

Reads Saved/egoscene_walks_manifest.json and builds one job per walk against
/Game/EgoScene/Sequences/SEQ_<walk_id>, map
/Game/EgoScene/Maps/Lighting/Apartment_01_<lighting> (the baked-lighting
duplicate matching that walk's manifest "lighting" entry -- build these first
with Scripts/make_lighting_maps.py; SEQ_<walk_id> itself carries no lighting,
see make_walk.py). Each job's own Configuration is seeded from the shared preset
/Game/EgoScene/Render/EgoScene_GT_Config via set_configuration() -- this
duplicates the preset's settings onto the job, it does NOT modify the shared
preset asset itself (ground rule: never touch EgoScene_GT_Config). The job's
own Output Setting is then overridden with a per-walk output directory,
"{render_pass}/{frame_number}" file name format (same as walk_01's existing
EgoScene_GT queue), and "Override existing output" turned on.

Saves the queue as /Game/EgoScene/Render/EgoScene_AllWalks (overwriting any
previous version of that asset) and loads it into the Movie Render Queue
window. Does NOT start the render -- press Render (Local) yourself.

Test mode -- a single walk, a custom (small) frame range, output kept under
its own _test subfolder so it never collides with a full-walk render:

    py "<UE project>\\Scripts\\make_render_queue.py" test walk_04 100 102

Builds /Game/EgoScene/Render/EgoScene_Test with exactly one job for the named
walk, restricted to [start_frame, end_frame], output to
Saved/EgoRenders/_test/SEQ_<walk_id>. The walk's lighting preset (and so its
map) is looked up from the manifest, same as the full build. Also loads it
into the MRQ window. Does NOT start the render.
"""
import unreal, json, sys

MANIFEST_PATH = unreal.Paths.project_saved_dir() + "egoscene_walks_manifest.json"
LIGHTING_MAP_DIR = "/Game/EgoScene/Maps/Lighting"
SEQ_DIR = "/Game/EgoScene/Sequences"
PRESET_PATH = "/Game/EgoScene/Render/EgoScene_GT_Config"
QUEUE_DIR = "/Game/EgoScene/Render"
QUEUE_NAME = "EgoScene_AllWalks"
TEST_QUEUE_NAME = "EgoScene_Test"
OUTPUT_ROOT = unreal.Paths.project_dir() + "Saved/EgoRenders"
TEST_OUTPUT_ROOT = unreal.Paths.project_dir() + "Saved/EgoRenders/_test"
FILE_NAME_FORMAT = "{render_pass}/{frame_number}"


def _load_preset():
    preset = unreal.load_asset(PRESET_PATH)
    if preset is None:
        raise RuntimeError(f"Could not load MRQ preset at {PRESET_PATH}")
    return preset


def _map_path_for(lighting):
    map_path = f"{LIGHTING_MAP_DIR}/Apartment_01_{lighting}"
    if not unreal.EditorAssetLibrary.does_asset_exist(map_path):
        raise RuntimeError(f"Baked lighting map {map_path} not found -- "
                            f"run Scripts/make_lighting_maps.py first.")
    return map_path


def _manifest_entry(manifest, walk_id):
    entry = next((e for e in manifest if e["walk_id"] == walk_id), None)
    if entry is None:
        raise RuntimeError(f"'{walk_id}' not found in {MANIFEST_PATH}")
    return entry


def _new_queue_asset(queue_path):
    # There is no Python-exposed factory for queue assets, so make the new queue
    # by duplicating the existing walk_01 queue (EgoScene_GT) and clearing its jobs.
    if unreal.EditorAssetLibrary.does_asset_exist(queue_path):
        unreal.EditorAssetLibrary.delete_asset(queue_path)
    queue = unreal.EditorAssetLibrary.duplicate_asset(f"{QUEUE_DIR}/EgoScene_GT", queue_path)
    if queue is None:
        raise RuntimeError(f"Could not duplicate {QUEUE_DIR}/EgoScene_GT to {queue_path}")
    queue.delete_all_jobs()
    return queue


def _build_job(queue, preset, walk_id, lighting, output_dir, custom_range=None):
    """Allocates one job for walk_id on `queue`, seeded from `preset`, against
    the baked map matching `lighting`, with its output directory/filename/
    override set. If custom_range=(start, end) is given, also restricts the
    job to that frame range instead of the full sequence. Returns None (and
    logs a warning) if SEQ_<walk_id> doesn't exist."""
    seq_path = f"{SEQ_DIR}/SEQ_{walk_id}"
    if not unreal.EditorAssetLibrary.does_asset_exist(seq_path):
        unreal.log_warning(f"{walk_id}: sequence {seq_path} not found, skipping")
        return None
    map_path = _map_path_for(lighting)

    job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
    job.job_name = walk_id
    # Soft object paths need the full "Package.Object" form.
    job.sequence = unreal.SoftObjectPath(f"{seq_path}.SEQ_{walk_id}")
    job.map = unreal.SoftObjectPath(f"{map_path}.{map_path.rsplit('/', 1)[-1]}")

    # Seed this job's own Configuration from the shared preset (duplicates the
    # preset's settings onto the job; EgoScene_GT_Config itself is untouched),
    # then override the per-job output settings on top of it.
    job.set_configuration(preset)
    config = job.get_configuration()
    output_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    output_setting.output_directory = unreal.DirectoryPath(output_dir)
    output_setting.file_name_format = FILE_NAME_FORMAT
    output_setting.override_existing_output = True
    if custom_range is not None:
        start_frame, end_frame = custom_range
        output_setting.use_custom_playback_range = True
        output_setting.custom_start_frame = start_frame
        output_setting.custom_end_frame = end_frame

    return job


def _save_and_load_into_mrq(queue, queue_path):
    unreal.EditorAssetLibrary.save_loaded_asset(queue)
    # Also load it into the Movie Render Queue window, so no "Load Queue" step is needed.
    mrq = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    mrq.get_queue().copy_from(queue)


def build_full_queue():
    with open(MANIFEST_PATH, "r") as fh:
        manifest = json.load(fh)

    preset = _load_preset()
    queue_path = f"{QUEUE_DIR}/{QUEUE_NAME}"
    queue = _new_queue_asset(queue_path)

    total_frames = 0
    jobs_built = 0
    for entry in manifest:
        walk_id = entry["walk_id"]
        lighting = entry["lighting"]
        output_dir = f"{OUTPUT_ROOT}/SEQ_{walk_id}"
        job = _build_job(queue, preset, walk_id, lighting, output_dir)
        if job is None:
            continue
        num_frames = entry.get("num_frames", 0)
        total_frames += num_frames
        jobs_built += 1
        unreal.log(f"{walk_id}: SEQ_{walk_id}  ({lighting})  {num_frames} frames  -> {output_dir}")

    _save_and_load_into_mrq(queue, queue_path)
    unreal.log(f"Saved {queue_path} with {jobs_built} jobs, {total_frames} total frames. "
               f"Not starting the render -- press Render (Local) yourself.")


def build_test_queue(walk_id, start_frame, end_frame):
    with open(MANIFEST_PATH, "r") as fh:
        manifest = json.load(fh)
    lighting = _manifest_entry(manifest, walk_id)["lighting"]

    preset = _load_preset()
    queue_path = f"{QUEUE_DIR}/{TEST_QUEUE_NAME}"
    queue = _new_queue_asset(queue_path)

    output_dir = f"{TEST_OUTPUT_ROOT}/SEQ_{walk_id}"
    job = _build_job(queue, preset, walk_id, lighting, output_dir, custom_range=(start_frame, end_frame))
    if job is None:
        raise RuntimeError(f"Could not build a test job for '{walk_id}' -- does SEQ_{walk_id} exist?")

    _save_and_load_into_mrq(queue, queue_path)
    num_frames = end_frame - start_frame + 1
    unreal.log(f"Saved {queue_path}: 1 job, '{walk_id}' ({lighting}) frames {start_frame}-{end_frame} "
               f"({num_frames} frames)  -> {output_dir}. Not starting the render -- "
               f"press Render (Local) yourself.")


if __name__ == "__main__":
    argv = sys.argv
    # When run via `py <script> a b c d`, UE's Python plugin passes the script's
    # own args starting at argv[0] == the script path.
    args = argv[1:] if argv and argv[0].lower().endswith(".py") else argv
    if args and args[0] == "test":
        if len(args) != 4:
            raise SystemExit("Usage: py make_render_queue.py test <walk_id> <start_frame> <end_frame>")
        _, test_walk_id, start_s, end_s = args
        build_test_queue(test_walk_id, int(start_s), int(end_s))
    else:
        build_full_queue()
