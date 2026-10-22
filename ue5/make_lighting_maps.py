"""
Bakes each lighting preset into its own duplicate map.

Keying the HDRI backdrop and PostProcessVolume exposure at runtime via
Sequencer looked correct in the editor viewport but was silently ignored
during an actual Movie Render Queue render (confirmed: walk_04/night still
rendered a fully bright daytime street through the window). Baking sidesteps
this: each preset's values are set directly on the actors, once, in the
editor, before the map is saved.

Run from the Unreal Editor Output Log:

    py "<UE project>\\Scripts\\make_lighting_maps.py"

For each preset in make_walk.LIGHTING_PRESETS (day, overcast, evening, night):
  1. Loads a clean copy of /Game/EgoScene/Maps/Apartment_01.
  2. Sets the preset's values directly on the actors: sun intensity/temperature/
     rotation; the HDRIBackdrop's BP "Intensity" variable followed by
     rerun_construction_scripts() (for the in-editor Details-panel-equivalent
     behaviour) AND, redundantly and unconditionally, the Skylight component's
     own Intensity plus the Intensity scalar parameter on both backdrop
     materials (HDRI_Floor, HDRI_Sky) directly -- belt-and-suspenders, so the
     bake is correct even if the construction script doesn't propagate the BP
     var as expected. For evening/overcast/night it also swaps the backdrop's
     Cubemap to a matching Poly Haven HDRI (day keeps the existing daytime
     street HDRI); SkyLight_0; the 3 room lamps + hallway light (intensity +
     temperature); and PostProcessVolume_01's exposure bias.
  3. Saves ("Save As") the modified world to
     /Game/EgoScene/Maps/Lighting/Apartment_01_<preset> -- Apartment_01 on
     disk is never itself modified.
Finally reloads /Game/EgoScene/Maps/Apartment_01 so the editor ends up back
where it started.

Why "load Apartment_01 fresh + Save As" rather than duplicate_asset +
load_level: duplicate_asset returns a live UWorld object, and holding that
Python reference across a subsequent load_level() call made the editor abort
with a fatal "World Memory Leaks" error. This version keeps no Python
reference to any world across a level transition, and also forces an actual
*engine* garbage-collection pass (unreal.SystemLibrary.collect_garbage()) --
not just Python's own gc.collect(), which only manages Python-side reference
cycles and does nothing for leaked UObjects/UWorlds -- before every load.

Values are baseline (Saved/egoscene_lighting_baseline.json) * preset
multiplier, using the exact same LIGHTING_PRESETS table as make_walk.py
(imported from it, not duplicated here).
"""
import unreal, os, sys, importlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import make_walk
importlib.reload(make_walk)   # pick up edits without restarting the editor

APARTMENT_MAP_PATH = "/Game/EgoScene/Maps/Apartment_01"
LIGHTING_MAP_DIR = "/Game/EgoScene/Maps/Lighting"

# Per-preset HDRI cubemap swap. day is omitted -- it keeps the backdrop's
# existing /Game/EgoScene/Env/HDRI_wide_street_01_2k (a clear, midday street).
# All three below are CC0 Poly Haven HDRIs (2k), chosen to match wide_street_01's
# "residential/city street" framing and confirmed via Poly Haven's own
# weather/time_of_day attributes, not just filenames:
#   overcast -> urban_street_03   ("overcast London residential street", weather=overcast, time=morning)
#   evening  -> twilight_sunset   ("twilight sunset over a suburban driveway", time=dusk)
#   night    -> cobblestone_street_night ("lamp-lit night street", time=night, light_type=artificial)
CUBEMAP_OVERRIDES = {
    "overcast": "/Game/EgoScene/Env/HDRI_urban_street_03_2k",
    "evening": "/Game/EgoScene/Env/HDRI_twilight_sunset_2k",
    "night": "/Game/EgoScene/Env/HDRI_cobblestone_street_night_2k",
}


# Fallbacks by class, for actors whose Outliner label differs from the object
# name recorded in the baseline (e.g. sun labelled "DirectionalLight", HDRI
# backdrop object "HDRIBackdrop_C_0" with a different label).
_CLASS_FALLBACKS = {
    "DirectionalLight_0": lambda a: isinstance(a, unreal.DirectionalLight),
    "SkyLight_0": lambda a: isinstance(a, unreal.SkyLight),
    "HDRIBackdrop_C_0": lambda a: "HDRIBackdrop" in a.get_class().get_name(),
}


def _find_actor(actors, label):
    """Find by Outliner label, then by object name, then by class."""
    actor = next((a for a in actors if a.get_actor_label() == label), None)
    if actor is None:
        actor = next((a for a in actors if a.get_name() == label), None)
    if actor is None and label in _CLASS_FALLBACKS:
        actor = next((a for a in actors if _CLASS_FALLBACKS[label](a)), None)
    if actor is not None and actor.get_actor_label() != label:
        unreal.log(f"  '{label}' -> using actor '{actor.get_actor_label()}' ({actor.get_name()})")
    return actor


def _find_component(actor, component_class):
    if actor is None:
        return None
    comps = actor.get_components_by_class(component_class)
    return comps[0] if comps else None


def _apply_preset(preset_name):
    preset = make_walk.LIGHTING_PRESETS[preset_name]
    baseline = make_walk._load_baseline()

    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = eas.get_all_level_actors()

    # Sun
    sun_actor = _find_actor(actors, make_walk.SUN_ACTOR_LABEL)
    if sun_actor is not None:
        sun_actor.modify()
        sun_yaw = baseline["sun"]["rotation"]["yaw"]
        loc = sun_actor.get_actor_location()
        scale = sun_actor.get_actor_scale3d()
        new_rot = unreal.Rotator(roll=0.0, pitch=preset["sun_pitch"], yaw=sun_yaw)
        sun_actor.set_actor_transform(unreal.Transform(location=loc, rotation=new_rot, scale=scale), False, False)
        sun_light = _find_component(sun_actor, unreal.DirectionalLightComponent)
        if sun_light is not None:
            sun_light.modify()
            sun_intensity = baseline["sun"]["intensity"] * preset["sun_mul"]
            sun_light.set_editor_property("Intensity", sun_intensity)
            sun_light.set_editor_property("Temperature", preset["sun_temp"])
    else:
        unreal.log_warning(f"{preset_name}: actor '{make_walk.SUN_ACTOR_LABEL}' not found")

    # HDRI backdrop. Set the BP "Intensity" var + force a construction-script
    # rerun (matches what editing it in the Details panel would do), AND --
    # unconditionally, not as a fallback -- also set the Skylight component's
    # Intensity and both materials' "Intensity" scalar parameter directly.
    # This makes the bake correct regardless of whether the construction
    # script actually propagates the BP var the way the Details panel does.
    hdri_actor = _find_actor(actors, make_walk.HDRI_ACTOR_LABEL)
    if hdri_actor is not None:
        hdri_base = make_walk._hdri_baseline(baseline)
        sky_mul = preset["sky_mul"]
        bp_intensity = hdri_base["bp_variable_DO_NOT_KEY"]["value"] * sky_mul
        skylight_intensity = hdri_base["skylight_component"]["baseline"] * sky_mul
        material_slots = hdri_base["mesh_component"]["material_slots"]

        hdri_actor.modify()
        # rerun_construction_scripts() isn't exposed to Python. Instead, set the BP
        # variables with notify ALWAYS: that fires PostEditChange, which re-runs the
        # construction script exactly like editing them in the Details panel.
        # Cubemap first, then Intensity; the direct Skylight/material writes below
        # come after, so the construction script can't overwrite them.
        notify = unreal.PropertyAccessChangeNotifyMode.ALWAYS
        cubemap_path = CUBEMAP_OVERRIDES.get(preset_name)
        if cubemap_path is not None:
            cubemap_asset = unreal.load_asset(cubemap_path)
            if cubemap_asset is not None:
                hdri_actor.set_editor_property("Cubemap", cubemap_asset, notify)
            else:
                unreal.log_warning(f"{preset_name}: could not load cubemap {cubemap_path}")
        hdri_actor.set_editor_property("Intensity", bp_intensity, notify)

        skylight_comp = hdri_actor.get_editor_property("Skylight")
        if skylight_comp is not None:
            skylight_comp.modify()
            skylight_comp.set_editor_property("Intensity", skylight_intensity)
        else:
            unreal.log_warning(f"{preset_name}: HDRIBackdrop has no 'Skylight' sub-object")

        mat_sky = hdri_actor.get_editor_property("materialInstanceSky")
        mat_floor = hdri_actor.get_editor_property("materialInstanceFloor")
        for slot in material_slots:
            mid = mat_floor if slot["index"] == 0 else mat_sky
            if mid is None:
                unreal.log_warning(f"{preset_name}: HDRIBackdrop has no material instance for slot {slot['index']}")
                continue
            mat_intensity = slot["baseline"] * sky_mul
            mid.set_scalar_parameter_value(slot["parameter"], mat_intensity)


        # Read back what's actually on the actor/components now, so the Output
        # Log confirms (or disproves) that this all actually took.
        readback_bp = hdri_actor.get_editor_property("Intensity")
        readback_sky = skylight_comp.get_editor_property("Intensity") if skylight_comp is not None else None
        try:
            readback_mat_sky = mat_sky.k2_get_scalar_parameter_value("Intensity") if mat_sky is not None else None
        except Exception as exc:
            readback_mat_sky = f"<readback failed: {exc}>"
        unreal.log(f"{preset_name}: HDRI bake -> BP Intensity={readback_bp} "
                   f"Skylight.Intensity={readback_sky} HDRI_Sky.Intensity={readback_mat_sky}")
    else:
        unreal.log_warning(f"{preset_name}: actor '{make_walk.HDRI_ACTOR_LABEL}' not found")

    # Extra standalone skylight
    extra_sky_actor = _find_actor(actors, make_walk.EXTRA_SKYLIGHT_LABEL)
    extra_sky_light = _find_component(extra_sky_actor, unreal.SkyLightComponent)
    if extra_sky_light is not None:
        extra_sky_light.modify()
        extra_sky_intensity = make_walk._sky_baseline(baseline, make_walk.EXTRA_SKYLIGHT_LABEL) * preset["sky_mul"]
        extra_sky_light.set_editor_property("Intensity", extra_sky_intensity)
    else:
        unreal.log_warning(f"{preset_name}: actor '{make_walk.EXTRA_SKYLIGHT_LABEL}' not found")

    # Lamps (3 room lights, all sharing the same baseline, + hallway light)
    lamp_baseline = baseline["lamps"][0]["intensity"]
    lamp_intensity = lamp_baseline * preset["lamp_mul"]
    lamp_temp = make_walk.LAMP_WARM_TEMPERATURE if preset["lamp_warm"] else 6500
    for label in make_walk.LAMP_ACTOR_LABELS:
        lamp_actor = _find_actor(actors, label)
        lamp_light = _find_component(lamp_actor, unreal.RectLightComponent)
        if lamp_light is None:
            unreal.log_warning(f"{preset_name}: lamp actor '{label}' not found")
            continue
        lamp_light.modify()
        lamp_light.set_editor_property("Intensity", lamp_intensity)
        lamp_light.set_editor_property("bUseTemperature", True)
        lamp_light.set_editor_property("Temperature", lamp_temp)

    hall_actor = _find_actor(actors, make_walk.HALLWAY_LIGHT_LABEL)
    hall_light = _find_component(hall_actor, unreal.RectLightComponent)
    if hall_light is not None:
        hall_light.modify()
        hallway_intensity = baseline["hallway_light"]["intensity"] * preset["lamp_mul"]
        hall_light.set_editor_property("Intensity", hallway_intensity)
        hall_light.set_editor_property("bUseTemperature", True)
        hall_light.set_editor_property("Temperature", lamp_temp)
    else:
        unreal.log_warning(f"{preset_name}: actor '{make_walk.HALLWAY_LIGHT_LABEL}' not found")

    # Exposure (PostProcessVolume_01's Settings struct -- modify a copy, write it back)
    exp_actor = _find_actor(actors, make_walk.EXPOSURE_VOLUME_LABEL)
    if exp_actor is not None:
        exp_actor.modify()
        settings = exp_actor.get_editor_property("Settings")
        settings.set_editor_property("AutoExposureBias", preset["exposure_bias"])
        exp_actor.set_editor_property("Settings", settings)
    else:
        unreal.log_warning(f"{preset_name}: actor '{make_walk.EXPOSURE_VOLUME_LABEL}' not found")


def _collect_garbage():
    # Python's gc.collect() only manages Python-side reference cycles -- it does
    # NOT reclaim leaked engine UObjects/UWorlds. Repeated Python-driven level
    # loads in one script run need an actual *engine* GC pass to avoid a fatal
    # "World Memory Leaks" abort, so call both.
    import gc
    gc.collect()
    unreal.SystemLibrary.collect_garbage()


def main():
    unreal.EditorAssetLibrary.make_directory(LIGHTING_MAP_DIR)
    level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    editor_subsystem = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)

    built = []
    for preset_name in make_walk.LIGHTING_PRESETS:
        dup_path = f"{LIGHTING_MAP_DIR}/Apartment_01_{preset_name}"

        _collect_garbage()
        if not level_subsystem.load_level(APARTMENT_MAP_PATH):
            unreal.log_warning(f"{preset_name}: could not load {APARTMENT_MAP_PATH}, skipping")
            continue

        # Apartment_01 is open now, so an older baked copy is safe to delete.
        if unreal.EditorAssetLibrary.does_asset_exist(dup_path):
            unreal.EditorAssetLibrary.delete_asset(dup_path)

        _apply_preset(preset_name)

        # Save As: the open world becomes dup_path; Apartment_01 on disk stays untouched.
        ok = unreal.EditorLoadingAndSavingUtils.save_map(editor_subsystem.get_editor_world(), dup_path)
        if ok:
            built.append(dup_path)
            unreal.log(f"{preset_name}: baked -> {dup_path}")
        else:
            unreal.log_warning(f"{preset_name}: save_map to {dup_path} failed")

        _collect_garbage()

    # Leave the editor back on the original map, without saving any changes to it.
    level_subsystem.load_level(APARTMENT_MAP_PATH)
    unreal.log(f"Built {len(built)} lighting map(s): {built}")


if __name__ == "__main__":
    main()
