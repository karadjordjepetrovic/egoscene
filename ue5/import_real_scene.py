"""Unreal Editor script: build a level from a real-video scene graph (scene_graph/real_scene_graph.py).

Run from the Output Log (Cmd):
    py "<UE project>/Scripts/import_real_scene.py" "C:/dev/egoscene/data/real/_out/home_walk_01/scene_graph/home_walk_01_graph.json"

Creates /Game/EgoScene/Real/Maps/Real_<video> with:
  - a wireframe box (12 thin edges) + label per scene-graph object, colour per class
  - the fitted walls as 2.4 m tall thin slabs (semi-hidden: toggle the "Walls" folder)
  - the camera path as small red spheres, and a floor plane at z = 0
The coloured point cloud (<video>_room_ue.ply, already in Unreal coordinates) is imported separately
with the LiDAR Point Cloud plugin; place that actor at location (0, 0, 0).

Coordinates: scene graph = right-handed, Z up, metres -> Unreal = left-handed, Z up, cm (y flipped).
"""
import json, math, os, sys, colorsys
import unreal

CUBE = "/Engine/BasicShapes/Cube.Cube"
SPHERE = "/Engine/BasicShapes/Sphere.Sphere"
PLANE = "/Engine/BasicShapes/Plane.Plane"
BASE_MAT = "/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"
MAT_DIR = "/Game/EgoScene/Real/Materials"
MAP_DIR = "/Game/EgoScene/Real/Maps"
EDGE = 2.0          # cm, wireframe edge thickness

eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
tools = unreal.AssetToolsHelpers.get_asset_tools()


def ue(p):                                   # metres, right-handed -> cm, Unreal
    return unreal.Vector(p[0] * 100.0, -p[1] * 100.0, p[2] * 100.0)


def material(name, rgb):
    path = f"{MAT_DIR}/MI_{name}"
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        return unreal.load_asset(path)
    mi = tools.create_asset(f"MI_{name}", MAT_DIR, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
    unreal.MaterialEditingLibrary.set_material_instance_parent(mi, unreal.load_asset(BASE_MAT))
    unreal.MaterialEditingLibrary.set_material_instance_vector_parameter_value(
        mi, "Color", unreal.LinearColor(rgb[0], rgb[1], rgb[2], 1.0))
    unreal.EditorAssetLibrary.save_loaded_asset(mi)
    return mi


def class_color(cls):
    h = (sum(map(ord, cls)) * 37 % 360) / 360.0
    return colorsys.hsv_to_rgb(h, 0.75, 0.95)


def mesh_actor(mesh, loc, scale, mat, label, folder):
    a = eas.spawn_actor_from_class(unreal.StaticMeshActor, loc, unreal.Rotator(0, 0, 0))
    c = a.static_mesh_component
    c.set_static_mesh(unreal.load_asset(mesh))
    c.set_material(0, mat)
    c.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
    a.set_actor_scale3d(scale)
    a.set_actor_label(label)
    a.set_folder_path(folder)
    return a


def wire_box(center, size, mat, label, folder):
    cx, cy, cz = center.x, center.y, center.z
    sx, sy, sz = abs(size[0]) * 100, abs(size[1]) * 100, abs(size[2]) * 100
    k = 0
    for dy in (-1, 1):                       # edges along X
        for dz in (-1, 1):
            mesh_actor(CUBE, unreal.Vector(cx, cy + dy * sy / 2, cz + dz * sz / 2), unreal.Vector(sx / 100, EDGE / 100, EDGE / 100),
                       mat, f"{label}_e{k}", folder); k += 1
    for dx in (-1, 1):                       # edges along Y
        for dz in (-1, 1):
            mesh_actor(CUBE, unreal.Vector(cx + dx * sx / 2, cy, cz + dz * sz / 2), unreal.Vector(EDGE / 100, sy / 100, EDGE / 100),
                       mat, f"{label}_e{k}", folder); k += 1
    for dx in (-1, 1):                       # vertical edges
        for dy in (-1, 1):
            mesh_actor(CUBE, unreal.Vector(cx + dx * sx / 2, cy + dy * sy / 2, cz), unreal.Vector(EDGE / 100, EDGE / 100, sz / 100),
                       mat, f"{label}_e{k}", folder); k += 1


def label_actor(text, loc, rgb, folder):
    a = eas.spawn_actor_from_class(unreal.TextRenderActor, loc, unreal.Rotator(0, 0, 0))
    t = a.text_render
    t.set_text(text)
    t.set_editor_property("world_size", 14.0)
    t.set_editor_property("horizontal_alignment", unreal.HorizTextAligment.EHTA_CENTER)
    t.set_text_render_color(unreal.Color(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), 255))
    a.set_actor_label(f"label_{text}")
    a.set_folder_path(folder)


def main(json_path):
    g = json.load(open(json_path))
    name = os.path.splitext(os.path.basename(g["video"]))[0]
    unreal.EditorAssetLibrary.make_directory(MAT_DIR)
    unreal.EditorAssetLibrary.make_directory(MAP_DIR)
    map_path = f"{MAP_DIR}/Real_{name}"
    if unreal.EditorAssetLibrary.does_asset_exist(map_path):
        les.load_level(map_path)
        for a in eas.get_all_level_actors():
            if str(a.get_folder_path()).startswith("SceneGraph"):
                eas.destroy_actor(a)
    else:
        les.new_level(map_path)
        sun = eas.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(0, 0, 500), unreal.Rotator(0, -50, 30))
        sun.set_folder_path("Lighting")
        eas.spawn_actor_from_class(unreal.SkyLight, unreal.Vector(0, 0, 400), unreal.Rotator(0, 0, 0)).set_folder_path("Lighting")

    n = 0
    for o in g["objects"]:
        rgb = class_color(o["class"])
        mat = material(o["class"], rgb)
        c = ue(o["centroid"])
        folder = f"SceneGraph/Objects/{o['class']}_{o['id']}"
        wire_box(c, o["size"], mat, f"{o['class']}_{o['id']}", folder)
        label_actor(f"{o['class']} #{o['id']} ({o['support']})", unreal.Vector(c.x, c.y, c.z + abs(o["size"][2]) * 50 + 12), rgb, folder)
        n += 1

    wall_mat = material("Wall", (0.75, 0.75, 0.78))
    for i, w in enumerate(g.get("walls", [])):
        p0, p1 = ue(w["p0"] + [0]), ue(w["p1"] + [0])
        dx, dy = p1.x - p0.x, p1.y - p0.y
        length = (dx * dx + dy * dy) ** 0.5
        a = mesh_actor(CUBE, unreal.Vector((p0.x + p1.x) / 2, (p0.y + p1.y) / 2, 120), unreal.Vector(length / 100, 0.03, 2.4),
                       wall_mat, f"wall_{i}", "SceneGraph/Walls")
        a.set_actor_rotation(unreal.Rotator(0, 0, math.degrees(math.atan2(dy, dx))), False)
        a.set_is_temporarily_hidden_in_editor(True)        # toggle with the eye icon on the Walls folder

    path_mat = material("CameraPath", (0.9, 0.1, 0.1))
    for i, p in enumerate(g.get("camera_path", [])[::3]):
        mesh_actor(SPHERE, ue(p), unreal.Vector(0.06, 0.06, 0.06), path_mat, f"cam_{i:03d}", "SceneGraph/CameraPath")
    floor_mat = material("Floor", (0.35, 0.35, 0.35))
    mesh_actor(PLANE, unreal.Vector(0, 0, -1), unreal.Vector(40, 40, 1), floor_mat, "floor_z0", "SceneGraph/Floor")
    les.save_current_level()
    unreal.log(f"Real scene: {n} objects, {len(g.get('walls', []))} walls, camera height {g['camera_height_m']:.2f} m -> {map_path}")


if __name__ == "__main__":
    args = sys.argv[1:] if sys.argv and sys.argv[0].lower().endswith(".py") else sys.argv
    if not args:
        raise SystemExit('Usage: py ".../import_real_scene.py" "<path to *_graph.json>"')
    main(args[0])
