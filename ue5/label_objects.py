import unreal, json, re

# Class rules: first regex that matches the actor label or mesh name wins.
# Target class list (append-only; existing entries keep their position so
# instance numbering for unaffected actors doesn't reshuffle):
# wall, floor, ceiling, door, window, table, chair, sofa, tv, counter,
# stove, sink, shelf, plant, lamp, bed, sign, outdoor.
CLASS_RULES = [
    ("chair",   r"chair|stool"),
    ("table",   r"table|desk"),
    ("sofa",    r"sofa|couch"),
    ("door",    r"(?<!out)door"),  # exclude "outdoor" so Outdoor_Backdrop falls through to the outdoor rule
    ("window",  r"window"),
    ("tv",      r"tv|monitor|screen"),
    ("counter", r"counter|worktop"),
    ("stove",   r"stove|oven|hob"),
    ("sink",    r"sink"),
    ("shelf",   r"shelf|bookcase"),
    ("plant",   r"plant|pot"),
    ("wall",    r"wall"),
    ("lamp",    r"lamp"),
    ("floor",   r"floor"),
    ("ceiling", r"ceiling"),
    # --- appended for Task 3 ---
    ("bed",     r"bed"),
    ("sign",    r"sign"),
    ("outdoor", r"outdoor"),
]

actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
labels, unmatched, next_id = {}, set(), 1

for actor in actors:
    for comp in actor.get_components_by_class(unreal.StaticMeshComponent):
        mesh = comp.static_mesh.get_name() if comp.static_mesh else ""
        text = f"{actor.get_actor_label()} {mesh}".lower()
        cls = next((c for c, rx in CLASS_RULES if re.search(rx, text)), None)
        if cls is None:
            unmatched.add(mesh)
            continue
        if next_id > 255:
            unreal.log_warning("More than 255 labelled objects; the rest stay unlabelled")
            break
        comp.set_editor_property("render_custom_depth", True)
        comp.set_editor_property("custom_depth_stencil_value", next_id)
        labels[next_id] = {"class": cls, "actor": actor.get_actor_label(), "mesh": mesh}
        # Signs carry their readable text on a separate TextRenderComponent;
        # give it the same stencil ID so the text isn't left out of the ID mask.
        if cls == "sign":
            for text_comp in actor.get_components_by_class(unreal.TextRenderComponent):
                text_comp.set_editor_property("render_custom_depth", True)
                text_comp.set_editor_property("custom_depth_stencil_value", next_id)
        next_id += 1

out = unreal.Paths.project_saved_dir() + "egoscene_labels.json"
with open(out, "w") as f:
    json.dump({"classes": sorted({v["class"] for v in labels.values()}),
               "instances": labels}, f, indent=2)
unreal.log(f"Labelled {len(labels)} objects -> {out}")
unreal.log("Unmatched meshes: " + ", ".join(sorted(unmatched)))