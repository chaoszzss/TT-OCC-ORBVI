#!/usr/bin/env python3
"""Render true semantic voxel surfaces and an index spanning all committed frames.

No inference is launched. Static white-background figures disclose display-only
filters, retain full-height counterparts, and carry source/view metadata. The
interactive HTML uses the existing read-only live server for all-frame browsing.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from live_viewer import RunReader, semantic_registry
from camera_delivery_evidence import matching_full_qa, run_identity

PROJECT = Path(__file__).resolve().parents[1]
POSES = {
    "official": {"target": [0., 0., .5], "yaw": math.pi, "pitch": 1.48, "distance": 35.,
                 "projection": "orthographic", "lighting": "flat"},
    "oblique": {"target": [0., 0., .5], "yaw": -.85, "pitch": .72, "distance": 35.},
    "top": {"target": [0., 0., 0.], "yaw": -math.pi/2, "pitch": 1.53, "distance": 35.},
    "low": {"target": [0., 0., .6], "yaw": -.85, "pitch": .22, "distance": 30.},
    "side": {"target": [0., 0., .5], "yaw": -math.pi/2, "pitch": .08, "distance": 30.},
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _font(size):
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def resolve_pose(value):
    if value in POSES:
        return {**POSES[value], "target": list(POSES[value]["target"])}
    obj = json.loads(Path(value).read_text())
    pose = obj.get("pose", obj)
    if (not isinstance(pose.get("target"), list) or len(pose["target"]) != 3
            or not np.isfinite(pose["target"]).all()
            or any(not math.isfinite(float(pose.get(key, float("nan")))) for key in ("yaw", "pitch", "distance"))
            or not 3 <= pose["distance"] <= 180 or abs(pose["pitch"]) > 1.53):
        raise ValueError("invalid saved camera view")
    if pose.get("projection", "perspective") not in ("perspective", "orthographic") or pose.get("lighting", "shaded") not in ("shaded", "flat"):
        raise ValueError("invalid saved projection or lighting")
    return pose


def _registry_palette(registry):
    palette = np.tile(np.array([145, 151, 163], dtype=np.uint8), (256, 1))
    for c in registry["classes"]:
        palette[c["id"]] = c["color"]
    return palette


def semantic_mesh(indices, bounds, size, palette, labels, eye, lighting="shaded", parallel_direction=None):
    """Exterior voxel cube faces; removing occluded internal faces is rendering."""
    occupied = {tuple(v) for v in indices}
    vertices, colors = [], []
    for axis in range(3):
        other = [v for v in range(3) if v != axis]
        for sign in (-1, 1):
            offset = np.zeros(3, dtype=np.int32)
            offset[axis] = sign
            exposed = np.array([tuple(v + offset) not in occupied for v in indices], dtype=bool)
            ids = np.flatnonzero(exposed)
            if not len(ids):
                continue
            lo = bounds[0] + indices[ids] * size
            center = lo + size/2
            front = (eye[axis] - center[:, axis]) * sign > 0 if parallel_direction is None else np.full(len(ids), parallel_direction[axis]*sign > 0)
            ids, lo = ids[front], lo[front]
            if not len(ids):
                continue
            face = np.repeat(lo[:, None, :], 4, axis=1)
            face[:, :, axis] += size if sign > 0 else 0
            face[:, [1, 2], other[0]] += size
            face[:, [2, 3], other[1]] += size
            normal = np.zeros(3); normal[axis] = sign
            light = np.array([.4, -.45, .8]); light /= np.linalg.norm(light)
            shade = 1. if lighting == "flat" else .58 + .42 * max(0., float(normal @ light))
            vertices.append(face)
            colors.append(np.clip(palette[labels[ids]] * shade, 0, 255).astype(np.uint8))
    if not vertices:
        return np.empty((0, 4, 3)), np.empty((0, 3), np.uint8)
    return np.concatenate(vertices), np.concatenate(colors)


def render_png(run, index, output, pose, mode="fused", hide_ceiling=False, width=1800, height=1200):
    metadata = json.loads((run / "run.json").read_text())
    registry = semantic_registry(metadata)
    artifact = run / "occupancy" / f"{index:06d}.npz"
    with np.load(artifact, allow_pickle=False) as stored:
        indices = stored[f"{mode}_indices"].copy()
        labels = stored[f"{mode}_labels"].astype(np.uint8)
        bounds = stored["bounds"].astype(float)
        size = float(stored["voxel_size"])
        stamp = str(int(stored["stamp_ns"]))
    original_count = len(indices)
    hidden_ids = [c["id"] for c in registry["classes"] if hide_ceiling and c["name"] == "ceiling"]
    keep = ~np.isin(labels, hidden_ids)
    indices, labels = indices[keep], labels[keep]
    target = np.asarray(pose["target"], float)
    yaw, pitch, distance = (pose[k] for k in ("yaw", "pitch", "distance"))
    projection, lighting = pose.get("projection", "perspective"), pose.get("lighting", "shaded")
    eye = target + distance * np.array([math.cos(pitch)*math.cos(yaw), math.cos(pitch)*math.sin(yaw), math.sin(pitch)])
    z = eye - target; z /= np.linalg.norm(z)
    x = np.cross([0., 0., 1.], z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    faces, colors = semantic_mesh(indices, bounds, size, _registry_palette(registry), labels, eye,
                                 lighting, eye-target if projection == "orthographic" else None)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    margin_top, margin_bottom = 85, 135
    view_h = height - margin_top - margin_bottom
    view_w = width - 30
    focal = (view_h/2) / math.tan(math.pi/8)
    if len(faces):
        relative = faces - eye
        vx, vy, depth = relative @ x, relative @ y, -(relative @ z)
        divisor = distance if projection == "orthographic" else depth
        px, py = width/2 + focal*vx/divisor, margin_top + view_h/2 - focal*vy/divisor
        order = np.argsort(depth.mean(axis=1))[::-1]
        for face_i in order:
            if np.any(depth[face_i] <= .1):
                continue
            xy = list(zip(px[face_i], py[face_i]))
            if not all(math.isfinite(a) and math.isfinite(b) for a,b in xy):
                continue
            draw.polygon(xy, fill=tuple(int(v) for v in colors[face_i]))
    title = f"TT-Occ / ORBVI Camera | Frame {index} | {mode} | {size:g} m voxel"
    draw.text((30, 20), title, fill=(35, 45, 58), font=_font(26))
    filter_note = "DISPLAY FILTER: ceiling hidden; saved occupancy unchanged" if hidden_ids else "FULL HEIGHT / ALL CLASSES: every saved occupied voxel retained"
    draw.text((30, 56), filter_note, fill=(140, 73, 30) if hidden_ids else (85, 95, 105), font=_font(18))
    counts = Counter(int(v) for v in labels)
    legend_x, legend_y = 30, height-106
    for c in registry["classes"]:
        count = counts.get(c["id"], 0)
        if not count:
            continue
        text = f'{c["name"]}: {count:,}'
        text_width = int(draw.textlength(text, font=_font(18)))
        if legend_x + text_width + 55 > width - 25:
            legend_x, legend_y = 30, legend_y + 30
        draw.rectangle((legend_x, legend_y+2, legend_x+18, legend_y+20), fill=tuple(c["color"]), outline=(110,110,110))
        draw.text((legend_x+26, legend_y), text, fill=(48,58,68), font=_font(18))
        legend_x += text_width + 65
    draw.text((30, height-32), f"{len(indices):,}/{original_count:,} displayed | occupied surfaces only; absence is unknown | stamp {stamp} ns", fill=(80,90,100), font=_font(16))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")
    sidecar = {"schema": "ttocc-view-v1", "pose": pose, "theme": "white", "color_mode": "semantic", "mode": mode,
               "projection": projection, "lighting": lighting, "helper_lines": False,
               "index": index, "stamp_ns_text": stamp, "filter": {"enabled": False, "hidden_semantic_ids": hidden_ids},
               "geometry_modified": False, "displayed_voxels": len(indices), "saved_voxels": original_count,
               "semantic_registry": registry, "source_npz": str(artifact), "source_npz_sha256": _sha256(artifact),
               "renderer": "Pillow projected exterior cube faces; no voxel decimation or geometry smoothing",
               "dimensions": [width,height], "semantic_counts_displayed": dict(counts),
               "source_point_provenance": str(run / "source_points" / f"{index:06d}.npz")}
    output.with_suffix(".view.json").write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n")
    return sidecar


def export_ply(run, index, output, mode="fused"):
    registry = semantic_registry(json.loads((run / "run.json").read_text()))
    source = run / "occupancy" / f"{index:06d}.npz"
    with np.load(source, allow_pickle=False) as stored:
        labels = stored[f"{mode}_labels"].astype(np.uint8)
        size = float(stored["voxel_size"])
        centers = stored["bounds"][0] + (stored[f"{mode}_indices"] + .5) * size
    dtype = [(k,"<f4") for k in ("x","y","z")] + [(k,"u1") for k in ("red","green","blue","semantic_id")]
    vertices = np.empty(len(centers), dtype=dtype)
    for i,k in enumerate(("x","y","z")):
        vertices[k] = centers[:,i]
    colors = _registry_palette(registry)[labels]
    for i,k in enumerate(("red","green","blue")):
        vertices[k] = colors[:,i]
    vertices["semantic_id"] = labels
    output.parent.mkdir(parents=True, exist_ok=True)
    header = (f"ply\nformat binary_little_endian 1.0\ncomment occupied voxel centers; all height; no display filtering\ncomment voxel_size_m {size}\nelement vertex {len(centers)}\n"
              "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar semantic_id\nend_header\n")
    with output.open("wb") as handle:
        handle.write(header.encode("ascii"));vertices.tofile(handle)
    output.with_suffix(".metadata.json").write_text(json.dumps({"source_npz": str(source), "source_npz_sha256": _sha256(source),
        "voxel_size": size, "mode": mode, "count": len(centers), "filter": None, "semantic_registry": registry},indent=2,ensure_ascii=False)+"\n")


def create_gallery(run, output, selected, pose, mode="fused", viewer_url="http://127.0.0.1:8767", render_all=False, all_ply=False,
                   require_complete=False, qa_path=None):
    reader = RunReader(run)
    latest, code = reader.frame()
    if code != 200 or latest.get("status") != "ready":
        raise ValueError(f"No valid committed frames: {latest}")
    metadata = json.loads((run / "run.json").read_text())
    qa_path = Path(qa_path) if qa_path is not None else run / "verification.json"
    qa = json.loads(qa_path.read_text()) if qa_path.is_file() else {}
    qa_full = matching_full_qa(qa, run, metadata, latest["expected_frames"])
    if require_complete and not (latest["complete"] and qa_full):
        raise ValueError("Final gallery refused: complete recording and matching independent full QA are required")
    indices = list(reader._rows)
    if require_complete and not render_all and any(i not in indices for i in selected):
        raise ValueError("Final gallery requested a frame that was not committed")
    rows = []
    aggregate = Counter()
    for index in indices:
        payload, code = reader.frame(index, mode)
        if code != 200 or payload.get("status") != "ready":
            raise ValueError(f"Committed frame {index} is invalid: {payload}")
        counts = payload["semantic_counts"]
        aggregate.update({int(cid): n for cid,n in counts.items()})
        sources, _ = reader.sources(index)
        rows.append({"index": index, "stamp_ns_text": payload["stamp_ns_text"], "voxel_count": len(payload["labels"]),
                     "semantic_counts": counts, "source_views": sources.get("sources", []),
                     "point_provenance": sources.get("point_provenance"),
                     "occupancy": f"occupancy/{index:06d}.npz"})
        if all_ply:
            export_ply(run, index, output / "ply" / f"{index:06d}_full.ply", mode)
    selected = indices if render_all else [i for i in selected if i in indices]
    samples = []
    for index in selected:
        for cut in (False, True):
            name = f'{index:06d}_{"ceiling_cut" if cut else "allheight"}.png'
            render_metadata = render_png(run, index, output / "png" / name, pose, mode, hide_ceiling=cut)
            samples.append({"index": index, "name": "png/"+name, "hidden_ceiling": cut,
                            "displayed_voxels": render_metadata["displayed_voxels"], "saved_voxels": render_metadata["saved_voxels"]})
        if not all_ply:
            export_ply(run, index, output / "ply" / f"{index:06d}_full.ply", mode)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "ttocc-semantic-gallery-v1", "run": str(run), "mode": mode, "frame_count": len(rows),
                "expected_frames": latest["expected_frames"], "complete": latest["complete"],
                "matching_independent_full_qa": qa_full, "run_identity": run_identity(run, metadata),
                "independent_qa": str(qa_path.resolve()),
                "delivery_state": "full_run_accepted" if latest["complete"] and qa_full else "interim_committed_frames",
                "semantic_registry": latest["semantic_registry"], "semantic_counts_all_frames": dict(aggregate),
                "pose": pose, "samples": samples, "rows": rows,
                "offline_3d_note": "Start scripts/live_viewer.sh --run with this run for interactive access to every committed frame.",
                "source_rgb_note": "Raw fish-eye and rectified virtual views are distinct pixel domains; saved masks are shown with their matching RGB."}
    (output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    (output/"view.json").write_text(json.dumps({"schema":"ttocc-view-v1","pose":pose,"theme":"white","color_mode":"semantic"},indent=2)+"\n")
    safe_url = json.dumps(viewer_url.rstrip("/"))
    state_note = ("完整时间线处理及独立全包验收已通过。" if latest["complete"] and qa_full else
                  "时间线处理已完成；尚无匹配当前文件的独立全包验收。" if latest["complete"] else
                  "阶段浏览：仅覆盖下列已提交时间步，尚未完成全包处理或验收。")
    options = ''.join(f'<option value="{row["index"]}">{row["index"]:04d} · {row["voxel_count"]:,} voxels</option>' for row in rows)
    cards = ''.join(f'<figure><a href="{sample["name"]}"><img src="{sample["name"]}" alt="Real frame {sample["index"]} semantic occupancy, {"ceiling hidden for display" if sample["hidden_ceiling"] else "all classes and all height"}"></a><figcaption>Frame {sample["index"]} · {"显示隐藏顶棚（占据未修改）" if sample["hidden_ceiling"] else "完整全高度"} · {sample["displayed_voxels"]:,}/{sample["saved_voxels"]:,} voxels · <a href="{sample["name"].replace(".png",".view.json")}">视角与来源</a></figcaption></figure>' for sample in samples)
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TT-Occ Camera · 相机语义图库</title><style>body{{font-family:system-ui,sans-serif;margin:24px;color:#263343;background:#f8fafc}}h1{{font-size:24px}}p{{line-height:1.7}}select,button{{padding:8px;margin-right:8px}}iframe{{display:block;width:100%;height:1100px;border:1px solid #cbd5e1;background:white}}.cards{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}}figure{{margin:0;background:white;border:1px solid #d4dce6}}img{{width:100%}}figcaption{{padding:10px;font-size:13px}}a{{color:#1769ab}}@media(max-width:900px){{.cards{{grid-template-columns:1fr}}}}</style><h1>TT-Occ Camera · {len(rows)}/{latest["expected_frames"] or "?"} 帧语义占据</h1><p>{state_note}</p><p>颜色来自真实保存的类别，未知保持灰色。车辆蓝、地面品红、墙淡紫；顶棚与柱采用车库扩展类别。地面类别不代表已确认可通行。PNG 同时提供完整全高度和明确标记的顶棚剖视。</p><p>已经提交的时间步均可选择。三维互动需要启动 <code>bash scripts/live_viewer.sh --run {html.escape(str(run))} --port 8767</code>；静态图片无需服务。</p><label for="frame">时间步</label> <select id="frame">{options}</select><button id="load">查看本帧真实三维结果</button><a href="manifest.json">全部类别计数与来源清单</a> · <a href="view.json">统一视角 JSON</a><iframe id="viewer" title="真实已提交占据浏览器" loading="lazy"></iframe><h2>统一白背景的静态对照</h2><div class="cards">{cards}</div><script>const base={safe_url};function load(){{document.getElementById('viewer').src=base+'/?index='+document.getElementById('frame').value+'&follow=0';}}document.getElementById('load').onclick=load;load();</script></html>'''
    (output/"gallery.html").write_text(page)
    return {"frame_count":len(rows),"sample_frames":selected,"complete":latest["complete"],
            "matching_independent_full_qa":qa_full,"gallery":str(output/"gallery.html")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", default="reports/camera_semantic_gallery")
    parser.add_argument("--frames", default="0,135,600,1174,1800,2347", help="Static sample indices (every committed frame remains browseable)")
    parser.add_argument("--pose", default="official", help="official (parallel/flat), oblique/top/low/side, or saved view JSON")
    parser.add_argument("--mode", choices=("current","fused"), default="fused")
    parser.add_argument("--viewer-url", default="http://127.0.0.1:8767")
    parser.add_argument("--require-complete", action="store_true", help="Final delivery only: require complete timeline and matching full QA identity")
    parser.add_argument("--qa", type=Path, help="Independent acceptance report (defaults to RUN/verification.json)")
    parser.add_argument("--render-all", action="store_true", help="Also render paired PNGs for all frames")
    parser.add_argument("--all-ply", action="store_true", help="Export full-height PLY for all committed frames")
    args = parser.parse_args()
    run = Path(args.run).resolve(); output = Path(args.output).resolve()
    if output == run or output.is_relative_to(run):
        parser.error("gallery output must be separate from the inference run")
    selected = [int(i) for i in args.frames.split(",") if i.strip()]
    result = create_gallery(run,output,selected,resolve_pose(args.pose),args.mode,args.viewer_url,args.render_all,args.all_ply,
                            args.require_complete,args.qa)
    print(json.dumps(result,ensure_ascii=False))


if __name__ == "__main__":
    main()
