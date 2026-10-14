#!/usr/bin/env python3
"""Compare two *completed* whole-bag runs and build a local, offline gallery.

No cross-sensor voxel alignment or accuracy estimate is performed. All plotted
states are read from the actual persisted NPZ outputs, never synthesized.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
CLASSES = {0: "unknown", 1: "blocked", 2: "partial_clearance", 3: "observed_clearance"}
CLASS_ZH = {"unknown": "未知", "blocked": "阻塞", "partial_clearance": "部分净空候选", "observed_clearance": "实测完整净空"}
PALETTE = np.array([[36, 47, 64], [235, 96, 103], [242, 187, 81], [63, 190, 149]], np.uint8)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(temporary, path)


def distribution(values):
    data = np.asarray(values, np.float64)
    if not data.size or not np.isfinite(data).all():
        raise ValueError("Metrics must contain finite recorded values")
    return {"mean": float(data.mean()), "median": float(np.median(data)),
            "p95": float(np.percentile(data, 95)), "min": float(data.min()), "max": float(data.max())}


def finite_max(values):
    valid = [float(x) for x in values if x is not None and np.isfinite(x)]
    return max(valid) if valid else None


def validated_run(directory, expected_frames, source_frames, source_hash):
    directory = Path(directory).resolve()
    summary_path = directory / "summary.json"
    if not summary_path.exists():
        raise ValueError(f"Run is unfinished: missing {summary_path}")
    summary = read_json(summary_path)
    if summary.get("complete") is not True or summary.get("acceptance", {}).get("accepted") is not True:
        raise ValueError(f"Refusing a final report for unfinished/unaccepted run: {directory}")
    run_metadata = read_json(directory / "run.json")
    rows = read_jsonl(directory / "frames.jsonl")
    if len(rows) != expected_frames or [r["index"] for r in rows] != list(range(expected_frames)):
        raise ValueError(f"Recording coverage is not contiguous 0..{expected_frames - 1}: {directory}")
    if summary["acceptance"].get("expected_frames") != expected_frames:
        raise ValueError(f"Unexpected full recording size in {directory}")
    if run_metadata.get("bag_sha256") != source_hash:
        raise ValueError(f"Run and original-bag provenance differ: {directory}")
    for row, source in zip(rows, source_frames):
        if row["stamp_ns"] != source["stamp_ns"]:
            raise ValueError(f"Acquisition timestamp mismatch at {row['index']}: {directory}")
        if row["raw_source_points"] != source["point_count"]:
            raise ValueError(f"Original scan point accounting mismatch at {row['index']}: {directory}")
        for kind in ["occupancy", "traversability"]:
            if not (directory / kind / f"{row['index']:06d}.npz").is_file():
                raise ValueError(f"Missing actual {kind} artifact at {row['index']}: {directory}")
    actual_accounted = sum(row["raw_source_points"] for row in rows)
    if summary.get("source_lidar_points_accounted") != actual_accounted or actual_accounted != summary.get("source_bag_lidar_points"):
        raise ValueError(f"Original LiDAR message point accounting is incomplete: {directory}")
    return {"directory": directory, "summary": summary, "metadata": run_metadata, "rows": rows}


def camera_segments(rows):
    segments = []
    start, active_id = 0, None
    previous_stamp = None
    rejected = []
    conditioning = []
    residuals = []
    continuity = []
    for i, row in enumerate(rows):
        motion = row.get("auxiliary", {}).get("camera_motion", {})
        if i and not motion.get("accepted", False):
            rejected.append(i)
        if motion.get("condition") is not None:
            conditioning.append(motion["condition"])
        if motion.get("point_plane_rmse_m") is not None:
            residuals.append(motion["point_plane_rmse_m"])
        if motion.get("global_segment_registration_valid") is False:
            continuity.append(i)
        segment_id = motion.get("segment_id")
        if segment_id is None:
            segment_id = (active_id or 0) + int(i > 0 and not motion.get("accepted", False))
        if active_id is None:
            active_id = segment_id
        elif segment_id != active_id:
            segments.append({"segment_id": int(active_id), "first_frame": start, "last_frame": i - 1,
                             "frames": i - start,
                             "duration_seconds": (previous_stamp - rows[start]["stamp_ns"]) / 1e9})
            start, active_id = i, segment_id
        previous_stamp = row["stamp_ns"]
    if rows:
        segments.append({"segment_id": int(active_id), "first_frame": start, "last_frame": len(rows) - 1,
                         "frames": len(rows) - start,
                         "duration_seconds": (rows[-1]["stamp_ns"] - rows[start]["stamp_ns"]) / 1e9})
    return {"independent_segment_count": len(segments), "segments": segments,
            "rejected_registration_count": len(rejected), "rejected_frames": rejected,
            "frames_without_valid_intersegment_registration": len(continuity),
            "continuous_global_registration_valid": not rejected,
            "global_pose_is_ground_truth": False,
            "condition": distribution(conditioning) if conditioning else None,
            "point_plane_rmse_m": distribution(residuals) if residuals else None,
            "pose_source": "camera predicted depth ICP only; no LiDAR/VIO trajectory input"}


def route_metrics(run):
    rows, summary, metadata = run["rows"], run["summary"], run["metadata"]
    current = np.asarray([r["current_occupied_voxels"] for r in rows], np.float64)
    fused = np.asarray([r["fused_occupied_voxels"] for r in rows], np.float64)
    if np.any(fused < current):
        raise ValueError("Past Gaussian fusion unexpectedly removed current occupied geometry")
    camera_gpu = [r.get("auxiliary", {}).get("camera_depth", {}).get("peak_allocated_gpu_mib") for r in rows]
    recorded_gpu = [summary.get(key) for key in ["peak_gpu_allocated_mib", "peak_allocated_gpu_mib", "peak_gpu_memory_mib"]]
    semantics_gpu = [r.get("auxiliary", {}).get("semantics", {}).get("peak_allocated_gpu_mib") for r in rows]
    gpu = finite_max(camera_gpu + recorded_gpu + semantics_gpu)
    duration = (rows[-1]["stamp_ns"] - rows[0]["stamp_ns"]) / 1e9
    processing = [r["processing_seconds"] for r in rows]
    native_areas = {name: distribution([r["class_area_m2"][name] for r in rows]) for name in CLASSES.values()}
    result = {"route": metadata["route"], "complete": True,
              "frames": len(rows), "recording_sensor_duration_seconds": duration,
              "source_bag_sha256": metadata["bag_sha256"],
              "source_lidar_points_accounted": summary["source_lidar_points_accounted"],
              "invalid_raw_lidar_returns": summary.get("invalid_raw_lidar_returns"),
              "latency_seconds": distribution(processing),
              "sum_frame_processing_seconds": float(np.sum(processing)),
              "processing_time_per_recording_time": float(np.sum(processing) / duration),
              "peak_loop_rss_mib": finite_max([summary.get("peak_process_rss_mib")] + [r.get("rss_mib") for r in rows]),
              "peak_torch_allocated_gpu_mib": gpu,
              "gpu_measurement_scope": "recorded full-run Torch allocation" if gpu is not None else "not recorded in full-run outputs",
              "model_loading_peak_rss_measured": False,
              "current_occupied_voxels": distribution(current),
              "fused_occupied_voxels": distribution(fused),
              "extra_occupied_voxels_from_past": distribution(fused - current),
              "past_voxel_gain_fraction": distribution((fused - current) / np.maximum(current, 1)),
              "frames_with_past_voxel_gain": int(np.count_nonzero(fused > current)),
              "native_class_area_m2": native_areas,
              "frames_with_ground": summary.get("frames_with_ground"),
              "frames_with_any_partial_clearance": sum(r["class_area_m2"]["partial_clearance"] > 0 for r in rows),
              "frames_with_any_observed_clearance": sum(r["class_area_m2"]["observed_clearance"] > 0 for r in rows),
              "native_frame": metadata.get("camera_reference_frame", "current gravity-aligned LiDAR coordinate frame"),
              "free_evidence": "measured LiDAR rays" if metadata["route"] == "lidar" else "predicted camera depth rays",
              "semantic_miou": None, "cross_sensor_voxel_iou": None,
              "config_sha256": metadata["config_sha256"], "footprint_scope": metadata.get("footprint_scope")}
    if metadata["route"] == "camera":
        for row in rows:
            depth = row.get("auxiliary", {}).get("camera_depth", {})
            if depth.get("input_lidar") is not False or depth.get("input_depth") is not False:
                raise ValueError("Camera-only model input provenance is missing or admits LiDAR/depth input")
        result["camera_temporal_registration"] = camera_segments(rows)
        result["camera_depth_seconds"] = distribution([r["auxiliary"]["camera_depth"]["runtime_ms"] / 1000 for r in rows])
        result["camera_cpu_fallback_frames"] = sum(bool(r["auxiliary"]["camera_depth"].get("fallback")) for r in rows)
    return result


def asset_url(path, report_directory):
    return quote(os.path.relpath(path, report_directory).replace(os.sep, "/"), safe="/")


def render_frame(run, row, report_directory, name):
    """Use actual states and Gaussian sparse voxels; do not infer missing frames."""
    index = row["index"]
    with np.load(run["directory"] / "traversability" / f"{index:06d}.npz", allow_pickle=False) as artifact:
        states = artifact["traversability"]
        resolution = float(artifact["resolution"])
        bounds = artifact["bounds"].astype(float)
    if (states.ndim != 2 or states.dtype.kind not in "iu" or np.any(states < 0) or np.any(states > 3)
            or not np.isfinite(resolution) or resolution <= 0 or bounds.shape != (6,) or not np.isfinite(bounds).all()):
        raise ValueError(f"Invalid actual traversability states at {name}/{index}")
    expected_shape = np.ceil((bounds[3:5] - bounds[:2]) / resolution).astype(int)
    if tuple(expected_shape) != states.shape:
        raise ValueError(f"Traversability shape disagrees with its native grid bounds at {name}/{index}")
    for code, class_name in CLASSES.items():
        observed_area = float(np.count_nonzero(states == code) * resolution ** 2)
        if abs(observed_area - row["class_area_m2"][class_name]) > 1e-5:
            raise ValueError(f"NPZ/log class area differs at {name}/{index}/{class_name}")
    # Grid coordinates are [x,y]. Pixel rows run down: transpose and flip y.
    state_image = PALETTE[states.T[::-1]]
    destination = report_directory / "assets" / name
    destination.mkdir(parents=True, exist_ok=True)
    state_path = destination / f"{index:06d}_traversability.png"
    Image.fromarray(state_image).save(state_path, optimize=True)
    with np.load(run["directory"] / "occupancy" / f"{index:06d}.npz", allow_pickle=False) as artifact:
        indices = artifact["fused_indices"]
        occupancy_bounds = artifact["bounds"].astype(float)
        voxel_size = float(artifact["voxel_size"])
        artifact_stamp = int(artifact["stamp_ns"])
    if artifact_stamp != row["stamp_ns"] or len(indices) != row["fused_occupied_voxels"]:
        raise ValueError(f"NPZ/log occupancy coverage differs at {name}/{index}")
    shape = np.ceil((occupancy_bounds[1] - occupancy_bounds[0]) / voxel_size).astype(int)
    if indices.ndim != 2 or indices.shape[1] != 3 or indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= shape):
        raise ValueError(f"Invalid sparse Gaussian indices at {name}/{index}")
    flat = (indices[:, 0].astype(np.int64) * shape[1] + indices[:, 1]) * shape[2] + indices[:, 2]
    if len(np.unique(flat)) != len(indices):
        raise ValueError(f"Gaussian occupied voxel indices contain duplicates at {name}/{index}")
    columns = np.zeros(tuple(shape[:2]), np.uint16)
    np.add.at(columns, (indices[:, 0], indices[:, 1]), 1)
    # Color = vertical occupied-voxel count, including floor and ceiling. This
    # is a full-volume projection, not the separate collision-only map.
    strength = np.clip(columns.astype(np.float32) / max(shape[2], 1), 0, 1)
    color = np.zeros((*columns.shape, 3), np.uint8)
    color[:] = [17, 24, 37]
    occupied = columns > 0
    color[occupied, 0] = (40 + 45 * strength[occupied]).astype(np.uint8)
    color[occupied, 1] = (100 + 120 * strength[occupied]).astype(np.uint8)
    color[occupied, 2] = (160 + 90 * strength[occupied]).astype(np.uint8)
    occupancy_path = destination / f"{index:06d}_occupancy.png"
    Image.fromarray(color.transpose(1, 0, 2)[::-1]).save(occupancy_path, optimize=True)
    return {"traversability": asset_url(state_path, report_directory),
            "occupancy": asset_url(occupancy_path, report_directory),
            "shape_xy": list(states.shape), "resolution_m": resolution,
            "bounds": bounds.tolist(), "vertical_occupied_count_max": int(columns.max())}


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ORBVI · TT-Occ 完整记录离线对比</title>
<style>
:root{color-scheme:dark;font-family:system-ui,"Microsoft YaHei",sans-serif;background:#0c1320;color:#e8eef6}
*{box-sizing:border-box}body{max-width:1480px;margin:auto;padding:22px}h1{font-size:24px;margin:0 0 10px}p{line-height:1.65;margin:10px 0}.muted{color:#a8b7ca;font-size:13px}
.note{background:#172234;border-left:3px solid #60c7e2;padding:12px 15px}.controls{position:sticky;top:0;background:#0c1320ee;z-index:5;padding:16px 0;backdrop-filter:blur(10px)}
.controlrow{display:flex;gap:10px;align-items:center;flex-wrap:wrap}button,input,select{accent-color:#51bce6}button,select,input[type=number]{background:#24344d;color:white;border:1px solid #4a6481;border-radius:6px;padding:7px 11px}input[type=range]{width:100%;margin-top:12px}input[type=number]{width:100px}
.routes{display:grid;grid-template-columns:1fr 1fr;gap:18px}.panel{background:#152032;border:1px solid #30415a;border-radius:10px;padding:16px}.panel h2{margin:0 0 10px;font-size:19px}.maps{display:grid;grid-template-columns:1fr 1fr;gap:10px}.map img{width:100%;aspect-ratio:1;image-rendering:pixelated;border:1px solid #40546b}.caption{font-size:12px;color:#b9c7d7;margin:5px 0}
.metrics{font-size:13px;line-height:1.9;margin-top:12px}.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;margin:14px 0}.legend span:before{content:"";display:inline-block;width:12px;height:12px;background:var(--color);margin-right:5px;border-radius:2px}.cameras{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:18px}.cameras img{width:100%;height:auto;border-radius:6px;border:1px solid #3c5069}.cameras figure{margin:0}.cameras figcaption{font-size:12px;color:#b9c7d7;margin:6px 0}canvas{width:100%;height:100px;background:#111c2d;border:1px solid #30415a;border-radius:8px}.links a{color:#82d5f2;margin-right:18px}
@media(max-width:900px){.routes{grid-template-columns:1fr}.cameras{grid-template-columns:1fr 1fr}body{padding:12px}}
</style>
<h1>ORBVI · TT-Occ 完整记录离线对比</h1>
<p class="muted">__FRAME_COUNT__ 个共同采集时间步 · __DURATION__ 秒 · 全部地图来自实际输出 NPZ；页面及图片均从本地读取。</p>
<div class="note">两路线显示各自传感器的局部坐标。设备 LiDAR—相机外参尚未验收，不能把相同格点当作同一个三维位置，也不能由这些图计算跨传感器 mIoU。相机净空来自预测深度射线。__CONTINUITY__</div>
<div class="controls"><div class="controlrow"><button id="previous">上一帧</button><button id="next">下一帧</button><button id="play">播放</button><label>帧 <input id="index" type="number" min="0" max="__MAX_FRAME__" value="0"></label><span id="stamp"></span><label>播放 <select id="fps"><option value="2">2 fps</option><option value="5" selected>5 fps</option><option value="10">10 fps</option></select></label></div><input id="slider" type="range" min="0" max="__MAX_FRAME__" value="0" step="1"></div>
<canvas id="latency" width="1280" height="140" aria-label="两路线每帧处理时间，白线表示当前帧"></canvas><p class="caption">逐帧处理时间：蓝色为 LiDAR，黄色为 Camera；白线为当前帧。它包含模型、占据融合、通行转换与本地结果写入。</p>
<div class="routes"><section class="panel"><h2>LiDAR 路线</h2><div class="maps"><div class="map"><img id="lidarOcc" alt="实际 LiDAR Gaussian 占据竖直列投影"><p class="caption">完整 Gaussian 占据的竖直列投影</p></div><div class="map"><img id="lidarMap" alt="实际 LiDAR 局部通行状态"><p class="caption">局部通行状态</p></div></div><div class="metrics" id="lidarMetrics"></div></section>
<section class="panel"><h2>Camera 路线</h2><div class="maps"><div class="map"><img id="cameraOcc" alt="实际 Camera Gaussian 占据竖直列投影"><p class="caption">完整 Gaussian 占据的竖直列投影</p></div><div class="map"><img id="cameraMap" alt="实际 Camera 局部通行状态"><p class="caption">局部通行状态</p></div></div><div class="metrics" id="cameraMetrics"></div></section></div>
<div class="legend"><span style="--color:#242f40">未知</span><span style="--color:#eb6067">阻塞</span><span style="--color:#f2bb51">部分净空候选</span><span style="--color:#3fbe95">实测完整净空</span></div>
<p class="caption">图横轴为局部 x，纵轴为局部 y，范围与分辨率显示在帧指标中。占据投影的蓝色越亮，表示该竖直列含更多 occupied 体素；它包含地面、天花板等完整几何。未知格点没有足够完整净空证据。</p>
<h2 style="font-size:19px;margin-top:22px">原始四相机图像</h2><p class="caption">使用对应时间步的实际因果同步图像；标题记录相机 ID 与相对于 LiDAR 采集锚点的时间差。点击图像可打开原始 JPEG。</p>
<div class="cameras" id="cameras"></div>
<p class="muted">采样尺寸为配置中的演示 footprint，尚未由真实车辆尺寸验收。该页面用于离线工程验证与检查，完整记录处理验收和场景几何精度分别报告。</p>
<p class="links"><a href="comparison.md">中文对比报告</a><a href="comparison.json">结构化对比指标</a></p>
<script>
const frames=__FRAME_DATA__;
const canvas=document.getElementById('latency'),ctx=canvas.getContext('2d');
let current=0,timer=null;
const fmt=(v,n=2)=>Number(v).toFixed(n);
function timeline(){ctx.clearRect(0,0,canvas.width,canvas.height);const max=Math.max(.1,...frames.map(f=>Math.max(f.lidar.latency,f.camera.latency)));for(const route of ['lidar','camera']){ctx.strokeStyle=route==='lidar'?'#65cbed':'#f2bb51';ctx.lineWidth=1.5;ctx.beginPath();frames.forEach((f,i)=>{const x=i/(frames.length-1)*(canvas.width-60)+50,y=canvas.height-20-f[route].latency/max*(canvas.height-40);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()}ctx.fillStyle='#aebcd0';ctx.font='14px sans-serif';ctx.fillText(fmt(max,1)+' s',3,16);ctx.fillText('0 s',3,canvas.height-12);const x=current/(frames.length-1)*(canvas.width-60)+50;ctx.strokeStyle='white';ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,canvas.height);ctx.stroke()}
function metrics(route){const d=frames[current][route];let message=`当前占据 ${d.current.toLocaleString()}，过去帧融合 ${d.fused.toLocaleString()}，增加 ${d.gain.toLocaleString()} 格 · 处理 ${fmt(d.latency,3)} s<br>未知 ${fmt(d.area.unknown)} m²；阻塞 ${fmt(d.area.blocked)} m²；部分净空候选 ${fmt(d.area.partial_clearance)} m²；实测完整净空 ${fmt(d.area.observed_clearance)} m²<br>网格 ${d.assets.shape_xy.join(' × ')}，${fmt(d.assets.resolution_m,2)} m/格；x [${d.assets.bounds[0]}, ${d.assets.bounds[3]}] m，y [${d.assets.bounds[1]}, ${d.assets.bounds[4]}] m`;
if(route==='camera'){message+=`<br>Camera-only ICP segment ${d.segment}；本步注册 ${d.registration_accepted?'通过':'未通过，已清空融合历史'}`;if(d.depth_latency!==null)message+=`；真实深度推理 ${fmt(d.depth_latency,3)} s`}
document.getElementById(route+'Metrics').innerHTML=message}
function show(i){current=Math.max(0,Math.min(frames.length-1,Math.round(i)));document.getElementById('slider').value=current;document.getElementById('index').value=current;document.getElementById('stamp').textContent=`t = ${fmt(frames[current].time_seconds,3)} s / ${fmt(frames.at(-1).time_seconds,3)} s`;
for(const route of ['lidar','camera']){const a=frames[current][route].assets;document.getElementById(route+'Occ').src=a.occupancy;document.getElementById(route+'Map').src=a.traversability;metrics(route)}
const cameras=document.getElementById('cameras');cameras.replaceChildren();for(const camera of frames[current].cameras){const figure=document.createElement('figure'),link=document.createElement('a'),im=document.createElement('img'),caption=document.createElement('figcaption');im.src=camera.url;im.alt='原始 camera '+camera.id;link.href=camera.url;link.target='_blank';link.appendChild(im);caption.textContent=`camera_${camera.id} · Δt ${fmt(camera.delta_ms,1)} ms`;figure.append(link,caption);cameras.appendChild(figure)}timeline()}
function stop(){if(timer)clearInterval(timer);timer=null;document.getElementById('play').textContent='播放'}
document.getElementById('slider').addEventListener('input',e=>show(Number(e.target.value)));document.getElementById('index').addEventListener('change',e=>show(Number(e.target.value)));document.getElementById('previous').onclick=()=>{stop();show(current-1)};document.getElementById('next').onclick=()=>{stop();show(current+1)};document.getElementById('play').onclick=()=>{if(timer){stop();return}document.getElementById('play').textContent='暂停';timer=setInterval(()=>{if(current===frames.length-1){stop();return}show(current+1)},1000/Number(document.getElementById('fps').value))};document.getElementById('fps').onchange=()=>{if(timer){stop();document.getElementById('play').click()}};window.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;if(e.key==='ArrowRight'){stop();show(current+1)}if(e.key==='ArrowLeft'){stop();show(current-1)}});show(0);
</script></html>'''


def create_gallery(runs, source_frames, prepared, report_directory, comparison):
    records = []
    first_stamp = source_frames[0]["stamp_ns"]
    for index, source in enumerate(source_frames):
        record = {"index": index, "stamp_ns": str(source["stamp_ns"]),
                  "time_seconds": (source["stamp_ns"] - first_stamp) / 1e9, "cameras": []}
        for camera_id in ["3", "2", "4", "5"]:
            camera = source["images"].get(camera_id)
            if not camera or not camera.get("valid_sync") or camera["stamp_ns"] > source["stamp_ns"]:
                raise ValueError(f"Missing valid causal source camera_{camera_id} at frame {index}")
            source_image = prepared / camera["path"]
            if not source_image.is_file():
                raise ValueError(f"Missing original source JPEG: {source_image}")
            record["cameras"].append({"id": camera_id, "url": asset_url(source_image, report_directory),
                                      "delta_ms": (camera["stamp_ns"] - source["stamp_ns"]) / 1e6})
        for name in ["lidar", "camera"]:
            row = runs[name]["rows"][index]
            assets = render_frame(runs[name], row, report_directory, name)
            motion = row.get("auxiliary", {}).get("camera_motion", {})
            depth = row.get("auxiliary", {}).get("camera_depth", {})
            record[name] = {"current": row["current_occupied_voxels"], "fused": row["fused_occupied_voxels"],
                            "gain": row["fused_occupied_voxels"] - row["current_occupied_voxels"],
                            "area": row["class_area_m2"], "latency": row["processing_seconds"], "assets": assets,
                            "segment": motion.get("segment_id", 0), "registration_accepted": motion.get("accepted"),
                            "depth_latency": depth["runtime_ms"] / 1000 if "runtime_ms" in depth else None}
        records.append(record)
        if index % 250 == 0 or index == len(source_frames) - 1:
            print(f"Gallery verified/rendered {index + 1}/{len(source_frames)} actual frame pairs", flush=True)
    temporal = comparison["routes"]["camera"]["camera_temporal_registration"]
    continuity = (f"相机 ICP 出现 {temporal['rejected_registration_count']} 次注册拒绝，记录分为 "
                  f"{temporal['independent_segment_count']} 个独立轨迹段；这些段之间尚未配准。"
                  if temporal["rejected_registration_count"] else
                  "相机本次记录的 ICP 质量门控持续通过；累计轨迹仍包含预测深度与注册误差。")
    serialized = json.dumps(records, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    rendered = (HTML_TEMPLATE.replace("__FRAME_COUNT__", str(len(records)))
                .replace("__MAX_FRAME__", str(len(records) - 1))
                .replace("__DURATION__", f"{comparison['recording_sensor_duration_seconds']:.3f}")
                .replace("__CONTINUITY__", html.escape(continuity)).replace("__FRAME_DATA__", serialized))
    (report_directory / "gallery.html").write_text(rendered)
    return {"path": "gallery.html", "actual_frame_pairs": len(records),
            "generated_pngs": 4 * len(records), "external_cdn": False,
            "original_images_copied": False, "source_images": "relative links to prepared/images",
            "native_npz_states_crosschecked_against_logs": True}


def markdown_report(comparison):
    lidar, camera = comparison["routes"]["lidar"], comparison["routes"]["camera"]
    temporal = camera["camera_temporal_registration"]
    def scalar(value, precision=2):
        return "完整运行未记录" if value is None else f"{value:,.{precision}f}"
    lines = ["# ORBVI 完整记录：TT-Occ 两路线对比", "",
             f"两路线均处理并验收 **{comparison['frames']} / {comparison['frames']}** 个相同采集时间步，"
             f"覆盖传感器时间 **{comparison['recording_sensor_duration_seconds']:.3f} 秒**。"
             "下述数值来自完整运行日志及实际 NPZ 输出。", "",
             "| 指标 | LiDAR 路线 | Camera 路线 |", "|---|---:|---:|",
             f"| 完整性验收 | {lidar['frames']} / {comparison['frames']} | {camera['frames']} / {comparison['frames']} |",
             f"| 每帧处理时间中位数 / P95（秒） | {lidar['latency_seconds']['median']:.3f} / {lidar['latency_seconds']['p95']:.3f} | {camera['latency_seconds']['median']:.3f} / {camera['latency_seconds']['p95']:.3f} |",
             f"| 逐帧处理时间合计（秒） | {lidar['sum_frame_processing_seconds']:.1f} | {camera['sum_frame_processing_seconds']:.1f} |",
             f"| 处理循环采样峰值 RSS（MiB） | {scalar(lidar['peak_loop_rss_mib'])} | {scalar(camera['peak_loop_rss_mib'])} |",
             f"| 完整运行记录的 Torch GPU 分配峰值（MiB） | {scalar(lidar['peak_torch_allocated_gpu_mib'])} | {scalar(camera['peak_torch_allocated_gpu_mib'])} |",
             f"| 当前 / 融合占据体素平均数 | {lidar['current_occupied_voxels']['mean']:,.0f} / {lidar['fused_occupied_voxels']['mean']:,.0f} | {camera['current_occupied_voxels']['mean']:,.0f} / {camera['fused_occupied_voxels']['mean']:,.0f} |",
             f"| 过去帧增加体素平均数 | {lidar['extra_occupied_voxels_from_past']['mean']:,.0f} | {camera['extra_occupied_voxels_from_past']['mean']:,.0f} |",
             f"| 过去帧体素增量比例平均值 | {100 * lidar['past_voxel_gain_fraction']['mean']:.1f}% | {100 * camera['past_voxel_gain_fraction']['mean']:.1f}% |",
             f"| 存在部分净空候选的时间步数 | {lidar['frames_with_any_partial_clearance']} | {camera['frames_with_any_partial_clearance']} |",
             f"| 存在实测完整净空的时间步数 | {lidar['frames_with_any_observed_clearance']} | {camera['frames_with_any_observed_clearance']} |", "",
             "峰值 RSS 采用运行循环中的进程采样，模型加载阶段未单独测峰值。Torch GPU 分配量属于对应进程，"
             "不包含桌面和其他用户进程，也不等于驱动报告的全部显存。时间合计包含模型、融合、通行转换和结果写入，"
             "多进程并发时不能直接视为独占设备吞吐率。时间融合体素增量反映输出覆盖变化，"
             "也会受深度和位姿噪声影响，不能等同几何精度提升。", "",
             "| 各自局部坐标内的平均状态面积（m²） | LiDAR 路线 | Camera 路线 |", "|---|---:|---:|"]
    for class_name in CLASSES.values():
        lines.append(f"| {CLASS_ZH[class_name]} | {lidar['native_class_area_m2'][class_name]['mean']:.2f} | {camera['native_class_area_m2'][class_name]['mean']:.2f} |")
    lines += ["", "状态面积表示各路线原生局部网格中的输出覆盖情况。两传感器原点尚未通过设备外参配准，"
              "这些面积不能解释成同一位置的预测正确率。完整 Gaussian 占据保留地面、天花板等几何；"
              "局部通行产品使用独立的地面、障碍膨胀和净空证据逻辑。", "",
              f"Camera 深度推理中位数为 **{camera['camera_depth_seconds']['median']:.3f} 秒**；"
              f"CPU fallback 记录数为 **{camera['camera_cpu_fallback_frames']}**。"
              "其网络只接收 RGB、相机内参和固定四相机 rig 外参，时间融合轨迹由预测深度 ICP 建立。", ""]
    if temporal["rejected_registration_count"]:
        lines += [f"相机 ICP 共拒绝 **{temporal['rejected_registration_count']}** 个时间步的注册，"
                  f"记录分为 **{temporal['independent_segment_count']}** 个独立轨迹段。每次拒绝均清空历史；"
                  "轨迹段之间没有有效配准，不能把这些段拼接成连续全局地图。", ""]
    else:
        lines += ["相机 ICP 的本次记录质量门控持续通过，未触发历史重置。累计轨迹仍是预测深度的注册结果，"
                  "没有独立全局位姿真值验证。", ""]
    if temporal["independent_segment_count"] <= 25:
        lines += ["| 相机独立段 | 首帧—末帧 | 时间步数 | 覆盖秒数 |", "|---:|---:|---:|---:|"]
        for segment in temporal["segments"]:
            lines.append(f"| {segment['segment_id']} | {segment['first_frame']}—{segment['last_frame']} | {segment['frames']} | {segment['duration_seconds']:.3f} |")
        lines.append("")
    else:
        lines += ["所有轨迹段的起止索引与长度见 comparison.json 中的 camera_temporal_registration.segments。", ""]
    lines += ["设备 LiDAR—相机外参尚未验收，当前没有用于语义或占据准确率评价的标注真值。"
              "本报告不计算跨传感器逐体素 IoU / mIoU，也不将两路线按同一个原点逐点叠加。"
              "LiDAR 空闲证据来自实测射线；Camera 空闲证据来自预测深度射线，两者证据来源不同。"
              "演示 footprint 可配置，尚未用真实车辆尺寸完成验收。", "",
              "[打开全部时间步的离线交互页面](gallery.html)；[查看结构化完整指标](comparison.json)。", "",
              f"原始 bag SHA256：`{comparison['source_bag_sha256']}`。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lidar-run", type=Path, default=PROJECT / "runs/lidar_full")
    parser.add_argument("--camera-run", type=Path, default=PROJECT / "runs/camera_full")
    parser.add_argument("--prepared", type=Path, default=PROJECT / "prepared")
    parser.add_argument("--output", type=Path, default=PROJECT / "reports")
    args = parser.parse_args()
    prepared = args.prepared.resolve()
    source_manifest = read_json(prepared / "manifest.json")
    source_frames = read_jsonl(prepared / "frames.jsonl")
    expected = source_manifest["counts"]["/orbvi/lidar/custom"]
    if source_manifest.get("complete") is not True or len(source_frames) != expected or expected != 2348:
        raise ValueError("This report requires the complete original 2348-scan ORBVI recording")
    if [f["index"] for f in source_frames] != list(range(expected)):
        raise ValueError("Prepared original acquisition index is not contiguous")
    source_hash = source_manifest["source_bag_sha256"]
    runs = {"lidar": validated_run(args.lidar_run, expected, source_frames, source_hash),
            "camera": validated_run(args.camera_run, expected, source_frames, source_hash)}
    if runs["lidar"]["metadata"]["route"] != "lidar" or runs["camera"]["metadata"]["route"] != "camera":
        raise ValueError("Route directory arguments were swapped or metadata is invalid")
    comparison = {"complete": True, "generated_utc": datetime.now(timezone.utc).isoformat(),
                  "frames": expected, "source_bag_sha256": source_hash,
                  "recording_sensor_duration_seconds": (source_frames[-1]["stamp_ns"] - source_frames[0]["stamp_ns"]) / 1e9,
                  "timestamp_basis": source_manifest.get("timestamp_basis"),
                  "routes": {name: route_metrics(run) for name, run in runs.items()},
                  "cross_sensor_alignment": {"available": False, "same_voxel_origin_assumed": False,
                                             "semantic_miou_computed": False, "voxel_iou_computed": False,
                                             "reason": "Device-specific LiDAR-camera extrinsic is unverified; labelled occupancy GT is unavailable"},
                  "report_scope": "whole recording engineering completeness, resource use, native geometry/temporal fusion and local state coverage"}
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    comparison["gallery"] = create_gallery(runs, source_frames, prepared, output, comparison)
    # Final acceptance/report files are written only after all actual artifacts
    # were read, timestamp-checked, and crosschecked against the complete logs.
    write_json(output / "comparison.json", comparison)
    (output / "comparison.md").write_text(markdown_report(comparison))
    print(json.dumps({"complete": True, "frames": expected, "report_directory": str(output),
                      "gallery_pngs": comparison["gallery"]["generated_pngs"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as error:
        print(f"Comparison refused: {error}", file=sys.stderr)
        raise SystemExit(2)
