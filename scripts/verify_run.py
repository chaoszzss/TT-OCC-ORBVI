#!/usr/bin/env python3
"""Independently verify complete ORBVI TT-Occ run artifacts; incomplete runs exit 2."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
EXPECTED_FRAMES = 2348
EXPECTED_RAW_POINTS = 46_929_024


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant {value}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_constant)


def read_jsonl(path):
    return [json.loads(line, parse_constant=_reject_constant)
            for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def verify(run_dir, prepared_dir=None, expected_frames=EXPECTED_FRAMES,
           expected_raw_points=EXPECTED_RAW_POINTS):
    """Read every saved frame without trusting runner's `complete` field."""
    run_dir = Path(run_dir).resolve()
    prepared = Path(prepared_dir or PROJECT / "prepared").resolve()
    checks, errors = [], []
    counts = Counter()

    def check(name, ok, detail=None):
        checks.append({"name": name, "passed": bool(ok), "detail": detail})

    def fail(frame, message):
        counts["artifact_errors"] += 1
        if len(errors) < 128:
            errors.append({"frame": frame, "error": str(message)})

    report = {"run_directory": str(run_dir), "prepared_directory": str(prepared),
              "expected_frames": expected_frames, "expected_raw_points": expected_raw_points,
              "checks": checks, "artifact_errors": errors}
    try:
        manifest = read_json(prepared / "manifest.json")
        source_frames = read_jsonl(prepared / "frames.jsonl")
        metadata = read_json(run_dir / "run.json")
        rows = read_jsonl(run_dir / "frames.jsonl")
        summary = read_json(run_dir / "summary.json")
        progress = read_json(run_dir / "progress.json")
    except (OSError, ValueError, KeyError) as exc:
        check("required_valid_json_documents", False, str(exc))
        report.update(accepted=False, statistics=dict(counts))
        return report
    route = metadata.get("route")
    report["route"] = route
    source_ids = [r["index"] for r in source_frames]
    ids = [r.get("index") for r in rows]
    source_stamps = [r["stamp_ns"] for r in source_frames]
    stamps = [r.get("stamp_ns") for r in rows]
    source_points = sum(int(f.get("point_count", 0)) for f in source_frames)
    accounted_points = sum(int(r.get("raw_source_points", 0)) for r in rows)
    check("source_manifest_complete", manifest.get("complete") is True)
    check("expected_source_frame_count", len(source_frames) == expected_frames,
          {"manifest": len(source_frames), "expected": expected_frames})
    check("run_expected_full_frame_count", metadata.get("expected_full_recording_frames") == expected_frames)
    check("source_index_contiguous", source_ids == list(range(expected_frames)))
    check("source_raw_point_accounting", source_points == expected_raw_points == manifest.get("lidar_point_count"),
          {"source_frame_sum": source_points, "manifest_sum": manifest.get("lidar_point_count"), "expected": expected_raw_points})
    check("all_recording_frames_logged", ids == list(range(expected_frames)),
          {"recorded": len(rows), "unique": len(set(ids)), "expected": expected_frames})
    check("sensor_timestamps_exactly_match_manifest", stamps == source_stamps,
          {"first_source": source_stamps[0] if source_stamps else None,
           "last_source": source_stamps[-1] if source_stamps else None,
           "first_result": stamps[0] if stamps else None, "last_result": stamps[-1] if stamps else None})
    check("sensor_timestamps_strictly_increase", bool(stamps) and all(a < b for a, b in zip(stamps, stamps[1:])))
    check("run_bag_fingerprint", metadata.get("bag_sha256") == manifest.get("source_bag_sha256"))
    config = metadata.get("config", {})
    provenance = metadata.get("input_provenance_sha256")
    fingerprint_input = {"config": config, "inputs": provenance} if provenance is not None else config
    fingerprint = hashlib.sha256(json.dumps(fingerprint_input, sort_keys=True).encode()).hexdigest()
    check("run_config_fingerprint", metadata.get("config_sha256") == fingerprint)
    if provenance is not None:
        mismatches, archived, unused = [], [], []
        for name, digest in provenance.items():
            path = Path(name)
            if not path.is_absolute():
                path = PROJECT / path
            snapshot = run_dir / "executed_source" / name if not Path(name).is_absolute() else None
            if snapshot is not None and snapshot.is_file() and hashlib.sha256(snapshot.read_bytes()).hexdigest() == digest:
                archived.append(name)
            elif route == "lidar" and name == "orbvi_occ/camera_depth.py":
                # This module is imported only inside the camera-route branch.
                # Earlier LiDAR fingerprints included all package files, even
                # unused modules; do not call a later camera-only fix a change
                # to the executed LiDAR Gaussian/safety implementation.
                unused.append(name)
            elif not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                mismatches.append(name)
        check("code_calibration_model_provenance_unchanged", not mismatches, mismatches)
        report["source_provenance"] = {"archived_executed_modules_matching_recorded_hash": archived,
                                      "route_unused_recorded_modules": unused,
                                      "changed_executed_or_input_files": mismatches,
                                      "fingerprint_uses_original_recorded_metadata": True}
    check("full_run_not_development_limit", metadata.get("limited_development_run") is False)
    check("runner_summary_complete", summary.get("complete") is True and summary.get("acceptance", {}).get("accepted") is True,
          "summary.json must contain runner acceptance; plot metadata must never overwrite it")
    check("runner_summary_frame_count", summary.get("acceptance", {}).get("recorded_frames") == expected_frames)
    check("runner_progress_complete", progress.get("complete") is True and progress.get("processed_frames") == expected_frames)
    check("full_raw_point_accounting", accounted_points == expected_raw_points == summary.get("source_lidar_points_accounted") == summary.get("source_bag_lidar_points"),
          {"frame_log_sum": accounted_points, "summary_sum": summary.get("source_lidar_points_accounted")})
    check("route_consistency", route in {"lidar", "camera"} and summary.get("route") == route)
    if route == "lidar":
        try:
            registration = read_json(prepared / "trajectory_report.json")
            check("complete_pose_registration_report", registration.get("complete") is True and registration.get("frame_count") == expected_frames)
            report["pose_registration"] = {k: registration.get(k) for k in ("method", "uses_vio", "uses_vio_translation", "uses_vio_scale")}
            report["pose_registration"]["rotation_validation"] = {k: registration.get("vio_rotation_report", {}).get(k) for k in ("validation_median_deg", "validation_p95_deg")}
        except (OSError, ValueError) as exc:
            check("complete_pose_registration_report", False, str(exc))
    check("no_unavailable_semantic_ground_truth_score", summary.get("semantic_miou") is None)
    if route == "lidar" and metadata.get("device_cross_sensor_calibration_valid") is not True:
        check("unavailable_lidar_camera_calibration_disclosed", summary.get("semantic_lidar_association_valid") is False)
    expected_names = {f"{i:06d}.npz" for i in range(expected_frames)}
    for kind in ("occupancy", "traversability"):
        names = {p.name for p in (run_dir / kind).glob("*.npz")}
        check(f"complete_{kind}_artifact_set", names == expected_names,
              {"found": len(names), "missing_count": len(expected_names - names), "extra_count": len(names - expected_names)})
    class_totals = Counter()
    for frame in source_frames:
        i = frame["index"]
        if i >= len(rows) or rows[i].get("index") != i:
            continue
        row = rows[i]
        try:
            with np.load(prepared / frame["lidar_path"], allow_pickle=False) as source:
                if len(source["points"]) != frame["point_count"] or int(source["stamp_ns"]) != frame["stamp_ns"]:
                    raise ValueError("prepared raw scan count or timestamp differs from manifest")
                valid_source_returns = int(np.count_nonzero(np.isfinite(source["points"]).all(axis=1) & (np.linalg.norm(source["points"], axis=1) >= 0.2)))
                if row["raw_source_points"] != frame["point_count"]:
                    raise ValueError("raw source point count differs at this frame")
                if row.get("invalid_raw_lidar_returns") != frame["point_count"] - valid_source_returns:
                    raise ValueError("invalid/zero source return accounting differs at this frame")
                if route == "lidar" and row["route_input_points"] != valid_source_returns:
                    raise ValueError("LiDAR construction includes invalid/zero returns or omits valid source returns")
                counts["invalid_raw_lidar_returns"] += frame["point_count"] - valid_source_returns
                counts["valid_raw_lidar_returns"] += valid_source_returns
            counts["checked_source_scans"] += 1
            with np.load(run_dir / "occupancy" / f"{i:06d}.npz", allow_pickle=False) as d:
                bounds = np.asarray(d["bounds"], dtype=float)
                resolution = float(d["voxel_size"])
                pose = np.asarray(d["T_world_sensor"], dtype=float)
                if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]) or not np.isfinite(resolution) or resolution <= 0:
                    raise ValueError("invalid occupancy bounds/resolution")
                if int(d["stamp_ns"]) != frame["stamp_ns"]:
                    raise ValueError("occupancy acquisition timestamp mismatch")
                if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5):
                    raise ValueError("non-finite or malformed sensor pose")
                if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-3):
                    raise ValueError("pose rotation is not a proper rigid rotation")
                shape = np.ceil((bounds[1] - bounds[0]) / resolution).astype(int)
                for kind in ("current", "fused"):
                    idx = d[kind + "_indices"]
                    labels = d[kind + "_labels"]
                    if idx.ndim != 2 or idx.shape[1] != 3 or not np.issubdtype(idx.dtype, np.integer):
                        raise ValueError(f"{kind} indices must be integer Nx3")
                    if np.any(idx < 0) or np.any(idx >= shape) or len(np.unique(idx, axis=0)) != len(idx):
                        raise ValueError(f"{kind} voxel index out of bounds or duplicated")
                    if labels.shape != (len(idx),) or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0) or np.any(labels > 255):
                        raise ValueError(f"{kind} semantic labels malformed")
                    if len(idx) != row[kind + "_occupied_voxels"]:
                        raise ValueError(f"{kind} count disagrees with JSON frame log")
                    if route == "lidar" and metadata.get("device_cross_sensor_calibration_valid") is not True and np.any(labels != 255):
                        raise ValueError("LiDAR labels claim calibrated semantics without available extrinsics")
                if row["new_voxels_from_history"] != row["fused_occupied_voxels"] - row["current_occupied_voxels"]:
                    raise ValueError("time-fusion new-voxel count differs from actual outputs")
                algorithm = row.get("algorithm", {})
                max_history = config.get("ttocc", {}).get("max_history_frames", 0)
                if algorithm.get("past_frames", 0) > min(i, max_history):
                    raise ValueError("temporal history references impossible future/extra frames")
                fused_idx = d["fused_indices"].copy()
                centres = bounds[0] + (fused_idx + 0.5) * resolution
            counts["valid_occupancy_artifacts"] += 1
            with np.load(run_dir / "traversability" / f"{i:06d}.npz", allow_pickle=False) as d:
                b = np.asarray(d["bounds"], float)
                r = float(d["resolution"])
                if b.shape != (6,) or not np.isfinite(b).all() or not np.isfinite(r) or r <= 0 or np.any(b[3:] <= b[:3]):
                    raise ValueError("invalid traversability bounds/resolution")
                grid_shape = tuple(np.ceil((b[3:] - b[:3]) / r).astype(int))
                free, occ, collision = d["observed_free"], d["occupied3d"], d["collision3d"]
                for name, mask in (("observed_free", free), ("occupied3d", occ), ("collision3d", collision)):
                    if mask.shape != grid_shape or mask.dtype != np.bool_:
                        raise ValueError(f"{name} grid shape/type mismatch")
                if np.any(free & collision):
                    raise ValueError("collision and observed free are not mutually exclusive")
                if np.any(collision & ~occ):
                    raise ValueError("collision occupancy contains geometry absent from TT occupancy")
                expected_occ = np.zeros(grid_shape, bool)
                mapped = np.floor((centres - b[:3]) / r).astype(int)
                inside = np.all((mapped >= 0) & (mapped < np.asarray(grid_shape)), axis=1)
                if inside.any():
                    expected_occ[tuple(mapped[inside].T)] = True
                if not np.array_equal(expected_occ, occ):
                    raise ValueError("saved TT occupied3d differs from saved TT fused indices")
                trav = d["traversability"]
                if trav.shape != grid_shape[:2] or not np.issubdtype(trav.dtype, np.integer) or np.any(trav > 3) or np.any(trav < 0):
                    raise ValueError("traversability classes/grid invalid")
                if int(trav.size) != row["traversability_cells"]:
                    raise ValueError("traversability output count differs from log")
                coverage = d["clearance_coverage"]
                if coverage.shape != trav.shape or not np.isfinite(coverage).all() or np.any(coverage < 0) or np.any(coverage > 1 + 1e-6):
                    raise ValueError("non-finite or out-of-range clearance coverage")
                if abs(float(d["timestamp"]) - frame["stamp_ns"] / 1e9) > 1e-6:
                    raise ValueError("traversability timestamp mismatch")
                plane = d["ground_plane"]
                if plane.shape != (4,) or (row["ground"].get("valid") and not np.isfinite(plane).all()):
                    raise ValueError("valid ground model has non-finite plane")
                if route == "camera" and np.any(trav == 3):
                    raise ValueError("predicted camera depth incorrectly certifies measured free clearance")
                for value, name in enumerate(("unknown", "blocked", "partial_clearance", "observed_clearance")):
                    n = int(np.count_nonzero(trav == value))
                    class_totals[name] += n
                    if not np.isclose(n * r * r, row["class_area_m2"][name], atol=1e-5):
                        raise ValueError("class areas disagree between NPZ and JSON log")
                counts["ground_supported_cells"] += int(np.count_nonzero(d["ground_support"]))
                counts["supported_partial_ray_evidence_cells"] += int(np.count_nonzero(d["ground_support"] & (coverage >= config.get("traversability", {}).get("min_partial_coverage", 0.2))))
            counts["valid_traversability_artifacts"] += 1
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            fail(i, exc)
    check("all_frame_artifacts_valid", counts["artifact_errors"] == 0 and counts["valid_traversability_artifacts"] == expected_frames,
          dict(counts))
    if rows:
        for name in ("unknown", "blocked", "partial_clearance", "observed_clearance"):
            values = [r.get("class_area_m2", {}).get(name, np.nan) for r in rows]
            declared = summary.get("class_area_m2", {}).get(name, {})
            check(f"summary_{name}_mean_matches_frames", np.isfinite(values).all() and np.isclose(declared.get("mean", np.nan), np.mean(values), atol=1e-5))
        latencies = np.asarray([r.get("processing_seconds", np.nan) for r in rows], float)
        check("finite_nonnegative_frame_latencies", np.isfinite(latencies).all() and np.all(latencies >= 0))
        check("summary_latency_matches_frames", np.isclose(summary.get("latency_seconds", {}).get("median", np.nan), np.median(latencies), atol=1e-6))
        check("summary_ground_valid_count", summary.get("frames_with_ground") == sum(bool(r["ground"].get("valid")) for r in rows))
    try:
        metric_rows = read_jsonl(run_dir / "heldout_metrics.jsonl")
        metric_ids = [r["index"] for r in metric_rows]
        check("heldout_metrics_unique", len(set(metric_ids)) == len(metric_ids))
        check("heldout_metrics_persisted_summary_matches", metric_rows == summary.get("heldout_metrics"))
        heldout = summary.get("heldout_geometry", {})
        held_returns = sum(r["holdout_returns"] for r in metric_rows)
        held_hits = sum(r["endpoint_hits"] for r in metric_rows)
        expected_rate = held_hits / held_returns if held_returns else None
        check("heldout_metrics_accumulator_matches", heldout.get("frames") == len(metric_rows)
              and heldout.get("holdout_returns") == held_returns and heldout.get("endpoint_hits") == held_hits
              and heldout.get("weighted_endpoint_hit_rate") == expected_rate)
        if route == "lidar":
            stride = config.get("evaluation", {}).get("holdout_stride", 50)
            expected_metric_ids = set(range(0, expected_frames, stride)) | {expected_frames - 1}
            check("scheduled_independent_holdout_complete", expected_metric_ids.issubset(set(metric_ids)))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        check("persisted_holdout_metrics_valid", False, str(exc))
    try:
        cameras = read_json(prepared / "camera_index.json")
        for cid in ("2", "3", "4", "5"):
            entries = cameras[cid]
            expected = manifest["counts"][f"/orbvi/raw/camera_{cid}/image/compressed"]
            check(f"all_raw_camera_{cid}_images_preserved", len(entries) == expected and all((prepared / e["path"]).is_file() for e in entries),
                  {"source_raw_images": expected, "indexed_images": len(entries)})
    except (OSError, ValueError, KeyError) as exc:
        check("raw_camera_extraction_complete", False, str(exc))
    if config.get("semantics", {}).get("enabled"):
        expected_semantics = sum(len(f["images"]) for f in source_frames)
        maps = list((run_dir / "semantic_maps").glob("*.npz"))
        check("all_anchor_pixel_semantic_maps_saved", len(maps) == expected_semantics,
              {"maps": len(maps), "expected": expected_semantics})
        semantic_failures = []
        for f in source_frames:
            for cid, image in f["images"].items():
                try:
                    with np.load(run_dir / "semantic_maps" / f"{f['index']:06d}_{cid}.npz", allow_pickle=False) as d:
                        labels = d["labels"]
                        if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0) or np.any(labels >= metadata["semantics"]["class_count"]):
                            raise ValueError("pixel semantic labels invalid")
                        if int(d["source_stamp_ns"]) != image["stamp_ns"]:
                            raise ValueError("pixel semantic source timestamp mismatch")
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    if len(semantic_failures) < 64:
                        semantic_failures.append(f"{f['index']} camera {cid}: {exc}")
        check("real_pixel_semantics_artifacts_valid", not semantic_failures, semantic_failures)
    if route == "camera":
        try:
            provider = metadata.get("camera_depth_provider", {})
            check("camera_frozen_weights_loaded_strictly", provider.get("weights_loaded_strict") is True and provider.get("trained_on_orbvi") is False)
            check("camera_geometry_does_not_use_lidar_depth_or_pose", provider.get("input_lidar") is False and provider.get("input_depth") is False and provider.get("input_poses") == "fixed camera-rig extrinsics only")
            check("camera_full_recording_10hz_anchor_coverage", len(rows) == expected_frames,
                  "camera inference uses all 2348 LiDAR-time anchors; all 20Hz source images are preserved, not all are inferred")
            camera_input_ok = all(len(f.get("images", {})) == 4 and all(e is not None and e.get("valid_sync") is True and e["stamp_ns"] <= f["stamp_ns"] for e in f["images"].values()) for f in source_frames)
            check("all_camera_anchor_inputs_are_four_causal_views", camera_input_ok)
            with np.load(run_dir / "trajectory.npz", allow_pickle=False) as d:
                check("camera_trajectory_complete", d["T_world_camera"].shape == (expected_frames, 4, 4) and np.isfinite(d["T_world_camera"]).all() and np.array_equal(d["stamp_ns"], source_stamps))
                expected_segments = np.asarray([r["auxiliary"]["camera_motion"]["segment_id"] for r in rows], dtype=int)
                expected_accepted = np.asarray([r["auxiliary"]["camera_motion"]["accepted"] for r in rows], dtype=bool)
                check("camera_trajectory_segment_flags_complete", np.array_equal(d["segment_id"], expected_segments) and np.array_equal(d["registration_accepted"], expected_accepted))
            camera_motion = [r["auxiliary"]["camera_motion"] for r in rows]
            gated = [r for r in camera_motion if r.get("accepted") and not r.get("initial_frame")]
            check("camera_accepted_registration_quality_gate", all(r.get("condition") is not None and np.isfinite(r["condition"]) and r["condition"] < 1e4 and r.get("rank") == 6 and r.get("point_plane_rmse_m") is not None and r["point_plane_rmse_m"] < .35 for r in gated))
            check("camera_segment_count_matches_frames", summary.get("camera_segments") == int(expected_segments.max() + 1) if len(expected_segments) else False)
        except (OSError, ValueError, KeyError) as exc:
            check("camera_input_and_trajectory_artifacts", False, str(exc))
    expected_visuals = {0, expected_frames - 1}
    stride = config.get("evaluation", {}).get("visualization_stride", 50)
    if isinstance(stride, int) and stride > 0:
        expected_visuals.update(range(0, expected_frames, stride))
    check("scheduled_real_map_figures_exist", all((run_dir / "visualizations" / f"{i:06d}.png").is_file() and (run_dir / "visualizations" / f"{i:06d}.json").is_file() for i in expected_visuals))
    check("full_run_timeline_artifacts_exist", (run_dir / "timeline.png").is_file() and (run_dir / "timeline.json").is_file())
    try:
        preview_name = "unregistered_segments_preview" if route == "camera" and summary.get("global_pose_continuity_valid") is False else "global_occupied_preview"
        with np.load(run_dir / (preview_name + ".npz"), allow_pickle=False) as d:
            points = d["points"]
            check("global_geometry_preview_valid", points.ndim == 2 and points.shape[1] == 3 and np.isfinite(points).all()
                  and int(d["source_frame_stride"]) == 10 and (run_dir / (preview_name + ".png")).is_file(),
                  {"preview_voxels": len(points), "source_frame_stride": 10, "all_frame_outputs_are_saved_separately": True,
                   "continuous_global_registration_claimed": preview_name == "global_occupied_preview"})
        if preview_name == "unregistered_segments_preview":
            check("camera_unregistered_segments_disclosed", (run_dir / "segments.json").is_file() and not (run_dir / "global_occupied_preview.npz").exists())
    except (OSError, ValueError, KeyError) as exc:
        check("global_geometry_preview_valid", False, str(exc))
    report.update(accepted=all(c["passed"] for c in checks), statistics=dict(counts),
                  class_cell_totals=dict(class_totals), source_raw_points=source_points,
                  accounted_raw_points=accounted_points,
                  scope="complete processing and artifact consistency; geometry quality is separate from labelled accuracy")
    return report


def write_report(report, json_path):
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    bad = [c for c in report["checks"] if not c["passed"]]
    text = ["# 全包工程验收", "", f"验收结果：{'通过' if report['accepted'] else '未通过'}。",
            f"预期完整处理 {report['expected_frames']} 帧，原始回波总数 {report['expected_raw_points']:,}。", "",
            f"运行目录：`{report['run_directory']}`。", "",
            "此验收核对全包覆盖、实际输出和数值一致性，不等于获得有标注的几何或语义准确率。"]
    if report.get("route") == "camera":
        text += ["", "相机路线覆盖全部 2348 个约 10 Hz 时步；约 20 Hz 原始相机图像全部保留，但并非每幅图像都单独推理。模型深度自由证据不认证真实传感器净空。"]
    if report.get("pose_registration", {}).get("uses_vio"):
        text += ["", "LiDAR 米制位姿来自 ICP，并使用经手眼关系验证的 VIO 旋转约束稳定 roll/pitch；不使用 VIO 平移或尺度。单帧回波 holdout 不使用该位姿拟合。"]
    provenance = report.get("source_provenance", {})
    if provenance.get("archived_executed_modules_matching_recorded_hash"):
        text += ["", "执行源码快照与启动时记录的原始 SHA-256 相符；验收保留原始指纹。"]
    if provenance.get("route_unused_recorded_modules"):
        text += ["", "该路线未导入的模块单独记录，不用于冒充执行源码：`" + "`, `".join(provenance["route_unused_recorded_modules"]) + "`。"]
    totals = report.get("class_cell_totals", {})
    if totals:
        total_cells = sum(totals.values())
        text += ["", "全部帧局部通行分类（逐帧体素格累积，重复观察同一位置会重复计数）：", "",
                 "| 状态 | 格数 | 占比 |", "|---|---:|---:|"]
        for key, label in (("unknown", "未知"), ("blocked", "阻塞"), ("partial_clearance", "部分净空，仍含未知"), ("observed_clearance", "实测完整净空")):
            count = totals.get(key, 0)
            text.append(f"| {label} | {count:,} | {count / total_cells:.2%} |")
        if not totals.get("observed_clearance", 0):
            text += ["", "没有输出经实测完整净空认证的通行格。工程贯通与可安全规划通行是不同结论，不能将未知区改称自由区。"]
    if bad:
        text += ["", "未通过检查：", ""] + [f"- {c['name']}：{c.get('detail')}" for c in bad]
    if report.get("artifact_errors"):
        text += ["", "逐帧错误（最多展示 128 条）：", ""] + [f"- 帧 {e['frame']}：{e['error']}" for e in report["artifact_errors"]]
    path.with_suffix(".md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--prepared", default=str(PROJECT / "prepared"))
    parser.add_argument("--report")
    args = parser.parse_args()
    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = PROJECT / run_dir
    report = verify(run_dir, args.prepared)
    output = args.report or str(run_dir / "verification.json")
    write_report(report, output)
    print(json.dumps({"accepted": report["accepted"], "checks": len(report["checks"]),
                      "failed_checks": [c["name"] for c in report["checks"] if not c["passed"]], "report": str(output)}, ensure_ascii=False))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    sys.exit(main())
