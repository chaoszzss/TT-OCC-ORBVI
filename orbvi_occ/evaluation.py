"""Geometry-only independent-return checks and full-bag acceptance accounting.

No semantic ground truth is assumed. Holdout returns are measurement evidence,
not complete 3D truth; observed-domain scores must not be called semantic mIoU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np


def _points(points: Any) -> np.ndarray:
    a = np.asarray(points, dtype=np.float64)
    if not a.size:
        return np.empty((0, 3))
    if a.ndim != 2 or a.shape[1] < 3:
        raise ValueError("expected XYZ points")
    return a[np.isfinite(a[:, :3]).all(axis=1), :3]


def split_holdout(points: np.ndarray, fraction: float = 0.1, seed: int = 42,
                  voxel_size: float = 0.3) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Partition whole spatial voxels so duplicated XYZ cannot leak across sets.

    Returns construction_xyz, holdout_xyz, construction_source_ids,
    holdout_source_ids. Source ids refer to the original input rows. The grouping
    grid uses sensor-origin metres; do not reuse the holdout for pose fitting,
    floor estimation, TT fusion, or clearance construction when reporting it.
    """
    if not 0 <= fraction < 1 or voxel_size <= 0:
        raise ValueError("fraction must be in [0,1) and voxel_size positive")
    original = np.asarray(points, dtype=np.float64)
    if not original.size:
        empty = np.empty((0, 3))
        ids = np.empty(0, dtype=np.int64)
        return empty, empty.copy(), ids, ids.copy()
    ids = np.flatnonzero(np.isfinite(original[:, :3]).all(axis=1))
    xyz = original[ids, :3]
    keys = np.floor(xyz / voxel_size).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    # Spatial group partition has no dependency on scan point order.
    rng = np.random.default_rng(seed)
    held_groups = rng.random(len(unique)) < fraction
    held = held_groups[inverse]
    return xyz[~held], xyz[held], ids[~held], ids[held]


def assert_no_source_leakage(construction_ids: Iterable[int], holdout_ids: Iterable[int]) -> None:
    if np.intersect1d(np.asarray(list(construction_ids)), np.asarray(list(holdout_ids))).size:
        raise ValueError("holdout source ids overlap occupancy construction ids")


def _structured(keys: np.ndarray) -> np.ndarray:
    keys = np.ascontiguousarray(keys, dtype=np.int64)
    return keys.view(np.dtype([("x", np.int64), ("y", np.int64), ("z", np.int64)])).reshape(-1)


def voxel_hit_mask(predicted_xyz: np.ndarray, query_xyz: np.ndarray, voxel_size: float = 0.3,
                   tolerance_m: float = 0.3, origin: np.ndarray | None = None) -> np.ndarray:
    """Voxel-neighbour association, with explicitly reported Chebyshev tolerance.

    This is not Euclidean nearest-neighbour distance. It accepts matching voxel
    or adjacent voxels as configured; returned denominators count query returns.
    """
    predicted, query = _points(predicted_xyz), _points(query_xyz)
    if voxel_size <= 0 or tolerance_m < 0:
        raise ValueError("invalid voxel size or tolerance")
    out = np.zeros(len(query), dtype=bool)
    if not len(query) or not len(predicted):
        return out
    origin = np.zeros(3) if origin is None else np.asarray(origin)
    pkeys = np.floor((predicted - origin) / voxel_size).astype(np.int64)
    qkeys = np.floor((query - origin) / voxel_size).astype(np.int64)
    pset = np.unique(_structured(pkeys))
    radius = int(np.ceil(tolerance_m / voxel_size - 1e-9))
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                out |= np.isin(_structured(qkeys + np.array([dx, dy, dz])), pset, assume_unique=False)
    return out


def grid_query(mask: np.ndarray, points: np.ndarray, resolution: float,
               bounds: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    """Returns value and in-grid mask; out-of-grid points are not called free."""
    xyz = _points(points)
    idx = np.floor((xyz - np.asarray(tuple(bounds)[:3])) / resolution).astype(np.int64)
    inside = np.all((idx >= 0) & (idx < np.asarray(mask.shape)), axis=1)
    values = np.zeros(len(xyz), dtype=mask.dtype)
    if inside.any():
        values[inside] = mask[tuple(idx[inside].T)]
    return values, inside


def evaluate_geometry(predicted_occupied_xyz: np.ndarray, holdout_xyz: np.ndarray,
                      voxel_size: float = 0.3, tolerance_m: float = 0.3,
                      observed_free: np.ndarray | None = None,
                      bounds: Iterable[float] | None = None,
                      construction_ids: Iterable[int] | None = None,
                      holdout_ids: Iterable[int] | None = None) -> dict[str, Any]:
    """Endpoint geometry hit rate and false-free rate on withheld measurements.

    Prediction may come from a LiDAR, camera or RGB-D algorithm. ``observed_free``
    must use construction evidence only. Ground truth empty space and semantic
    accuracy cannot be deduced from endpoint correspondence.
    """
    if construction_ids is not None and holdout_ids is not None:
        assert_no_source_leakage(construction_ids, holdout_ids)
    predicted, holdout = _points(predicted_occupied_xyz), _points(holdout_xyz)
    if bounds is not None:
        b = np.asarray(tuple(bounds))
        if b.shape != (6,):
            raise ValueError("bounds must contain six values")
        inside = np.all((holdout >= b[:3]) & (holdout < b[3:]), axis=1)
        holdout = holdout[inside]
    hit = voxel_hit_mask(predicted, holdout, voxel_size, tolerance_m)
    report = {"holdout_returns": len(holdout), "predicted_occupied_voxels": len(predicted),
              "endpoint_hits": int(hit.sum()), "endpoint_misses": int((~hit).sum()),
              "endpoint_hit_rate": float(hit.mean()) if len(hit) else None,
              "voxel_size_m": voxel_size, "chebyshev_voxel_tolerance_m": tolerance_m,
              "false_free_count": None, "free_observation_denominator": 0,
              "false_free_rate": None,
              "metric_scope": "withheld LiDAR-return consistency; not semantic mIoU or complete occupancy truth"}
    if observed_free is not None:
        if bounds is None:
            raise ValueError("bounds are required to query a free-observation grid")
        value, inside = grid_query(observed_free, holdout, voxel_size, bounds)
        report["false_free_count"] = int(np.count_nonzero(value & inside))
        report["free_observation_denominator"] = int(inside.sum())
        report["false_free_rate"] = report["false_free_count"] / int(inside.sum()) if inside.any() else None
    return report


@dataclass
class GeometryMetricAccumulator:
    frames: int = 0
    holdout_returns: int = 0
    endpoint_hits: int = 0
    false_free_count: int = 0
    free_observation_denominator: int = 0
    zero_holdout_frames: int = 0

    def add(self, report: Mapping[str, Any]) -> None:
        self.frames += 1
        count = int(report.get("holdout_returns", 0))
        self.holdout_returns += count
        self.zero_holdout_frames += count == 0
        self.endpoint_hits += int(report.get("endpoint_hits", 0))
        self.false_free_count += int(report.get("false_free_count") or 0)
        self.free_observation_denominator += int(report.get("free_observation_denominator") or 0)

    def summary(self) -> dict[str, Any]:
        return {"frames": self.frames, "holdout_returns": self.holdout_returns,
                "endpoint_hits": self.endpoint_hits,
                "weighted_endpoint_hit_rate": self.endpoint_hits / self.holdout_returns if self.holdout_returns else None,
                "false_free_count": self.false_free_count,
                "free_observation_denominator": self.free_observation_denominator,
                "weighted_false_free_rate": self.false_free_count / self.free_observation_denominator if self.free_observation_denominator else None,
                "zero_holdout_frames": self.zero_holdout_frames,
                "metric_scope": "withheld-return geometry consistency; no semantic ground truth"}


@dataclass
class FullBagAcceptance:
    """Account for every expected LiDAR frame, including empty or failed frames."""
    expected_frames: int
    expected_start_ns: int | None = None
    expected_end_ns: int | None = None
    records: list[dict[str, Any]] = field(default_factory=list)

    def add(self, frame_id: int, timestamp_ns: int, success: bool,
            input_points: int = 0, occupancy_voxels: int = 0,
            traversability_cells: int = 0, error: str | None = None) -> None:
        self.records.append({"frame_id": int(frame_id), "timestamp_ns": int(timestamp_ns),
                             "success": bool(success), "input_points": int(input_points),
                             "occupancy_voxels": int(occupancy_voxels),
                             "traversability_cells": int(traversability_cells), "error": error})

    def summary(self) -> dict[str, Any]:
        ids = [r["frame_id"] for r in self.records]
        timestamps = [r["timestamp_ns"] for r in self.records]
        unique = set(ids)
        missing = sorted(set(range(self.expected_frames)) - unique)
        extra = sorted(unique - set(range(self.expected_frames)))
        failed = [r["frame_id"] for r in self.records if not r["success"]]
        ordered = all(a < b for a, b in zip(ids, ids[1:]))
        monotonic = all(a < b for a, b in zip(timestamps, timestamps[1:]))
        start_matches = self.expected_start_ns is None or bool(timestamps and timestamps[0] == self.expected_start_ns)
        end_matches = self.expected_end_ns is None or bool(timestamps and timestamps[-1] == self.expected_end_ns)
        complete = (len(self.records) == self.expected_frames and not missing and not extra
                    and not failed and ordered and monotonic and start_matches and end_matches)
        return {"accepted": complete, "expected_frames": self.expected_frames,
                "recorded_frames": len(self.records), "unique_frames": len(unique),
                "missing_frame_ids": missing, "extra_frame_ids": extra,
                "duplicate_records": len(ids) - len(unique), "failed_frame_ids": failed,
                "frame_ids_ordered": ordered, "timestamps_monotonic": monotonic,
                "expected_first_timestamp_ns": self.expected_start_ns,
                "expected_last_timestamp_ns": self.expected_end_ns,
                "first_timestamp_matches_expected": start_matches,
                "last_timestamp_matches_expected": end_matches,
                "first_timestamp_ns": timestamps[0] if timestamps else None,
                "last_timestamp_ns": timestamps[-1] if timestamps else None,
                "empty_input_frames": sum(r["input_points"] == 0 for r in self.records),
                "empty_occupancy_frames": sum(r["occupancy_voxels"] == 0 for r in self.records),
                "frames_with_traversability_output": sum(r["traversability_cells"] > 0 for r in self.records),
                "acceptance_scope": "all LiDAR frames processed; geometric quality is evaluated separately"}
