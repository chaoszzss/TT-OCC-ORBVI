"""ROS-independent, zero-iteration TT-Occ LiDAR Gaussian occupancy adapter.

This is a CPU port of the *actual* upstream Gaussian initialization and gs2voxel
operators, not a raycasting replacement. Free/unknown-space reasoning belongs
to the downstream safety layer. See docs/UPSTREAM_ADAPTATION.md for deviations.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Mapping
import ast
import time

import numpy as np

UPSTREAM_COMMIT = "0cdb8c9e8abc8124fd956c1e0cb582d3ad18b333"
UNKNOWN_SEMANTIC = 255


@dataclass
class TTConfig:
    voxel_size: float = 0.3
    bounds: tuple = ((-12.0, -12.0, -2.0), (12.0, 12.0, 4.0))
    gaussian_sigma: float = 0.3
    history_seconds: float = 1.0
    max_history_frames: int = 12
    input_voxel_size: float = 0.0
    voxel_indexing: str = "floor"
    chunk_size: int = 10000


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Transform row-vector XYZ coordinates with a rigid 4x4 transform."""
    return np.asarray(points) @ np.asarray(transform)[:3, :3].T + np.asarray(transform)[:3, 3]


def gaussian_offsets(sigma: float, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Upstream local_range_offset_full + spatial_weight > 0.6065.

    The upstream code quantizes its 0.3 m scales to integer voxel radii before
    evaluation. Keeping that quantization matters at 0.2 m voxel resolution.
    """
    if sigma <= 0 or voxel_size <= 0:
        raise ValueError("Gaussian scale and voxel size must be positive")
    radius_cells = int(np.clip(np.rint(np.float32(sigma / voxel_size + 1e-3)), 0, 3))
    if radius_cells == 0:
        return np.zeros((1, 3), np.float32), np.ones(1, np.float32)
    axis = np.arange(-radius_cells, radius_cells + 1, dtype=np.float32)
    offsets = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    offsets *= np.float32(voxel_size)
    quantized_sigma = np.float32(radius_cells * voxel_size)
    weights = np.exp(-(offsets ** 2).sum(axis=1) / (2 * quantized_sigma ** 2)).astype(np.float32)
    keep = weights > np.float32(0.6065)
    return offsets[keep], weights[keep]


def _indices(points, bounds, voxel_size, indexing):
    if indexing == "upstream_trunc":
        return np.trunc(points / voxel_size).astype(np.int32) - np.trunc(bounds[0] / voxel_size).astype(np.int32)
    if indexing == "floor":
        return np.floor((points - bounds[0]) / voxel_size).astype(np.int32)
    raise ValueError("voxel_indexing must be floor or upstream_trunc")


def gaussian_voxelize(points, semantic=None, *, bounds, voxel_size=0.3,
                      sigma=0.3, indexing="floor", chunk_size=10000):
    """Initialize 0.3 m isotropic Gaussians and accumulate semantic votes.

    semantic is either N integer labels (unknown=255), or an N x C array of
    non-negative class evidence. With one-hot evidence this reproduces the
    upstream no-training gs2voxel, except when the explicit floor indexing
    correction is selected. Return sparse grid indices and class IDs.
    """
    points = np.asarray(points, dtype=np.float32)
    bounds = np.asarray(bounds, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or bounds.shape != (2, 3):
        raise ValueError("Expected points Nx3 and bounds 2x3")
    if not np.all(np.isfinite(bounds)) or np.any(bounds[1] <= bounds[0]):
        raise ValueError("Invalid occupancy bounds")
    if semantic is None:
        semantic = np.full(len(points), UNKNOWN_SEMANTIC, np.uint8)
    semantic = np.asarray(semantic)
    if len(semantic) != len(points) or semantic.ndim not in (1, 2):
        raise ValueError("Semantic evidence must have one row per point")
    if semantic.ndim == 1 and (np.any(semantic < 0) or np.any(semantic > 255)):
        raise ValueError("Semantic label IDs must be between 0 and 255")
    if semantic.ndim == 2 and (np.any(~np.isfinite(semantic)) or np.any(semantic < 0)):
        raise ValueError("Class evidence must be finite and nonnegative")
    keep = np.isfinite(points).all(1) & (points >= bounds[0]).all(1) & (points < bounds[1]).all(1)
    points, semantic = points[keep], semantic[keep]
    if not len(points):
        return np.empty((0, 3), np.int32), np.empty(0, np.uint8)
    offsets, weights = gaussian_offsets(sigma, voxel_size)
    radius_cells = int(np.clip(np.rint(np.float32(sigma / voxel_size + 1e-3)), 0, 3))
    effective_sigma = np.float32(radius_cells * voxel_size)
    uniform_label = semantic.ndim == 1 and np.all(semantic == semantic[0])
    shape = np.ceil((bounds[1] - bounds[0]) / voxel_size).astype(np.int64)
    flat_parts, source_parts, weight_parts = [], [], []
    for start in range(0, len(points), chunk_size):
        xyz = points[start:start + chunk_size, None, :] + offsets[None, :, :]
        if radius_cells and not uniform_label:
            # The original evaluates (expanded_xyz - base_xyz), rather than the
            # nominal offset. Reproducing float32 subtraction avoids changing
            # semantic argmax at almost-tied class votes.
            # Use the same CPU Torch elementary kernels as upstream: NumPy exp
            # can differ by one ULP and change class argmax for tied one-hot
            # votes. Geometry/bookkeeping remains bounded CPU NumPy work.
            import torch
            diff = torch.from_numpy(xyz) - torch.from_numpy(points[start:start + chunk_size, None, :])
            scales = torch.full((len(diff), 3), float(effective_sigma))
            scaled_difference = diff.pow(2) / (2 * scales.pow(2).unsqueeze(1))
            point_weights = torch.exp(-scaled_difference.sum(-1)).numpy()
        elif not uniform_label:
            point_weights = np.ones(xyz.shape[:2], np.float32)
        valid = (xyz >= bounds[0]).all(-1) & (xyz < bounds[1]).all(-1)
        idx = _indices(xyz[valid], bounds, voxel_size, indexing)
        valid_idx = (idx >= 0).all(1) & (idx < shape).all(1)
        idx = idx[valid_idx]
        flat_parts.append((idx[:, 0] * shape[1] + idx[:, 1]) * shape[2] + idx[:, 2])
        if not uniform_label:
            source = np.broadcast_to(np.arange(start, min(start + chunk_size, len(points)))[:, None], valid.shape)[valid]
            source_parts.append(source[valid_idx])
            weight_parts.append(point_weights[valid][valid_idx])
    flat = np.concatenate(flat_parts)
    if uniform_label:
        # Exact sparse equivalent of an argmax with a single populated class:
        # positive Gaussian vote magnitudes cannot change the winning class.
        flat_unique = np.unique(flat)
        labels = np.full(len(flat_unique), semantic[0], np.uint8)
    else:
        source = np.concatenate(source_parts)
        weight = np.concatenate(weight_parts)
    if semantic.ndim == 1 and not uniform_label:
        # Sparse reduction is mathematically the upstream index_add on one-hot
        # semantics, without allocating a point_count x 256 one-hot tensor.
        composite = flat * 256 + semantic[source].astype(np.int64)
        unique, inverse = np.unique(composite, return_inverse=True)
        summed = np.zeros(len(unique), np.float32)
        np.add.at(summed, inverse, weight)
        cells, classes = unique // 256, unique % 256
        order = np.lexsort((classes, -summed, cells))
        cells, classes = cells[order], classes[order]
        best = np.r_[True, cells[1:] != cells[:-1]]
        flat_unique, labels = cells[best], classes[best]
    elif semantic.ndim == 2:
        flat_unique, inverse = np.unique(flat, return_inverse=True)
        accumulated = np.zeros((len(flat_unique), semantic.shape[1]), np.float32)
        # Chunking bounds peak memory for full OpenSeeD-style evidence arrays.
        for start in range(0, len(source), chunk_size):
            stop = start + chunk_size
            np.add.at(accumulated, inverse[start:stop], semantic[source[start:stop]] * weight[start:stop, None])
        labels = accumulated.argmax(1)
    indices = np.stack((flat_unique // (shape[1] * shape[2]),
                        (flat_unique // shape[2]) % shape[1], flat_unique % shape[2]), axis=1).astype(np.int32)
    return indices, np.asarray(labels, np.uint8)


class TTBackend:
    """Causal past-only TT-Occ Gaussian occupancy, independent of ROS and CUDA.

    Input points and output centers use the current *gravity-aligned* lidar
    coordinate system. T_world_lidar must describe this same input frame.
    Historical dynamic returns are never retained. The semantic adapter may
    supply dynamic_mask from real image semantics only after calibration passes.
    """
    def __init__(self, config: TTConfig | Mapping | None = None):
        if config is None:
            config = TTConfig()
        elif isinstance(config, Mapping):
            names = {f.name for f in fields(TTConfig)}
            # Accept root-level config with a nested tt_occ/backend section.
            config = config.get("tt_occ", config.get("backend", config))
            config = TTConfig(**{k: v for k, v in config.items() if k in names})
        self.config = config
        self.bounds = np.asarray(config.bounds, np.float32)
        if self.bounds.shape != (2, 3) or np.any(self.bounds[1] <= self.bounds[0]):
            raise ValueError("bounds must be a valid 2x3 array")
        if config.history_seconds < 0 or config.max_history_frames < 0:
            raise ValueError("History limits must be nonnegative")
        gaussian_offsets(config.gaussian_sigma, config.voxel_size)
        self.history = deque()
        self.last_stamp_ns = None
        self.frame_count = 0

    def reset(self):
        self.history.clear()
        self.last_stamp_ns = None
        self.frame_count = 0

    def _rasterize(self, points, semantic):
        idx, labels = gaussian_voxelize(points, semantic, bounds=self.bounds,
                                       voxel_size=self.config.voxel_size,
                                       sigma=self.config.gaussian_sigma,
                                       indexing=self.config.voxel_indexing,
                                       chunk_size=self.config.chunk_size)
        centers = self.bounds[0] + (idx.astype(np.float32) + 0.5) * self.config.voxel_size
        return idx, labels, centers

    def update(self, points, T_world_lidar, stamp_ns, semantic=None,
               dynamic_mask=None, **kwargs):
        """Process exactly one complete scan; reject noncausal timestamps."""
        started = time.perf_counter()
        stamp_ns = int(stamp_ns)
        if self.last_stamp_ns is not None and stamp_ns <= self.last_stamp_ns:
            raise ValueError("TTBackend requires strictly increasing timestamps")
        points = np.asarray(points, np.float32)
        pose = np.asarray(T_world_lidar, np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("Expected points Nx3 and finite T_world_lidar 4x4")
        if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5):
            raise ValueError("Invalid homogeneous transform")
        if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-3):
            raise ValueError("Pose rotation must be orthonormal")
        if semantic is None:
            semantic = kwargs.get("labels", np.full(len(points), UNKNOWN_SEMANTIC, np.uint8))
        semantic = np.asarray(semantic)
        if len(semantic) != len(points):
            raise ValueError("Semantic point count mismatch")
        if dynamic_mask is None:
            dynamic_mask = np.zeros(len(points), bool)
        dynamic_mask = np.asarray(dynamic_mask, bool)
        if dynamic_mask.shape != (len(points),):
            raise ValueError("Dynamic mask point count mismatch")
        valid = np.isfinite(points).all(1)
        points, semantic, dynamic_mask = points[valid], semantic[valid], dynamic_mask[valid]
        if self.config.input_voxel_size > 0 and len(points):
            # Optional bounded memory policy. Select the first original return
            # in each source voxel, preserving measured coordinates and labels.
            # It is disabled by default, and is explicitly recorded in metadata.
            source_voxels = np.floor(points / self.config.input_voxel_size).astype(np.int32)
            _, first = np.unique(source_voxels, axis=0, return_index=True)
            first.sort()
            points, semantic, dynamic_mask = points[first], semantic[first], dynamic_mask[first]
        cutoff = stamp_ns - int(self.config.history_seconds * 1e9)
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()
        while len(self.history) > self.config.max_history_frames:
            self.history.popleft()
        current_idx, current_labels, current_centers = self._rasterize(points, semantic)
        inverse_pose = np.linalg.inv(pose)
        past_points, past_semantic = [], []
        for past_stamp, world_points, past_labels in self.history:
            if past_stamp >= stamp_ns:
                raise RuntimeError("Future data entered the causal history")
            local = transform_points(world_points, inverse_pose).astype(np.float32)
            in_bounds = (local >= self.bounds[0]).all(1) & (local < self.bounds[1]).all(1)
            if np.any(in_bounds):
                past_points.append(local[in_bounds])
                past_semantic.append(past_labels[in_bounds])
        if past_points:
            fused_input = np.concatenate([points] + past_points)
            fused_semantic = np.concatenate([semantic] + past_semantic)
            fused_dynamic = np.r_[dynamic_mask, np.zeros(sum(map(len, past_points)), bool)]
            fused_idx, fused_labels, fused_centers = self._rasterize(fused_input, fused_semantic)
        else:
            fused_input = points
            fused_semantic = semantic
            fused_dynamic = dynamic_mask
            fused_idx, fused_labels, fused_centers = current_idx, current_labels, current_centers
        history_frames_used = len(self.history)
        history_points_used = sum(map(len, past_points))
        # Current dynamic points contribute now but do not leave temporal trails.
        static = ~dynamic_mask
        if self.config.history_seconds > 0 and self.config.max_history_frames > 0:
            self.history.append((stamp_ns, transform_points(points[static], pose).astype(np.float32), semantic[static].copy()))
            while len(self.history) > self.config.max_history_frames:
                self.history.popleft()
        self.last_stamp_ns, self.frame_count = stamp_ns, self.frame_count + 1
        diagnostics = {
            "algorithm": "TT-Occ LiDAR zero-iteration isotropic Gaussian CPU adaptation",
            "upstream_commit": UPSTREAM_COMMIT, "iterations": 0,
            "input_points": len(points), "dynamic_points": int(dynamic_mask.sum()),
            "past_frames": history_frames_used, "past_points": history_points_used,
            "history_seconds": self.config.history_seconds,
            "gaussian_sigma_m": self.config.gaussian_sigma,
            "voxel_size_m": self.config.voxel_size, "voxel_indexing": self.config.voxel_indexing,
            "input_voxel_size_m": self.config.input_voxel_size,
            "current_occupied_voxels": len(current_idx), "fused_occupied_voxels": len(fused_idx),
            "new_voxels_from_history": len(fused_idx) - len(current_idx),
            "runtime_ms": (time.perf_counter() - started) * 1000,
        }
        return {
            "current_indices": current_idx, "current_labels": current_labels,
            "current_centers": current_centers,
            "fused_indices": fused_idx, "fused_labels": fused_labels,
            "fused_centers": fused_centers,
            "indices": fused_idx, "labels": fused_labels, "centers": fused_centers,
            "current_source_points": points, "current_source_labels": semantic,
            "current_source_dynamic": dynamic_mask,
            "fused_source_points": fused_input, "fused_source_labels": fused_semantic,
            "fused_source_dynamic": fused_dynamic,
            "diagnostics": diagnostics,
        }


def upstream_reference_voxelize(points, semantic, bounds, voxel_size=0.3,
                                sigma=0.3, upstream_root=None):
    """Execute the original gs2voxel methods from the pinned source on CPU.

    Only the hardcoded 'cuda' device literal is changed to 'cpu'. This reference
    deliberately retains upstream negative-coordinate truncation; it is used
    by differential tests, not by the production occupancy adapter.
    """
    import torch
    source = Path(upstream_root or Path(__file__).resolve().parents[1] / "upstream") / "scene/gaussian_model.py"
    tree = ast.parse(source.read_text())
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GaussianModel")
    names = {"local_range_offset_full", "gaussian_spatial_weight_simplified", "gaussian_spatial_weight", "gs2voxel"}
    methods = [node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(methods) != 4:
        raise RuntimeError("Pinned upstream operators changed unexpectedly")
    class CpuDevice(ast.NodeTransformer):
        def visit_Constant(self, node):
            return ast.copy_location(ast.Constant("cpu"), node) if node.value == "cuda" else node
    module = ast.Module(body=[ast.ClassDef(name="ReferenceGaussian", bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    module = ast.fix_missing_locations(CpuDevice().visit(module))
    namespace = {"torch": torch}
    exec(compile(module, str(source), "exec"), namespace)
    reference = namespace["ReferenceGaussian"]()
    reference.scaling_activation = torch.sigmoid
    reference.rotation_activation = lambda x: torch.nn.functional.normalize(x)
    xyz = torch.as_tensor(points, dtype=torch.float32)
    sem = torch.as_tensor(semantic, dtype=torch.float32)
    scale = torch.full((len(xyz), 3), float(sigma))
    scales_logit = torch.log(scale / (1 - scale))
    rot = torch.zeros((len(xyz), 4)); rot[:, 0] = 1
    detach = {"_xyz": xyz, "_semantic": sem, "_scaling": scales_logit, "_rotation": rot}
    idx, cls = reference.gs2voxel(torch.eye(4), torch.as_tensor(bounds[0], dtype=torch.float32),
                                torch.as_tensor(bounds[1], dtype=torch.float32), voxel_size, detach)
    return idx.cpu().numpy(), cls.cpu().numpy()
