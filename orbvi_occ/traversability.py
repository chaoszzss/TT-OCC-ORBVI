"""Conservative local clearance from TT-Occ occupied voxels and observed LiDAR rays.

TT-Occ supplies occupied geometry; this module supplies the missing distinction
between empty predictions and actually observed free space. All public point
inputs are XYZ metres in the current LiDAR/ego frame (x forward, y left, z up).
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

UNKNOWN, BLOCKED, PARTIAL_CLEARANCE, OBSERVED_CLEARANCE = 0, 1, 2, 3
CLASS_NAMES = {0: "unknown", 1: "blocked", 2: "partial_clearance", 3: "observed_clearance"}


@dataclass
class TraversabilityConfig:
    resolution: float = 0.3
    bounds: tuple[float, ...] = (-12.0, -12.0, -2.0, 12.0, 12.0, 4.0)
    footprint_length: float = 0.7
    footprint_width: float = 0.55
    safety_margin: float = 0.1
    vehicle_height: float = 1.0
    clearance_bottom: float = 0.15
    max_slope_degrees: float = 10.0
    ground_max_height: float = -0.15
    ground_min_height: float = -2.5
    ground_tolerance: float = 0.08
    ground_support_radius: float = 0.4
    min_ground_inliers: int = 25
    ray_max_points: int = 6000
    ray_angular_degrees: float = 0.2
    ray_step_fraction: float = 0.5
    free_history_frames: int = 3
    free_history_seconds: float = 0.4
    min_partial_coverage: float = 0.2
    random_seed: int = 42
    gaussian_sigma: float = 0.3

    def __post_init__(self) -> None:
        self.bounds = tuple(float(x) for x in self.bounds)
        if len(self.bounds) != 6 or any(self.bounds[i] >= self.bounds[i + 3] for i in range(3)):
            raise ValueError("bounds must be (xmin,ymin,zmin,xmax,ymax,zmax)")
        if self.resolution <= 0 or self.vehicle_height <= self.clearance_bottom:
            raise ValueError("resolution and body clearance interval must be positive")
        if not 0 <= self.min_partial_coverage <= 1:
            raise ValueError("min_partial_coverage must be in [0,1]")
        if self.free_history_frames < 1 or self.ray_max_points < 1:
            raise ValueError("history and ray point limits must be positive")
        if min(self.footprint_width, self.footprint_length) <= 0 or self.safety_margin < 0:
            raise ValueError("footprint dimensions must be positive and margin nonnegative")


@dataclass
class TraversabilityResult:
    timestamp: float | None
    traversability: np.ndarray
    occupancy2d: np.ndarray
    observed_free: np.ndarray
    occupied3d: np.ndarray
    collision3d: np.ndarray
    ground_support: np.ndarray
    clearance_coverage: np.ndarray
    ground_plane: np.ndarray
    summary: dict[str, Any]
    resolution: float
    bounds: tuple[float, ...]

    def save(self, path: str) -> None:
        """Save arrays without pickle; raster axis 0=x, axis 1=y, axis 2=z."""
        np.savez_compressed(path, timestamp=np.nan if self.timestamp is None else self.timestamp,
                            traversability=self.traversability, occupancy2d=self.occupancy2d,
                            observed_free=self.observed_free, occupied3d=self.occupied3d,
                            collision3d=self.collision3d,
                            ground_support=self.ground_support,
                            clearance_coverage=self.clearance_coverage,
                            ground_plane=self.ground_plane,
                            resolution=self.resolution, bounds=np.asarray(self.bounds))


def _xyz(points: Any) -> np.ndarray:
    if isinstance(points, Mapping):
        points = points.get("xyz", points.get("points"))
    a = np.asarray(points, dtype=np.float64)
    if a.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if a.ndim != 2 or a.shape[1] < 3:
        raise ValueError("points must have shape (N,3+) in XYZ metres")
    a = a[:, :3]
    return a[np.isfinite(a).all(axis=1)]


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def _shift(mask: np.ndarray, dx: int, dy: int, outside: bool = False) -> np.ndarray:
    out = np.full_like(mask, outside)
    nx, ny = mask.shape
    if abs(dx) >= nx or abs(dy) >= ny:
        return out
    sx = slice(max(0, -dx), min(nx, nx - dx))
    sy = slice(max(0, -dy), min(ny, ny - dy))
    tx = slice(max(0, dx), min(nx, nx + dx))
    ty = slice(max(0, dy), min(ny, ny + dy))
    out[tx, ty] = mask[sx, sy]
    return out


def _disk_offsets(radius: float, resolution: float) -> list[tuple[int, int]]:
    cells = int(np.ceil(radius / resolution))
    return [(x, y) for x in range(-cells, cells + 1) for y in range(-cells, cells + 1)
            if x * x + y * y <= (radius / resolution + 0.5) ** 2]


def dilate(mask: np.ndarray, radius: float, resolution: float) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    for dx, dy in _disk_offsets(radius, resolution):
        out |= _shift(mask, dx, dy)
    return out


def erode(mask: np.ndarray, radius: float, resolution: float) -> np.ndarray:
    out = np.ones_like(mask, dtype=bool)
    for dx, dy in _disk_offsets(radius, resolution):
        out &= _shift(mask, dx, dy)
    return out


class LocalTraversability:
    """Bounded rolling free-ray evidence, slope-aware floor and footprint clearance.

    ``pose`` is T_world_from_current. Supplying no pose disables motion-aware
    history (only current rays are used). ``fused_occ`` is TT-Occ voxel-centre
    XYZ, not voxel indices. Ray endpoints are raw training/construction points,
    never holdout points. Occupied voxels always override ray-free observations.
    """
    def __init__(self, config: TraversabilityConfig | Mapping[str, Any] | None = None,
                 *, collision_voxelizer=None):
        self.config = config if isinstance(config, TraversabilityConfig) else TraversabilityConfig(**dict(config or {}))
        self.origin = np.asarray(self.config.bounds[:3])
        self.shape = tuple(np.ceil((np.asarray(self.config.bounds[3:]) - self.origin) / self.config.resolution).astype(int))
        if np.prod(self.shape) > 15_000_000:
            raise ValueError("local grid exceeds 15 million voxels; use coarser resolution or smaller bounds")
        self._history: deque[tuple[float, np.ndarray, np.ndarray]] = deque(maxlen=self.config.free_history_frames)
        self._previous_plane_world: np.ndarray | None = None
        self._previous_plane_time = -np.inf
        self._frame_count = 0
        if collision_voxelizer is not None and not callable(collision_voxelizer):
            raise TypeError('collision_voxelizer must be a callable or None')
        self._collision_voxelizer = collision_voxelizer

    def voxel_indices(self, xyz: np.ndarray) -> np.ndarray:
        idx = np.floor((xyz - self.origin) / self.config.resolution).astype(np.int32)
        good = np.all((idx >= 0) & (idx < np.asarray(self.shape)), axis=1)
        return idx[good]

    def _raster(self, xyz: np.ndarray) -> np.ndarray:
        mask = np.zeros(self.shape, dtype=bool)
        idx = self.voxel_indices(xyz)
        if len(idx):
            mask[tuple(idx.T)] = True
        return mask

    def _centres(self, mask: np.ndarray) -> np.ndarray:
        return (np.argwhere(mask).astype(np.float64) + 0.5) * self.config.resolution + self.origin

    def _floor(self, points: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        c = self.config
        candidates = points[(points[:, 2] >= c.ground_min_height) & (points[:, 2] <= c.ground_max_height)]
        if len(candidates) < c.min_ground_inliers:
            return np.full(4, np.nan), {"valid": False, "reason": "insufficient_low_returns", "inliers": 0}
        # One lowest return per XY tile avoids giving one dense surface excess weight.
        tiles = np.floor(candidates[:, :2] / 0.75).astype(np.int32)
        order = np.argsort(candidates[:, 2], kind="stable")
        _, first = np.unique(tiles[order], axis=0, return_index=True)
        candidates = candidates[order[first]]
        if len(candidates) > 2500:
            candidates = candidates[np.linspace(0, len(candidates) - 1, 2500).astype(int)]
        if len(candidates) < c.min_ground_inliers:
            return np.full(4, np.nan), {"valid": False, "reason": "insufficient_spatial_support", "inliers": 0}
        rng = np.random.default_rng(c.random_seed)
        best: np.ndarray | None = None
        best_count = 0
        slope_limit = np.tan(np.deg2rad(c.max_slope_degrees))
        for _ in range(64):
            sample = candidates[rng.choice(len(candidates), 3, replace=False)]
            try:
                coefficients = np.linalg.solve(np.c_[sample[:, :2], np.ones(3)], sample[:, 2])
            except np.linalg.LinAlgError:
                continue
            if np.linalg.norm(coefficients[:2]) > slope_limit:
                continue
            errors = np.abs(candidates[:, 2] - (candidates[:, 0] * coefficients[0] + candidates[:, 1] * coefficients[1] + coefficients[2]))
            inliers = errors < c.ground_tolerance
            count = int(inliers.sum())
            if count > best_count:
                best, best_count = inliers, count
        if best is None or best_count < c.min_ground_inliers:
            return np.full(4, np.nan), {"valid": False, "reason": "no_plausible_ground_plane", "inliers": best_count}
        selected = candidates[best]
        if min(np.ptp(selected[:, 0]), np.ptp(selected[:, 1])) < 1.0:
            return np.full(4, np.nan), {"valid": False, "reason": "ground_support_too_narrow", "inliers": best_count}
        coef = np.linalg.lstsq(np.c_[selected[:, :2], np.ones(best_count)], selected[:, 2], rcond=None)[0]
        slope_degrees = float(np.rad2deg(np.arctan(np.linalg.norm(coef[:2]))))
        if slope_degrees > c.max_slope_degrees:
            return np.full(4, np.nan), {"valid": False, "reason": "ground_slope_exceeded", "inliers": best_count}
        plane = np.array([-coef[0], -coef[1], 1.0, -coef[2]])
        return plane, {"valid": True, "source": "current_scan_ransac", "inliers": best_count,
                       "slope_degrees": slope_degrees,
                       "residual_median_m": float(np.median(np.abs(np.c_[selected, np.ones(best_count)] @ plane)))}

    def ray_free_mask(self, points: np.ndarray, ray_origin: np.ndarray | None = None) -> np.ndarray:
        """Mark only traversed voxels, excluding a full endpoint voxel margin."""
        c = self.config
        original = np.asarray(points, dtype=np.float64)
        valid = np.isfinite(original[:, :3]).all(axis=1) if original.size else np.zeros(0, dtype=bool)
        points = _xyz(points)
        if ray_origin is None:
            origins = np.zeros_like(points)
        else:
            origins = np.asarray(ray_origin, dtype=np.float64)
            if origins.size == 3:
                origins = np.broadcast_to(origins.reshape(1, 3), points.shape)
            elif origins.shape == (len(original), 3):
                origins = origins[valid]
            else:
                raise ValueError("ray_origin must be XYZ or one XYZ origin per source point")
            if not np.isfinite(origins).all():
                raise ValueError("ray origins must be finite")
        delta = points - origins
        distances = np.linalg.norm(delta, axis=1)
        good = distances > c.resolution * 2
        delta, distances, origins = delta[good], distances[good], origins[good]
        if not len(delta):
            return np.zeros(self.shape, dtype=bool)
        # Keep the nearest return in an angular bin; farther returns cannot certify
        # space behind the first return as free.
        angle = np.deg2rad(c.ray_angular_degrees)
        azimuth = np.arctan2(delta[:, 1], delta[:, 0])
        elevation = np.arctan2(delta[:, 2], np.linalg.norm(delta[:, :2], axis=1))
        bins = np.c_[np.floor(np.c_[azimuth, elevation] / angle).astype(np.int32),
                     np.rint(origins / 0.001).astype(np.int32)]
        order = np.argsort(distances, kind="stable")
        _, first = np.unique(bins[order], axis=0, return_index=True)
        keep = order[first]
        if len(keep) > c.ray_max_points:
            keep = keep[np.linspace(0, len(keep) - 1, c.ray_max_points).astype(int)]
        delta, distances, origins = delta[keep], distances[keep], origins[keep]
        directions = delta / distances[:, None]
        max_ray_range = float(np.max(np.linalg.norm(np.maximum(np.abs(np.asarray(c.bounds[:3]) - origins), np.abs(np.asarray(c.bounds[3:]) - origins)), axis=1)))
        stops = np.minimum(distances - c.resolution * 1.5, max_ray_range)
        step = c.resolution * c.ray_step_fraction
        steps = np.arange(step, min(float(stops.max()), max_ray_range) + step, step)
        mask = np.zeros(self.shape, dtype=bool)
        # Bound intermediate arrays independently of the number of LiDAR returns.
        for start in range(0, len(delta), 512):
            rays = origins[start:start + 512, None, :] + directions[start:start + 512, None, :] * steps[None, :, None]
            active = steps[None, :] < stops[start:start + 512, None]
            idx = self.voxel_indices(rays[active])
            if len(idx):
                mask[tuple(idx.T)] = True
        # An endpoint can coincide with another ray. Do not let that certify empty.
        endpoint_idx = self.voxel_indices(points)
        if len(endpoint_idx):
            mask[tuple(endpoint_idx.T)] = False
        return mask

    def update(self, points_current: Any, fused_occ: Any, pose: np.ndarray | None = None,
               timestamp: float | None = None, ray_origins: np.ndarray | None = None,
               fused_source_points: Any | None = None,
               free_evidence_source: str = "lidar_measurement") -> TraversabilityResult:
        c = self.config
        points, occupied_xyz = _xyz(points_current), _xyz(fused_occ)
        if free_evidence_source not in {"lidar_measurement", "predicted_camera_depth"}:
            raise ValueError("free_evidence_source must be lidar_measurement or predicted_camera_depth")
        t = float(self._frame_count if timestamp is None else timestamp)
        self._frame_count += 1
        pose_provided = pose is not None
        pose = np.eye(4) if pose is None else np.asarray(pose, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("pose must be finite 4x4 T_world_from_current")
        if ray_origins is not None:
            origins = np.asarray(ray_origins, dtype=np.float64)
            if origins.size != 3:
                raw = np.asarray(points_current, dtype=np.float64)
                if origins.shape != (len(raw), 3):
                    raise ValueError("ray_origins must be XYZ or one XYZ origin per input point")
                ray_origins = origins[np.isfinite(raw[:, :3]).all(axis=1)]
        free = self.ray_free_mask(points, ray_origins)
        current_free = free.copy()
        plane, ground_info = self._floor(points)
        if ground_info["valid"]:
            self._previous_plane_world = np.linalg.inv(pose).T @ plane
            self._previous_plane_time = t
        elif pose_provided and self._previous_plane_world is not None and 0 <= t - self._previous_plane_time <= c.free_history_seconds:
            plane = pose.T @ self._previous_plane_world
            if abs(plane[2]) > 1e-6:
                ground_info = {"valid": True, "source": "recent_motion_compensated_plane", "inliers": 0,
                               "slope_degrees": float(np.rad2deg(np.arctan(np.linalg.norm(plane[:2]) / abs(plane[2]))))}
        if pose_provided:
            if self._history and t < self._history[-1][0]:
                self._history.clear()
            while self._history and t - self._history[0][0] > c.free_history_seconds:
                self._history.popleft()
            current_from_world = np.linalg.inv(pose)
            for _, free_world, _ in self._history:
                idx = self.voxel_indices(transform_points(free_world, current_from_world))
                if len(idx):
                    free[tuple(idx.T)] = True
            # Store only current observations, not the union, to enforce expiration.
            ground_points = np.empty((0, 3))
            if ground_info["valid"]:
                ground_points = points[np.abs(np.c_[points, np.ones(len(points))] @ plane) <= c.ground_tolerance]
            self._history.append((t, transform_points(self._centres(current_free), pose), transform_points(ground_points, pose)))
        else:
            self._history.clear()
        occupied = self._raster(occupied_xyz)
        collision = occupied.copy()
        floor_source_count = 0
        if fused_source_points is not None and ground_info["valid"]:
            # Provenance, not voxel-centre height, distinguishes floor Gaussian
            # thickness from low obstacles. An overlapping obstacle source still
            # contributes its full TT Gaussian. Raw TT occupied is not modified.
            from .tt_backend import gaussian_voxelize
            source = _xyz(fused_source_points)
            floor_source = np.abs(np.c_[source, np.ones(len(source))] @ plane) <= c.ground_tolerance
            floor_source_count = int(floor_source.sum())
            voxelizer=gaussian_voxelize if self._collision_voxelizer is None else self._collision_voxelizer
            collision_idx, _ = voxelizer(source[~floor_source],
                                                bounds=np.array([c.bounds[:3], c.bounds[3:]], dtype=np.float32),
                                                voxel_size=c.resolution, sigma=c.gaussian_sigma)
            if self._collision_voxelizer is not None:
                collision_idx=np.asarray(collision_idx)
                if collision_idx.ndim!=2 or collision_idx.shape[1]!=3 or not np.issubdtype(collision_idx.dtype,np.integer) or np.any(collision_idx<0) or np.any(collision_idx>=np.asarray(self.shape)):
                    raise ValueError('Injected collision voxelizer returned invalid grid indices')
            collision = np.zeros(self.shape, dtype=bool)
            if len(collision_idx):
                collision[tuple(collision_idx.T)] = True
        # Raw observed endpoints supplement only the observation mask; TT occupied
        # geometry is kept independently as the occupancy prediction.
        endpoint = self._raster(points)
        free[collision | endpoint] = False
        nx, ny, nz = self.shape
        support = np.zeros((nx, ny), dtype=bool)
        coverage = np.zeros((nx, ny), dtype=np.float32)
        obstacles = np.any(collision, axis=2)
        traversability = np.zeros((nx, ny), dtype=np.uint8)
        floor_height = np.full((nx, ny), np.nan)
        if ground_info["valid"] and abs(plane[2]) > 1e-6:
            x = self.origin[0] + (np.arange(nx) + 0.5) * c.resolution
            y = self.origin[1] + (np.arange(ny) + 0.5) * c.resolution
            floor_height = -(plane[0] * x[:, None] + plane[1] * y[None, :] + plane[3]) / plane[2]
            z = self.origin[2] + (np.arange(nz) + 0.5) * c.resolution
            in_body = (z[None, None, :] >= floor_height[:, :, None] + c.clearance_bottom) & (z[None, None, :] <= floor_height[:, :, None] + c.vehicle_height)
            required = in_body.sum(axis=2)
            coverage = np.divide((free & in_body).sum(axis=2), required, out=np.zeros((nx, ny), dtype=np.float32), where=required > 0)
            obstacles = np.any(collision & in_body, axis=2)
            ground_points = points[np.abs(np.c_[points, np.ones(len(points))] @ plane) <= c.ground_tolerance]
            if pose_provided:
                for _, _, ground_world in self._history:
                    prior = transform_points(ground_world, np.linalg.inv(pose))
                    prior = prior[np.abs(np.c_[prior, np.ones(len(prior))] @ plane) <= c.ground_tolerance]
                    ground_points = np.concatenate([ground_points, prior])
            idx = self.voxel_indices(ground_points)
            if len(idx):
                support[idx[:, 0], idx[:, 1]] = True
            support = dilate(support, c.ground_support_radius, c.resolution)
            # Circumscribed footprint circle is conservative for any heading.
            radius = 0.5 * np.hypot(c.footprint_length, c.footprint_width) + c.safety_margin
            footprint_blocked = dilate(obstacles, radius, c.resolution)
            partial = support & (coverage >= c.min_partial_coverage) & ~obstacles
            full = support & (coverage >= 1.0 - 1e-6) & ~obstacles
            full &= (floor_height + c.vehicle_height <= c.bounds[5]) & (floor_height + c.clearance_bottom >= c.bounds[2])
            partial = erode(partial, radius, c.resolution) & ~footprint_blocked
            full = erode(full, radius, c.resolution) & ~footprint_blocked
            traversability[partial] = PARTIAL_CLEARANCE
            traversability[full] = OBSERVED_CLEARANCE
            traversability[footprint_blocked] = BLOCKED
        else:
            # Without a plausible floor, report all TT occupied XY columns as
            # blocked; no automatic flat-floor assumption creates false free cells.
            radius = 0.5 * np.hypot(c.footprint_length, c.footprint_width) + c.safety_margin
            traversability[dilate(obstacles, radius, c.resolution)] = BLOCKED
        if free_evidence_source == "predicted_camera_depth":
            # Model-predicted camera depth is not an independent observation of
            # empty space, even where every requested body voxel has a ray.
            traversability[traversability == OBSERVED_CLEARANCE] = PARTIAL_CLEARANCE
        occupancy2d = np.zeros((nx, ny), dtype=np.uint8)
        occupancy2d[np.any(free, axis=2)] = 1
        occupancy2d[np.any(occupied, axis=2)] = 2
        counts = {CLASS_NAMES[k]: int(np.count_nonzero(traversability == k)) for k in CLASS_NAMES}
        summary = {"frame": self._frame_count - 1, "timestamp": timestamp, "input_points": len(points),
                   "tt_occupied_voxels": int(occupied.sum()), "collision_voxels": int(collision.sum()),
                   "floor_source_returns": floor_source_count,
                   "floor_provenance_available": fused_source_points is not None,
                   "ray_free_voxels": int(free.sum()),
                   "local_grid_shape": [int(n) for n in self.shape], "class_counts": counts,
                   "class_area_m2": {name: count * c.resolution ** 2 for name, count in counts.items()},
                   "ground": ground_info, "history_frames": len(self._history),
                   "strict_clearance_fraction": counts["observed_clearance"] / (nx * ny),
                   "partial_is_verified_free": False, "config": asdict(c)}
        summary["free_evidence_source"] = free_evidence_source
        if self._collision_voxelizer is not None:
            module=getattr(self._collision_voxelizer,'__module__',type(self._collision_voxelizer).__module__)
            name=getattr(self._collision_voxelizer,'__qualname__',type(self._collision_voxelizer).__name__)
            summary['collision_voxelizer']=f'{module}.{name}'
        summary["free_evidence_is_sensor_confirmed"] = free_evidence_source == "lidar_measurement"
        return TraversabilityResult(timestamp, traversability, occupancy2d, free, occupied, collision, support,
                                    coverage, plane, summary, c.resolution, c.bounds)
