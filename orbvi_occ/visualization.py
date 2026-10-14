"""Static figures from actual local grids and per-frame results.

No interpolation or synthetic scene geometry is used. Local PNGs have paired
JSON metadata and may be paired with TraversabilityResult.save NPZ source grids.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .traversability import TraversabilityResult

CLASS_LABELS = ["Unknown", "Blocked (footprint inflated)", "Partial clearance (unknown remains)", "Observed full clearance"]
CLASS_COLORS = ["#b8b8b8", "#d55e00", "#f0e442", "#009e73"]


def _pyplot():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt


def save_local_map(result: TraversabilityResult, path: str | Path,
                   title: str | None = None, source_npz: str | Path | None = None) -> Path:
    """Two panels use the same metric extent; unknown is explicitly rendered."""
    plt = _pyplot()
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Circle, Patch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    b = result.bounds
    extent = [b[0], b[3], b[1], b[4]]
    with plt.rc_context({"font.size": 10, "axes.titlesize": 11, "savefig.facecolor": "white"}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
        axes[0].imshow(result.occupancy2d.T, origin="lower", extent=extent,
                       cmap=ListedColormap(["#b8b8b8", "#56b4e9", "#d55e00"]),
                       norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3), interpolation="nearest")
        camera_predicted = result.summary.get("free_evidence_source") == "predicted_camera_depth"
        evidence_label = "Predicted camera-depth ray voxel" if camera_predicted else "At least one free LiDAR ray voxel"
        axes[0].set_title("TT-Occ occupied + " + ("predicted camera-depth rays" if camera_predicted else "LiDAR observed-ray evidence"))
        axes[0].legend(handles=[Patch(color=c, label=l) for c, l in zip(
            ["#b8b8b8", "#56b4e9", "#d55e00"],
            ["Unknown", evidence_label, "TT occupied voxel in column"])],
            loc="lower left", fontsize=8, framealpha=0.95)
        axes[1].imshow(result.traversability.T, origin="lower", extent=extent,
                       cmap=ListedColormap(CLASS_COLORS), norm=BoundaryNorm(np.arange(-0.5, 4.5), 4),
                       interpolation="nearest")
        axes[1].set_title("Local floor + footprint + vertical-clearance classification")
        axes[1].legend(handles=[Patch(color=c, label=l) for c, l in zip(CLASS_COLORS, CLASS_LABELS)],
                       loc="lower left", fontsize=8, framealpha=0.95)
        conf = result.summary.get("config", {})
        radius = 0.5 * np.hypot(conf.get("footprint_length", 0.7), conf.get("footprint_width", 0.55)) + conf.get("safety_margin", 0.1)
        for ax in axes:
            ax.plot(0, 0, marker="+", color="black", markersize=10)
            ax.add_patch(Circle((0, 0), radius, fill=False, edgecolor="black", linestyle="--", linewidth=1))
            ax.arrow(0, 0, 1.5, 0, width=0.025, head_width=0.3, color="black", length_includes_head=True)
            ax.set(xlabel="LiDAR-frame x, forward (m)", ylabel="LiDAR-frame y, left (m)",
                   xlim=extent[:2], ylim=extent[2:], aspect="equal")
        fig.suptitle(title or f"Frame {result.summary.get('frame')} | timestamp {result.timestamp} | {result.resolution:g} m voxels")
        fig.savefig(path, dpi=130, facecolor="white")
        plt.close(fig)
    metadata = {"source_npz": str(source_npz) if source_npz is not None else None,
                "timestamp": result.timestamp, "bounds_m": list(b), "voxel_size_m": result.resolution,
                "axis_order": "source arrays [x,y,z]; figure x forward horizontal, y left vertical",
                "classes": dict(enumerate(CLASS_LABELS)), "colors": CLASS_COLORS,
                "transformations": ["XY column projection for occupancy panel", "nearest-neighbour raster display, no smoothing"],
                "partial_clearance_is_confirmed_free": False,
                "free_evidence_source": result.summary.get("free_evidence_source", "lidar_measurement"),
                "alt_text": "Two local XY maps. Grey is unknown. TT occupied geometry is orange. Yellow clearance retains unobserved volume; green requires observed empty body volume over the full footprint.",
                "summary": result.summary}
    path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_summary_plot(records: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Plot full-run per-frame areas and latency without inventing missing samples."""
    rows = list(records)
    if not rows:
        raise ValueError("no frame records to plot")
    plt = _pyplot()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.array([float(r.get("timestamp", r.get("frame", i))) for i, r in enumerate(rows)])
    t = t - t[0]
    areas = {key: np.array([r.get("class_area_m2", {}).get(key, np.nan) for r in rows], dtype=float)
             for key in ("unknown", "blocked", "partial_clearance", "observed_clearance")}
    with plt.rc_context({"font.size": 10, "savefig.facecolor": "white"}):
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained")
        for color, label, values in zip(CLASS_COLORS, CLASS_LABELS, areas.values()):
            axes[0].plot(t, values, color=color, linewidth=0.8, label=label)
        axes[0].set(ylabel="Local classified area (m²)", ylim=(0, None), title="Every processed LiDAR frame")
        axes[0].legend(fontsize=8)
        latency = np.array([r.get("processing_seconds", np.nan) for r in rows], dtype=float)
        axes[1].plot(t, latency, color="#0072b2", linewidth=0.8)
        axes[1].set(xlabel="Elapsed LiDAR timestamp (s)", ylabel="Frame processing time (s)", ylim=(0, None))
        fig.savefig(path, dpi=130, facecolor="white")
        plt.close(fig)
    path.with_suffix(".json").write_text(json.dumps({"frames": len(rows), "source": "per-frame result records",
        "transformations": ["timestamp relative to first processed frame", "no smoothing or interpolation"],
        "missing_values": "retained as NaN gaps", "partial_clearance_is_confirmed_free": False}, indent=2), encoding="utf-8")
    return path


def save_geometry_xy(occupied_world_xyz: np.ndarray, path: str | Path,
                     trajectory_world_xyz: np.ndarray | None = None,
                     max_points: int = 200_000, title: str = "Accumulated occupied geometry") -> Path:
    """Actual world-frame geometry; deterministic display subsampling is disclosed."""
    plt = _pyplot()
    points = np.asarray(occupied_world_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("occupied_world_xyz must be Nx3")
    points = points[np.isfinite(points[:, :3]).all(axis=1), :3]
    original_count = len(points)
    if len(points) > max_points:
        points = points[np.linspace(0, len(points) - 1, max_points).astype(int)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 10}):
        fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
        if len(points):
            scatter = ax.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=1, cmap="viridis", rasterized=True)
            fig.colorbar(scatter, ax=ax, label="World z (m)")
        if trajectory_world_xyz is not None:
            trajectory = np.asarray(trajectory_world_xyz)
            ax.plot(trajectory[:, 0], trajectory[:, 1], color="#d55e00", linewidth=1, label="Estimated LiDAR trajectory")
            ax.legend()
        ax.set(xlabel="World x (m)", ylabel="World y (m)", aspect="equal", title=title)
        fig.savefig(path, dpi=140, facecolor="white")
        plt.close(fig)
    path.with_suffix(".json").write_text(json.dumps({"source_points": original_count, "displayed_points": len(points),
        "display_subsampling": "uniform source-row indices only when source exceeds configured limit",
        "coordinates": "world metres", "z_color": "actual occupied voxel center height"}, indent=2), encoding="utf-8")
    return path
