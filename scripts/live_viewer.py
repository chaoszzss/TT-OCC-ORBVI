#!/usr/bin/env python3
"""Read-only, localhost viewer for frames committed by either TT-Occ route.

The frame journal is the commit boundary: unlogged NPZ files and an unterminated
journal line are never displayed. The server neither imports nor runs inference.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import io
from pathlib import Path
import re
import threading
from urllib.parse import parse_qs, urlsplit
import zipfile
import zlib

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
PAGE = Path(__file__).with_suffix(".html")

# Official TT-Occ palette, keyed by meaning, never by an ADE numeric index.
OFFICIAL_PALETTE = {
    "noise": [200, 200, 200], "barrier": [255, 120, 50],
    "bicycle": [255, 192, 203], "bus": [255, 255, 0],
    "car": [0, 150, 245], "construction_vehicle": [0, 255, 255],
    "motorcycle": [255, 127, 0], "pedestrian": [255, 0, 0],
    "traffic_cone": [255, 240, 150], "trailer": [135, 60, 0],
    "truck": [160, 32, 240], "driveable_surface": [255, 0, 255],
    "other_flat": [175, 0, 75], "sidewalk": [75, 0, 75],
    "terrain": [150, 240, 80], "manmade": [230, 230, 250],
    "vegetation": [0, 175, 0], "sky": [100, 150, 190],
    "unknown": [145, 151, 163],
}
EXPERT_REGISTRY = {
    0: {"name": "官方 OpenSeeD", "color": [54, 94, 210]},
    1: {"name": "室内 ADE 补充未知", "color": [245, 155, 50]},
    2: {"name": "室内 ADE 细分柱", "color": [20, 166, 132]},
    255: {"name": "语义未知", "color": [145, 145, 145]},
}
SEMANTIC_VARIANTS = {"final": "labels", "official": "official_labels",
                     "indoor": "b5_labels", "expert": "expert_id"}

ALIASES = {"floor": "driveable_surface", "road": "driveable_surface",
           "person": "pedestrian", "van": "car", "minibike": "motorcycle",
           "wall": "manmade", "building": "manmade", "ceiling": "manmade",
           "column": "manmade", "door": "manmade", "pillar": "manmade",
           "fence": "barrier", "railing": "barrier", "grass": "vegetation",
           "tree": "vegetation", "plant": "vegetation", "earth": "terrain"}
DISPLAY_NAMES = {"wall": "墙", "floor": "地面", "ceiling": "顶棚", "column": "柱",
                 "pillar": "柱", "car": "车辆", "person": "行人", "pedestrian": "行人",
                 "door": "门", "other_obstacle": "其他障碍", "unknown": "语义未知"}


def semantic_registry(metadata):
    """Normalize explicit native registries and legacy ADE class names safely."""
    source = metadata.get("semantic_registry") or metadata.get("semantics", {})
    if not isinstance(source, dict):
        source = {}
    raw = source.get("classes", source.get("class_names", {}))
    if isinstance(raw, dict):
        raw = [{"id": key, **(value if isinstance(value, dict) else {"name": value})}
               for key, value in raw.items()]
    if not isinstance(raw, list):
        raw = []
    classes, seen = [], set()
    for offset, value in enumerate(raw):
        value = {"id": offset, "name": value} if isinstance(value, str) else value
        if not isinstance(value, dict):
            continue
        try:
            cid = int(value.get("id", offset))
        except (ValueError, TypeError):
            continue
        if not 0 <= cid < 255 or cid in seen:
            continue
        name = str(value.get("name", f"class_{cid}")).strip()
        mapping = value.get("nuscenes_name", value.get("official_name", ALIASES.get(name, name)))
        color = value.get("color", OFFICIAL_PALETTE.get(mapping, [150, 140, 130]))
        if (not isinstance(color, (list, tuple)) or len(color) != 3
                or any(not isinstance(x, (int, float)) or not np.isfinite(x) or not 0 <= x <= 255 for x in color)):
            color = OFFICIAL_PALETTE.get(mapping, [150, 140, 130])
        classes.append({"id": cid, "name": name,
                        "display_name": str(value.get("display_name", value.get("name_zh", DISPLAY_NAMES.get(name, name)))),
                        "color": [int(x) for x in color], "nuscenes_name": mapping})
        seen.add(cid)
    classes.append({"id": 255, "name": "unknown", "display_name": "语义未知",
                    "color": source.get("unknown", {}).get("color", OFFICIAL_PALETTE["unknown"]), "nuscenes_name": None})
    return {"schema_version": source.get("name", source.get("schema_version", source.get("schema", "legacy-ade20k" if classes[:-1] else "unspecified"))),
            "unknown_id": 255, "classes": classes,
            "source_registry": source.get("source_registry", source.get("sourceRegistry")),
            "palette_note": "Official TT-Occ colors mapped by class meaning. Floor color does not establish driveability."}


def _contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("artifact path escapes its declared directory")
    return path


def _reject_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def _json_loads(value):
    return json.loads(value, parse_constant=_reject_constant)


def _fingerprint(path):
    try:
        stat = path.stat()
        return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev
    except FileNotFoundError:
        return None


class ArtifactsPending(Exception):
    """An archive is missing or cannot yet be decoded as complete NPZ data."""


def _artifact_fingerprint(path):
    """Fingerprint NPZ contents using ZIP directory CRCs, without decoding arrays.

    Same-size rewrites can share a filesystem timestamp tick. Per-member CRCs
    and ZIP layout distinguish those writes cheaply even when stat metadata is
    identical; full array archives are decoded only on a new cache key.
    """
    stat = _fingerprint(path)
    if stat is None:
        return None
    try:
        with zipfile.ZipFile(path, "r") as archive:
            directory = tuple((entry.filename, entry.CRC, entry.file_size,
                               entry.compress_size, entry.compress_type,
                               entry.header_offset) for entry in archive.infolist())
    except (OSError, EOFError, zipfile.BadZipFile, zlib.error, ValueError) as exc:
        raise ArtifactsPending(f"Invalid NPZ ZIP directory: {type(exc).__name__}") from exc
    return stat, directory


def _read_arrays(path, names):
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in names}
    except (OSError, EOFError, zipfile.BadZipFile, zlib.error, KeyError, ValueError) as exc:
        raise ArtifactsPending(type(exc).__name__) from exc


class RunReader:
    """Incrementally read committed rows and a small cache of artifact payloads."""

    def __init__(self, run_directory):
        self.run = Path(run_directory).resolve()
        self.lock = threading.Lock()
        self._journal_stat = None
        self._journal_offset = 0
        self._pending = b""
        self._rows = OrderedDict()
        self._cache = OrderedDict()
        self._verified_frames = set()
        self._prepared_index = None

    def _reset(self):
        self._journal_offset = 0
        self._pending = b""
        self._rows.clear()
        self._cache.clear()
        self._verified_frames.clear()

    def _refresh_journal(self):
        path = self.run / "frames.jsonl"
        stat = _fingerprint(path)
        if stat == self._journal_stat:
            return
        if stat is None:
            self._reset()
            self._journal_stat = None
            return
        if (self._journal_stat is None or stat[0] != self._journal_stat[0]
                or stat[4] != self._journal_stat[4]
                or stat[1] <= self._journal_offset):
            self._reset()
        with path.open("rb") as journal:
            journal.seek(self._journal_offset)
            extra = journal.read()
        data = self._pending + extra
        lines = data.split(b"\n")
        # Validate all new complete lines before advancing the reader offset.
        additions = []
        new_ids = set()
        for line in lines[:-1]:
            if not line.strip():
                continue
            row = _json_loads(line)
            index = row.get("index") if isinstance(row, dict) else None
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ValueError("committed journal row has an invalid frame index")
            if index in self._rows or index in new_ids:
                raise ValueError("committed journal contains duplicate frame indices")
            stamp = row.get("stamp_ns")
            if isinstance(stamp, bool) or not isinstance(stamp, int):
                raise ValueError("committed journal row has an invalid stamp_ns")
            additions.append(row)
            new_ids.add(index)
        for row in additions:
            self._rows[row["index"]] = row
        self._pending = lines[-1]
        self._journal_offset += len(extra)
        self._journal_stat = stat

    def _metadata(self, name):
        try:
            value = _json_loads((self.run / name).read_bytes())
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            # These files may be between truncate and close during a run.
            return {}

    def frame(self, index="latest", mode="fused"):
        if mode not in ("fused", "current"):
            raise ValueError("mode must be fused or current")
        if index != "latest" and (isinstance(index, bool) or not isinstance(index, int) or index < 0):
            raise ValueError("index must be latest or a nonnegative integer")
        with self.lock:
            try:
                self._refresh_journal()
            except (OSError, ValueError, UnicodeError) as exc:
                return {"status": "error", "status_detail": f"Invalid committed frame journal: {exc}"}, 500
            metadata = self._metadata("run.json")
            expected = metadata.get("expected_full_recording_frames")
            if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
                expected = None
            summary = self._metadata("summary.json")
            complete = (summary.get("complete") is True and expected is not None
                        and list(self._rows) == list(range(expected))
                        and metadata.get("limited_development_run") is not True)
            base = {"semantic_registry": semantic_registry(metadata), "pipeline": metadata.get("pipeline"),
                    "presentation": metadata.get("presentation"), "route": metadata.get("route"), "processed_frames": len(self._rows),
                    "expected_frames": expected, "complete": complete}
            if not self._rows:
                return {**base, "status": "waiting", "status_detail": "Waiting for the first committed frame."}, 200
            selected = next(reversed(self._rows)) if index == "latest" else index
            if selected not in self._rows:
                return {**base, "status": "waiting", "status_detail": f"Frame {selected} has not been committed."}, 200
            row = self._rows[selected]
            occupancy_path = self.run / "occupancy" / f"{selected:06d}.npz"
            grid_path = self.run / "traversability" / f"{selected:06d}.npz"
            verified_before = selected in self._verified_frames
            try:
                stat = _artifact_fingerprint(occupancy_path), _artifact_fingerprint(grid_path)
            except ArtifactsPending as exc:
                return {**base, "index": selected, "status": "error" if verified_before else "waiting",
                        "status_detail": f"Committed frame archive is corrupt or incomplete: {exc}"}, 500 if verified_before else 200
            if None in stat:
                return {**base, "index": selected, "status": "waiting",
                        "status_detail": "Committed frame artifacts are not yet available."}, 200
            key = selected, mode, stat
            if key not in self._cache:
                try:
                    payload = self._artifacts(occupancy_path, grid_path, row, mode)
                except ArtifactsPending as exc:
                    return {**base, "index": selected, "status": "error" if verified_before else "waiting",
                            "status_detail": f"Frame artifacts are corrupt or incomplete: {exc}"}, 500 if verified_before else 200
                except (ValueError, TypeError, OverflowError) as exc:
                    return {**base, "index": selected, "status": "error",
                            "status_detail": f"Invalid committed artifacts: {exc}"}, 500
                try:
                    after = _artifact_fingerprint(occupancy_path), _artifact_fingerprint(grid_path)
                except ArtifactsPending as exc:
                    return {**base, "index": selected, "status": "error" if verified_before else "waiting",
                            "status_detail": f"Frame archive changed while being read: {exc}"}, 500 if verified_before else 200
                if after != stat:
                    return {**base, "index": selected, "status": "waiting",
                            "status_detail": "Committed frame archives changed during reading; retrying a consistent snapshot."}, 200
                self._cache[key] = payload
                while len(self._cache) > 4:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(key)
            self._verified_frames.add(selected)
            return {**base, **self._cache[key], "status": "ready"}, 200

    @staticmethod
    def _artifacts(occupancy_path, grid_path, row, mode):
        occ = _read_arrays(occupancy_path, ("bounds", "voxel_size", "stamp_ns", "current_indices", "fused_indices",
                                          f"{mode}_labels"))
        bounds = np.asarray(occ["bounds"], dtype=np.float64)
        size = float(occ["voxel_size"])
        stamp = int(occ["stamp_ns"])
        indices = np.asarray(occ[f"{mode}_indices"])
        labels = np.asarray(occ[f"{mode}_labels"])
        current_count = len(occ["current_indices"])
        fused_count = len(occ["fused_indices"])
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
            raise ValueError("invalid occupancy bounds")
        if not np.isfinite(size) or size <= 0:
            raise ValueError("invalid voxel size")
        if stamp != row["stamp_ns"]:
            raise ValueError("artifact stamp_ns does not match committed row")
        if indices.ndim != 2 or indices.shape[1] != 3 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("voxel indices must be an integer Nx3 array")
        shape = np.ceil((bounds[1] - bounds[0]) / size).astype(np.int64)
        if np.any(indices < 0) or np.any(indices >= shape):
            raise ValueError("voxel indices are outside configured bounds")
        if (labels.shape != (len(indices),) or not np.issubdtype(labels.dtype, np.integer)
                or np.any(labels < 0) or np.any(labels > 255)):
            raise ValueError("labels must be uint8-compatible, one per voxel")
        if (current_count != row.get("current_occupied_voxels")
                or fused_count != row.get("fused_occupied_voxels")):
            raise ValueError("artifact voxel counts do not match committed row")
        centers = bounds[0] + (indices.astype(np.float64) + 0.5) * size
        stored = _read_arrays(grid_path, ("traversability", "bounds", "resolution", "timestamp"))
        traversability = np.asarray(stored["traversability"])
        grid_bounds = np.asarray(stored["bounds"], dtype=np.float64)
        resolution = float(stored["resolution"])
        timestamp = float(stored["timestamp"])
        if grid_bounds.shape != (6,) or not np.isfinite(grid_bounds).all() or np.any(grid_bounds[3:] <= grid_bounds[:3]):
            raise ValueError("invalid traversability bounds")
        if not np.isfinite(resolution) or resolution <= 0 or not np.isfinite(timestamp):
            raise ValueError("invalid grid resolution or timestamp")
        expected_shape = tuple(np.ceil((grid_bounds[3:5] - grid_bounds[:2]) / resolution).astype(int))
        if (traversability.shape != expected_shape or not np.issubdtype(traversability.dtype, np.integer)
                or np.any(traversability < 0) or np.any(traversability > 3)):
            raise ValueError("traversability must be an x-major 2D grid with classes 0..3")
        if abs(timestamp - stamp / 1e9) > 1e-5:
            raise ValueError("grid timestamp does not match committed row")
        grid = {"shape": list(traversability.shape), "resolution": resolution,
                "bounds": grid_bounds.tolist(), "traversability": traversability.ravel(order="C").tolist()}
        aux = {key: row[key] for key in ("processing_seconds", "new_voxels_from_history", "class_area_m2", "ground", "auxiliary") if key in row}
        return {"index": row["index"], "stamp_ns": stamp, "stamp_ns_text": str(stamp), "timestamp": timestamp,
                "mode": mode, "current_count": current_count, "fused_count": fused_count,
                "voxel_size": size, "bounds2x3": bounds.tolist(), "bounds": bounds.tolist(),
                "centers": centers.tolist(), "labels": labels.astype(np.uint8).tolist(),
                "confidence": RunReader._confidence(occupancy_path, mode, len(indices)),
                "semantic_counts": {str(int(cid)): int(n) for cid, n in zip(*np.unique(labels, return_counts=True))},
                "grid": grid, "rowAux": aux}


    @staticmethod
    def _confidence(path, mode, count):
        with np.load(path, allow_pickle=False) as stored:
            name = f"{mode}_confidence"
            if name not in stored.files:
                return None
            values = np.asarray(stored[name], dtype=np.float64)
        if values.shape != (count,) or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError("confidence must be finite [0,1], one per voxel")
        return values.tolist()

    def sources(self, index):
        """Expose only image references of a committed frame, not arbitrary files."""
        payload, code = self.frame(index)
        if code != 200 or payload.get("status") != "ready":
            return payload, code
        metadata = self._metadata("run.json")
        prepared = Path(metadata.get("config", {}).get("prepared", "prepared"))
        if not prepared.is_absolute():
            prepared = PROJECT / prepared
        prepared = prepared.resolve()
        index_path = prepared / "frames.jsonl"
        fingerprint = _fingerprint(index_path)
        if self._prepared_index is None or self._prepared_index[0] != fingerprint:
            try:
                records = [_json_loads(line) for line in index_path.read_bytes().splitlines() if line.strip()]
                self._prepared_index = fingerprint, {int(row["index"]): row for row in records}
            except (OSError, ValueError, KeyError, TypeError):
                return {"status": "unavailable", "index": index, "sources": [], "detail": "Prepared RGB index unavailable."}, 200
        row = self._prepared_index[1].get(index, {})
        sources = []
        views_path = self.run / "views" / f"{index:06d}.json"
        try:
            views = _json_loads(views_path.read_bytes())
            views = views.get("views", []) if isinstance(views, dict) else views
        except (OSError, ValueError):
            views = []
        if isinstance(views, list) and views:
            for view in views:
                view_id = str(view.get("view_id", ""))
                camera = str(view.get("physical_camera_id", ""))
                if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", view_id) or not re.fullmatch(r"[0-9]+", camera):
                    continue
                try:
                    rgb = _contained(self.run, view.get("semantic_image_path", view["image_path"]))
                    semantic = _contained(self.run, view["semantic_maps_path"])
                    raw_source = view.get("source_image_path", row.get("images", {}).get(camera, {}).get("path"))
                    raw = _contained(prepared, raw_source) if raw_source else None
                except (KeyError, TypeError, ValueError):
                    continue
                source_stamp = str(view.get("source_stamp_ns", ""))
                sources.append({"camera": camera, "view_id": view_id, "stamp_ns_text": source_stamp,
                                "rgb_available": rgb.is_file(), "semantic_available": semantic.is_file(),
                                "rgb_url": f"/api/image?index={index}&camera={view_id}&kind=rgb",
                                "semantic_url": f"/api/image?index={index}&camera={view_id}&kind=semantic",
                                "raw_url": f"/api/image?index={index}&camera={view_id}&kind=raw" if raw and raw.is_file() else None,
                                "source": str(rgb.relative_to(self.run)), "semantic_path": str(semantic.relative_to(self.run)),
                                "raw_source": raw_source, "pixel_domain": view.get("pixel_domain", "unspecified"),
                                "K": view.get("K"), "T_rig_camera": view.get("T_rig_camera")})
        else:
            for camera, reference in row.get("images", {}).items():
                if not re.fullmatch(r"[0-9]+", str(camera)):
                    continue
                try:
                    rgb = _contained(prepared, reference["path"])
                except (KeyError, ValueError):
                    continue
                semantic = self.run / "semantic_maps" / f"{index:06d}_{camera}.npz"
                sources.append({"camera": str(camera), "stamp_ns_text": str(reference.get("stamp_ns", "")),
                                "delta_ns": reference.get("delta_ns"), "valid_sync": reference.get("valid_sync"),
                                "rgb_available": rgb.is_file(), "semantic_available": semantic.is_file(),
                                "rgb_url": f"/api/image?index={index}&camera={camera}&kind=rgb",
                                "semantic_url": f"/api/image?index={index}&camera={camera}&kind=semantic",
                                "source": reference.get("path"), "pixel_domain": "raw_fisheye"})
        for item in sources:
            path = _contained(self.run, item.get("semantic_path", f"semantic_maps/{index:06d}_{item['camera']}.npz"))
            variants, counts = self._semantic_options(path)
            view = item.get("view_id", item["camera"])
            item["semantic_variants"] = variants
            item["semantic_urls"] = {variant: f"/api/image?index={index}&camera={view}&kind={('semantic' if variant=='final' else variant)}" for variant in variants}
            item["expert_counts"] = counts
        lineage = self.run / "source_points" / f"{index:06d}.npz"
        if not lineage.is_file():
            lineage = self.run / "point_provenance" / f"{index:06d}.npz"
        return {"status": "ready", "index": index, "sources": sources,
                "point_provenance_available": lineage.is_file(),
                "point_provenance": str(lineage.relative_to(self.run)) if lineage.is_file() else None,
                "association_note": "2D labels are the saved inference output. Source camera and UV lineage is a separate point artifact; no LiDAR semantics are synthesized."}, 200

    @staticmethod
    def _semantic_options(path):
        try:
            with np.load(path, allow_pickle=False) as stored:
                variants = [variant for variant, name in SEMANTIC_VARIANTS.items() if name in stored.files]
                counts = {str(int(cid)): int(n) for cid, n in zip(*np.unique(stored["expert_id"],return_counts=True))} if "expert_id" in stored.files else None
            return variants, counts
        except (OSError, ValueError, zipfile.BadZipFile, EOFError, zlib.error):
            return [], None

    def image(self, index, camera, kind):
        source_payload, code = self.sources(index)
        if code != 200 or source_payload.get("status") != "ready":
            raise FileNotFoundError("frame sources not available")
        source = next((item for item in source_payload["sources"] if item.get("view_id", item["camera"]) == camera), None)
        if source is None:
            raise FileNotFoundError("camera not present in committed source frame")
        if kind in ("rgb", "raw"):
            metadata = self._metadata("run.json")
            prepared = Path(metadata.get("config", {}).get("prepared", "prepared"))
            if not prepared.is_absolute():
                prepared = PROJECT / prepared
            if source.get("view_id") and kind == "rgb":
                image = _contained(self.run, source["source"])
            else:
                image = _contained(prepared, source.get("raw_source", source["source"]))
            return image.read_bytes(), "image/jpeg"
        variant = "final" if kind == "semantic" else kind
        if variant not in SEMANTIC_VARIANTS:
            raise ValueError("kind must be rgb, raw, semantic, official, indoor or expert")
        if variant not in source.get("semantic_variants", []):
            raise FileNotFoundError(f"Saved {variant} semantic output is unavailable for this provider")
        from PIL import Image
        path = _contained(self.run, source.get("semantic_path", f"semantic_maps/{index:06d}_{camera}.npz"))
        field = SEMANTIC_VARIANTS[variant]
        labels = _read_arrays(path, (field,))[field]
        if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0) or np.any(labels > 255):
            raise ValueError("invalid saved pixel semantic labels")
        if variant == "expert":
            if np.any(~np.isin(labels, list(EXPERT_REGISTRY))):
                raise ValueError("expert_id must be 0=official, 1=supplement, 2=column refinement or 255=unknown")
            palette = np.tile(np.array(EXPERT_REGISTRY[255]["color"],dtype=np.uint8),(256,1))
            for expert_id, entry in EXPERT_REGISTRY.items():
                palette[expert_id] = entry["color"]
            buffer = io.BytesIO()
            Image.fromarray(palette[labels.astype(np.uint8)]).save(buffer,format="PNG")
            return buffer.getvalue(), "image/png"
        # New caches can declare their own source registry (e.g. preserved ADE).
        with np.load(path, allow_pickle=False) as stored:
            source_registry = stored["semantic_registry_json"].item() if "semantic_registry_json" in stored.files else None
        metadata = self._metadata("run.json")
        if source_registry:
            registry = semantic_registry({"semantic_registry": _json_loads(str(source_registry))})
        elif metadata.get("semantic_registry", {}).get("pixel_registry"):
            registry = semantic_registry({"semantic_registry": metadata["semantic_registry"]["pixel_registry"]})
        else:
            registry = semantic_registry(metadata)
        if variant in ("official", "indoor") and np.any(~np.isin(labels, [entry["id"] for entry in registry["classes"]])):
            raise ValueError("Saved provider labels do not belong to the native semantic registry; raw ADE IDs must not use the garage palette")
        palette = np.tile(np.array(OFFICIAL_PALETTE["unknown"], dtype=np.uint8), (256, 1))
        for entry in registry["classes"]:
            palette[entry["id"]] = entry["color"]
        buffer = io.BytesIO()
        Image.fromarray(palette[labels.astype(np.uint8)]).save(buffer, format="PNG")
        return buffer.getvalue(), "image/png"


def make_server(run_directory, host="127.0.0.1", port=8766, page_path=PAGE):
    """Create a server without starting a background task; port=0 is useful in tests."""
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("live viewer binds only to localhost")
    reader = RunReader(run_directory)
    page_path = Path(page_path)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Polls happen twice a second; don't flood inference terminals.
            pass

        def _send(self, code, body, content_type):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _json(self, code, payload):
            self._send(code, json.dumps(payload, ensure_ascii=False, allow_nan=False,
                                       separators=(",", ":")).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path in ("/", "/index.html"):
                try:
                    body = page_path.read_bytes()
                except OSError:
                    self._json(503, {"status": "error", "status_detail": "Viewer page is unavailable."})
                    return
                self._send(200, body, "text/html; charset=utf-8")
            elif parsed.path == "/favicon.ico" and not parsed.query:
                self._send(204, b"", "image/x-icon")
            elif parsed.path in ("/api/sources", "/api/image"):
                try:
                    query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True, max_num_fields=5)
                    permitted = {"index"} if parsed.path == "/api/sources" else {"index", "camera", "kind"}
                    if set(query) != permitted or any(len(v) != 1 for v in query.values()):
                        raise ValueError("invalid source request parameters")
                    raw_index = query["index"][0]
                    if not re.fullmatch(r"[0-9]{1,12}", raw_index):
                        raise ValueError("index must be a nonnegative integer")
                    index = int(raw_index)
                    if parsed.path == "/api/sources":
                        payload, code = reader.sources(index)
                        self._json(code, payload)
                    else:
                        camera, kind = query["camera"][0], query["kind"][0]
                        if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", camera) or kind not in ("rgb", "raw", "semantic", "official", "indoor", "expert"):
                            raise ValueError("invalid camera or image kind")
                        body, content_type = reader.image(index, camera, kind)
                        self._send(200, body, content_type)
                except (FileNotFoundError, ArtifactsPending) as exc:
                    self._json(404, {"status": "unavailable", "status_detail": str(exc)})
                except (ValueError, KeyError, OSError) as exc:
                    self._json(400, {"status": "error", "status_detail": str(exc)})
            elif parsed.path == "/api/frame":
                try:
                    query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True,
                                     max_num_fields=8)
                    if set(query) - {"index", "mode"} or any(len(v) != 1 for v in query.values()):
                        raise ValueError("only one index and mode query parameter are permitted")
                    index = query.get("index", ["latest"])[0]
                    if index != "latest":
                        if not re.fullmatch(r"[0-9]{1,12}", index):
                            raise ValueError("index must be latest or a nonnegative integer")
                        index = int(index)
                    mode = query.get("mode", ["fused"])[0]
                    payload, code = reader.frame(index, mode)
                except ValueError as exc:
                    self._json(400, {"status": "error", "status_detail": str(exc)})
                    return
                self._json(code, payload)
            else:
                self._json(404, {"status": "error", "status_detail": "Unknown endpoint."})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/lidar_retest", help="Output directory to watch (it may not exist yet)")
    parser.add_argument("--host", choices=("127.0.0.1", "localhost"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")
    run = Path(args.run)
    if not run.is_absolute():
        run = PROJECT / run
    try:
        server = make_server(run, args.host, args.port)
    except OSError as exc:
        parser.exit(1, f"Cannot start live viewer: {exc}\n")
    print(f"Watching {run.resolve()}\nOpen http://{args.host}:{args.port}/ (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
