"""Frozen official MapAnything metric depth for the independent camera route.

Only rectified RGB, pinhole K and fixed camera-rig calibration are accepted.
LiDAR returns, LiDAR depth, and LiDAR-derived time-varying poses cannot be
passed through this interface. Output XYZ is in the caller's fixed rig frame.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import gc
import hashlib
import json
import io
import os
import sys
import tarfile
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
MODEL_REPO = "facebook/map-anything-apache"
MODEL_REVISION = "00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a"
MODEL_SOURCE_COMMIT = "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9"
DINOV2_SOURCE_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DEFAULT_MODEL_DIR = PROJECT / "models/map-anything-apache"
MAPANYTHING_SOURCE = PROJECT / "models/mapanything-source"
DINOV2_SOURCE = PROJECT / "models/dinov2-source"


def download_sources():
    """Download exact public architecture sources without redundant model assets."""
    import requests
    sources = [("facebookresearch/map-anything", MODEL_SOURCE_COMMIT, MAPANYTHING_SOURCE,
                lambda name: name.startswith("mapanything/") or name in {"LICENSE", "README.md", "pyproject.toml", "setup.py"}),
               ("facebookresearch/dinov2", DINOV2_SOURCE_COMMIT, DINOV2_SOURCE,
                lambda name: name.startswith("dinov2/") or name in {"hubconf.py", "LICENSE", "README.md", "setup.py"})]
    for repo, commit, destination, include in sources:
        provenance = destination / "provenance.json"
        if provenance.exists() and json.loads(provenance.read_text()).get("commit") == commit:
            continue
        url = f"https://codeload.github.com/{repo}/tar.gz/{commit}"
        for attempt in range(6):
            try:
                response = requests.get(url, timeout=45); response.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 5:
                    raise
                time.sleep(1)
        archive = response.content
        files = {}
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                relative = Path(*Path(member.name).parts[1:])
                if ".." in relative.parts or relative.is_absolute():
                    raise RuntimeError("Unsafe official architecture archive path")
                if not include(relative.as_posix()):
                    continue
                content = tar.extractfile(member).read()
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                files[relative.as_posix()] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        provenance.write_text(json.dumps({"repository": "https://github.com/" + repo, "commit": commit,
                                          "archive_sha256": hashlib.sha256(archive).hexdigest(), "files": files}, indent=2))


def download_model(model_dir=DEFAULT_MODEL_DIR):
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download
    model_dir = Path(model_dir)
    snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_dir=model_dir,
                      cache_dir=model_dir.parent / "hf-cache", token=False,
                      allow_patterns=["config.json", "model.safetensors", "README.md", "LICENSE"])
    files = {}
    for p in model_dir.iterdir():
        if p.is_file() and p.name != "provenance.json":
            digest = hashlib.sha256()
            with p.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 ** 2), b""):
                    digest.update(chunk)
            files[p.name] = {"bytes": p.stat().st_size, "sha256": digest.hexdigest()}
    (model_dir / "provenance.json").write_text(json.dumps({"repository": MODEL_REPO,
                                                         "revision": MODEL_REVISION, "files": files}, indent=2))
    return model_dir


@contextmanager
def _local_dinov2_init():
    """Resolve DINO architecture locally, without downloading redundant weights.

    The complete official MapAnything checkpoint includes its DINO encoder.
    strict=True below guarantees no random initialized tensor remains in use.
    No source file or globally installed package is patched.
    """
    import torch
    if not (DINOV2_SOURCE / "hubconf.py").exists():
        raise FileNotFoundError(f"Missing pinned DINOv2 architecture: {DINOV2_SOURCE}")
    original = torch.hub.load
    def local_load(repo_or_dir, model, *args, **kwargs):
        if "facebookresearch/dinov2" in str(repo_or_dir):
            kwargs.pop("force_reload", None)
            kwargs.pop("trust_repo", None)
            kwargs["pretrained"] = False
            return original(str(DINOV2_SOURCE), model, *args, source="local", **kwargs)
        raise RuntimeError(f"Unexpected remote TorchHub architecture requested: {repo_or_dir}")
    torch.hub.load = local_load
    try:
        yield
    finally:
        torch.hub.load = original


def prepare_camera_views(views, image_size=224):
    """Resize RGB and K together, with deterministic DINOv2 normalization."""
    import torch
    import cv2
    if image_size < 112 or image_size % 14:
        raise ValueError("MapAnything image_size must be >=112 and a multiple of 14")
    if not views:
        raise ValueError("At least one camera is required")
    forbidden = {"depth", "depth_z", "points_lidar", "lidar", "T_world_lidar", "lidar_pose"}
    prepared, auxiliary = [], []
    mean = np.asarray([0.485, 0.456, 0.406], np.float32)
    std = np.asarray([0.229, 0.224, 0.225], np.float32)
    for i, view in enumerate(views):
        invalid_keys = forbidden.intersection(view)
        if invalid_keys:
            raise ValueError(f"Camera-only adapter forbids LiDAR/depth input: {sorted(invalid_keys)}")
        image = np.asarray(view.get("image", view.get("img")))
        K = np.asarray(view.get("K", view.get("intrinsics")), np.float32)
        pose = np.asarray(view.get("T_rig_camera", np.eye(4)), np.float32)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Each view must contain an RGB uint8 image")
        if K.shape != (3, 3) or pose.shape != (4, 4) or not np.isfinite(K).all() or not np.isfinite(pose).all():
            raise ValueError("Each view needs finite K 3x3 and T_rig_camera 4x4")
        if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(pose[3], [0, 0, 0, 1]):
            raise ValueError("Invalid camera calibration")
        if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-3):
            raise ValueError("Camera-rig rotation must be orthonormal")
        resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        resized_K = K.copy()
        resized_K[0] *= image_size / image.shape[1]
        resized_K[1] *= image_size / image.shape[0]
        normalized = ((resized.astype(np.float32) / 255 - mean) / std).transpose(2, 0, 1).copy()
        valid = np.asarray(view.get("valid_mask", np.ones(image.shape[:2], bool)), bool)
        if valid.shape != image.shape[:2]:
            raise ValueError("valid_mask must match the rectified RGB shape")
        validity_sampling=view.get('validity_sampling',view.get('valid_mask_sampling','legacy_opencv_nearest'))
        if validity_sampling=='pixel_center':
            # The source semantic lift uses pixel-center domain mapping. The
            # new route must mask exactly those pixels, especially the ego-body
            # polygon edge; old views retain their original nearest behavior.
            source_u=np.floor((np.arange(image_size)+.5)*valid.shape[1]/image_size).astype(int)
            source_v=np.floor((np.arange(image_size)+.5)*valid.shape[0]/image_size).astype(int)
            resized_valid=valid[source_v[:,None],source_u[None,:]]
        elif validity_sampling=='legacy_opencv_nearest':
            resized_valid = cv2.resize(valid.astype(np.uint8), (image_size, image_size), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            raise ValueError('Unknown camera valid-mask sampling convention')
        prepared.append({"img": torch.from_numpy(normalized)[None], "data_norm_type": ["dinov2"],
                         "intrinsics": torch.from_numpy(resized_K)[None],
                         "camera_poses": torch.from_numpy(pose)[None],
                         "is_metric_scale": torch.tensor([True]), "instance": [str(view.get("camera_id", i))],
                         "idx": [i], "true_shape": [(image_size, image_size)]})
        auxiliary.append({"image": resized, "K": resized_K, "pose": pose,
                          "valid_mask": resized_valid, "camera_id": str(view.get("camera_id", i)),
                          "view_id": str(view.get("view_id", view.get("camera_id", i))),
                          "validity_sampling":validity_sampling})
    return prepared, auxiliary


def predictions_to_points(predictions, auxiliary, *, pixel_stride=2,
                          max_range=20.0, min_range=0.25):
    """Convert metric camera pointmaps into fixed calibrated rig coordinates."""
    if pixel_stride < 1:
        raise ValueError("pixel_stride must be >=1")
    if len(predictions) != len(auxiliary):
        raise ValueError("Model prediction count differs from the camera count")
    def as_numpy(tensor):
        return tensor.detach().float().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    all_points, all_colors, all_origins, all_ids = [], [], [], []
    # Additive provenance: a point must receive semantics from the very pixel
    # which generated it, rather than an unrelated overlapping camera ray.
    all_uv, all_source_depth, all_source_confidence, all_camera_points = [], [], [], []
    depth_maps, confidence_maps, valid_masks = [], [], []
    for i, (prediction, view) in enumerate(zip(predictions, auxiliary)):
        camera_points = as_numpy(prediction["pts3d_cam"])[0]
        depth = as_numpy(prediction["depth_z"])[0].squeeze(-1)
        confidence = as_numpy(prediction["conf"])[0]
        model_mask = as_numpy(prediction["mask"])[0].squeeze(-1).astype(bool)
        valid = model_mask & view["valid_mask"] & np.isfinite(camera_points).all(-1)
        ranges = np.linalg.norm(camera_points, axis=-1)
        valid &= (ranges >= min_range) & (ranges <= max_range) & (camera_points[..., 2] > 0)
        sampled = np.zeros_like(valid); sampled[::pixel_stride, ::pixel_stride] = True
        selected = valid & sampled
        pose = view["pose"]
        points = camera_points[selected] @ pose[:3, :3].T + pose[:3, 3]
        all_points.append(points.astype(np.float32))
        all_origins.append(np.broadcast_to(pose[:3, 3], points.shape).copy())
        all_colors.append(view["image"][selected])
        all_ids.append(np.full(len(points), i, np.uint8))
        vv, uu = np.nonzero(selected)
        all_uv.append(np.stack([uu, vv], -1).astype(np.int32))
        all_source_depth.append(depth[selected].astype(np.float32))
        all_source_confidence.append(confidence[selected].astype(np.float32))
        all_camera_points.append(camera_points[selected].astype(np.float32))
        depth_maps.append(depth.astype(np.float32)); confidence_maps.append(confidence.astype(np.float32))
        valid_masks.append(valid)
    return {"points": np.concatenate(all_points), "origins": np.concatenate(all_origins),
            "colors": np.concatenate(all_colors), "camera_ids": np.concatenate(all_ids),
            "depth_maps": depth_maps, "confidence_maps": confidence_maps,
            "valid_masks": valid_masks,
            "source_uv": np.concatenate(all_uv),
            "source_view_ids": np.concatenate(all_ids),
            "source_depth": np.concatenate(all_source_depth),
            "source_confidence": np.concatenate(all_source_confidence),
            "source_camera_points": np.concatenate(all_camera_points),
            "source_poses": [np.asarray(v["pose"], np.float32) for v in auxiliary],
            "source_intrinsics": [np.asarray(v.get("K", np.eye(3)), np.float32) for v in auxiliary],
            "source_image_shapes": [list(v["image"].shape[:2]) for v in auxiliary],
            "source_validity_sampling": [v.get('validity_sampling','legacy_opencv_nearest') for v in auxiliary],
            "source_view_names": [str(v.get("view_id", v.get("camera_id", i))) for i, v in enumerate(auxiliary)]}


class MapAnythingDepth:
    def __init__(self, model_dir=DEFAULT_MODEL_DIR, device="auto", image_size=224,
                 pixel_stride=2, max_range=20.0, fallback_cpu=True):
        import torch
        from safetensors.torch import load_model
        self.model_dir = Path(model_dir).resolve()
        weights = self.model_dir / "model.safetensors"
        if not weights.exists():
            raise FileNotFoundError(f"Missing official MapAnything checkpoint: {weights}")
        if image_size < 112 or image_size % 14:
            raise ValueError("image_size must be a multiple of 14 and >=112")
        if device == "auto":
            free_bytes = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
            device = "cuda" if free_bytes >= 3 * 1024 ** 3 else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device, self.image_size = device, int(image_size)
        self.pixel_stride, self.max_range = int(pixel_stride), float(max_range)
        self.fallback_cpu = bool(fallback_cpu)
        sys.path.insert(0, str(MAPANYTHING_SOURCE))
        from mapanything.models import MapAnything
        config = json.loads((self.model_dir / "config.json").read_text())
        # This controls architecture initialization only. Every tensor is later
        # replaced by the complete, official MapAnything pretrained checkpoint.
        config["encoder_config"]["torch_hub_pretrained"] = False
        load_started = time.perf_counter()
        with _local_dinov2_init():
            self.model = MapAnything(**config).eval()
        # Official safetensors omit duplicate alias names (DPT heads are shared
        # under several module paths). load_model(strict=True) verifies complete
        # parameter coverage while preserving those aliases. A plain strict
        # state_dict load would incorrectly reject the legitimate shared names.
        missing, unexpected = load_model(self.model, str(weights), strict=True, device="cpu")
        if missing or unexpected:
            raise RuntimeError(f"Incomplete pretrained weights: {(missing, unexpected)}")
        gc.collect()
        self.model.requires_grad_(False)
        self.initial_fallback = None
        geometric_names = ["ray_dirs_encoder", "depth_encoder", "depth_scale_encoder",
                           "cam_rot_encoder", "cam_trans_encoder", "cam_trans_scale_encoder",
                           "fusion_norm_layer", "dense_head", "pose_head", "scale_head"]
        self._geometric_fp32 = geometric_names
        original_geometric_fusion = self.model._encode_and_fuse_optional_geometric_inputs
        def geometric_fusion_fp32(views, feature_list):
            # Official source disables autocast for geometry but assumes FP32
            # image features at its indexed ray assignment. Quantized bf16
            # backbones produce bf16 features, so restore the intended FP32
            # geometry branch explicitly, then return the backbone dtype.
            original_dtype = feature_list[0].dtype
            outputs = original_geometric_fusion(views, [x.float() for x in feature_list])
            return [x.to(dtype=original_dtype) for x in outputs]
        self.model._encode_and_fuse_optional_geometric_inputs = geometric_fusion_fp32
        original_downstream_head = self.model.downstream_head
        def downstream_head_fp32(dense_head_inputs, scale_head_inputs, **kwargs):
            if isinstance(dense_head_inputs, list):
                dense_head_inputs = [x.float() for x in dense_head_inputs]
            else:
                dense_head_inputs = dense_head_inputs.float()
            return original_downstream_head(dense_head_inputs=dense_head_inputs,
                                           scale_head_inputs=scale_head_inputs.float(), **kwargs)
        self.model.downstream_head = downstream_head_fp32
        try:
            if self.device.startswith("cuda"):
                # Keep the small optional-geometric encoders exactly FP32; they
                # run with autocast disabled in the official forward method.
                geometry_state = {name: {key: value.detach().clone() for key, value in getattr(self.model, name).state_dict().items()}
                                  for name in geometric_names}
                self.model.to(dtype=torch.bfloat16)
                for name in geometric_names:
                    module = getattr(self.model, name).float()
                    module.load_state_dict(geometry_state[name], strict=True)
                del geometry_state
                self.model.to(self.device)
            else:
                self.model.to(device="cpu", dtype=torch.float32)
        except torch.cuda.OutOfMemoryError:
            if not self.fallback_cpu:
                raise
            self.model.to(device="cpu", dtype=torch.float32)
            self.device = "cpu"; torch.cuda.empty_cache()
            self.initial_fallback = "CUDA OOM loading bf16 weights: using official model on CPU"
        self.load_seconds = time.perf_counter() - load_started
        self.last_info = {}

    def metadata(self):
        return {"provider": "Official pretrained MapAnything Apache",
                "repository": MODEL_REPO, "revision": MODEL_REVISION,
                "source": "Pinned official facebookresearch/map-anything; TT-Occ vendor predates this checkpoint's register-token config",
                "source_commit": MODEL_SOURCE_COMMIT, "dinov2_source_commit": DINOV2_SOURCE_COMMIT,
                "device": self.device, "image_size": self.image_size,
                "dtype": "bfloat16 backbone/transformer + float32 geometry encoders/heads" if self.device.startswith("cuda") else "float32",
                "bf16_geometry_dtype_adapter": True,
                "memory_efficient_inference": True, "dense_head_minibatch": 1,
                "input_depth": False, "input_lidar": False,
                "input_poses": "fixed camera-rig extrinsics only",
                "input_intrinsics": True, "trained_on_orbvi": False,
                "pixel_stride": self.pixel_stride, "max_range_m": self.max_range,
                "weights_loaded_strict": True, "load_seconds": self.load_seconds}

    def predict(self, views):
        import torch
        started = time.perf_counter()
        prepared, auxiliary = prepare_camera_views(views, self.image_size)
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        fallback = self.initial_fallback
        def infer():
            with torch.inference_mode():
                return self.model.infer(prepared, memory_efficient_inference=True, minibatch_size=1,
                              use_amp=self.device.startswith("cuda"), amp_dtype="bf16",
                              apply_mask=True, mask_edges=True,
                              apply_confidence_mask=True, confidence_percentile=10,
                              ignore_depth_inputs=True)
        try:
            predictions = infer()
        except torch.cuda.OutOfMemoryError:
            if not self.fallback_cpu or not self.device.startswith("cuda"):
                raise
            self.model.to(device="cpu", dtype=torch.float32)
            self.device = "cpu"; torch.cuda.empty_cache()
            # infer() moves validated input dictionaries in place. Build clean
            # CPU tensors again before retrying the complete real model.
            prepared, auxiliary = prepare_camera_views(views, self.image_size)
            fallback = "CUDA OOM in inference: reran complete official model on CPU"
            predictions = infer()
        result = predictions_to_points(predictions, auxiliary, pixel_stride=self.pixel_stride,
                                        max_range=self.max_range)
        self.last_info = {"runtime_ms": (time.perf_counter() - started) * 1000,
                          "device": self.device, "fallback": fallback,
                          "views": len(views), "point_count": len(result["points"]),
                          "valid_pixels_per_view": [int(x.sum()) for x in result["valid_masks"]],
                          "median_depth_m_per_view": [float(np.median(d[m])) if np.any(m) else None
                                                       for d, m in zip(result["depth_maps"], result["valid_masks"])],
                          "peak_allocated_gpu_mib": (torch.cuda.max_memory_allocated() / 2 ** 20
                                                      if self.device.startswith("cuda") else 0),
                          "input_lidar": False, "input_depth": False}
        result["diagnostics"] = self.last_info.copy()
        del predictions
        return result


def main():
    import argparse
    import cv2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--views-json", help="JSON list of image paths, K, T_rig_camera; RGB is read from image_path")
    parser.add_argument("--output", default="results/camera_smoke")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--pixel-stride", type=int, default=2)
    args = parser.parse_args()
    if args.download:
        download_sources()
        download_model(args.model_dir)
    if args.views_json:
        output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
        view_descriptors = json.loads(Path(args.views_json).read_text())
        views = []
        for descriptor in view_descriptors:
            bgr = cv2.imread(descriptor["image_path"])
            if bgr is None:
                raise FileNotFoundError(descriptor["image_path"])
            view = {k: v for k, v in descriptor.items() if k != "image_path"}
            view["image"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            views.append(view)
        provider = MapAnythingDepth(args.model_dir, args.device, args.image_size, args.pixel_stride)
        result = provider.predict(views)
        np.savez_compressed(output / "camera_points.npz", points=result["points"], origins=result["origins"],
                            colors=result["colors"], camera_ids=result["camera_ids"])
        for i, (depth, valid) in enumerate(zip(result["depth_maps"], result["valid_masks"])):
            np.savez_compressed(output / f"depth_{i}.npz", depth=depth, valid=valid,
                                confidence=result["confidence_maps"][i])
            image = cv2.applyColorMap(np.clip(depth / 20 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            image[~valid] = 0
            cv2.imwrite(str(output / f"depth_{i}.png"), image)
        report = {"model": provider.metadata(), "inference": result["diagnostics"],
                  "views_json": str(Path(args.views_json).resolve()),
                  "scope": "Real pretrained camera-depth smoke; this file alone does not certify a whole-bag camera run."}
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report["inference"], indent=2))


if __name__ == "__main__":
    main()
