"""Real frozen pretrained image semantics for the ORBVI TT-Occ adapter.

SegFormer B0 is a lightweight closed-vocabulary replacement for upstream
OpenSeeD/Rex-Omni. ADE20K IDs are retained; they are not nuScenes class IDs.
No image-to-LiDAR semantic projection is permitted without calibrated geometry.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import time

import numpy as np

MODEL_REPO = "nvidia/segformer-b0-finetuned-ade-512-512"
MODEL_REVISION = "489d5cd81a0b59fab9b7ea758d3548ebe99677da"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models/segformer-b0-ade20k"
UNKNOWN_SEMANTIC = 255


def download_model(model_dir=DEFAULT_MODEL_DIR):
    """Fetch an immutable public checkpoint into the project, without HF tokens."""
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download
    model_dir = Path(model_dir)
    snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_dir=model_dir,
                      cache_dir=model_dir.parent / "hf-cache", token=False,
                      allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors", "README.md"])
    files = {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in model_dir.iterdir() if p.is_file() and p.name != "provenance.json"}
    (model_dir / "provenance.json").write_text(json.dumps({"repository": MODEL_REPO,
                                                         "revision": MODEL_REVISION, "files": files}, indent=2))
    return model_dir


class SegformerSemantics:
    def __init__(self, model_dir=DEFAULT_MODEL_DIR, device="auto", image_size=512,
                 fallback_cpu=True):
        import torch
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        self.model_dir = Path(model_dir).resolve()
        if not (self.model_dir / "model.safetensors").exists():
            raise FileNotFoundError(f"Pretrained semantics are missing: {self.model_dir}. Run scripts/setup_tt_backend.sh")
        if image_size < 128 or image_size % 32:
            raise ValueError("SegFormer image_size must be >=128 and divisible by 32")
        if device == "auto":
            free_bytes = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
            device = "cuda" if free_bytes >= 1024 ** 3 else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device = device
        self.image_size = int(image_size)
        self.fallback_cpu = bool(fallback_cpu)
        self.processor = SegformerImageProcessor.from_pretrained(self.model_dir, local_files_only=True)
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_dir, local_files_only=True,
                                                                     use_safetensors=True).eval().to(device)
        self.model.requires_grad_(False)
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        # "Dynamic" means potentially moving and is used conservatively to
        # prevent temporal trails. Parked cars are still occupied in each frame.
        tokens = {"person", "car", "bus", "truck", "bicycle", "minibike", "motorcycle", "train"}
        self.dynamic_class_ids = sorted(k for k, name in self.id2label.items()
                                        if any(t in name.lower().replace(";", ",").split(", ") for t in tokens))
        self.last_info = {}

    def metadata(self):
        return {"provider": "SegFormer B0 ADE20K (OpenSeeD replacement)",
                "repository": MODEL_REPO, "revision": MODEL_REVISION,
                "class_count": len(self.id2label), "class_names": self.id2label,
                "device": self.device, "image_size": self.image_size,
                "dynamic_class_ids": self.dynamic_class_ids,
                "open_vocabulary": False, "trained_on_orbvi": False,
                "pixel_semantics_valid": True, "lidar_association_requires_calibration": True}

    def _run(self, rgb):
        import torch
        import cv2
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Semantics expects an RGB uint8 HxWx3 image")
        inputs = self.processor(images=image, return_tensors="pt",
                                size={"height": self.image_size, "width": self.image_size})
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=self.device.startswith("cuda")):
                logits = self.model(**inputs).logits
            # Upsample logits only to inference resolution; the full raw image
            # can otherwise allocate ~786 MB for four 1280x1024 cameras.
            logits_resized = torch.nn.functional.interpolate(logits.float(),
                           size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            labels = logits_resized.argmax(1)[0].cpu().numpy().astype(np.uint8)
            confidence = logits.float().softmax(1).amax(1)[0].cpu().numpy()
        labels = cv2.resize(labels, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        confidence = cv2.resize(confidence, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        return labels, confidence.astype(np.float32)

    def predict_with_confidence(self, image_rgb):
        import torch
        started = time.perf_counter()
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        fallback = None
        try:
            labels, confidence = self._run(image_rgb)
        except torch.cuda.OutOfMemoryError:
            if not self.fallback_cpu or not self.device.startswith("cuda"):
                raise
            fallback = "CUDA out of memory: continued with real pretrained model on CPU"
            self.model.to("cpu"); self.device = "cpu"
            torch.cuda.empty_cache()
            labels, confidence = self._run(image_rgb)
        self.last_info = {"runtime_ms": (time.perf_counter() - started) * 1000,
                          "device": self.device, "fallback": fallback,
                          "pixel_count": int(labels.size),
                          "classes_present": np.unique(labels).tolist(),
                          "mean_confidence": float(confidence.mean()),
                          "peak_allocated_gpu_mib": (torch.cuda.max_memory_allocated() / 2 ** 20
                                                      if self.device.startswith("cuda") else 0)}
        return labels, confidence

    def predict(self, image_rgb):
        return self.predict_with_confidence(image_rgb)[0]

    def dynamic_mask(self, label_map):
        return np.isin(label_map, self.dynamic_class_ids)


def main():
    import argparse
    import cv2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--image")
    parser.add_argument("--output", default="results/semantics_smoke")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()
    if args.download:
        download_model(args.model_dir)
    if args.image:
        output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
        bgr = cv2.imread(args.image)
        if bgr is None:
            raise FileNotFoundError(args.image)
        provider = SegformerSemantics(args.model_dir, args.device, args.image_size)
        labels, confidence = provider.predict_with_confidence(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        np.savez_compressed(output / "semantic.npz", labels=labels, confidence=confidence)
        rng = np.random.default_rng(0)
        palette = rng.integers(32, 256, (256, 3), dtype=np.uint8)
        overlay = cv2.addWeighted(bgr, 0.55, palette[labels], 0.45, 0)
        cv2.imwrite(str(output / "overlay.jpg"), overlay)
        report = {"input": str(Path(args.image).resolve()), "model": provider.metadata(),
                  "inference": provider.last_info, "image_shape": list(bgr.shape),
                  "semantic_lidar_valid": False,
                  "note": "Independent image semantics; geometric LiDAR association is separately calibration-gated."}
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report["inference"], indent=2))


if __name__ == "__main__":
    main()
