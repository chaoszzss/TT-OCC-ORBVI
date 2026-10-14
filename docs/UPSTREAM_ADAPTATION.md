# ORBVI TT-Occ adaptation and provenance

This engineering adapter uses the **TT-Occ LiDAR zero-iteration Gaussian
occupancy mechanism**, and adds sensor-specific I/O, bounded causal history,
free-space evidence and local traversability. It does not reproduce the
published nuScenes semantic mIoU experiment. A CUDA Gaussian renderer is
unnecessary when `iterations=0`; occupancy rasterization still uses the actual
upstream Gaussian mathematics.

## Immutable upstream source

- Repository: <https://github.com/Xian-Bei/TT-Occ>
- Commit: `0cdb8c9e8abc8124fd956c1e0cb582d3ad18b333`
- Original source is retained unchanged in `upstream/`, with original licenses.
- Initialization: `upstream/scene/gaussian_model.py:create_from_pcd` assigns
  isotropic scale **0.3 m**, opacity **0.5**, identity quaternion. Its earlier
  nearest-neighbor scale computation is overwritten by this constant.
- Occupancy: `local_range_offset_full`, `gaussian_spatial_weight_simplified`,
  `gaussian_spatial_weight` and `gs2voxel` expand each Gaussian on a voxel
  lattice, retain samples with weight **>0.6065**, sum semantic evidence per
  voxel, and select argmax. Opacity does not enter `gs2voxel`.
- Past fusion: `add_from_pcd` transforms/maintains world-coordinate Gaussians,
  removes dynamic history and concatenates new points. The optional E-style
  radiometric/semantic KNN fusion remains disabled (`use_fusion=False`).

`orbvi_occ/tt_backend.py` implements this inference path with CPU NumPy grid
bookkeeping and the **same Torch CPU elementary Gaussian kernels**. There is
no OCC network training or scene optimization. The direct reference test parses
the pinned original source with Python AST, extracts the four original methods,
and changes only the hardcoded `cuda` device literal to `cpu`. It executes
those original methods independently of the adapter. At voxel sizes 0.1, 0.2,
0.3, 0.5 and 0.8 m, `upstream_trunc` compatibility mode yields identical sparse
indices and one-hot class IDs. A separate test compares soft class evidence.

## Deliberate production differences

| Change | Reason and practical consequence |
|---|---|
| Four actual ORBVI cameras rather than six fabricated nuScenes cameras | The original data reader and dynamic-mask branch contain `range(6)`. Sensor loops in this project use actual camera IDs. |
| Correct triple-sphere lens projection and pinhole rectification | ORBVI's 9 parameters are `[fx,fy,cx,cy,xi,lambda,alpha,b,c]`, not a four-coefficient OpenCV fisheye or MEI model. Device model verification belongs to calibration diagnostics. |
| CPU Gaussian occupancy, zero iterations | Runs without unsupported CUDA rasterizer builds and preserves GPU memory for frozen pretrained image models. |
| Floor-based voxel indexing by default | Original `Tensor.int()` truncates negative coordinates towards zero, merging cells around the origin. `floor` provides regular spatial cells consistent with free-space ray traversal. Original behavior is available with `voxel_indexing=upstream_trunc`. |
| Local spatial bounds and finite past-only time window | Prevents indefinite accumulation and memory growth over all 2,348 scans. No future frame contributes to an output. The actual window and point counts are recorded per frame. |
| Potentially dynamic classes contribute to the current scan, but are not retained | Conservative replacement for RAFT masks / instance ICP tracking; avoids motion trails when semantic projection is valid. It also excludes parked cars from temporal persistence. |
| Keep LiDAR returns outside camera coverage as unknown semantic class 255 | The original nuScenes reader drops all camera-invisible returns. Sparse Livox and four fish-eye coverage should not erase measured geometry simply because semantics are unavailable. |
| Calibration gate for LiDAR semantic association | If device-specific LiDAR-camera projection is unverified, independent image semantics remain available but mapped voxel semantics must be marked invalid. The measured Gaussian geometry remains valid under the LiDAR-only coordinate checks. |
| Optional representative source-voxel reduction | `input_voxel_size` defaults to zero. Enabling it changes point density and therefore semantic voting; the selected value is recorded in diagnostics. |
| Additional free / occupied / unknown and traversability products | TT-Occ only outputs occupied geometry. Ray-observed free space and a conservatively inflated collision map are downstream products, kept separate from the raw full Gaussian occupied volume. Unobserved voxels are never declared free solely because no occupied prediction exists. |

`current_source_points` and `fused_source_points` are the actual measured
returns entering the Gaussian rasterizer, in the current gravity-aligned local
frame. They permit a separate collision-only Gaussian volume to exclude
reliably identified floor returns, while the raw TT-Occ volume retains the floor
and all measured scene geometry. Dropping all low-z occupied voxels is unsafe
and is not the intended floor handling.

## Frozen real semantic component

The original online providers are OpenSeeD and Rex-Omni. This adapter uses
**SegFormer B0 pretrained on ADE20K**, a deliberately lighter semantic provider
for this shared 8 GB laptop GPU. This is a closed-vocabulary segmentation model,
not an open-vocabulary VLM. Its ADE20K IDs and class names are retained and are
not silently remapped to nuScenes IDs. The checkpoint is not random and has not
been fine-tuned on ORBVI.

- Model: <https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512>
- Revision: `489d5cd81a0b59fab9b7ea758d3548ebe99677da`
- Safetensors size: **15,036,944 bytes**.
- SHA256: `6ae39addd01de6b1b8bde2cf677d43a5cd733424b8d186de3f95d1c51fee23f9`.
- Local provider: `orbvi_occ/semantics.py`; offline loading is required during
  dataset runs. Model evaluation mode and frozen parameters are checked.
- Full raw-resolution logits are avoided: bilinear logits are evaluated at the
  configured model input resolution, then the argmax map is nearest-resized to
  the original image. This reduces memory, with less boundary precision.
- `results/semantics_bag_smoke/report.json` records inference on an original
  image extracted from this exact ROSbag (1280x1080), with
  actual classes/confidence/latency and allocated GPU memory. That smoke does
  not itself claim calibrated LiDAR semantic association.

OpenSeeD is intentionally not reported as having been run. Its custom
Detectron2/CUDA dependencies and larger checkpoint are unnecessary for the
first geometry loop; substitution means the project should be called a
**TT-Occ adaptation**, not a bit-identical full OpenSeeD nuScenes reproduction.

## Independent camera geometry component

`orbvi_occ/camera_depth.py` loads the **official MapAnything Apache checkpoint**
and pinned official MapAnything code in `models/mapanything-source`. The TT-Occ
vendor predates the current checkpoint's `use_register_tokens_from_encoder`
setting. Silently dropping that setting would alter the actual pretrained
forward pass, so the corresponding official implementation is used instead;
the original TT-Occ source remains unchanged. Inputs are
rectified RGB images, pinhole intrinsics, and fixed calibrated rig extrinsics.
The interface rejects LiDAR/depth fields. No LiDAR-derived depth, online
LiDAR tracking pose or LiDAR metric scaling is supplied to the neural predictor.

- Model: <https://huggingface.co/facebook/map-anything-apache>
- Revision: `00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a`
- Official MapAnything source commit:
  `3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9`.
- Original FP32 safetensors size: **4,914,062,480 bytes**.
- DINOv2 architecture source: <https://github.com/facebookresearch/dinov2>,
  commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.
- The DINO encoder architecture is initialized with `pretrained=False` only to
  avoid downloading redundant DINO weights. A **strict complete MapAnything
  Safetensors model load** covers every parameter before any prediction,
  preserving shared DPT module aliases omitted from the serialized key list. Random
  initialized tensors are never used for inference.
- The weights are loaded on CPU, converted to BF16 before CUDA transfer, and
  used with AMP, `memory_efficient_inference=True`, and the public
  `minibatch_size=1` argument; source files are unchanged.
- Official geometry fusion disables autocast but attempts an indexed FP32 ray
  assignment into BF16 image features. An instance-level precision wrapper
  restores FP32 features for that branch and returns the backbone dtype after
  fusion. Geometry encoders and downstream heads retain their exact FP32
  pretrained weights, while the backbone and transformer use BF16. The
  downstream-head inputs are also explicitly converted to FP32, matching the
  official source's disabled-autocast block and DPT casts. This resolves the
  actual dtype exceptions without editing official source or dropping
  calibration.
- All input image resizing updates K by the same width/height factors. Pixel
  stride, range limits, confidence masks and valid rectification masks are
  explicitly recorded. Output camera pointmaps are transformed by fixed rig
  calibration, and per-point ray origins are retained for free-space traversal.
- CUDA OOM falls back to the **same actual model** on CPU and is recorded. User
  GPU processes are never stopped. The camera smoke report clearly distinguishes
  model readiness from a completed whole-bag camera execution.

Model SHA256 values and official source commits are stored in the models'
`provenance.json` files; model files
remain local in this project. `scripts/setup_tt_backend.sh --camera` installs
the minimal isolated-environment dependencies and downloads the pinned public
checkpoints. It does not install external training, VGGT, RAFT or 3DGS code.

The exact backend dependencies in the verified isolated environment are
recorded in `models/backend_dependencies.json`. This includes Torch
2.7.1+cu126, Transformers 4.51.3, Hugging Face Hub 0.36.2, UniCeption 0.1.7,
and Safetensors 0.8.0. Reproduction should retain these versions rather than
upgrading the model stack implicitly during a dataset run.

## Camera temporal pose and precision boundaries

The network's `camera_poses` input consists of **fixed rig extrinsics**, all
from the archived device camera calibration, composed with a constant optical
to forward/left/up coordinate rotation. The translation units are converted
from the board file's millimetres to metres. This coordinate conversion is not
a time-varying trajectory estimate and carries no LiDAR/VIO tracking information.
`T_rig_camera` explicitly maps each camera's optical XYZ into this fixed rig.

Camera time fusion uses point-to-plane ICP between **predicted camera depth
clouds only**. Current-to-previous registration is composed into the previous
camera pose; rank, correspondence ratio, residual, transform magnitude and
conditioning gates determine whether the pose may support temporal fusion.
If those gates reject a transform, occupancy/free-space history is cleared.
There is no identity-transform substitution presented as successful ICP, and
there is no fallback to a LiDAR/VIO pose. An unconnected camera trajectory
segment has its own arbitrary anchor and must not be mistaken for a validated
continuous global map. Per-frame local camera occupancy can remain available
even when continuous world-frame alignment is not reliable.

The initial 4-view smoke used 224x224 rectified images and the complete official
checkpoint. It produced 28,170 valid sampled points (stride 2), approximately
0.75 s first inference latency, and 2,694.45 MiB peak Torch-allocated GPU memory.
These are measured smoke values, not throughput or accuracy guarantees. The
final camera-run configuration and report determine its actual stride, timing,
coverage, temporal resets and full-recording acceptance.
