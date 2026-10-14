# Camera semantic occupancy adaptation

This route predicts metric geometry and semantic occupancy from the four actual
ORBVI cameras, using the fixed device camera-to-camera calibration. It never
inputs LiDAR points, LiDAR depth, `/orbvi/depth/points`, or VIO poses to either
neural model. All pretrained weights remain frozen. New modules and output
directories keep the previous geometry-only results unchanged.

## Source-view lifting

1. Each physical Triple-Sphere image is remapped to a calibrated virtual pinhole
   view. RGB, valid mask, intrinsics and `T_rig_camera` describe the same view.
   Square 518 × 518, 110° horizontal FOV retains more low-image vehicle content
   than the old 518 × 392, 100° view. Additional yaw/pitch views, when configured,
   rotate the real physical camera calibration; they are not additional sensors.
2. Semantic inference runs on this rectified image rather than a distorted raw
   fisheye image. Depth inference receives the same rectified RGB and resized
   intrinsics. This is essential: image content cropped before depth inference
   cannot be recovered by a later semantic recoloring operation.
3. For every accepted depth point, `camera_depth.py` retains its source view ID,
   integer sampled UV, camera-coordinate XYZ, depth, native model confidence,
   input image shape, K and calibrated pose. These fields are additive; the old
   point-selection behavior is unchanged.
4. `lift_source_semantics` obtains semantic evidence from that same view and
   source pixel. Pixel-center resizing uses `floor((uv + 0.5) * new / old)`.
   No overlapping camera can take over this point's label merely because its
   projection has a smaller central angle.

The raw-fisheye segmentation cache in old runs is not the semantic image domain
for this route. Viewer overlays use the new rectified RGB. Provenance includes
the actual physical camera ID and image acquisition timestamp for every view.

The persisted `semantic_images/*.jpg` are JPEG-quality-95 display copies in that
same rectified pixel domain. Neural inference consumes the arrays reconstructed
from the original prepared compressed images and fixed calibration, before this
display encoding; it never reloads the display JPEGs as neural inputs. Their RGB
bytes are therefore not an exact record of the neural input pixels. Exact pixel
reconstruction uses the retained raw image, source-view calibration and executed
software snapshot.

## Semantic model and named class registry

The initial executable provider supports frozen official NVIDIA SegFormer
ADE20K checkpoints. This substitutes SegFormer for the original TT-Occ
OpenSeeD/Rex-Omni component and must be reported as a method difference.
The project also supports the original OpenSeeD Swin-T pretrained component
through `orbvi_occ/openseed_provider.py` in an isolated dependency environment.
Its actual panoptic winning query supplies image labels and semantic evidence;
rejected segments stay unknown. The chosen provider, checkpoint hash, vocabulary
and confidence definition are explicit in each run's metadata. SegFormer remains
an executable alternative; this document's geometry and source-UV contract
applies to both providers.

The stronger evaluated checkpoint is
[`nvidia/segformer-b5-finetuned-ade-640-640`](https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640),
revision `739f5d4692954e4a185eac280dec1ba5a7d52f1d`. Its official `.bin` state
dictionary is loaded with `torch.load(weights_only=True)` and copied without
tensor changes to a local safetensors file. The original and converted file
SHA-256 values are recorded in `models/segformer-b5-ade20k/provenance.json`.
The safetensors JSON header is serialized in sorted order while keeping tensor
payload bytes unchanged. This removes nondeterministic metadata-key ordering;
the reproducible converted file SHA is
`4ff9bef4fc23da75ec1a6199f327a80963a1faa9a7ebc45ea73fc6c53c87cda3`.
Loading rejects missing, unexpected, mismatched or erroneous tensors; no random
initialized semantic head is accepted. Inference at 512 is supported separately
from the checkpoint's original 640 training resolution.

`taxonomy/garage.json` fixes the following IDs: wall 0, floor 1, ceiling 2,
column 3, car 4, person 5, door 6, vegetation 7, other 8, road 9, sky 10,
semantic unknown 255. Native classes are mapped by explicit names, not numeric
IDs. Wall 0 is never nuScenes noise 0. TT-Occ's active car, floor, wall and
vegetation colors are reused; ceiling, column and door receive documented
garage-specific colors.

The provider groups source class softmax evidence into this registry and stores
the top three known-class masses. The sum need not equal one because omitted
class evidence is not fabricated. A pixel below the native confidence gate,
outside the calibrated image mask, or classified as sky has zero known semantic
evidence and receives label 255. Model softmax values are confidence scores;
they are not calibrated correctness probabilities. A semantic floor or road
label does not certify traversability or observed free space.

## Separate geometry and semantic evidence

`semantic_gaussian_voxelize` uses the pinned TT-Occ zero-iteration isotropic
Gaussian stencil: spatial weight greater than 0.6065, integer-quantized scale,
with explicitly documented floor voxel indexing. Geometry includes all accepted
depth points. Unknown semantics has zero class evidence; a large number of
unknown points cannot outvote a known class as an artificial 255 category.

The grid shape and relative-coordinate floor now use float64 arithmetic, with
explicit per-axis integer bounds checks before flattening. A real frame 68
source point plus its Gaussian stencil had expanded x=11.999999046325684,
strictly below the physical upper bound 12 m. The previous float32 subtraction
rounded x+12 to 24 and incorrectly produced X index 80 for an 80-cell axis.
Y/Z overflow could also alias another axis when flattened. The corrected code
keeps the valid physical contribution in index 79 and filters invalid integer
axes before flattening; source XYZ and the old LiDAR module are unchanged.
Actual source frames 62–68, using recorded camera-only poses and six-frame
causal history, independently reproduce the corrected voxel index sets and
canonical nearest-source support with no OOB indices. The diagnostic is in
`diagnostics/camera_semantic_backend/boundary_frame68/report.json`.
The rejected incomplete run is retained separately; its inference cache can
be reused, but occupancy/fusion/traversability must be reconstructed rather
than relabeled as successful complete results.

After that correction, an actual frame 9 also exposed a downstream coupling
error: floor-source removal still used the legacy float32 collision raster,
so collision cell [71,69,1] did not belong to the camera's float64 occupied
grid. `LocalTraversability` now accepts an optional `collision_voxelizer`
callback. The dedicated camera route injects `geometry_collision_voxelize`,
which uses the identical camera raster with a dummy zero-evidence channel.
Its non-floor physical source members therefore form a true subset of occupied
geometry. No collision/intersection clipping hides mismatches. Default callers
continue using the untouched legacy raster, preserving LiDAR/old-camera behavior.
Actual six-history replays at frames 9 and 68 independently rebuild every
in-bounds non-floor Gaussian member; collision masks match exactly and
collision minus occupied is empty. Diagnostics are in
`diagnostics/camera_semantic_backend/collision_grid/report.json`.

Every occupied voxel stores:

| Field | Meaning |
| --- | --- |
| `labels` | Winning garage class, or semantic unknown 255 |
| `confidence` | Winning class mass / total retained known semantic mass |
| `known_support` | Sum of spatially weighted retained known evidence |
| `geometry_support` | Sum of geometric Gaussian weights, including unknown semantics |
| `support_count` | Number of positive-known source contributions |
| `entropy` | Normalized entropy of retained known class evidence |
| `evidence` | Spatially accumulated retained known-class evidence |

When an occupied voxel's winning score is below the configured acceptance
threshold, its semantic label becomes unknown 255 while its candidate confidence
and entropy remain available for diagnosis. This score does not make the
rejected class a known label. Point-level unknown observations still have zero
known evidence and zero semantic confidence.

These describe inferred occupied surfaces. Missing sparse cells remain unknown
unless a separate ray-evidence layer establishes an appropriate free-space
claim. Predicted camera depth cannot certify measured free space.

## Causal temporal policy and limitations

The configurable initial policy uses 0.6 seconds of history, at most six past
frames, with semantic exponential decay of 0.25 seconds. Current class evidence
with confidence at least 0.45 takes priority over conflicting accumulated past
class votes. This class-independent priority stops repeatedly sampled historical
walls/ground from suppressing a newly observed vehicle. When it applies, semantic
confidence/support/evidence are those of the current observation; fused geometry
support is retained. Override counts are reported.

All classes, including parked cars, can enter history. No blanket vehicle-class
exclusion is used. History storage selects the strongest known evidence within
a configured source voxel, rather than allowing the first unknown sample to
starve known semantic evidence. Current source points are not downsampled by
this history policy.

An optional geometric history gate removes a past surface only when a current
predicted ray in the same direction reaches a clearly farther first surface.
The initial 1°/0.6 m gate uses each current ray's real calibrated optical center
and requires three nearby supporting current rays. History surfaces closer than
2 m are excluded from this removal gate because near-body geometry and small
projection errors need especially conservative handling. When a caller omits
origins, a documented rig-centered approximation is used. Uncovered directions
and past surfaces behind the current return are retained. This is evidence of
geometric inconsistency, not an estimate of
object velocity. The bounded history limits damage from remaining uncertainty.
Ray-neighbor searches are bounded by the same angular acceptance gate, with
uncovered directions rejected and points below the existing minimum range
skipped before querying. This is an exact search optimization, not a different
removal rule. Actual pilot differential checks cover 463,842 historical points
with zero removal-mask differences; 16-frame causal replay preserves all
persisted fused indices and labels. Mature-history fusion falls from roughly
4.1 seconds to 0.4 seconds on this machine. Diagnostics are in
`diagnostics/camera_semantic_backend/runtime_profile_bounded.json` and
`runtime_profile_optimized.json`.
Camera-depth ICP quality rejection resets all history and creates a new local
segment; segments are not silently asserted to share a reliable global map.

This adaptation does **not** reproduce original TT-Occ RAFT scene flow,
instance association or dynamic-object ICP. Those mechanisms,
independent geometric scale validation and 3D semantic
ground truth remain separate work. A completed whole-bag engineering run is not
evidence of official benchmark accuracy.

## Real representative diagnostics

`diagnostics/camera_semantic_backend/probe.py` runs actual B0/B5 semantic
inference on frames 0, 135, 1174 and 2347 using identical calibrated square
110° views, then actual MapAnything 224/336 inference and source-UV lifting.
`probe_summary.json`, `compact_summary.json`, original RGB, overlays and source
NPZ files record the results. These independent representative inferences are
explicitly not the whole-recording acceptance run.

Frame 135's lower image contains a very near vehicle body visible to the rig.
Many predicted car points belong to this foreground, plausibly the acquisition
vehicle itself. Car pixel/voxel counts must therefore be separated by range;
more blue voxels alone do not demonstrate improved external-car detection.
No automatic near-obstacle removal is applied without an explicit body mask.
The B5 overlays are visibly more continuous for floor, vehicle body and columns
than B0 in these representatives, but no ground-truth accuracy is claimed.

The subsequent accepted route can configure a manually reviewed fixed ego-body
polygon per virtual view in `configs/camera_ego_mask.json`. This excludes only
the visible acquisition-vehicle pixels and a recorded image-space padding.
RGB stays unchanged, so the input can be reviewed; excluded pixels are unknown
and provide neither occupied points nor free rays. Other structures inside
2 m remain eligible. Mask size, FOV and yaw/pitch must match the configured
virtual view exactly, otherwise inference fails. The mask file's SHA is part
of cache/run provenance. A fixed image-body mask is not a measured vehicle
envelope or an externally validated safety footprint. The initial representative
probe files above precede this explicit mask and retain the visible body.

New virtual views explicitly select pixel-center valid-mask sampling during
depth resizing, matching the semantic UV lift. The previous route retains its
legacy OpenCV nearest sampling. This distinction prevents a mask-edge depth
point from slipping through one resizing convention and falling inside the
ego polygon under the other convention.

Independent QA validates actual source-pixel labels, rigid-coordinate lifting,
class IDs, semantic confidence/support, unknown preservation and Gaussian
surface support. Ground-truth mIoU is not reported without a labeled evaluation
set. The final runner must cover all 2348 aligned acquisition steps and retain
per-frame committed source, semantic, occupancy and traversability artifacts.

`semantic_confidence` validates the source semantic gate; its soft confidence is
already present in the top-k evidence and is not multiplied a second time.
The MapAnything native confidence is retained and validated, but is not treated
as a calibrated probability or arbitrarily multiplied into semantic votes.
The official depth-provider confidence/edge masks determine accepted pixels.

## Reproducible model setup

Use the existing environment, with no dependency upgrade:

```bash
.venv/bin/python scripts/setup_camera_semantic_models.py --check
.venv/bin/python scripts/setup_camera_semantic_models.py --download
```

`--check` is offline, reads SHA/revision manifests and never initializes a GPU
model. `--download` fetches only missing pinned model files, converts B5 safely
and checks deterministic hashes. A valid complete MapAnything checkpoint is
reused; its 4.9 GB file is not downloaded again. Pin checks include original
B5 `.bin`, converted safetensors, configuration and image preprocessor, plus
the official MapAnything and DINOv2 architecture commits and source-file hashes.
