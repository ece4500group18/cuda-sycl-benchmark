# stencil / convolution / image processing collection

Owner: Zijian

## Backlog

`cuda_to_sycl_kernel_source_backlog_top100.xlsx` — a 100-candidate backlog
of CUDA image/vision kernels, with sheets:

- **Top100_Kernels**: candidates from mmcv (35), NVIDIA/DALI (24),
  opencv_contrib (20), pytorch (10), torchvision (6), detectron2 (5).
  Columns include difficulty / extractability / verification-ease scores,
  an input→output contract, suggested reference baseline, license + caveat,
  path confidence, and a priority score.
- **Summary**, **License_Notes**, **Benchmark_Extraction_Guide**.

This spreadsheet uses Zijian's own schema (not the shared
`candidates.csv` columns described in `../README.md`). Reconciling the two
formats — or exporting the backlog to a `candidates.csv` + per-id
`sources/` snapshots like the other categories — is a follow-up the team
can decide on.

## Cases already adapted

34 of these candidates are already built as full cases under `cases/`
(same unit format as the pilot: `original/main.cu` + `CMakeLists` +
deterministic inputs + `metadata.json` + `tests/verify.py`). All 34 pass
`verify.py --selftest`. They run against the shared
`benchmark/collection/tools/verify_lib.py`.

- 20 DALI: dali{BoundingBoxFlip, BoxEncoder, Cast, ColorSpaceConversion,
  ColorTwist, Crop, CropMirrorNormalize, ElementExtract, Flip,
  MakeContiguous, NormalizePermute, Paste, RandomResizedCrop,
  ResizeBilinear, ResizeCropMirror, Rotate, Slice3d, TransposeHwcChw,
  WarpAffine, WaterWarp}
- 12 mmcv: mmcv{BallQuery, BboxOverlaps, FurthestPointSample, GatherPoints,
  GroupPoints, Knn, Nms, PointsInBoxes, RoiAlign, RoiPool,
  ThreeInterpolate, ThreeNN}
- 2 pytorch: pytorch{MaxPool2d, UpsampleNearest2d}

These are the collected/adapted cases for this category, parallel to the
other members' categories under `benchmark/collection/`. They are kept
out of `pilot_benchmark/` (the original 50 self-authored pilot cases stay
frozen).
