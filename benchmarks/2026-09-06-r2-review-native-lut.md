# r2 result and optional native-LUT candidate

Base: ec5103a5de25b6cd7fcf4ea8a4cf670c222bb251.
The existing live defaults, model assets, provider policies and handoff plans are
unchanged. This is not an optimization-complete or commercial-release declaration.

## Measured on the user's M5, not rerun on this host

The r2 report completes four 900-frame passes, same original prefix and 968 person
observations per pass. Same source pixels, boxes and prediction hashes in all arms.
Pooled FPS uses total frames / total loop time:

| Mode | FPS each | Pooled FPS | Mean host submit-to-pose ms | Mean pose submit gap ms |
| --- | --- | ---: | ---: | ---: |
| overlap-LUT | 50.4086 / 51.4475 | 50.9227 | 35.9764 | 2.2980 |
| ready-handoff-LUT | 61.1278 / 63.6901 | 62.3827 | 28.8824 | 0.2123 |

Observed throughput gain: 22.5045%. The 60Hz mean file-capacity milestone has been
reached for this prefix. It is NOT a full 108.5-second clip, camera or display test.
The candidate's verified-output p95 is 23.5665 / 22.4652 ms; 361/1798 intervals
exceed 16.67 ms (20.08%), not frame drops or measured live deadline misses.
Ready handoff admitted 828/899 and 832/899 already-detected next poses. All owners
released normally with no admitted work lost. Larger 2-person workloads remain.
Numerical comparisons exclude 68 ambiguous multi-person frames per pass; full
hashes include them. Ground-truth and the earlier crop-sensitive outlier issue
remain unresolved. The log records ec5103a plus dirty=true, not a clean checkout.

## Added candidate and evidence

OpenCV LUT uses the SAME immutable 256x3 float32 table as NumPy take. It performs
one multi-channel lookup and an explicit HWC-to-contiguous-NCHW copy. Affine crop,
precision, score policy, models, all source frames and all persons are unchanged.
Existing NumPy mode remains selectable; no backend or numeric fallback is added.

Primary API convention:
https://docs.opencv.org/4.12.0/d2/de8/group__core__array.html
LUT permits a lookup table with the source channel count; output depth follows
the table. This does not by itself prove speed or numerical parity on M5.

Local synthetic-ROI affine+lookup+layout microbenchmark: median-round mean
0.458270 -> 0.383036 ms (16.42% reduction of that kernel ONLY), six alternating
rounds of 400 calls. All 32 distinct ROI pairs matched in all returned tensor and
geometry bytes. Unit tests additionally cover every uint8 value, non-contiguous
crops, float32/float64 ULP-edge boxes and source/past-output preservation.
These are Linux/OpenCV4.13 measurements, not the user's M5 or a neural benchmark.

All arms now include bounded output-cadence metadata: original-PTS 10-second bins,
within-bin actual host completion rates, interval statistics by ending-frame person
count, longest consecutive interval-budget exceedances and 16 worst interval IDs.
There is no nominal-FPS time rewriting or new buffering. Boundary intervals remain
in the global N-1 sample set but are excluded from within-bin rates. Partial final
bins and failures remain visible. Additional observer cost is separately measured;
its cost remains in full loop time. New logs must be compared internally first.

## Full-clip confirmation and candidate comparison

```bash
git fetch origin
git switch feat/wholebody-onnx-lab
git pull --ff-only origin feat/wholebody-onnx-lab

uv run --with-requirements tools/requirements-wholebody.txt \
  python -m motioncapture.wholebody_optimize_bench \
  "$HOME/Downloads/20260906_030954.mp4" \
  --detector-provider coreml-all --pose-provider coreml-all \
  --allow-cpu-partitions --research-only --suite native-normalize \
  --max-frames 0 --output sessions/wholebody-optimize-r3-full.json
```

A/B/B/A = proven ready-NumPy / ready-OpenCV / ready-OpenCV / ready-NumPy, with fresh
sessions and the ENTIRE original file each time. This also tests source content
beyond the prior 900 frames. It is not a multi-camera, 3D, facial-blendshape or live
system acceptance test. Keep local changes intact on a Git conflict. Send the main
JSON; if interrupted, the terminal/started checkpoints remain relevant.

No model-weight downloads or new optional package dependencies are added. Existing
--suite handoff remains the original r2 comparison. --max-frames 900 is still an
explicitly labelled prefix option, not the full test shown above.
