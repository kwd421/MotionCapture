# R10 review: same-frame batch2 -> exact-60Hz replay gate

Source report: `wholebody-pose-batch-pipeline-r10.json` supplied 2026-09-07.

## R10 finding
The four full-file arms completed 6,442 frames each. Pixel hashes and detector-box hashes were equal across all arms. Batch2 changed prediction records only on the 131 multi-person frames, as designed. The run records revision `261fed41bb3a7c28b329ae02d36325074c80203e` with `dirty=true`; this review does not relabel it as a clean checkout.

### Scheduled-source age
- serial A1: mean 35.7805 ms, p95 55.7493 ms, max 316.333 ms, >100 ms 263.
- batch2 B1: mean 28.4318 ms, p95 34.9718 ms, max 88.7084 ms, >100 ms 0.
- batch2 B2: mean 28.6372 ms, p95 36.9512 ms, max 79.8793 ms, >100 ms 0.
- serial A2: mean 44.3881 ms, p95 131.716 ms, max 567.101 ms, >100 ms 366.

Equal-frame pooled mean source age: serial 40.084300 ms vs batch2 28.534492 ms, a 28.814% reduction. The two batch2 arms beat both serial controls on p95, maximum age, and >100 ms count.

### Multi-person pose burst
Two-person elapsed pose-stage mean: serial 24.845347 ms vs batch2 17.290441 ms, a 30.408% reduction.

Across the two serial controls there were 629 source-age samples over 100 ms; across the two batch2 arms there were 0. The 50 ms counts were 850 vs 181.

## Accuracy/correctness boundary
R9's isolated serial-vs-batch2 comparison found only 8 matched points above 1 px, none above 10 px, and no validity changes; that is numerical agreement evidence, not ground-truth accuracy. R10 confirmed that all single-person and zero-person frames remained bitwise identical and detector boxes did not change. Ground-truth accuracy remains unverified.

## Decision
Same-frame batch2 is retained as the leading performance candidate. It is not yet promoted to the default/live path because the recording's original PTS span is only about 59.373 Hz. The next gate replays the same 6,442 original frames on an exact fixed 60 Hz host-release schedule without rewriting PTS, skipping frames, or batching future frames.

Assistant-side pre-push validation compiled all three new Python files successfully and exercised the fixed-rate scheduler with a deterministic clock: 60 Hz due times were absolute, source PTS remained unchanged, and the reported schedule rate was exactly 60.0 Hz. Native M5/CoreML execution of the new fixed-60Hz gate remains pending.
