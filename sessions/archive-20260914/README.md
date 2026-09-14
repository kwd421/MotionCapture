# Session archive — 2026-09-14

Text-only snapshot of locally retained experiments, including failed/partial
runs, native profiling, reproduction probes and later low-load retries.
Results retain their original validity/scope; archiving is not a new benchmark.

`ARCHIVE-MANIFEST.json` inventories included and excluded files. It records both
original and archived SHA-256; paths were sanitized in the copies only. Absolute
workspace paths became `.`, user homes became `<USER_HOME>`, and any local temp
paths became `<LOCAL_TEMP_PATH>`. Trailing whitespace and blank EOF lines in `.txt` copies were normalized and
flagged in the manifest. Original files remain unchanged outside this
archive. Original script-hash receipts still describe original bytes; consult
the archive manifest when a sanitized script has a different hash.

Images (including body/hand crops and 3D plots), caches and filesystem metadata
are excluded. Input videos and model weights were never part of this archive.
Model/source receipts retain provenance; upstream code retains its original
attribution and is not relicensed by archival.

This is an evidence snapshot, not an installed execution package. Historical
scripts may require the repository root as working directory, the documented
pinned optional dependencies, private inputs/model files, and their original
`sessions/...` layout or temporary helper locations. Review/adapt paths and use
fresh output names before execution; do not overwrite prior results. The archive
was parsed/hash-checked but experiments were not rerun for this commit.
