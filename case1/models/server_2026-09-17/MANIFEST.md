# MANIFEST — models synced from GPU server (team1), 2026-09-17/18

Synced via `scp team1:...` on 2026-09-18. Source host: remote GPU container (`ssh team1`), project root `/workspace`.
No files on the server were modified or deleted. Local `case1/models/weed_detector_finetuned.pt` (repo root copy) was left untouched.

| File | Source path on server | sha256 | Size (bytes) | Server mtime |
|---|---|---|---|---|
| weed_detector_finetuned.pt | /workspace/case1/models/weed_detector_finetuned.pt | 34b0935e31b13262b5893e2deea9adef4ba8f7c3591ea014f6768910a35a0fb4 | 22517482 | 2026-09-17 20:40 UTC |
| multitask_weeds_best.pt | /workspace/case1/models/multitask_weeds_best.pt | 46b671f4315f5a1c795ed7dd94dfe9aec1904d32a7fb89dd14b27249e91a4294 | 19000309 | 2026-09-17 19:30 UTC |
| species_mapping.json | /workspace/case1/models/species_mapping.json | 8f347b60d1ccc28390dd54059e47243ff40d5f02451d5bd7db46309670eb9eee | 1692 | 2026-09-17 19:09 UTC |
| weed_classifier_mobile.torchscript | /workspace/case1/models/weed_classifier_mobile.torchscript | 9c4d017a336f478e886f1de2d539d74a331c88c7d2bec12b4003a312ad503804 | 19457164 | 2026-09-17 19:38 UTC |

## Notes
- `weed_detector_finetuned.pt`: sha256 is **identical** to the existing repo-root copy at `case1/models/weed_detector_finetuned.pt` (34b0935e...) — already in sync, no divergence. Copied here anyway for a complete, dated snapshot.
- `multitask_weeds_best.pt`: sha256 **differs** from the existing repo-root copy (local repo has `cd987073...`, server has `46b671f4...`) — the server has a newer/different training run of the classifier than what's committed locally. Repo-root file was NOT overwritten; this is a separate copy for comparison.
- `multitask_weeds_focal.pt` was present on both sides but was **not** requested for sync (not in the sync list) and was skipped.
- Skipped per instructions: `epoch*.pt` checkpoints under `/workspace/runs/detect/case1/models/detector_runs/weed_crop_detector/weights/` (epoch0/5/10/15/20.pt, ~87 MB each) and `last.pt` (duplicate of `best.pt` — detector was trained with `val=False`, so `best.pt` == `last.pt`, both are the final-epoch weights).
- Detector weights (`weed_detector_finetuned.pt` == server `weights/best.pt` == `weights/last.pt`) were trained with `val=False`; only epoch 25 in `results.csv` carries validation metrics (mAP50 0.804, mAP50-95 0.496, P 0.706, R 0.769 — see `case1/output/server_2026-09-17/results.csv` and `detector_training_metrics.json`).
