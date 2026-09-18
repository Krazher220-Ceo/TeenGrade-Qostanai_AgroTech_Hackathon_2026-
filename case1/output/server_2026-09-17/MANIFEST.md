# MANIFEST — training/eval outputs synced from GPU server (team1), 2026-09-17/18

Synced via `scp team1:...` on 2026-09-18. Source host: remote GPU container (`ssh team1`), project root `/workspace`.

| File | Source path on server | sha256 | Size (bytes) | Server mtime (UTC) |
|---|---|---|---|---|
| results.csv | /workspace/runs/detect/case1/models/detector_runs/weed_crop_detector/results.csv | b6e09e7db3cf074465c09bfdb1db94e017e476ec0b2b30298a64b50bb87ea429 | 2302 | 2026-09-17 20:40:13 |
| args.yaml | /workspace/runs/detect/case1/models/detector_runs/weed_crop_detector/args.yaml | 93368c8e0fa0f0530a6976658ab1fb994453f3c18cb5268cd9a45f6fe5429aa7 | 1789 | 2026-09-17 20:30:15 |
| detector_training_metrics.json | /workspace/case1/output/detector_training_metrics.json | 2d64772adc9564257a8e2cd83a42bdc6e82119ccd0153a1b4dc1e5ed31183b05 | 201 | 2026-09-17 20:40:21 |
| classifier_training_metrics.json | /workspace/case1/output/classifier_training_metrics.json | 70c66c642328eec205c117faa292a00202755bb1fcc42caac8abade219e88ba0 | 3520 | 2026-09-17 19:33:36 |
| latency_benchmark_report.json | /workspace/case1/output/latency_benchmark_report.json | 3f7dd0694a8673de8e8bd163e61149adda85d769191417cef9c23fa2bc303451 | 2266 | 2026-09-17 20:44:39 |

## Notes
- YOLO detector was trained with `val=False` (see `args.yaml`), so `results.csv` only carries validation-split metrics on the last logged epoch (epoch 25): mAP50 = 0.804, mAP50-95 = 0.496, precision = 0.706, recall = 0.769. All other epochs show train losses only, no val metrics.
- `detector_training_metrics.json` reports **test-split** metrics from a separate run: mAP50 = 0.7709, mAP50-95 = 0.4865.
- These are training-time numbers only. Independent test-set evaluation with per-class breakdown was run separately — see `eval/` subdirectory in this same folder (task 2, 2026-09-18) for `detector_test_metrics.json` and confusion-matrix plots.
- Local repo files with the same basenames (e.g. `case1/output/detector_training_metrics.json`, `case1/output/classifier_training_metrics.json`, `case1/output/latency_benchmark_report.json` at the top-level `case1/output/` dir) were NOT modified — these are separate dated copies for traceability.

## eval/ subdirectory (task 2 — independent test-set evaluation, 2026-09-18)
Produced by running `ultralytics.YOLO(...).val()` on the server against `/workspace/case1/configs/combined_detector.yaml`, weights `weed_detector_finetuned.pt`. Server working dir: `/workspace/case1/output/eval_2026-09-18/`.
- `detector_test_metrics.json` — full test split (381 images: 313 real + 56 synthetic copy-paste + 12 empty-soil negatives). mAP50 0.7715, mAP50-95 0.4858, per-class P/R/mAP, raw confusion matrix (crop/weed/background).
- `detector_test_real_only_metrics.json` — same eval restricted to the 313 "real" test images (excludes `synth_weed_*` and `bg_soil_*` by filename). mAP50 0.7571, mAP50-95 0.4713.
- `negatives_fp_report.json` — `model.predict()` on the 12 `bg_soil_*` empty-soil negative tiles at conf ≥ 0.25: 1 false-positive box total (1 "weed" FP across 12 images).
- `subset_counts.json` — counts used to split the test set (total 381 / synth 56 / negative 12 / real 313).
- `detector_test/`, `detector_test_real_only/` — confusion matrix PNGs (raw + normalized) and PR curve for each run.

## field_eval/ subdirectory (task 3 — real drone frames check, 2026-09-18)
Produced by a standalone script (`field_eval_run.py`, kept only on the server under `/workspace/case1/output/eval_2026-09-18/`, NOT committed to the repo) that monkeypatches `case1_main.FIELD_DIR` to a stratified sample and calls the existing `case1_main.cmd_process()` pipeline unmodified (finetuned detector + `multitask_weeds_best.pt` classifier), writing to a fresh `--output` dir so it never touched `case1/output/all_fields_*`.
- `sample_manifest.json` — the 40 images used: 10 per field (Поле 114/36/37/40 СПП1), picked at evenly spaced dates within each field's date folders (mid-sequence frame per date), fixed random seed 42.
- `all_fields_report.json` / `all_fields_detections.csv` — full per-image / per-detection pipeline output for the 40-image sample (711 weed detections total).
- `field_eval_summary.json` — aggregated stats: detections/frame, detector-confidence and classifier-species-confidence distributions, review-required share, species/stage distribution, per-frame runtime, relative-altitude stats.
- `negative_relalt_full_dataset.json` — separate full-dataset scan (all 1178 drone JPGs under `/data/ФотоПолей`, not just the 40-sample) for negative DJI XMP `RelativeAltitude`: 140/1178 (11.9%) negative.
- `examples/` — 8 resized (1400px wide, JPEG q82) annotated example frames chosen for range: highest detection count, a negative-RelAlt frame, two zero-detection frames, and mid-range frames across different fields.
