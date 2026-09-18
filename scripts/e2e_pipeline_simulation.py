#!/usr/bin/env python3
"""Deterministic synthetic end-to-end harness for the Qostanai weed pipeline.

This program does *not* load the trained checkpoints and its scores must not be
reported as field/model validation.  It exercises contracts and safety gates:

camera frame -> 1024 px slicing -> ExG candidates -> detector fragments ->
IoM clump merge -> species/stage classification -> offline review queue ->
human overrides -> idempotent batch sync -> GeoJSON + ISOXML-like TASKDATA.XML.

The generated plots are deliberately watermarked ``SYNTHETIC SIMULATION``.
Run:
    .venv/bin/python scripts/e2e_pipeline_simulation.py --output /private/tmp/qostanai-e2e
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import statistics
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qostanai-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "qostanai-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEED = 20260918
SPECIES = (
    "field_thistle", "perennial_sowthistle", "field_bindweed", "leafy_spurge",
    "tatarian_lettuce", "wormwood_bitter", "mugwort", "statice", "dandelion",
    "horse_sorrel", "redroot_pigweed", "common_lambsquarters", "tartary_buckwheat",
    "smartweed", "spear_saltbush", "stickseed", "canadian_fleabane",
    "scentless_mayweed", "redstem_filaree", "volunteer_flax",
    "volunteer_sunflower", "couch_grass", "wild_oat", "barnyard_grass",
    "foxtail", "volunteer_wheat",
)
CROP = "crop_wheat"
ALL_CLASSES = SPECIES + (CROP,)
STAGES = ("cotyledon_to_2_leaves", "4_to_6_leaves", "over_6_leaves_or_flowering")
REVIEW_THRESHOLD = 0.60
SPRAY_RATE_L_HA = 150.0


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    source: str

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass(frozen=True)
class Decision:
    event_id: str
    frame_id: str
    object_id: int
    bbox_xyxy: tuple[int, int, int, int]
    detector_conf: float
    predicted_species: str
    species_conf: float
    predicted_stage: str
    stage_conf: float
    action: str
    review_required: bool
    lon: float
    lat: float
    source: str = "ai"
    verified_by: str | None = None
    verified_at: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tile_frame(width: int, height: int, tile_size: int = 1024, overlap: float = 0.20) -> list[tuple[int, int, int, int]]:
    stride = int(tile_size * (1.0 - overlap))
    tiles = set()
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            x2, y2 = min(x + tile_size, width), min(y + tile_size, height)
            tiles.add((max(0, x2 - tile_size), max(0, y2 - tile_size), x2, y2))
    return sorted(tiles)


def simulate_frame(rng: np.random.Generator, width: int = 4096, height: int = 3072) -> tuple[np.ndarray, list[Box], list[str]]:
    """Create RGB vegetation blobs and object-level synthetic truth."""
    yy, xx = np.mgrid[0:height:8j, 0:width:8j]
    small = np.zeros((8, 8, 3), dtype=np.float32)
    small[..., 0] = 98 + 18 * np.sin(xx / 700)
    small[..., 1] = 86 + 14 * np.cos(yy / 600)
    small[..., 2] = 60
    frame = np.repeat(np.repeat(small, height // 8, axis=0), width // 8, axis=1).astype(np.uint8)
    truth: list[Box] = []
    labels: list[str] = []
    chosen = ["field_thistle", "field_bindweed", CROP, "couch_grass", "wild_oat", "redroot_pigweed"]
    for i, label in enumerate(chosen):
        cx = int(rng.integers(300, width - 300)); cy = int(rng.integers(250, height - 250))
        rx = int(rng.integers(70, 180)); ry = int(rng.integers(60, 160))
        y0, y1 = max(0, cy - ry), min(height, cy + ry)
        x0, x1 = max(0, cx - rx), min(width, cx + rx)
        patch_y, patch_x = np.ogrid[y0:y1, x0:x1]
        mask = ((patch_x - cx) / rx) ** 2 + ((patch_y - cy) / ry) ** 2 <= 1
        frame[y0:y1, x0:x1][mask] = (35 + i * 2, 145 + i * 5, 38)
        truth.append(Box(x0, y0, x1, y1, 0.99, "truth")); labels.append(label)
    return frame, truth, labels


def exg_candidates(frame: np.ndarray, truth: Sequence[Box], threshold: float = 30.0) -> tuple[list[Box], float]:
    """Apply ExG=2G-R-B and approximate connected components with known blobs."""
    rgb = frame.astype(np.float32)
    exg = 2.0 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]
    vegetation_fraction = float(np.mean(exg > threshold))
    candidates = [replace(b, score=0.70, source="exg") for b in truth if np.mean(exg[int(b.y1):int(b.y2), int(b.x1):int(b.x2)] > threshold) > 0.20]
    return candidates, vegetation_fraction


def simulate_detector_fragments(rng: np.random.Generator, truth: Sequence[Box]) -> list[Box]:
    boxes: list[Box] = []
    for obj in truth:
        fragments = 7 if obj.area > 50_000 else 3
        boxes.append(replace(obj, score=float(rng.uniform(0.68, 0.94)), source="detector"))
        for _ in range(fragments):
            w, h = obj.x2 - obj.x1, obj.y2 - obj.y1
            dx1, dy1 = rng.uniform(0.0, 0.45, 2); dx2, dy2 = rng.uniform(0.55, 1.0, 2)
            boxes.append(Box(obj.x1 + dx1*w, obj.y1 + dy1*h, obj.x1 + dx2*w, obj.y1 + dy2*h,
                             float(rng.uniform(0.30, 0.80)), "detector_fragment"))
    boxes.append(Box(20, 20, 35, 37, 0.31, "micro_noise"))
    return boxes


def iom(a: Box, b: Box) -> float:
    ix1, iy1, ix2, iy2 = max(a.x1, b.x1), max(a.y1, b.y1), min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / max(1e-9, min(a.area, b.area))


def merge_contained(boxes: Iterable[Box], iom_threshold: float = 0.35) -> list[Box]:
    work = sorted((b for b in boxes if b.x2-b.x1 >= 30 and b.y2-b.y1 >= 30), key=lambda b: b.score, reverse=True)
    merged: list[Box] = []
    while work:
        seed, rest = work[0], work[1:]
        cluster, remaining = [seed], []
        for candidate in rest:
            if max(iom(member, candidate) for member in cluster) > iom_threshold:
                cluster.append(candidate)
            else:
                remaining.append(candidate)
        merged.append(Box(min(b.x1 for b in cluster), min(b.y1 for b in cluster),
                          max(b.x2 for b in cluster), max(b.y2 for b in cluster),
                          max(b.score for b in cluster), "+".join(sorted({b.source for b in cluster}))))
        work = remaining
    return merged


def nearest_truth(box: Box, truth: Sequence[Box]) -> int:
    cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
    return min(range(len(truth)), key=lambda i: ((truth[i].x1+truth[i].x2)/2-cx)**2 + ((truth[i].y1+truth[i].y2)/2-cy)**2)


def classify(rng: np.random.Generator, frame_id: str, boxes: Sequence[Box], truth: Sequence[Box], labels: Sequence[str]) -> tuple[list[Decision], list[str]]:
    decisions, y_true = [], []
    for i, box in enumerate(boxes, 1):
        truth_i = nearest_truth(box, truth); actual = labels[truth_i]
        predicted = actual if rng.random() > 0.18 else str(rng.choice(ALL_CLASSES))
        confidence = float(rng.uniform(0.45, 0.98))
        stage = str(rng.choice(STAGES)); stage_conf = float(rng.uniform(0.52, 0.97))
        review = confidence < REVIEW_THRESHOLD
        action = "manual_review" if review else ("do_not_spray" if predicted == CROP else "spray_weed")
        center_x, center_y = (box.x1+box.x2)/2, (box.y1+box.y2)/2
        lon = 63.6240 + center_x / 4096 * 0.002
        lat = 53.2190 + center_y / 3072 * 0.002
        decisions.append(Decision(str(uuid.uuid5(uuid.NAMESPACE_URL, f"{frame_id}:{i}")), frame_id, i,
                                  tuple(map(round, (box.x1, box.y1, box.x2, box.y2))), round(box.score, 4),
                                  predicted, round(confidence, 4), stage, round(stage_conf, 4), action, review,
                                  round(lon, 7), round(lat, 7)))
        y_true.append(actual)
    return decisions, y_true


def assert_safety(decisions: Sequence[Decision]) -> None:
    for d in decisions:
        assert not (d.predicted_species == CROP and d.action != "do_not_spray"), f"crop safety violation: {d}"
        assert not (d.species_conf < REVIEW_THRESHOLD and d.action != "manual_review"), f"abstention violation: {d}"


class OfflineQueue:
    """SQLite analogue of the PWA IndexedDB queue, with durable event IDs."""
    def __init__(self, path: Path):
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS events(event_id TEXT PRIMARY KEY, payload TEXT NOT NULL, synced INTEGER NOT NULL DEFAULT 0)")

    def add(self, decision: Decision) -> None:
        self.db.execute("INSERT OR REPLACE INTO events(event_id,payload,synced) VALUES(?,?,0)",
                        (decision.event_id, json.dumps(asdict(decision), ensure_ascii=False)))
        self.db.commit()

    def pending(self) -> list[Decision]:
        rows = self.db.execute("SELECT payload FROM events WHERE synced=0 ORDER BY event_id").fetchall()
        return [Decision(**json.loads(row[0])) for row in rows]

    def mark_synced(self, ids: Sequence[str]) -> None:
        self.db.executemany("UPDATE events SET synced=1 WHERE event_id=?", ((i,) for i in ids)); self.db.commit()

    def close(self) -> None:
        self.db.close()


def agronomist_review(decisions: Sequence[Decision], y_true: Sequence[str]) -> list[Decision]:
    reviewed = []
    for d, actual in zip(decisions, y_true):
        if d.review_required or d.predicted_species != actual:
            action = "do_not_spray" if actual == CROP else "spray_weed"
            reviewed.append(replace(d, predicted_species=actual, species_conf=1.0, action=action,
                                    review_required=False, source="human_override", verified_by="field_agronomist",
                                    verified_at=utc_now()))
        else:
            reviewed.append(d)
    return reviewed


def server_sync(batch: Sequence[Decision], ledger_path: Path) -> dict:
    """Idempotent last-human-write-wins sync; AI may never overwrite a human record."""
    ledger = json.loads(ledger_path.read_text("utf-8")) if ledger_path.exists() else {}
    applied, duplicate, rejected = [], [], []
    for item in batch:
        old = ledger.get(item.event_id)
        if old == asdict(item):
            duplicate.append(item.event_id); continue
        if old and old.get("source") == "human_override" and item.source == "ai":
            rejected.append(item.event_id); continue
        ledger[item.event_id] = asdict(item); applied.append(item.event_id)
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), "utf-8")
    return {"applied": applied, "duplicate": duplicate, "rejected": rejected, "server_count": len(ledger)}


def write_geojson(decisions: Sequence[Decision], path: Path) -> None:
    features = []
    for d in decisions:
        features.append({"type": "Feature", "id": d.event_id,
                         "geometry": {"type": "Point", "coordinates": [d.lon, d.lat]},
                         "properties": {"species": d.predicted_species, "stage": d.predicted_stage,
                                        "action": d.action, "rate_l_ha": SPRAY_RATE_L_HA if d.action == "spray_weed" else 0.0,
                                        "source": d.source, "crs": "EPSG:4326"}})
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2), "utf-8")


def write_taskdata(decisions: Sequence[Decision], path: Path) -> None:
    """Write a well-formed ISOXML-like demo artifact, not a certified TC file."""
    root = ET.Element("ISO11783_TaskData", VersionMajor="4", VersionMinor="3", DataTransferOrigin="1",
                      ManagementSoftwareManufacturer="Qostanai AgroVision", ManagementSoftwareVersion="simulation-1")
    task = ET.SubElement(root, "TSK", A="TSK-1", B="Synthetic spot-spray prescription", G="1")
    zone = ET.SubElement(task, "TZN", A="1", B="Reviewed targets")
    for index, d in enumerate(decisions, 1):
        point = ET.SubElement(zone, "PNT", A="2", B=f"{d.lat:.7f}", C=f"{d.lon:.7f}", D="0", E=str(index))
        ET.SubElement(point, "PDV", A="1", B=str(int(SPRAY_RATE_L_HA * 1000) if d.action == "spray_weed" else 0), C="0")
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    ET.parse(path)  # well-formedness check only


def confusion(y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]) -> np.ndarray:
    index = {c: i for i, c in enumerate(classes)}; matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for actual, pred in zip(y_true, y_pred): matrix[index[actual], index[pred]] += 1
    return matrix


def watermark(ax) -> None:
    ax.text(0.5, 0.5, "SYNTHETIC SIMULATION", transform=ax.transAxes, ha="center", va="center",
            fontsize=20, color="crimson", alpha=0.16, rotation=22, weight="bold")


def write_plots(out: Path, rng: np.random.Generator, decisions: Sequence[Decision], y_true: Sequence[str], latencies_ms: Sequence[float]) -> None:
    plots = out / "plots"; plots.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.hist(latencies_ms, bins=24, color="#2a9d8f", alpha=.85)
    for q, label in ((50, "p50"), (95, "p95"), (99, "p99")):
        value = float(np.percentile(latencies_ms, q)); ax.axvline(value, label=f"{label}={value:.1f} ms")
    ax.set(title="Synthetic pipeline latency", xlabel="milliseconds", ylabel="events"); ax.legend(); watermark(ax)
    fig.tight_layout(); fig.savefig(plots / "latency_distribution.png", dpi=160); plt.close(fig)

    active = sorted(set(y_true) | {d.predicted_species for d in decisions})
    cm = confusion(y_true, [d.predicted_species for d in decisions], active)
    fig, ax = plt.subplots(figsize=(8, 7)); im = ax.imshow(cm, cmap="Blues"); fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(active)), active, rotation=45, ha="right"); ax.set_yticks(range(len(active)), active)
    ax.set(xlabel="predicted", ylabel="actual", title="Synthetic species confusion matrix")
    for i in range(len(active)):
        for j in range(len(active)): ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    watermark(ax); fig.tight_layout(); fig.savefig(plots / "confusion_matrix.png", dpi=160); plt.close(fig)

    # One-vs-rest ROC for crop safety, produced from synthetic confidence scores.
    truth_crop = np.array([v == CROP for v in y_true], dtype=int)
    crop_scores = np.array([d.species_conf if d.predicted_species == CROP else 1-d.species_conf for d in decisions])
    thresholds = np.linspace(1.0, 0.0, 101); tpr, fpr = [], []
    for threshold in thresholds:
        pred = crop_scores >= threshold
        tp = np.sum(pred & (truth_crop == 1)); fp = np.sum(pred & (truth_crop == 0))
        tpr.append(tp / max(1, np.sum(truth_crop == 1))); fpr.append(fp / max(1, np.sum(truth_crop == 0)))
    order = np.argsort(fpr); auc = float(np.trapezoid(np.array(tpr)[order], np.array(fpr)[order]))
    fig, ax = plt.subplots(figsize=(6, 6)); ax.plot(np.array(fpr)[order], np.array(tpr)[order], label=f"crop OvR AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray"); ax.set(xlabel="false-positive rate", ylabel="true-positive rate", title="Synthetic ROC: crop safety class"); ax.legend(); watermark(ax)
    fig.tight_layout(); fig.savefig(plots / "roc_crop_safety.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6)); spray = [d for d in decisions if d.action == "spray_weed"]
    if spray:
        h = ax.hexbin([d.lon for d in spray], [d.lat for d in spray], gridsize=14, cmap="YlOrRd", mincnt=1); fig.colorbar(h, ax=ax, label="reviewed targets")
    ax.scatter([d.lon for d in decisions if d.action != "spray_weed"], [d.lat for d in decisions if d.action != "spray_weed"], marker="x", c="blue", label="no spray")
    ax.set(title="Synthetic reviewed prescription heatmap", xlabel="longitude", ylabel="latitude"); ax.legend(); watermark(ax)
    fig.tight_layout(); fig.savefig(plots / "field_heatmap.png", dpi=160); plt.close(fig)


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED)
    timings: dict[str, float] = {}; start = time.perf_counter()
    frame, truth, labels = simulate_frame(rng); timings["capture"] = (time.perf_counter()-start)*1000
    start = time.perf_counter(); tiles = tile_frame(frame.shape[1], frame.shape[0]); timings["tiling"] = (time.perf_counter()-start)*1000
    start = time.perf_counter(); exg, vegetation_fraction = exg_candidates(frame, truth); timings["exg"] = (time.perf_counter()-start)*1000
    start = time.perf_counter(); fragments = simulate_detector_fragments(rng, truth); timings["detector_stub"] = (time.perf_counter()-start)*1000
    start = time.perf_counter(); merged = merge_contained(fragments + exg); timings["iom_merge"] = (time.perf_counter()-start)*1000
    decisions, y_true = classify(rng, "DJI_SYNTHETIC_0001", merged, truth, labels); assert_safety(decisions)

    queue = OfflineQueue(output / "offline_queue.sqlite")
    for decision in decisions: queue.add(decision)
    pending_before = len(queue.pending()); reviewed = agronomist_review(queue.pending(), y_true); assert_safety(reviewed)
    sync_result = server_sync(reviewed, output / "server_ledger.json"); queue.mark_synced(sync_result["applied"] + sync_result["duplicate"])
    duplicate_result = server_sync(reviewed, output / "server_ledger.json")
    pending_after = len(queue.pending()); queue.close()

    write_geojson(reviewed, output / "prescription.geojson"); write_taskdata(reviewed, output / "TASKDATA.XML")
    latency = np.concatenate([rng.lognormal(math.log(6.8), .28, 900), rng.lognormal(math.log(58), .35, 100)])
    write_plots(output, rng, decisions, y_true, latency)
    report = {
        "provenance": "SYNTHETIC_SIMULATION_NOT_MODEL_VALIDATION", "seed": SEED,
        "configuration": {"review_threshold": REVIEW_THRESHOLD, "iom_threshold": .35, "tile_size": 1024, "overlap": .20},
        "counts": {"tiles": len(tiles), "truth_objects": len(truth), "exg_candidates": len(exg),
                   "raw_detector_boxes": len(fragments), "merged_objects": len(merged),
                   "offline_before_sync": pending_before, "offline_after_sync": pending_after},
        "safety": {"crop_sprayed": sum(d.predicted_species == CROP and d.action != "do_not_spray" for d in reviewed),
                   "low_confidence_auto_actioned": sum(d.species_conf < REVIEW_THRESHOLD and d.action != "manual_review" for d in decisions)},
        "sync": sync_result, "idempotent_replay": duplicate_result,
        "latency_ms": {"p50": round(float(np.percentile(latency, 50)), 3), "p95": round(float(np.percentile(latency, 95)), 3),
                       "p99": round(float(np.percentile(latency, 99)), 3), "mean": round(statistics.mean(latency), 3)},
        "vegetation_fraction": round(vegetation_fraction, 5), "stage_timings_ms": {k: round(v, 3) for k, v in timings.items()},
        "artifact_sha256": {},
        "limitations": ["Synthetic detector/classifier stubs; no checkpoint inference.", "GPS is fabricated and not photogrammetric georeferencing.",
                        "TASKDATA.XML is well-formed demonstration output, not ISO 11783 conformance certification."],
    }
    for name in ("prescription.geojson", "TASKDATA.XML", "server_ledger.json"):
        report["artifact_sha256"][name] = hashlib.sha256((output/name).read_bytes()).hexdigest()
    (output / "simulation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("case1/output/e2e_simulation"))
    args = parser.parse_args(); report = run(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2)); print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
