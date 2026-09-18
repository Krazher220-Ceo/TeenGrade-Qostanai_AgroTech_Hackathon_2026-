# Original User Request

## Initial Request — 2026-09-18T02:13:28Z

# Teamwork Project Prompt

Finalize the release deliverables and presentation package for Case №1 («Олжа Агро», Qostanai AgroTech Hackathon 2026), creating a production-grade multimodal README.md, verifying and generating prescription map and ISO-XML demo artifacts, designing the offline PWA «АгроСкаут» interface representation, and authoring a 5-minute pitch deck with speaker script for the jury.

Working directory: /Users/kr220/Documents/Projects/Qostanai_AgroTech_Hackathon_2026
Integrity mode: demo

## Requirements

### R1. Multimodal Root README.md
Update and polish the root `README.md` to a presentation-grade standard:
- Prominent badges: Python 3.10+, PyTorch, FastAPI, YOLOv8, PWA Offline, ISO 11783, Streamlit.
- ASCII and Mermaid architecture diagrams illustrating the end-to-end pipeline: UAV Edge survey → ExG filter & 1024px tiling → YOLOv8s weed candidate detection → IoM Clump Merging (anti-matryoshka) → Multi-Task EfficientNet-B0 (26 species + growth stages) → Safety Quarantine Gate (<60% confidence) → Field Agronomist Offline PWA («АгроСкаут») → Server Synchronization Ledger → ISO-XML / GeoJSON Spraying Map Export.
- Audited performance and agronomic metric tables:
  - Detection & Classification: mAP@50 80.4%, Test mAP@50 77.1%, 26 weeds + wheat.
  - Kinematics & Latency: p50 = 6.8ms, p95 = 68.0ms; at 18–20 km/h (5.0–5.56 m/s) spray offset is 3.4–37.8 cm.
  - Economic Impact: 82.4% herbicide savings (+1,450 ₸/ha savings, 15M ₸ saved per 1000 ha).
- Exhaustive step-by-step launch and usage instructions:
  - Starting the FastAPI backend (`case1/server/app.py`).
  - Running and testing the offline PWA «АгроСкаут» on Android / mobile browser.
  - Launching the Streamlit dashboard on Windows/macOS (`case1/dashboard/app.py`).
  - Generating and exporting Spot-Spraying maps (GeoJSON & ISO-XML).

### R2. Visual Demonstration & Artifact Verification
- Verify and demonstrate programmatic generation of Spot-Spraying Prescription Map (`prescription.geojson`) and ISO 11783 TaskController XML (`TASKDATA.XML`) based on verified field detections.
- Structure and document a high-contrast visual ASCII / Markdown layout of the mobile PWA screen «АгроСкаут» (Tinder-UX inspection card, offline sync badge, GPS coordinates, wheat override button, species selector modal).

### R3. 5-Minute Pitch Deck & Jury Defense Script
Author a comprehensive pitch guide in `docs/pitch_deck_5min_jury.md` with exact timing (0:00 to 5:00), slide cues, and word-for-word speaker notes in Russian for 6 slides:
- Slide 1 (0:00 - 0:45): Agronomist pain point at «Олжа Агро» (15M ₸ pesticide waste per 1000 ha, soil toxicity, weed resistance).
- Slide 2 (0:45 - 1:35): Edge detection & overcoming "matryoshka" nesting (ExG vegetative index + IoM Clump Merging reducing 55 redundant boxes to 15 clean weed rosettes).
- Slide 3 (1:35 - 2:30): Crop safety (Multi-Task Focal Loss for 26 weeds + wheat, <60% quarantine to `manual_review`, 0% accidental wheat spraying).
- Slide 4 (2:30 - 3:15): Field «АгроСкаут» PWA (Offline-first IndexedDB + Service Worker vs native mobile apps, 100% operation in remote steppe).
- Slide 5 (3:15 - 4:05): Agricultural machinery integration (ISO 11783 Task Controller XML / Shapefile directly into John Deere CommandCenter / Amazone Amatron 4).
- Slide 6 (4:05 - 4:45): Economic impact and unit economics (82.4% herbicide savings, UAV hardware payback in 1 single season).
- Appendix (4:45 - 5:00 + Q&A): Anticipated tricky jury questions & bulletproof technical answers.

## Acceptance Criteria

### Documentation Quality
- [ ] Root `README.md` is updated with all requested badges, valid Mermaid diagrams, metric tables, and verified CLI instructions.
- [ ] Metric tables in `README.md` accurately match the audited project reports (`mAP@50 80.4%`, `p50=6.8ms`, `82.4% savings`).

### Code & Artifact Verification
- [ ] Spot-Spraying prescription GeoJSON and ISO 11783 `TASKDATA.XML` generation is verified and outputs are well-formed.
- [ ] High-contrast visual mock and state structure of the «АгроСкаут» offline PWA interface is documented with offline sync badges and review flows.

### Pitch Readiness
- [ ] `docs/pitch_deck_5min_jury.md` contains exact 5-minute timestamps, slide contents, word-for-word speaker lines in Russian, and jury Q&A defense arguments.
