---
license: agpl-3.0
library_name: ultralytics
pipeline_tag: object-detection
tags:
  - yolov8
  - object-detection
  - precision-agriculture
  - weed-detection
  - crop-detection
  - computer-vision
  - transfer-learning
  - autonomous-robotics
  - laser-weeding
datasets:
  - cropandweed/cropandweed-dataset
metrics:
  - precision
model-index:
  - name: yolov8s-weedblaster-cropandweed
    results:
      - task:
          type: object-detection
        dataset:
          name: CropAndWeed (CropsOrWeed9 variant)
          type: cropandweed/cropandweed-dataset
        metrics:
          - name: Overall Precision
            type: precision
            value: 0.96
          - name: Weed Precision
            type: precision
            value: 0.85
---

# WeedBlaster Vision Model — YOLOv8s on CropAndWeed

> Detection backbone for a laser-powered autonomous weed-killing robot. Identifies 8 individual crop species and a weed superclass in real time to enable precision laser ablation without chemicals.

## Overview

This is a **YOLOv8-s (small)** model fine-tuned via transfer learning on the [CropAndWeed dataset](https://github.com/cropandweed/cropandweed-dataset) (Steininger et al., WACV 2023). The model uses the **CropsOrWeed9** label mapping: 8 individual crop classes plus a single weed superclass, optimized for the real-world task of distinguishing "what to protect" from "what to destroy."

**Why this matters:** Herbicides account for ~$30B in global ag-chem spend annually and are linked to soil degradation, water contamination, and biodiversity loss. Precision laser weeding eliminates chemical use entirely by targeting individual weed plants at the stem — but only if the vision system can reliably distinguish crops from weeds in real time, across variable field conditions.

## Model Performance

**Version:** v1.9 · **Architecture:** YOLOv8-s · **Input Resolution:** 1280×1280 px · **Augmentation:** Enabled

| Class | Precision |
|:------|:---------:|
| **Overall** | **0.960** |
| Bean | 0.988 |
| Sunflower | 0.985 |
| Pumpkin | 0.980 |
| Pea | 0.975 |
| Maize | 0.973 |
| Sugar Beet | 0.973 |
| Soy | 0.950 |
| Potato | 0.920 |
| **Weed (superclass)** | **0.850** |

The model achieves 96% overall precision, meaning fewer than 4% of detections are false positives — critical for a system that fires a laser at whatever it classifies as a weed.

## Training Curves

![Training Results](results/results.png)

![Confusion Matrix](results/confusion_matrix_normalized.png)

![PR Curve](results/PR_curve.png)

## Dataset

**[CropAndWeed Dataset](https://github.com/cropandweed/cropandweed-dataset)** — Steininger et al., WACV 2023.

- **8,034 annotated images** from 929 recording sessions across Austrian agricultural sites and experimental plots
- **~112k annotated plant instances** across 74 species (16 crops, 58 weeds)
- Bounding boxes, semantic masks, and stem positions
- High variability in lighting, soil type, moisture, and growth stage
- Top-down capture at ~1.1m height with 50mm focal length — representative of robot-mounted camera perspectives

We use the **CropsOrWeed9** variant, which maps the 74 original classes into 8 crop species (Maize, Sugar Beet, Soy, Sunflower, Potato, Pea, Bean, Pumpkin) and a single Weed superclass aggregating all 58 weed species.

## Training Configuration

| Parameter | Value |
|:----------|:------|
| Base model | `yolov8s.pt` (COCO pretrained) |
| Epochs | 100 |
| Image size | 1280×1280 px |
| Batch size | 8 |
| Optimizer | AdamW |
| Learning rate | 0.001 → cosine anneal to 0.00001 (lrf=0.01) |
| Weight decay | 0.0005 |
| Warmup | 3 epochs |
| Precision | AMP (FP16) |
| Hardware | NVIDIA Quadro RTX 6000 (24 GB) |

### Data Augmentation

Augmentation was tuned for agricultural top-down imagery. Rotations and perspective transforms are disabled to preserve plant orientation; mosaic and copy-paste are enabled to improve small-object detection for weeds.

| Augmentation | Value | Rationale |
|:-------------|:------|:----------|
| HSV Hue | ±0.015 | Simulate lighting variation |
| HSV Saturation | ±0.7 | Soil color / moisture differences |
| HSV Value | ±0.4 | Shadow and exposure changes |
| Scale | ±0.5 | Growth-stage variation |
| Translate | ±0.1 | Camera positioning jitter |
| Horizontal Flip | 0.5 | Safe for top-down crop views |
| Vertical Flip | 0.0 | Crops have vertical orientation |
| Rotation | 0.0° | Preserves row structure |
| Mosaic | 1.0 | Excellent for small weed instances |
| Mixup | 0.1 | Helps with overlapping boundaries |
| Copy-Paste | 0.1 | Synthetic weed density variation |

## Quick Start

```python
from ultralytics import YOLO

# Load model
model = YOLO("best.pt")

# Run inference
results = model.predict("field_image.jpg", imgsz=1280, conf=0.25)

# Access detections
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_names = ['Maize', 'Sugar Beet', 'Soy', 'Sunflower',
                       'Potato', 'Pea', 'Bean', 'Pumpkin', 'Weed']
        print(f"{class_names[cls]}: {conf:.3f}")
```

## Intended Use

This model is the perception component of an autonomous laser weeding system. The operational pipeline is:

1. **Detect** — this model identifies all crop and weed instances in the camera frame
2. **Classify** — crops are protected; weeds are targeted
3. **Localize** — bounding box centers (or future stem-point regression) guide laser aim
4. **Ablate** — a focused laser destroys the weed at the stem, no chemicals required

### Limitations

- Trained on Austrian agricultural data; generalization to other geographies, soil types, and crop varieties requires validation
- Weed superclass does not differentiate between weed species — sufficient for "destroy all weeds" but not for selective herbicide application
- Performance on tiny instances (<16² px) is limited, consistent with dataset design
- Not validated for safety-critical deployment without additional testing and redundancy

## Citation

If you use this model, please cite the underlying dataset:

```bibtex
@inproceedings{steininger2023cropandweed,
  title={The CropAndWeed Dataset: A Multi-Modal Learning Approach for Efficient Crop and Weed Manipulation},
  author={Steininger, Daniel and Trondl, Andreas and Croonen, Gerardus and Simon, Julia and Widhalm, Verena},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={3729--3738},
  year={2023}
}
```

## License

This model is released under [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html). The CropAndWeed dataset is available for academic use — see the [dataset repository](https://github.com/cropandweed/cropandweed-dataset) for terms.
