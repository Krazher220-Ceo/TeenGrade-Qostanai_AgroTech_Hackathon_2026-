# Dataset Information

## Source

**CropAndWeed Dataset** — Steininger et al., WACV 2023
- Repository: [cropandweed/cropandweed-dataset](https://github.com/cropandweed/cropandweed-dataset)
- Paper: [The CropAndWeed Dataset: A Multi-Modal Learning Approach for Efficient Crop and Weed Manipulation](https://openaccess.thecvf.com/content/WACV2023/papers/Steininger_The_CropAndWeed_Dataset_A_Multi-Modal_Learning_Approach_for_Efficient_Crop_WACV_2023_paper.pdf)

## Dataset Statistics

| Metric | Experimental | Application | Total |
|:-------|:------------|:-----------|:------|
| Sessions recorded | 1,363 | 738 | 2,101 |
| Sessions annotated | 665 | 264 | 929 |
| Images recorded | 22,597 | 21,217 | 43,814 |
| Images annotated | 4,990 | 3,044 | 8,034 |
| Instances annotated | 66,877 | 45,076 | 111,953 |

## Label Mapping: CropsOrWeed9

We use the **CropsOrWeed9** variant, which consolidates the original 74 fine-grained species labels into 9 classes:

| Class ID | Class Name | Type | Description |
|:---------|:-----------|:-----|:------------|
| 0 | Maize | Crop | *Zea mays* |
| 1 | Sugar Beet | Crop | *Beta vulgaris* |
| 2 | Soy | Crop | *Glycine max* |
| 3 | Sunflower | Crop | *Helianthus annuus* |
| 4 | Potato | Crop | *Solanum tuberosum* |
| 5 | Pea | Crop | *Pisum sativum* |
| 6 | Bean | Crop | *Phaseolus vulgaris* |
| 7 | Pumpkin | Crop | *Cucurbita* spp. |
| 8 | Weed | Weed superclass | 58 weed species merged (Grasses, Amaranth, Goosefoot, Knotweed, Corn spurry, Chickweed, Solanales, Potato weed, Chamomile, Thistle, Mercuries, Geranium, Crucifer, Poppy, Plantago, Labiate) |

This mapping is chosen because the downstream task (laser weeding) requires:
1. **Identifying specific crops** to protect them from laser exposure
2. **Detecting any weed** for targeted destruction — species-level weed ID is not needed

## Data Split

Standard 70/15/15 train/validation/test split (random, consistent across all experiments).

## Capture Conditions

- **Camera:** Semi-professional SLR, full-frame sensor, 50mm focal length
- **Perspective:** Top-down at ~1.1m height (representative of robot-mounted cameras)
- **Mode:** Manual capture, auto-exposure
- **Period:** March–July, collected over 4 years
- **Location:** Austrian agricultural sites and experimental plots
- **Variability:** Multiple soil types, lighting conditions (sunny to diffuse), moisture levels (dry to wet), and soil granularity (fine to coarse)

## Key Design Choices from the Paper

1. **Weed incorporation improves crop detection.** The paper demonstrates that training with weed classes alongside crops (CropsOrWeed9) achieves better crop detection AP (71.5 overall) than single-crop models (57.3 average) or binary crop/weed models (60.7).

2. **Instance size distribution matters.** Weeds tend to be more frequent but smaller than crops. The augmentation strategy (mosaic, copy-paste) specifically targets small-object performance.

3. **Vegetation fallback class.** Instances smaller than 16² pixels are assigned to a "Vegetation" class and excluded from training/evaluation, as they cannot be reliably identified even by human annotators.
