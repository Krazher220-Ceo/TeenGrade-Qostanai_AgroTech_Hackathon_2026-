"""
Датасет и загрузчик данных для многозадачной модели классификации сорняков.
"""

import csv
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict, Any
import torch
from torch.utils.data import Dataset
from PIL import Image

SPECIES_TO_IDX = {
    "field_thistle": 0,
    "field_bindweed": 1,
    "couch_grass": 2,
    "crop_wheat": 3,
}
IDX_TO_SPECIES = {v: k for k, v in SPECIES_TO_IDX.items()}

# Базовые 3 агрономические фазы по шпаргалке ментора «Олжа Агро»:
STAGE_TO_IDX = {
    # 1. Семядоли — 2 листа (Оптимальное окно, базовая/минимальная дозировка)
    "cotyledon_to_2_leaves": 0,
    "rosette": 0,
    "всходы": 0,
    "розетка": 0,

    # 2. 4–6 листьев (Сорняк грубеет, дозировка +15–20%)
    "4_to_6_leaves": 1,
    "stem_elongation": 1,
    "стеблевание": 1,

    # 3. Более 6 листьев / цветение (Упущенное окно, риск фитотоксичности культуры)
    "over_6_leaves_or_flowering": 2,
    "цветение": 2,
    "плодоношение": 2,
    "flowering": 2,
    "fruiting": 2,
}
IDX_TO_STAGE = {0: "cotyledon_to_2_leaves", 1: "4_to_6_leaves", 2: "over_6_leaves_or_flowering"}
STAGE_NAMES = ["cotyledon_to_2_leaves", "4_to_6_leaves", "over_6_leaves_or_flowering"]


class WeedMultiTaskDataset(Dataset):
    """
    Датасет эталонных фотографий сорняков (до 26 видов и 3 фаз по шпаргалке «Олжа Агро»).
    """
    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        species_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.split = split
        self.transform = transform
        self.samples = []

        manifest_file = Path(manifest_path)
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_file}")

        with open(manifest_file, "r", encoding="utf-8") as f:
            all_rows = list(csv.DictReader(f))

        # Динамическое определение видов сорняков
        if species_to_idx is None:
            unique_species = sorted(list(set(row["species"] for row in all_rows if row.get("species"))))
            self.species_to_idx = {sp: idx for idx, sp in enumerate(unique_species)}
        else:
            self.species_to_idx = species_to_idx

        self.idx_to_species = {v: k for k, v in self.species_to_idx.items()}

        for row in all_rows:
            if row.get("split") == split:
                sp_str = row.get("species", "")
                st_str = row.get("stage_code") or row.get("stage") or row.get("stage_folder", "")

                sp_idx = self.species_to_idx.get(sp_str, -1)
                st_idx = STAGE_TO_IDX.get(st_str.lower(), -1)

                has_valid_stage = (st_idx != -1)
                stage_mask = 1.0 if has_valid_stage else 0.0

                self.samples.append({
                    "path": row["path"],
                    "species_idx": max(0, sp_idx),
                    "stage_idx": max(0, st_idx),
                    "stage_mask": stage_mask,
                    "group_id": row.get("group_id", ""),
                    "filename": row.get("filename", ""),
                    "species": sp_str,
                    "stage": st_str,
                    "weed_type": row.get("weed_type", "annual"),
                    "botanical_class": row.get("botanical_class", "A"),
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, float, Dict[str, Any]]:
        item = self.samples[idx]
        img_path = item["path"]

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return (
            img,
            torch.tensor(item["species_idx"], dtype=torch.long),
            torch.tensor(item["stage_idx"], dtype=torch.long),
            torch.tensor(item["stage_mask"], dtype=torch.float32),
            item
        )
