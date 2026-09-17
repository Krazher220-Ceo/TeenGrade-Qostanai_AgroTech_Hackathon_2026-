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

STAGE_TO_IDX = {
    "rosette": 0,
    "stem_elongation": 1,
}
IDX_TO_STAGE = {v: k for k, v in STAGE_TO_IDX.items()}


class WeedMultiTaskDataset(Dataset):
    """
    Датасет эталонных фотографий и вырезок сорняков и культурных растений.
    Возвращает:
      image: Tensor [3, H, W]
      species_target: LongTensor (0..3)
      stage_target: LongTensor (0..1)
      stage_mask: FloatTensor (1.0 если стадия известна и применима, иначе 0.0)
      meta: Dict (путь, имя файла, группа)
    """
    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        transform: Optional[Callable] = None
    ):
        self.split = split
        self.transform = transform
        self.samples = []

        manifest_file = Path(manifest_path)
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_file}")

        with open(manifest_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    sp_str = row["species"]
                    st_str = row["stage"]

                    sp_idx = SPECIES_TO_IDX.get(sp_str, -1)
                    st_idx = STAGE_TO_IDX.get(st_str, -1)

                    # Стадия маскируется для пшеницы и для пырея (если розетка)
                    if sp_str == "crop_wheat":
                        has_valid_stage = False
                    elif sp_str == "couch_grass":
                        has_valid_stage = (st_idx != -1 and st_str == "stem_elongation")
                    else:
                        has_valid_stage = (st_idx != -1)

                    stage_mask = 1.0 if has_valid_stage else 0.0

                    self.samples.append({
                        "path": row["path"],
                        "species_idx": sp_idx,
                        "stage_idx": max(0, st_idx),
                        "stage_mask": stage_mask,
                        "group_id": row.get("group_id", ""),
                        "filename": row.get("filename", "")
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
