#!/usr/bin/env python3
"""
Генератор полного манифеста датасета сорняков (10 226 изображений)
в строгом соответствии с агрономической шпаргалкой «Олжа Агро»:
1. Класс A (Двудольные) vs Класс B (Злаковые)
2. Малолетние vs Многолетние (критический приоритет!)
3. 3 технологических окна фаз развития
"""

import os
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

# Базовые пути
ROOT_DIR = Path(__file__).resolve().parents[2]
SERVER_DATA = Path("/data/Сорняки")
LOCAL_DATA = ROOT_DIR / "Dataset 1 кейс" / "Сорняки"

WEEDS_DIR = SERVER_DATA if SERVER_DATA.exists() else LOCAL_DATA

# 1. БОТАНИЧЕСКИЙ СПРАВОЧНИК ПО ШПАРГАЛКЕ «ОЛЖА АГРО»
BOTANICAL_TAXONOMY: Dict[str, Dict[str, str]] = {
    # КЛАСС A: Двудольные (Широколистные)
    # Многолетники (самый опасный и приоритетный сектор! ЭПВ >= 2 шт/м²):
    "Бодяк полевой": {"class": "A", "type": "perennial", "ru": "Бодяк полевой", "en": "field_thistle"},
    "Осот полевой": {"class": "A", "type": "perennial", "ru": "Осот полевой", "en": "perennial_sowthistle"},
    "Вьюнок полевой": {"class": "A", "type": "perennial", "ru": "Вьюнок полевой", "en": "field_bindweed"},
    "Молочай лозный": {"class": "A", "type": "perennial", "ru": "Молочай лозный", "en": "leafy_spurge"},
    "Молокан татарский": {"class": "A", "type": "perennial", "ru": "Молокан татарский", "en": "tatarian_lettuce"},
    "Полынь горькая": {"class": "A", "type": "perennial", "ru": "Полынь горькая", "en": "wormwood_bitter"},
    "Полынь обыкновенная": {"class": "A", "type": "perennial", "ru": "Полынь обыкновенная", "en": "mugwort"},
    "Кермек широколистный": {"class": "A", "type": "perennial", "ru": "Кермек широколистный", "en": "statice"},
    "Одуванчик лекарственный": {"class": "A", "type": "perennial", "ru": "Одуванчик лекарственный", "en": "dandelion"},
    "Конский щавель": {"class": "A", "type": "perennial", "ru": "Конский щавель", "en": "horse_sorrel"},

    # Малолетние двудольные (ЭПВ: <=5 не опрыскивать, 6-15 норма, >15 максимум):
    "Щирица запрокинутая": {"class": "A", "type": "annual", "ru": "Щирица запрокинутая", "en": "redroot_pigweed"},
    "Марь белая": {"class": "A", "type": "annual", "ru": "Марь белая", "en": "common_lambsquarters"},
    "Гречишка татарская": {"class": "A", "type": "annual", "ru": "Гречишка татарская", "en": "tartary_buckwheat"},
    "Горцы (виды)": {"class": "A", "type": "annual", "ru": "Горцы (виды)", "en": "smartweed"},
    "Лебеда копьелистная": {"class": "A", "type": "annual", "ru": "Лебеда копьелистная", "en": "spear_saltbush"},
    "Липучка ежевидная_оттопыренная": {"class": "A", "type": "annual", "ru": "Липучка ежевидная", "en": "stickseed"},
    "Мелколепестник канадский": {"class": "A", "type": "annual", "ru": "Мелколепестник канадский", "en": "canadian_fleabane"},
    "Ромашка непахучая": {"class": "A", "type": "annual", "ru": "Ромашка непахучая", "en": "scentless_mayweed"},
    "Аистник цикутовый": {"class": "A", "type": "annual", "ru": "Аистник цикутовый", "en": "redstem_filaree"},
    "Падалица льна": {"class": "A", "type": "annual", "ru": "Падалица льна", "en": "volunteer_flax"},
    "Падалица подсолнечника": {"class": "A", "type": "annual", "ru": "Падалица подсолнечника", "en": "volunteer_sunflower"},

    # КЛАСС B: Злаковые (Узколистные)
    # Многолетники (ЭПВ >= 2 шт/м²):
    "Пырей ползучий": {"class": "B", "type": "perennial", "ru": "Пырей ползучий", "en": "couch_grass"},

    # Малолетние злаковые (ЭПВ: 6-15 норма, >15 максимум):
    "Овсюг обыкновенный": {"class": "B", "type": "annual", "ru": "Овсюг обыкновенный", "en": "wild_oat"},
    "Куриное просо": {"class": "B", "type": "annual", "ru": "Куриное просо", "en": "barnyard_grass"},
    "Щетинник": {"class": "B", "type": "annual", "ru": "Щетинник", "en": "foxtail"},
    "Падалица пшеницы": {"class": "B", "type": "annual", "ru": "Падалица пшеницы", "en": "volunteer_wheat"},
}

# 2. ФАЗЫ РАЗВИТИЯ ПО ШПАРГАЛКЕ «ОЛЖА АГРО»
STAGE_AGRO_MAPPING: Dict[str, Dict[str, str]] = {
    "Всходы": {
        "stage_code": "cotyledon_to_2_leaves",
        "stage_ru": "Семядоли — 2 листа",
        "dosage_recommendation": "Базовая / минимальная дозировка (уязвимая фаза)",
        "dosage_modifier": "1.0",
    },
    "Розетка": {
        "stage_code": "cotyledon_to_2_leaves",
        "stage_ru": "Семядоли — 2 листа (Розетка)",
        "dosage_recommendation": "Базовая / минимальная дозировка",
        "dosage_modifier": "1.0",
    },
    "Стеблевание": {
        "stage_code": "4_to_6_leaves",
        "stage_ru": "4–6 листьев (Стеблевание)",
        "dosage_recommendation": "Увеличить дозировку на 15–20% (сорняк грубеет)",
        "dosage_modifier": "1.18",
    },
    "Цветение": {
        "stage_code": "over_6_leaves_or_flowering",
        "stage_ru": "Более 6 листьев / Цветение",
        "dosage_recommendation": "ПРЕДУПРЕЖДЕНИЕ: Окно упущено! Сорняк устойчив, риск фитотоксичности",
        "dosage_modifier": "0.0",
    },
    "Плодоношение": {
        "stage_code": "over_6_leaves_or_flowering",
        "stage_ru": "Плодоношение",
        "dosage_recommendation": "ПРЕДУПРЕЖДЕНИЕ: Окно упущено! Требуется десикация или механическая обработка",
        "dosage_modifier": "0.0",
    },
}


def build_manifest(output_path: Path = None, seed: int = 42) -> Path:
    """Формирует CSV-манифест по всем файлам в каталоге сорняков."""
    if output_path is None:
        output_path = ROOT_DIR / "case1" / "data" / "manifest_full_10k.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== Сканирование каталога: {WEEDS_DIR} ===")
    if not WEEDS_DIR.exists():
        raise FileNotFoundError(f"Каталог сорняков {WEEDS_DIR} не существует!")

    random.seed(seed)
    records: List[Dict[str, str]] = []

    species_folders = sorted([d for d in WEEDS_DIR.iterdir() if d.is_dir()])
    print(f"Найдено классов сорняков: {len(species_folders)}")

    for sp_dir in species_folders:
        sp_name = sp_dir.name
        tax = BOTANICAL_TAXONOMY.get(sp_name, {
            "class": "A", "type": "annual", "ru": sp_name, "en": sp_name.lower().replace(" ", "_")
        })

        phase_folders = sorted([d for d in sp_dir.iterdir() if d.is_dir()])
        for ph_dir in phase_folders:
            ph_name = ph_dir.name
            agro_info = STAGE_AGRO_MAPPING.get(ph_name, {
                "stage_code": "cotyledon_to_2_leaves",
                "stage_ru": ph_name,
                "dosage_recommendation": "Стандартная дозировка",
                "dosage_modifier": "1.0",
            })

            photos = sorted(list(ph_dir.glob("*.[jJ][pP][gG]")) + list(ph_dir.glob("*.[pP][nN][gG]")))
            for img_p in photos:
                records.append({
                    "path": str(img_p),
                    "filename": img_p.name,
                    "species": tax["en"],
                    "species_ru": tax["ru"],
                    "botanical_class": tax["class"],    # A (двудольные) / B (злаковые)
                    "weed_type": tax["type"],            # annual / perennial
                    "stage_folder": ph_name,
                    "stage_code": agro_info["stage_code"],
                    "stage_ru": agro_info["stage_ru"],
                    "dosage_recommendation": agro_info["dosage_recommendation"],
                    "dosage_modifier": agro_info["dosage_modifier"],
                })

    print(f"Всего проиндексировано снимков: {len(records)}")

    # Стратифицированное разбиение на Train (70%), Val (15%), Test (15%)
    # Группируем по (species, stage_code)
    groups: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for r in records:
        key = (r["species"], r["stage_code"])
        groups.setdefault(key, []).append(r)

    for key, items in groups.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        for i, item in enumerate(items):
            if i < n_train:
                item["split"] = "train"
            elif i < n_train + n_val:
                item["split"] = "val"
            else:
                item["split"] = "test"

    fieldnames = [
        "path", "filename", "species", "species_ru", "botanical_class",
        "weed_type", "stage_folder", "stage_code", "stage_ru",
        "dosage_recommendation", "dosage_modifier", "split"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"[УСПЕХ] Манифест сохранен в: {output_path} ({len(records)} строк)")
    return output_path


if __name__ == "__main__":
    build_manifest()
