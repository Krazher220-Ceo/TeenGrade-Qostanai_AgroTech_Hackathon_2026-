"""
Агрономический интерпретатор правил на основе шпаргалки ментора (Qostanai AgroTech 2026).
Правила загружаются из версионированного файла case1/configs/agronomy_rules.json.
Принцип Human-In-The-Loop:
- Все решения формируются как advisory (консультативные).
- human_confirmation_required ВСЕГДА равен True.
- Никакая рекомендация не транслируется в автоматическую команду распылителю.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

RULES_FILE = Path(__file__).resolve().parents[1] / "configs" / "agronomy_rules.json"

# Полный справочник агрономической классификации «Олжа Агро» (26 сорняков):
# Класс A: Двудольные (Широколистные), Класс B: Злаковые (Узколистные).
# Многолетники — самый опасный и приоритетный сектор. Список содержит и коды
# видов (SPECIES_RU_MAP), и русские названия/синонимы для сопоставления по
# произвольному текстовому полю species/species_ru. Единственный источник
# истины для "что считается многолетником" в проекте — используйте этот
# набор (например, case1/data/perennial_registry.py), не дублируйте список.
PERENNIAL_SPECIES_MARKERS = {
    # Класс A — многолетники:
    "field_thistle", "бодяк полевой", "бодяк",
    "field_bindweed", "вьюнок полевой", "вьюнок",
    "осот полевой", "осот", "perennial_sowthistle",
    "молочай лозный", "молочай", "leafy_spurge",
    "молокан татарский", "молокан", "tatarian_lettuce",
    "полынь горькая", "полынь обыкновенная", "полынь", "wormwood",
    "кермек широколистный", "кермек", "statice",
    "одуванчик лекарственный", "одуванчик", "dandelion",
    "конский щавель", "щавель", "horse_sorrel",
    # Класс B — многолетники:
    "couch_grass", "пырей ползучий", "пырей",
}

# Коды видов (совпадающие с SPECIES_RU_MAP из case1.ml.multitask_model),
# которые относятся к многолетним сорнякам. Подмножество PERENNIAL_SPECIES_MARKERS,
# содержащее только машинные идентификаторы (без русских синонимов).
PERENNIAL_SPECIES_CODES = {
    "field_thistle", "field_bindweed", "perennial_sowthistle", "leafy_spurge",
    "tatarian_lettuce", "wormwood_bitter", "mugwort", "statice", "dandelion",
    "horse_sorrel", "couch_grass",
}


class AgronomyRuleEngine:
    """Движок применения агрономических порогов к данным засорённости."""

    def __init__(self, rules_path: Optional[Path] = None):
        self.rules_path = rules_path or RULES_FILE
        self.rules = self._load_rules()

    def _load_rules(self) -> Dict[str, Any]:
        if not self.rules_path.exists():
            raise FileNotFoundError(f"Файл правил {self.rules_path} не найден")
        with open(self.rules_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @property
    def version(self) -> str:
        return self.rules.get("rules_version", "unknown")

    def evaluate_weed_patch(
        self,
        annual_density_per_m2: float = 0.0,
        perennial_density_per_m2: float = 0.0,
        growth_stage: str = "cotyledon_to_2_leaves",
        model_version_or_hash: Optional[str] = None,
        is_unknown: bool = False,
    ) -> Dict[str, Any]:
        """
        Оценка участка засорённости по агрономической шпаргалке ментора:
        - малолетние <= 5 шт/м² -> слабая засорённость -> не опрыскивать (ниже ЭПВ)
        - малолетние 6–15 шт/м² (5 < d <= 15) -> средняя -> стандартная норма
        - малолетние > 15 шт/м² -> сильная -> повышенная обработка
        - многолетние >= 2 шт/м² -> критическая угроза -> срочная обработка
        - семядоли — 2 листа -> оптимальное технологическое окно
        - 4–6 листьев -> рекомендация об увеличении дозировки на 15–20%
        - более 6 листьев / цветение -> предупреждение о пропущенном окне
        - unknown -> не опрыскивать, только manual_review
        - human_confirmation_required ВСЕГДА равен True
        """
        annual_cfg = self.rules["density_thresholds"]["annual_weeds"]
        perennial_cfg = self.rules["density_thresholds"]["perennial_weeds"]
        stage_cfg = self.rules.get("growth_stages", {})

        explanations = []
        action = "do_not_spray"
        action_ru = "не опрыскивать"
        threat_level = "low"

        if is_unknown:
            threat_level = "review_required"
            action = "manual_review"
            action_ru = "ручная проверка агрономом"
            explanations.append(
                "Вид сорняка не определён достоверно (уверенность < 65%). "
                "Автоматическое опрыскивание запрещено. Требуется ручной осмотр специалистом."
            )
        # 1. Анализ многолетних сорняков (высший приоритет)
        elif perennial_density_per_m2 >= perennial_cfg["critical"]["min_density_per_m2"]:
            threat_level = "critical"
            action = perennial_cfg["critical"]["action"]
            action_ru = perennial_cfg["critical"]["action_ru"]
            crit_thresh = perennial_cfg["critical"]["min_density_per_m2"]
            explanations.append(
                f"Многолетние сорняки ({perennial_density_per_m2:.1f} шт/м² >= {crit_thresh} шт/м²): "
                f"критическая угроза (Класс A/B). Корнеотпрысковые сорняки подавляют культуру, требуется срочная локальная обработка."
            )
        else:
            # 2. Анализ малолетних сорняков
            if annual_density_per_m2 <= annual_cfg["low"]["max_density_per_m2"]:
                threat_level = "low"
                action = annual_cfg["low"]["action"]
                action_ru = annual_cfg["low"]["action_ru"]
                explanations.append(
                    f"Малолетние сорняки ({annual_density_per_m2:.1f} шт/м² <= {annual_cfg['low']['max_density_per_m2']} шт/м²): "
                    f"ниже экономического порога вредоносности. Обработка не рекомендуется для экономии гербицида."
                )
            elif annual_density_per_m2 <= annual_cfg["medium"]["max_density_per_m2"]:
                threat_level = "medium"
                action = annual_cfg["medium"]["action"]
                action_ru = annual_cfg["medium"]["action_ru"]
                explanations.append(
                    f"Малолетние сорняки ({annual_density_per_m2:.1f} шт/м²): средняя засорённость (6–15 шт/м²). "
                    f"Рекомендована стандартная норма расхода зарегистрированного гербицида."
                )
            else:
                threat_level = "high"
                action = annual_cfg["high"]["action"]
                action_ru = annual_cfg["high"]["action_ru"]
                explanations.append(
                    f"Малолетние сорняки ({annual_density_per_m2:.1f} шт/м² > 15 шт/м²): высокая засорённость. "
                    f"Рекомендована повышенная обработка или применение баковых смесей."
                )

        # 3. Анализ фазы вегетации
        stage_info = stage_cfg.get(growth_stage)
        dosage_note = ""
        if stage_info:
            explanations.append(
                f"Фаза сорняков «{stage_info['stage_ru']}»: {stage_info['description']}"
            )
            if "dosage_change_pct" in stage_info:
                dosage_note = f"Рекомендация: {stage_info['dosage_change_pct']} к базовой дозе"
        else:
            stage_info = {
                "stage_ru": growth_stage,
                "window_status": "unknown",
                "window_status_ru": "Фаза требует осмотра",
            }

        return {
            "rules_version": self.version,
            "threat_level": threat_level,
            "recommended_action": action,
            "recommended_action_ru": action_ru,
            "growth_stage_status": stage_info.get("window_status", "unknown"),
            "growth_stage_status_ru": stage_info.get("window_status_ru", ""),
            "dosage_adjustment_note": dosage_note,
            "explanation": " \n".join(explanations),
            "model_version_or_hash": model_version_or_hash or "unspecified",
            "human_confirmation_required": True,
            "auto_spray_enabled": False,
        }

    def evaluate_field_detections(
        self,
        detections: list,
        field_area_m2: float = 12.0,
        dominant_stage: Optional[str] = None,
        model_version_or_hash: Optional[str] = None,
        execution_latency_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Расчёт плотности и агрономического заключения для набора обнаруженных объектов кадра:
        - Исключает культуру (crop_wheat).
        - Разделяет сорняки на многолетние (Класс A: бодяк, вьюнок; Класс B: пырей) и малолетние.
        - Учитывает долю сомнительных детекций (unknown).
        - Оценивает смещение штанги опрыскивателя на скоростях 18 и 20 км/ч.
        """
        area = max(float(field_area_m2), 0.1)

        # Единый справочник многолетников проекта — см. PERENNIAL_SPECIES_MARKERS
        # в начале модуля (используется также case1/data/perennial_registry.py).
        perennial_names = PERENNIAL_SPECIES_MARKERS

        annual_count = 0
        perennial_count = 0
        unknown_count = 0
        crop_count = 0
        stage_votes: Dict[str, int] = {}

        for det in detections:
            sp = str(det.get("species", "") or det.get("top_species_ru", "") or det.get("species_ru", "")).strip().lower()
            stage = str(det.get("stage", "") or det.get("stage_ru", "")).strip().lower()
            rev = det.get("review_required", False)

            if "wheat" in sp or "пшениц" in sp or "культур" in sp:
                crop_count += 1
                continue

            if rev or sp in {"unknown", "uncertain", "неизвестно"}:
                unknown_count += 1
                continue

            if any(p in sp for p in perennial_names):
                perennial_count += 1
            else:
                annual_count += 1

            if stage and stage not in {"unknown", "неизвестно"}:
                stage_votes[stage] = stage_votes.get(stage, 0) + 1

        total_weeds = perennial_count + annual_count
        annual_density = round(annual_count / area, 2)
        perennial_density = round(perennial_count / area, 2)

        # Определение доминирующей фазы вегетации по шпаргалке «Олжа Агро»:
        stage_map = {
            "всходы": "cotyledon_to_2_leaves",
            "розетка": "cotyledon_to_2_leaves",
            "rosette": "cotyledon_to_2_leaves",
            "seedling": "cotyledon_to_2_leaves",
            "cotyledon_to_2_leaves": "cotyledon_to_2_leaves",
            "стеблевание": "4_to_6_leaves",
            "stem_elongation": "4_to_6_leaves",
            "4_to_6_leaves": "4_to_6_leaves",
            "цветение": "over_6_leaves_or_flowering",
            "плодоношение": "over_6_leaves_or_flowering",
            "flowering": "over_6_leaves_or_flowering",
            "fruiting": "over_6_leaves_or_flowering",
            "over_6_leaves_or_flowering": "over_6_leaves_or_flowering",
        }

        if dominant_stage:
            resolved_stage = stage_map.get(str(dominant_stage).lower(), "cotyledon_to_2_leaves")
        elif stage_votes:
            top_stage = max(stage_votes.items(), key=lambda item: item[1])[0]
            resolved_stage = stage_map.get(str(top_stage).lower(), "cotyledon_to_2_leaves")
        else:
            resolved_stage = "cotyledon_to_2_leaves"

        if total_weeds == 0 and unknown_count == 0:
            result = {
                "rules_version": self.version,
                "threat_level": "clean",
                "recommended_action": "do_not_spray",
                "recommended_action_ru": "не опрыскивать",
                "growth_stage_status": "not_applicable",
                "growth_stage_status_ru": "Сорняки отсутствуют",
                "dosage_adjustment_note": "",
                "explanation": "На снимке не обнаружено сорных растений (фон / культура). Опрыскивание не требуется.",
                "model_version_or_hash": model_version_or_hash or "unspecified",
                "human_confirmation_required": True,
                "auto_spray_enabled": False,
            }
        elif total_weeds == 0 and unknown_count > 0:
            result = self.evaluate_weed_patch(
                annual_density_per_m2=0.0,
                perennial_density_per_m2=0.0,
                growth_stage=resolved_stage,
                model_version_or_hash=model_version_or_hash,
                is_unknown=True,
            )
        else:
            result = self.evaluate_weed_patch(
                annual_density_per_m2=annual_density,
                perennial_density_per_m2=perennial_density,
                growth_stage=resolved_stage,
                model_version_or_hash=model_version_or_hash,
                is_unknown=False,
            )

        # Статистика и задержка
        result["field_area_m2"] = round(area, 2)
        result["annual_density_per_m2"] = annual_density
        result["perennial_density_per_m2"] = perennial_density
        result["counts"] = {
            "perennial": perennial_count,
            "annual": annual_count,
            "unknown": unknown_count,
            "crop": crop_count,
            "total_weeds": total_weeds,
        }

        if execution_latency_s is not None and execution_latency_s > 0:
            result["sprayer_displacement"] = {
                "speed_18kmh_m_s": 5.0,
                "displacement_18kmh_m": round(5.0 * execution_latency_s, 3),
                "speed_20kmh_m_s": 5.56,
                "displacement_20kmh_m": round(5.56 * execution_latency_s, 3),
                "latency_s": round(execution_latency_s, 4),
                "recommendation": "Post-flight Advisory Mapping" if execution_latency_s > 0.1 else "Real-time Capable",
            }

        return result
