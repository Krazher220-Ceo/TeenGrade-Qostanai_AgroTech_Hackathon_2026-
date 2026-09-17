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
    ) -> Dict[str, Any]:
        """
        Оценка участка засорённости:
        - малолетние <= 5 шт/м2 -> слабая засорённость -> не опрыскивать
        - малолетние 6-15 шт/м2 -> средняя -> стандартная норма
        - малолетние > 15 шт/м2 -> сильная -> повышенная обработка
        - многолетние >= 2 шт/м2 -> критическая угроза -> срочная обработка
        - семядоли-2 листа -> оптимальное окно
        - 4-6 листьев -> рекомендация об увеличении дозировки на 15-20%
        - более 6 листьев / цветение -> предупреждение о пропущенном окне
        """
        annual_cfg = self.rules["density_thresholds"]["annual_weeds"]
        perennial_cfg = self.rules["density_thresholds"]["perennial_weeds"]
        stage_cfg = self.rules.get("growth_stages", {})

        explanations = []
        action = "do_not_spray"
        action_ru = "не опрыскивать"
        threat_level = "low"

        # 1. Анализ многолетних сорняков (высший приоритет)
        crit_thresh = perennial_cfg["critical"]["min_density_per_m2"]
        if perennial_density_per_m2 >= crit_thresh:
            threat_level = "critical"
            action = perennial_cfg["critical"]["action"]
            action_ru = perennial_cfg["critical"]["action_ru"]
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
