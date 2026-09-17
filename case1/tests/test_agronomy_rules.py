import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from case1.fleet.agronomy_rules import AgronomyRuleEngine


def test_agronomy_rules_thresholds():
    engine = AgronomyRuleEngine()
    assert engine.version == "2026.1-cheatsheet"

    # 1. Малолетние <= 5 шт/м² -> не опрыскивать
    res_low = engine.evaluate_weed_patch(annual_density_per_m2=4.5, perennial_density_per_m2=0.0)
    assert res_low["recommended_action"] == "do_not_spray"
    assert res_low["threat_level"] == "low"
    assert res_low["human_confirmation_required"] is True
    assert res_low["auto_spray_enabled"] is False

    # Граничное значение 5.0
    res_5 = engine.evaluate_weed_patch(annual_density_per_m2=5.0, perennial_density_per_m2=0.0)
    assert res_5["recommended_action"] == "do_not_spray"

    # 2. Малолетние 6–15 шт/м² -> стандартная норма
    res_med = engine.evaluate_weed_patch(annual_density_per_m2=10.0, perennial_density_per_m2=0.0)
    assert res_med["recommended_action"] == "standard_spray"
    assert res_med["threat_level"] == "medium"

    # Граничное значение 15.0
    res_15 = engine.evaluate_weed_patch(annual_density_per_m2=15.0, perennial_density_per_m2=0.0)
    assert res_15["recommended_action"] == "standard_spray"

    # 3. Малолетние > 15 шт/м² -> сильная засорённость
    res_high = engine.evaluate_weed_patch(annual_density_per_m2=18.0, perennial_density_per_m2=0.0)
    assert res_high["recommended_action"] == "increased_spray"
    assert res_high["threat_level"] == "high"

    # 4. Многолетние >= 2 шт/м² -> критическая угроза (перебивает даже слабые малолетние)
    res_perennial = engine.evaluate_weed_patch(
        annual_density_per_m2=2.0, perennial_density_per_m2=2.5
    )
    assert res_perennial["recommended_action"] == "urgent_spray"
    assert res_perennial["threat_level"] == "critical"
    assert "критическая угроза" in res_perennial["explanation"]


def test_growth_stage_recommendations():
    engine = AgronomyRuleEngine()

    # Семядоли - 2 листа (оптимальное окно)
    res_opt = engine.evaluate_weed_patch(
        annual_density_per_m2=8.0, growth_stage="cotyledon_to_2_leaves"
    )
    assert res_opt["growth_stage_status"] == "optimal"

    # 4-6 листьев -> увеличение дозы на 15-20%
    res_4_6 = engine.evaluate_weed_patch(
        annual_density_per_m2=8.0, growth_stage="4_to_6_leaves"
    )
    assert res_4_6["growth_stage_status"] == "acceptable_with_increased_dose"
    assert "+15-20%" in res_4_6["dosage_adjustment_note"]

    # Более 6 листьев / цветение -> предупреждение о пропущенном окне
    res_late = engine.evaluate_weed_patch(
        annual_density_per_m2=8.0, growth_stage="over_6_leaves_or_flowering"
    )
    assert res_late["growth_stage_status"] == "missed_window"
    assert "Пропущено оптимальное окно" in res_late["explanation"]
