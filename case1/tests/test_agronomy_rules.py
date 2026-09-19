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

    # Граничные значения 5.0, 6.0, 15.0, 16.0 из шпаргалки
    res_5 = engine.evaluate_weed_patch(annual_density_per_m2=5.0, perennial_density_per_m2=0.0)
    assert res_5["recommended_action"] == "do_not_spray"

    res_6 = engine.evaluate_weed_patch(annual_density_per_m2=6.0, perennial_density_per_m2=0.0)
    assert res_6["recommended_action"] == "standard_spray"
    assert res_6["threat_level"] == "medium"

    # Граничное значение 15.0
    res_15 = engine.evaluate_weed_patch(annual_density_per_m2=15.0, perennial_density_per_m2=0.0)
    assert res_15["recommended_action"] == "standard_spray"

    # Граничное значение 16.0
    res_16 = engine.evaluate_weed_patch(annual_density_per_m2=16.0, perennial_density_per_m2=0.0)
    assert res_16["recommended_action"] == "increased_spray"
    assert res_16["threat_level"] == "high"

    # 4. Многолетние: 1.9 (мониторинг) vs 2.0 (критическая угроза)
    res_perennial_low = engine.evaluate_weed_patch(
        annual_density_per_m2=2.0, perennial_density_per_m2=1.9
    )
    assert res_perennial_low["threat_level"] == "low"
    assert res_perennial_low["recommended_action"] == "do_not_spray"

    res_perennial_2 = engine.evaluate_weed_patch(
        annual_density_per_m2=2.0, perennial_density_per_m2=2.0
    )
    assert res_perennial_2["recommended_action"] == "urgent_spray"
    assert res_perennial_2["threat_level"] == "critical"
    assert "критическая угроза" in res_perennial_2["explanation"]

    # 5. Unknown статус никогда не даёт команду опрыскивания
    res_unknown = engine.evaluate_weed_patch(is_unknown=True)
    assert res_unknown["recommended_action"] == "manual_review"
    assert res_unknown["auto_spray_enabled"] is False
    assert res_unknown["human_confirmation_required"] is True


def test_unknown_species_growth_stage_is_undetermined_not_optimal():
    """Регресс: при is_unknown=True фаза не может быть достоверно определена,
    поэтому growth_stage_status обязан быть 'undetermined', а не 'optimal'
    (даже если внутренне используется growth_stage по умолчанию
    'cotyledon_to_2_leaves', которому в agronomy_rules.json соответствует
    window_status='optimal' — это значение не должно "просачиваться" наружу
    для неопознанного объекта)."""
    engine = AgronomyRuleEngine()
    res_unknown = engine.evaluate_weed_patch(is_unknown=True)
    assert res_unknown["growth_stage_status"] == "undetermined"
    assert res_unknown["growth_stage_status"] != "optimal"
    assert "не определ" in res_unknown["growth_stage_status_ru"].lower()

    # Тот же инвариант должен соблюдаться и на уровне агрегации кадра, когда
    # все детекции на снимке — неопознанные объекты.
    res_field = engine.evaluate_field_detections(
        detections=[{"species": "unknown", "stage": "unknown", "review_required": True}],
        field_area_m2=1.0,
    )
    assert res_field["growth_stage_status"] == "undetermined"


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


def test_botanical_reference_structure():
    engine = AgronomyRuleEngine()
    bot_ref = engine.rules.get("botanical_reference", {})
    assert "class_A_broadleaf" in bot_ref
    assert "class_B_grass" in bot_ref

    class_a = bot_ref["class_A_broadleaf"]
    class_b = bot_ref["class_B_grass"]
    assert len(class_a["annual_species"]) >= 3
    assert len(class_a["perennial_species"]) >= 3
    assert len(class_b["annual_species"]) >= 3
    assert len(class_b["perennial_species"]) >= 2


def test_evaluate_field_detections_with_crop_exclusion():
    engine = AgronomyRuleEngine()

    # Смешанный кадр: пшеница (культура) + бодяк (многолетний) + вьюнок + сомнительный
    detections = [
        {"species": "crop_wheat", "stage": "rosette", "review_required": False},
        {"species": "crop_wheat", "stage": "rosette", "review_required": False},
        {"species": "field_thistle", "stage": "rosette", "review_required": False},
        {"species": "field_bindweed", "stage": "rosette", "review_required": False},
        {"species": "unknown", "stage": "unknown", "review_required": True},
    ]

    # Площадь кадра 1.0 м² -> 2 многолетних / 1.0 = 2.0 шт/м² -> критическая угроза
    res = engine.evaluate_field_detections(
        detections=detections,
        field_area_m2=1.0,
        dominant_stage="cotyledon_to_2_leaves",
        execution_latency_s=0.25,
    )
    assert res["counts"]["crop"] == 2
    assert res["counts"]["perennial"] == 2
    assert res["counts"]["unknown"] == 1
    assert res["threat_level"] == "critical"
    assert res["recommended_action"] == "urgent_spray"
    assert res["human_confirmation_required"] is True

    # Проверка расчёта смещения штанги опрыскивателя на скоростях 18 и 20 км/ч
    assert "sprayer_displacement" in res
    disp = res["sprayer_displacement"]
    assert disp["displacement_18kmh_m"] == 1.25  # 5.0 * 0.25
    assert disp["displacement_20kmh_m"] == 1.39  # 5.56 * 0.25 = 1.39
    assert disp["recommendation"] == "Post-flight Advisory Mapping"


def test_evaluate_field_detections_clean_field():
    engine = AgronomyRuleEngine()
    # Только культура
    detections = [
        {"species": "crop_wheat", "stage": "rosette", "review_required": False},
        {"species": "crop_wheat", "stage": "stem_elongation", "review_required": False},
    ]
    res = engine.evaluate_field_detections(detections=detections, field_area_m2=10.0)
    assert res["threat_level"] == "clean"
    assert res["recommended_action"] == "do_not_spray"
    assert res["counts"]["total_weeds"] == 0
    assert res["counts"]["crop"] == 2

