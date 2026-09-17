import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from case1.fleet.models import (
    LandingPad,
    SafetyPolicy,
    MissionState,
)
from case1.fleet.demo_field import get_kostanay_demo_field, get_demo_landing_pads
from case1.fleet.planner import plan_fleet_coverage
from case1.fleet.safety import SafetyValidator, transition_plan_state


def test_safety_rejects_identical_landing_pad():
    """
    Одновременная посадка нескольких аппаратов в одну точку (или ближе 5 м)
    должна вызывать блокирующую ошибку безопасности.
    """
    field = get_kostanay_demo_field()
    # Создаём две площадки в одной точке
    colliding_pads = [
        LandingPad(pad_id="P1", system_id=1, lat=53.2100, lon=63.6200),
        LandingPad(pad_id="P2", system_id=2, lat=53.2100, lon=63.6200),  # Та же точка!
    ]

    plan = plan_fleet_coverage(
        field=field,
        num_drones=2,
        pads=colliding_pads,
    )

    assert not plan.is_safe
    assert any("слишком близко" in v or "одну точку" in v for v in plan.safety_violations)
    assert plan.state == MissionState.DRAFT


def test_safety_rejects_duplicate_system_ids():
    """Повторяющиеся MAVLink System ID должны быть заблокированы."""
    field = get_kostanay_demo_field()
    plan = plan_fleet_coverage(field=field, num_drones=2)

    # Искусственно дублируем system_id
    plan.vehicles[1] = plan.vehicles[1].model_copy(update={"system_id": plan.vehicles[0].system_id})

    validator = SafetyValidator()
    is_safe, violations, _, _ = validator.validate_plan(plan, field, [v.pad for v in plan.vehicles])

    assert not is_safe
    assert any("повторяющиеся MAVLink System ID" in v for v in violations)


def test_safety_rejects_extreme_altitude():
    """Высота вне разрешённого диапазона [10, 120] м блокируется."""
    field = get_kostanay_demo_field()

    # Слишком низко (< 10 м)
    plan_low = plan_fleet_coverage(field=field, altitude_m=5.0)
    assert not plan_low.is_safe
    assert any("ниже безопасного минимума" in v for v in plan_low.safety_violations)

    # Слишком высоко (> 120 м)
    plan_high = plan_fleet_coverage(field=field, altitude_m=150.0)
    assert not plan_high.is_safe
    assert any("превышает допустимый предел" in v for v in plan_high.safety_violations)


def test_state_machine_transition():
    """Нельзя перевести небезопасный план в статус SIMULATION_READY."""
    field = get_kostanay_demo_field()
    plan = plan_fleet_coverage(field=field, altitude_m=5.0)  # Ошибочная высота
    assert not plan.is_safe

    with pytest.raises(ValueError, match="Невозможно перевести план"):
        transition_plan_state(
            plan=plan,
            target_state=MissionState.SIMULATION_READY,
            field=field,
            pads=[v.pad for v in plan.vehicles],
        )


def test_mandatory_advisories_present():
    """Все обязательные по заданию плашки и дисклеймеры должны присутствовать в плане."""
    field = get_kostanay_demo_field()
    plan = plan_fleet_coverage(field=field, num_drones=3)

    adv_text = " ".join(plan.advisories)
    assert "Демонстрационная симуляция" in adv_text
    assert "Не является разрешением на реальный групповой полёт" in adv_text
    assert "Рекомендация требует обязательного подтверждения агрономом" in adv_text
