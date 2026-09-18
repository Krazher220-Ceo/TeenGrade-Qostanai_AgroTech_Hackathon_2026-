import sys
import math
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from case1.fleet.models import (
    DockStation,
    DroneProfile,
    SafetyPolicy,
    FieldPolygon,
    LandingPad,
)
from case1.fleet.demo_field import get_kostanay_demo_field
from case1.fleet.planner import plan_fleet_coverage
from case1.fleet.safety import SafetyValidator
from case1.fleet.dock import (
    dock_station_to_landing_pads,
    split_lanes_into_sorties,
    compute_sortie_data_flow,
    plan_dock_fleet_mission,
    recommend_dock_position,
    scouting_safety_policy,
    sample_transect_lane_ids,
)


def _make_big_field(base_lat: float = 53.20, base_lon: float = 63.60, half_side_m: float = 1000.0) -> FieldPolygon:
    """Синтетическое поле ~2x2 км (~400 га) для проверки многовылетного планирования."""
    lat_step = half_side_m / 111320.0
    lon_step = half_side_m / (111320.0 * math.cos(math.radians(base_lat)))
    coords = [
        (base_lat - lat_step, base_lon - lon_step),
        (base_lat - lat_step, base_lon + lon_step),
        (base_lat + lat_step, base_lon + lon_step),
        (base_lat + lat_step, base_lon - lon_step),
    ]
    return FieldPolygon(field_id="big_400ha_test", name="Тестовое поле 400 га", coordinates=coords)


def test_dock_station_to_landing_pads_spacing_and_uniqueness():
    """Гнёзда станции должны быть уникальны по system_id и разнесены на ~slot_offset_m."""
    dock = DockStation(dock_id="UAZ1", lat=53.2150, lon=63.6250, num_slots=4, slot_offset_m=2.0)
    pads = dock_station_to_landing_pads(dock)

    assert len(pads) == 4
    sys_ids = [p.system_id for p in pads]
    assert sys_ids == [1, 2, 3, 4]
    assert len(set(p.pad_id for p in pads)) == 4

    # Расстояние между соседними гнёздами ~ slot_offset_m (в пределах допуска проекции)
    for i in range(len(pads) - 1):
        d_lat = (pads[i + 1].lat - pads[i].lat) * 111320.0
        d_lon = (pads[i + 1].lon - pads[i].lon) * 111320.0 * math.cos(math.radians(pads[i].lat))
        dist_m = math.hypot(d_lat, d_lon)
        assert dist_m == pytest.approx(dock.slot_offset_m, rel=0.05)


def test_dock_pads_pass_safety_with_relaxed_policy():
    """
    В dock-режиме (require_independent_pads=False) близко расположенные гнёзда
    НЕ должны блокировать план, в отличие от стандартной политики независимых площадок.
    """
    field = get_kostanay_demo_field()
    dock = DockStation(dock_id="UAZ1", lat=53.2150, lon=63.6250, num_slots=3, slot_offset_m=2.0)
    pads = dock_station_to_landing_pads(dock)

    relaxed_policy = SafetyPolicy(require_independent_pads=False, dock_min_slot_separation_m=1.0)
    plan_relaxed = plan_fleet_coverage(field=field, num_drones=3, pads=pads, safety_policy=relaxed_policy)
    assert not any("слишком близко" in v or "практически" in v for v in plan_relaxed.safety_violations)

    strict_policy = SafetyPolicy(require_independent_pads=True)
    validator = SafetyValidator(policy=strict_policy)
    is_safe, violations, _, _ = validator.validate_plan(plan_relaxed, field, pads)
    assert not is_safe
    assert any("слишком близко" in v for v in violations)


def test_plan_dock_fleet_mission_basic_demo_field():
    """Базовый прогон dock-режима на демо-поле: план безопасен, все галсы распределены без потерь/дублей."""
    field = get_kostanay_demo_field()
    dock = DockStation(dock_id="UAZ1", lat=53.2127, lon=63.6250, num_slots=4)

    plan = plan_dock_fleet_mission(field=field, dock=dock, num_drones=4, altitude_m=30.0, ground_speed_m_s=8.0)

    assert plan.is_safe, plan.safety_violations
    assert plan.dock is not None
    assert plan.dock.dock_id == "UAZ1"
    assert plan.data_flow_total is not None
    assert plan.data_flow_total.frames_count > 0
    assert plan.data_flow_total.edge_output_mb < plan.data_flow_total.raw_data_mb

    all_lane_ids = []
    for v in plan.vehicles:
        assert v.num_sorties == len(v.sorties)
        for s in v.sorties:
            all_lane_ids.extend(s.lane_ids)
        # Сумма длин полос по вылетам должна совпадать с флот-планом дрона
        assert sum(s.flight_length_m for s in v.sorties) == pytest.approx(
            sum(l.length_m for l in v.lanes), rel=1e-2
        )

    assert len(all_lane_ids) == plan.total_lanes_count
    assert len(set(all_lane_ids)) == plan.total_lanes_count


def test_plan_dock_fleet_mission_takeoff_staggering_and_altitude_echelons():
    """Каждый дрон должен взлетать со смещением takeoff_interval_s и на своём эшелоне высоты."""
    field = get_kostanay_demo_field()
    dock = DockStation(
        dock_id="UAZ1", lat=53.2127, lon=63.6250, num_slots=4,
        takeoff_interval_s=18.0, base_transit_altitude_m=30.0, transit_altitude_step_m=5.0,
    )
    plan = plan_dock_fleet_mission(field=field, dock=dock, num_drones=4, altitude_m=30.0, ground_speed_m_s=8.0)

    vehicles_sorted = sorted(plan.vehicles, key=lambda v: v.system_id)
    for i, v in enumerate(vehicles_sorted):
        assert v.takeoff_offset_s == pytest.approx(i * dock.takeoff_interval_s, abs=0.5)
        assert v.transit_altitude_m == pytest.approx(dock.base_transit_altitude_m + i * dock.transit_altitude_step_m)

    # Разные эшелоны высоты у разных дронов не совпадают (защита от столкновения при перелёте)
    altitudes = [v.transit_altitude_m for v in vehicles_sorted]
    assert len(set(altitudes)) == len(altitudes)


def test_multi_sortie_split_when_battery_insufficient():
    """
    На большом поле (~400 га) с ограниченным временем полёта дрон не должен облететь
    весь свой сектор за один вылет — миссия обязана разбиться на несколько вылетов
    с зарядкой/заменой батареи между ними, вместо блокировки плана.
    """
    field = _make_big_field(half_side_m=700.0)
    dock = DockStation(dock_id="UAZ1", lat=53.20 - 700.0 / 111320.0, lon=63.60, num_slots=4, charge_time_min=25.0)
    drone = DroneProfile(system_id=1, survey_speed_m_s=8.0, max_flight_time_min=18.0, battery_reserve_pct=20.0)

    plan = plan_dock_fleet_mission(
        field=field, dock=dock, num_drones=4, altitude_m=30.0, ground_speed_m_s=8.0,
        drone_profile=drone, battery_reserve_pct=20.0,
    )

    assert plan.num_cycles > 1
    assert plan.is_safe, plan.safety_violations

    for v in plan.vehicles:
        assert v.num_sorties >= 1
        assert v.battery_safe
        for s in v.sorties:
            assert s.battery_safe
            assert s.battery_consumed_pct <= (100.0 - 20.0) + 1e-6
        # Зарядка идёт после каждого вылета, кроме последнего
        for s in v.sorties[:-1]:
            assert s.charge_time_s > 0.0
        assert v.sorties[-1].charge_time_s == 0.0
        # Makespan с зарядкой должен быть больше суммарного времени полёта (иначе зарядка не учтена)
        assert v.mission_time_with_charging_s >= sum(s.flight_time_s for s in v.sorties)

    # Никакая полоса не теряется и не дублируется по всем дронам/вылетам
    all_lane_ids = [lid for v in plan.vehicles for s in v.sorties for lid in s.lane_ids]
    assert len(all_lane_ids) == plan.total_lanes_count
    assert len(set(all_lane_ids)) == plan.total_lanes_count


def test_battery_swap_mode_uses_swap_time_not_charge_time():
    """Если battery_swap_enabled=True, между вылетами должно использоваться battery_swap_time_min, а не charge_time_min."""
    field = _make_big_field(half_side_m=600.0)
    dock = DockStation(
        dock_id="UAZ1", lat=53.20 - 600.0 / 111320.0, lon=63.60, num_slots=2,
        charge_time_min=30.0, battery_swap_enabled=True, battery_swap_time_min=2.0,
    )
    drone = DroneProfile(system_id=1, survey_speed_m_s=8.0, max_flight_time_min=15.0, battery_reserve_pct=20.0)

    plan = plan_dock_fleet_mission(
        field=field, dock=dock, num_drones=2, altitude_m=30.0, ground_speed_m_s=8.0,
        drone_profile=drone, battery_reserve_pct=20.0,
    )

    swap_seconds = dock.battery_swap_time_min * 60.0
    found_multi_sortie = False
    for v in plan.vehicles:
        for s in v.sorties[:-1]:
            found_multi_sortie = True
            assert s.charge_time_s == pytest.approx(swap_seconds, abs=0.5)

    assert found_multi_sortie, "Тест должен покрывать хотя бы один дрон с >1 вылетом"


def test_compute_sortie_data_flow_reduces_volume():
    """Проверка модели потока данных: edge-детектор должен резко сокращать объём против сырых кадров."""
    field = get_kostanay_demo_field()
    plan = plan_fleet_coverage(field=field, num_drones=1)
    lanes = plan.vehicles[0].lanes

    report = compute_sortie_data_flow(
        lanes=lanes,
        trigger_distance_m=plan.trigger_distance_m,
        uplink_mbps=20.0,
        mb_per_frame_raw=12.0,
        edge_reduction_ratio=0.02,
    )

    assert report.frames_count > 0
    assert report.raw_data_mb == pytest.approx(report.frames_count * 12.0, rel=1e-6)
    assert report.edge_output_mb == pytest.approx(report.raw_data_mb * 0.02, rel=1e-6)
    assert report.edge_uplink_time_s < report.raw_uplink_time_s
    assert report.raw_uplink_time_s == pytest.approx((report.raw_data_mb * 8.0) / 20.0, rel=1e-3)


def test_recommend_dock_position_returns_perimeter_candidates():
    """Рекомендация позиции машины должна перебирать точки НА ГРАНИЦЕ поля и выбирать лучший makespan."""
    field = get_kostanay_demo_field()
    rec = recommend_dock_position(field=field, num_drones=3, altitude_m=30.0, ground_speed_m_s=8.0)

    assert len(rec.candidates) > 0
    assert any("угол поля" in c.label for c in rec.candidates)
    assert any("середина стороны" in c.label for c in rec.candidates)

    # Выбранный кандидат должен иметь минимальный (или равный минимуму) makespan среди всех
    min_makespan = min(c.estimated_makespan_s for c in rec.candidates)
    assert rec.chosen.estimated_makespan_s == pytest.approx(min_makespan, rel=1e-6)
    assert "makespan" in rec.chosen.explanation.lower() or "makespan" in rec.chosen.explanation


def test_scouting_safety_policy_relaxes_overlap_and_pad_separation():
    """
    Дефолтная SafetyPolicy требует перекрытие >=40%/40% (фотограмметрический стандарт).
    Профиль скаутинга должен допускать низкое перекрытие (напр. 0.20/0.20) и близкие
    гнёзда станции, НЕ трогая глобальный дефолт SafetyPolicy().
    """
    field = get_kostanay_demo_field()
    dock = DockStation(dock_id="UAZ1", lat=53.2150, lon=63.6250, num_slots=3, slot_offset_m=2.0)

    default_policy = SafetyPolicy()
    assert default_policy.min_side_overlap == pytest.approx(0.40)
    assert default_policy.min_front_overlap == pytest.approx(0.40)
    assert default_policy.require_independent_pads is True

    policy = scouting_safety_policy(dock, min_side_overlap=0.20, min_front_overlap=0.20)
    assert policy.min_side_overlap == pytest.approx(0.20)
    assert policy.min_front_overlap == pytest.approx(0.20)
    assert policy.require_independent_pads is False

    plan = plan_dock_fleet_mission(
        field=field, dock=dock, num_drones=3, altitude_m=30.0, ground_speed_m_s=8.0,
        side_overlap=0.20, front_overlap=0.20, safety_policy=policy,
    )
    assert plan.is_safe, plan.safety_violations
    assert not any("перекрытие" in v.lower() for v in plan.safety_violations)

    # С дефолтной политикой те же 20% перекрытия должны блокироваться
    strict_plan = plan_dock_fleet_mission(
        field=field, dock=dock, num_drones=3, altitude_m=30.0, ground_speed_m_s=8.0,
        side_overlap=0.20, front_overlap=0.20,
        safety_policy=SafetyPolicy(require_independent_pads=False),
    )
    assert not strict_plan.is_safe
    assert any("перекрытие" in v.lower() for v in strict_plan.safety_violations)


def test_sample_transect_lane_ids_uniform_stride():
    """Прореживание галсов: каждый stride-й индекс из отсортированного списка, детерминированно."""
    ids = list(range(1, 21))  # 1..20
    kept = sample_transect_lane_ids(ids, stride=8)
    assert kept == [1, 9, 17]

    kept_offset = sample_transect_lane_ids(ids, stride=8, offset=3)
    assert kept_offset == [4, 12, 20]

    # stride<=1 не должен ничего менять
    assert sample_transect_lane_ids(ids, stride=1) == ids
    assert sample_transect_lane_ids(ids, stride=0) == ids


def test_plan_dock_fleet_mission_lane_sampling_reduces_coverage():
    """
    Режим выборочных транссект (lane_sample_stride) должен резко снижать число облетаемых
    галсов и площадь покрытия по сравнению со сплошным обзором, без потери/дублирования
    оставленных полос между дронами.
    """
    field = _make_big_field(half_side_m=1000.0)
    dock = DockStation(dock_id="UAZ1", lat=53.20 - 1000.0 / 111320.0, lon=63.60, num_slots=4)
    policy = scouting_safety_policy(dock)
    drone = DroneProfile(system_id=1, survey_speed_m_s=8.0, battery_reserve_pct=20.0)

    full_plan = plan_dock_fleet_mission(
        field=field, dock=dock, num_drones=4, altitude_m=30.0, ground_speed_m_s=8.0,
        side_overlap=0.25, front_overlap=0.25, safety_policy=policy, drone_profile=drone,
    )
    sampled_plan = plan_dock_fleet_mission(
        field=field, dock=dock, num_drones=4, altitude_m=30.0, ground_speed_m_s=8.0,
        side_overlap=0.25, front_overlap=0.25, safety_policy=policy, drone_profile=drone,
        lane_sample_stride=8,
    )

    assert sampled_plan.is_safe, sampled_plan.safety_violations
    assert sampled_plan.total_lanes_count < full_plan.total_lanes_count
    assert sampled_plan.coverage_percentage < full_plan.coverage_percentage
    assert sampled_plan.coverage_percentage < 20.0  # заведомо не сплошной облёт

    all_sampled_ids = [l.lane_id for v in sampled_plan.vehicles for l in v.lanes]
    assert len(all_sampled_ids) == len(set(all_sampled_ids)) == sampled_plan.total_lanes_count

    full_ids = {l.lane_id for v in full_plan.vehicles for l in v.lanes}
    assert set(all_sampled_ids).issubset(full_ids)


def test_recommend_dock_position_handles_tiny_field_gracefully():
    """Если поле не даёт ни одного галса, функция не должна падать, а должна вернуть fallback-кандидата."""
    tiny_field = FieldPolygon(
        field_id="tiny",
        name="Крошечное поле",
        coordinates=[
            (53.2000, 63.6000),
            (53.20001, 63.6000),
            (53.20001, 63.60001),
            (53.2000, 63.60001),
        ],
    )
    rec = recommend_dock_position(field=tiny_field, num_drones=2, altitude_m=110.0, side_overlap=0.4, front_overlap=0.4)
    assert rec.chosen is not None
