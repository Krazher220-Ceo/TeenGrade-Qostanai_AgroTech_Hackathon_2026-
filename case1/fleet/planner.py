"""
Главный диспетчер и построитель комплексного плана флота (Fleet Coverage Planner).
Связывает воедино:
- Оптику и геометрию
- Выбор оптимального угла галсов
- Динамическое непрерывное распределение между 1–5 БПЛА
- Валидатор безопасности (Safety Validator)
- Формирование полного отчёта и экспорт в форматы JSON и GeoJSON.
"""

import time
import uuid
from typing import Optional, List, Dict, Any, Tuple
from shapely.geometry import Polygon

from case1.fleet.models import (
    FieldPolygon,
    CameraProfile,
    DroneProfile,
    LandingPad,
    SafetyPolicy,
    FleetPlan,
    MissionState,
)
from case1.fleet.geometry import (
    LocalCoordinateTransformer,
    calculate_optical_parameters,
    build_field_polygon,
    calculate_polygon_center_wgs84,
)
from case1.fleet.coverage_planner import (
    optimize_sweep_angle,
    calculate_coverage_ratio,
    calculate_plan_time_metrics,
)
from case1.fleet.partitioner import partition_fleet_missions
from case1.fleet.safety import SafetyValidator, transition_plan_state
from case1.fleet.demo_field import get_demo_landing_pads


def plan_fleet_coverage(
    field: FieldPolygon,
    num_drones: int = 3,
    altitude_m: float = 30.0,
    ground_speed_m_s: float = 5.0,
    camera: Optional[CameraProfile] = None,
    side_overlap: float = 0.70,
    front_overlap: float = 0.80,
    battery_reserve_pct: float = 20.0,
    pads: Optional[List[LandingPad]] = None,
    safety_policy: Optional[SafetyPolicy] = None,
    preferred_angle_deg: Optional[float] = None,
) -> FleetPlan:
    """
    Основная точка входа для построения миссии 1–5 дронов на одном поле.
    """
    policy = safety_policy or SafetyPolicy(min_battery_reserve_pct=battery_reserve_pct)
    cam = camera or CameraProfile.default_research_camera()

    # 1. Оптические параметры
    opt_params = calculate_optical_parameters(
        altitude_m=altitude_m,
        ground_speed_m_s=ground_speed_m_s,
        camera=cam,
        side_overlap=side_overlap,
        front_overlap=front_overlap,
    )

    # 2. Локальная метрическая проекция
    center_lat, center_lon = calculate_polygon_center_wgs84(field.coordinates)
    transformer = LocalCoordinateTransformer(center_lat, center_lon)

    # 3. Построение полигона поля (с вырезом зон отчуждения)
    poly = build_field_polygon(field, transformer, boundary_buffer_m=policy.boundary_buffer_m)
    field_area_ha = round(poly.area / 10000.0, 2)

    # 4. Профиль дрона
    drone_profile = DroneProfile(
        system_id=1,
        survey_speed_m_s=ground_speed_m_s,
        battery_reserve_pct=battery_reserve_pct,
    )

    # 5. Выбор угла и нарезка галсов
    if preferred_angle_deg is not None:
        from case1.fleet.coverage_planner import generate_lanes_for_angle
        best_angle = preferred_angle_deg % 180.0
        best_lanes = generate_lanes_for_angle(poly, best_angle, opt_params["line_spacing_m"], transformer)
        time_metrics = calculate_plan_time_metrics(
            best_lanes, ground_speed_m_s, drone_profile.turn_speed_m_s, drone_profile.turn_time_penalty_s
        )
    else:
        best_angle, best_lanes, time_metrics = optimize_sweep_angle(
            poly=poly,
            line_spacing_m=opt_params["line_spacing_m"],
            survey_speed_m_s=ground_speed_m_s,
            turn_speed_m_s=drone_profile.turn_speed_m_s,
            turn_penalty_s=drone_profile.turn_time_penalty_s,
            transformer=transformer,
        )

    # 6. Расчёт процента покрытия
    cov_ratio = calculate_coverage_ratio(poly, best_lanes, opt_params["line_spacing_m"])
    covered_area_ha = round(field_area_ha * cov_ratio, 2)
    cov_percentage = round(cov_ratio * 100.0, 1)

    # 7. Подготовка посадочных площадок
    landing_pads = pads if pads else get_demo_landing_pads(field)

    # 8. Разделение галсов между 1..5 дронами
    vehicle_missions, effective_drones, partition_notes = partition_fleet_missions(
        lanes=best_lanes,
        num_drones_requested=num_drones,
        pads=landing_pads,
        drone_profile=drone_profile,
        safety_policy=policy,
    )

    # Временные метрики
    single_drone_time = time_metrics.get("total_time_s", 0.0)
    makespan = max((v.total_estimated_time_s for v in vehicle_missions), default=0.0)
    speedup = round(single_drone_time / makespan, 2) if makespan > 0 else 1.0

    plan_id = f"plan_{int(time.time())}_{uuid.uuid4().hex[:6]}"

    # Формирование начального плана (DRAFT)
    fleet_plan = FleetPlan(
        plan_id=plan_id,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        field_id=field.field_id,
        state=MissionState.DRAFT,
        num_drones_requested=num_drones,
        num_drones_assigned=effective_drones,
        altitude_m=altitude_m,
        ground_speed_m_s=ground_speed_m_s,
        side_overlap=side_overlap,
        front_overlap=front_overlap,
        footprint_width_m=opt_params["footprint_width_m"],
        footprint_height_m=opt_params["footprint_height_m"],
        gsd_x_cm_px=opt_params["gsd_x_cm_px"],
        gsd_y_cm_px=opt_params["gsd_y_cm_px"],
        line_spacing_m=opt_params["line_spacing_m"],
        trigger_distance_m=opt_params["trigger_distance_m"],
        trigger_interval_s=opt_params["trigger_interval_s"],
        sweep_angle_deg=round(best_angle, 1),
        total_lanes_count=len(best_lanes),
        field_area_ha=field_area_ha,
        covered_area_ha=covered_area_ha,
        coverage_percentage=cov_percentage,
        makespan_s=round(makespan, 1),
        theoretical_single_drone_time_s=round(single_drone_time, 1),
        speedup_factor=speedup,
        vehicles=vehicle_missions,
        advisories=partition_notes,
    )

    # 9. Проверка безопасности (Safety Validation)
    validator = SafetyValidator(policy=policy)
    is_safe, violations, warnings, mandatory_advisories = validator.validate_plan(
        plan=fleet_plan,
        field=field,
        pads=landing_pads[:effective_drones],
    )

    fleet_plan.is_safe = is_safe
    fleet_plan.safety_violations = violations
    fleet_plan.safety_warnings = warnings
    fleet_plan.advisories.extend(mandatory_advisories)

    if is_safe:
        fleet_plan.state = MissionState.VALIDATED

    return fleet_plan
