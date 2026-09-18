"""
Dock-режим планирования флота: станция на раскладной крыше машины агронома.

Концепция (от ментора): у агронома машина (УАЗ/минивэн) с раскладной крышей.
На крыше — станция (dock) с 3-4 гнёздами для зарядки дронов. Все дроны взлетают
из ОДНОЙ точки (машины), облетают свой сектор поля, возвращаются на станцию.
Станция сама заряжает дроны, забирает с них фото, прогоняет edge-детектор
(1-я модель, в машине) и отправляет через Starlink на сервер только кропы-боксы
+ метаданные (сырые фото — слишком большой объём для Starlink за разумное время;
оригиналы синхронизируются позже на базе).

Модуль обратно совместим с case1.fleet.partitioner / planner: он строится
ПОВЕРХ существующего plan_fleet_coverage() и ничего в нём не меняет.
"""

import math
from typing import List, Optional, Tuple, Dict, Any

from case1.fleet.models import (
    FieldPolygon,
    CameraProfile,
    DroneProfile,
    LandingPad,
    SafetyPolicy,
    FleetPlan,
    VehicleMission,
    FlightLane,
    DockStation,
    Sortie,
    DataFlowReport,
    DockPositionCandidate,
    DockPositionRecommendation,
)
from case1.fleet.geometry import (
    LocalCoordinateTransformer,
    calculate_optical_parameters,
    build_field_polygon,
    calculate_polygon_center_wgs84,
)
from case1.fleet.coverage_planner import optimize_sweep_angle, calculate_coverage_ratio
from case1.fleet.partitioner import partition_lanes_dp
from case1.fleet.planner import plan_fleet_coverage


def scouting_safety_policy(
    dock: DockStation,
    battery_reserve_pct: float = 20.0,
    min_side_overlap: float = 0.15,
    min_front_overlap: float = 0.15,
) -> SafetyPolicy:
    """
    Политика безопасности для РЕЖИМА СКАУТИНГА (фитомониторинг / визуальная разведка сорняков).

    Дефолтная SafetyPolicy() требует перекрытие >=40%/40% — это фотограмметрический
    стандарт для построения бесшовной ортомозаики (нужен для картирования полигонов
    сорняков субметровой точности). Для быстрой разведки (найти очаги, оценить
    заселённость) такое перекрытие избыточно: снимки нужны для детекции объектов,
    а не для сшивки — перекрытие 15-30% достаточно и кратно снижает число кадров.

    НЕ меняет глобальный дефолт SafetyPolicy() — вызывающий код должен явно запросить
    этот профиль. Также включает dock-специфику (гнёзда рядом допустимы).
    """
    return SafetyPolicy(
        min_battery_reserve_pct=battery_reserve_pct,
        min_side_overlap=min_side_overlap,
        min_front_overlap=min_front_overlap,
        require_independent_pads=False,
        dock_min_slot_separation_m=max(0.5, dock.slot_offset_m * 0.5),
    )


def sample_transect_lane_ids(all_lane_ids: List[int], stride: int, offset: int = 0) -> List[int]:
    """
    Выбирает КАЖДУЮ stride-ю полосу из отсортированного списка lane_id — стандартный
    приём выборочного фитомониторинга (обследование трансект вместо сплошного облёта).
    Например, stride=8 -> облетается ~1/8 (~12.5%) площади поля.
    """
    if stride <= 1:
        return list(all_lane_ids)
    sorted_ids = sorted(all_lane_ids)
    return sorted_ids[offset::stride]


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Расстояние по большому кругу между двумя точками WGS84, метры."""
    r = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.asin(min(1.0, math.sqrt(a)))


def dock_station_to_landing_pads(dock: DockStation) -> List[LandingPad]:
    """
    Строит список LandingPad (для совместимости с существующим partitioner/planner)
    из гнёзд станции DockStation. Гнёзда расставляются в ряд с шагом slot_offset_m,
    центрированным на координатах машины (dock.lat, dock.lon).
    """
    pads: List[LandingPad] = []
    n = dock.num_slots
    lat_step = dock.slot_offset_m / 111320.0
    for i in range(n):
        sys_id = i + 1
        offset_index = i - (n - 1) / 2.0
        pad_lat = dock.lat + offset_index * lat_step
        pads.append(
            LandingPad(
                pad_id=f"{dock.dock_id}-S{sys_id}",
                system_id=sys_id,
                lat=round(pad_lat, 8),
                lon=round(dock.lon, 8),
                alt_m=dock.alt_m,
            )
        )
    return pads


def recompute_transit_from_pad(v: VehicleMission, drone_profile: DroneProfile) -> Dict[str, float]:
    """Публичная обёртка над пересчётом транзита площадка<->галс по фактическим координатам площадки."""
    return _recompute_vehicle_transit(v, drone_profile)


def _recompute_vehicle_transit(v: VehicleMission, drone_profile: DroneProfile) -> Dict[str, float]:
    """
    Пересчёт транзита площадка<->галс по РЕАЛЬНЫМ координатам площадки (haversine),
    т.к. estimate_subplan_duration в partitioner.py считает транзит от локального
    начала координат (центроид поля), что не отражает фактическое положение станции.
    """
    if not v.lanes:
        return {"transit_length_m": 0.0, "transit_time_s": 0.0}

    first, last = v.lanes[0], v.lanes[-1]
    dist_entry = _haversine_m(v.pad.lat, v.pad.lon, first.start_wgs84[0], first.start_wgs84[1])
    dist_exit = _haversine_m(last.end_wgs84[0], last.end_wgs84[1], v.pad.lat, v.pad.lon)
    transit_len = dist_entry + dist_exit
    transit_time = transit_len / drone_profile.transit_speed_m_s
    return {"transit_length_m": round(transit_len, 2), "transit_time_s": round(transit_time, 1)}


def _chunk_time_s(chunk: List[FlightLane], pad: LandingPad, drone_profile: DroneProfile) -> float:
    """Оценка полного времени вылета (подлёт + съёмка + развороты + возврат) для набора галсов."""
    if not chunk:
        return 0.0
    survey_time = sum(l.length_m for l in chunk) / drone_profile.survey_speed_m_s
    turns = max(0, len(chunk) - 1)
    turn_dist = 0.0
    for i in range(len(chunk) - 1):
        p1 = chunk[i].end_metric
        p2 = chunk[i + 1].start_metric
        turn_dist += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    turn_time = turns * drone_profile.turn_time_penalty_s + turn_dist / drone_profile.turn_speed_m_s
    entry = _haversine_m(pad.lat, pad.lon, chunk[0].start_wgs84[0], chunk[0].start_wgs84[1])
    exit_ = _haversine_m(chunk[-1].end_wgs84[0], chunk[-1].end_wgs84[1], pad.lat, pad.lon)
    transit_time = (entry + exit_) / drone_profile.transit_speed_m_s
    return survey_time + turn_time + transit_time


def split_lanes_into_sorties(
    lanes: List[FlightLane],
    pad: LandingPad,
    drone_profile: DroneProfile,
    battery_reserve_pct: float,
) -> List[List[FlightLane]]:
    """
    Разбивает непрерывный блок галсов дрона на минимально возможное число вылетов (sorties),
    каждый из которых укладывается в безопасный батарейный бюджет одного заряда.
    Жадный алгоритм: добавляем галсы, пока время вылета не превысит бюджет; тогда
    закрываем текущий вылет и начинаем новый. Если ОДНА полоса сама по себе превышает
    бюджет — она всё равно образует отдельный вылет (полосу дальше делить нельзя).
    """
    if not lanes:
        return []

    max_flight_sec = drone_profile.max_flight_time_min * 60.0
    budget_sec = max_flight_sec * (1.0 - battery_reserve_pct / 100.0)

    chunks: List[List[FlightLane]] = []
    current: List[FlightLane] = []

    for lane in lanes:
        trial = current + [lane]
        if current and _chunk_time_s(trial, pad, drone_profile) > budget_sec:
            chunks.append(current)
            current = [lane]
        else:
            current = trial

    if current:
        chunks.append(current)

    return chunks


def compute_sortie_data_flow(
    lanes: List[FlightLane],
    trigger_distance_m: float,
    uplink_mbps: float,
    mb_per_frame_raw: float = 12.0,
    edge_reduction_ratio: float = 0.02,
) -> DataFlowReport:
    """
    Модель потока данных одного вылета:
    - число кадров = длина галсов / шаг триггера камеры (+1 на галс на "закрывающий" кадр);
    - объём сырых данных = кадры * mb_per_frame_raw;
    - объём после edge-детектора (кропы-боксы) = raw * edge_reduction_ratio;
    - время выгрузки через Starlink-uplink для обоих вариантов.
    """
    if trigger_distance_m <= 0:
        trigger_distance_m = 1.0

    frames = 0
    for lane in lanes:
        frames += max(1, int(lane.length_m / trigger_distance_m) + 1)

    raw_mb = frames * mb_per_frame_raw
    edge_mb = raw_mb * edge_reduction_ratio
    raw_uplink_s = (raw_mb * 8.0) / uplink_mbps
    edge_uplink_s = (edge_mb * 8.0) / uplink_mbps

    return DataFlowReport(
        frames_count=frames,
        raw_data_mb=round(raw_mb, 1),
        edge_output_mb=round(edge_mb, 2),
        raw_uplink_time_s=round(raw_uplink_s, 1),
        edge_uplink_time_s=round(edge_uplink_s, 2),
        mb_per_frame_raw=mb_per_frame_raw,
        edge_reduction_ratio=edge_reduction_ratio,
    )


def _aggregate_data_flow(reports: List[DataFlowReport], uplink_mbps: float) -> DataFlowReport:
    """Суммирование потока данных по всем вылетам всех дронов в единый отчёт по операции."""
    if not reports:
        return DataFlowReport(mb_per_frame_raw=12.0, edge_reduction_ratio=0.02)

    frames = sum(r.frames_count for r in reports)
    raw_mb = sum(r.raw_data_mb for r in reports)
    edge_mb = sum(r.edge_output_mb for r in reports)
    raw_uplink_s = (raw_mb * 8.0) / uplink_mbps
    edge_uplink_s = (edge_mb * 8.0) / uplink_mbps

    return DataFlowReport(
        frames_count=frames,
        raw_data_mb=round(raw_mb, 1),
        edge_output_mb=round(edge_mb, 2),
        raw_uplink_time_s=round(raw_uplink_s, 1),
        edge_uplink_time_s=round(edge_uplink_s, 2),
        mb_per_frame_raw=reports[0].mb_per_frame_raw,
        edge_reduction_ratio=reports[0].edge_reduction_ratio,
    )


def plan_dock_fleet_mission(
    field: FieldPolygon,
    dock: DockStation,
    num_drones: Optional[int] = None,
    altitude_m: float = 30.0,
    ground_speed_m_s: float = 5.0,
    camera: Optional[CameraProfile] = None,
    side_overlap: float = 0.70,
    front_overlap: float = 0.80,
    drone_profile: Optional[DroneProfile] = None,
    safety_policy: Optional[SafetyPolicy] = None,
    battery_reserve_pct: float = 20.0,
    mb_per_frame_raw: float = 12.0,
    edge_reduction_ratio: float = 0.02,
    lane_sample_stride: Optional[int] = None,
    lane_sample_offset: int = 0,
) -> FleetPlan:
    """
    Точка входа для dock-режима: полный план флота (1 машина, N дронов из общих гнёзд),
    с многовылетным планированием (при нехватке батареи на всё поле за один вылет),
    очерёдностью взлёта/посадки, эшелонированием высот и моделью потока данных станции.

    Опирается на существующий plan_fleet_coverage() для геометрии галсов, распределения
    между дронами и базовой безопасности; далее достраивает dock-специфику поверх плана.

    lane_sample_stride: если задан (>1), включает РЕЖИМ ВЫБОРОЧНЫХ ТРАНССЕКТ — из полного
    набора галсов остаётся только каждый stride-й (равномерно по всему полю), остальные
    отбрасываются ДО разбиения на вылеты. Стандартный приём фитомониторинга по выборке
    (не путать со сплошным облётом для ортомозаики).
    """
    n = min(num_drones or dock.num_slots, dock.num_slots)
    pads = dock_station_to_landing_pads(dock)
    cam = camera or CameraProfile.default_research_camera()

    policy = safety_policy or SafetyPolicy(
        min_battery_reserve_pct=battery_reserve_pct,
        require_independent_pads=False,
        dock_min_slot_separation_m=max(0.5, dock.slot_offset_m * 0.5),
    )
    dprofile = drone_profile or DroneProfile(system_id=1, survey_speed_m_s=ground_speed_m_s, battery_reserve_pct=battery_reserve_pct)

    base_plan = plan_fleet_coverage(
        field=field,
        num_drones=n,
        altitude_m=altitude_m,
        ground_speed_m_s=ground_speed_m_s,
        camera=cam,
        side_overlap=side_overlap,
        front_overlap=front_overlap,
        battery_reserve_pct=battery_reserve_pct,
        pads=pads,
        safety_policy=policy,
    )

    if lane_sample_stride and lane_sample_stride > 1:
        all_lane_ids = [l.lane_id for v in base_plan.vehicles for l in v.lanes]
        keep_ids = set(sample_transect_lane_ids(all_lane_ids, lane_sample_stride, lane_sample_offset))

        thinned_vehicles = [
            v.model_copy(update={"lanes": [l for l in v.lanes if l.lane_id in keep_ids]})
            for v in base_plan.vehicles
        ]
        all_kept_lanes = [l for v in thinned_vehicles for l in v.lanes]

        center_lat, center_lon = calculate_polygon_center_wgs84(field.coordinates)
        transformer = LocalCoordinateTransformer(center_lat, center_lon)
        poly = build_field_polygon(field, transformer, boundary_buffer_m=policy.boundary_buffer_m)
        opt_params = calculate_optical_parameters(altitude_m, ground_speed_m_s, cam, side_overlap, front_overlap)
        cov_ratio = calculate_coverage_ratio(poly, all_kept_lanes, opt_params["line_spacing_m"]) if all_kept_lanes else 0.0

        base_plan = base_plan.model_copy(
            update={
                "vehicles": thinned_vehicles,
                "total_lanes_count": len(all_kept_lanes),
                "covered_area_ha": round(base_plan.field_area_ha * cov_ratio, 2),
                "coverage_percentage": round(cov_ratio * 100.0, 1),
                "advisories": base_plan.advisories + [
                    f"Режим выборочных транссект: облётано {len(all_kept_lanes)} из {len(all_lane_ids)} галсов "
                    f"(каждый {lane_sample_stride}-й) — ~{cov_ratio * 100.0:.1f}% площади поля. "
                    f"Это стандарт фитомониторинга по выборке, а не сплошная фотограмметрическая съёмка."
                ],
            }
        )

    updated_vehicles: List[VehicleMission] = []
    all_data_flow_reports: List[DataFlowReport] = []
    max_sorties = 1
    battery_unresolved: List[str] = []

    for v in base_plan.vehicles:
        # 1. Пересчитываем транзит по фактическим координатам станции (не по центроиду поля)
        recomputed = _recompute_vehicle_transit(v, dprofile)

        # 2. Разбиваем блок галсов дрона на вылеты, укладывающиеся в батарейный бюджет
        lane_chunks = split_lanes_into_sorties(v.lanes, v.pad, dprofile, battery_reserve_pct)
        max_sorties = max(max_sorties, len(lane_chunks))

        turnaround_s = dock.turnaround_time_min() * 60.0
        takeoff_offset_s = (v.system_id - 1) * dock.takeoff_interval_s
        transit_altitude_m = dock.base_transit_altitude_m + (v.system_id - 1) * dock.transit_altitude_step_m
        landing_offset_s = (n - v.system_id) * dock.takeoff_interval_s

        sorties: List[Sortie] = []
        cumulative_time_s = 0.0
        max_flight_sec = dprofile.max_flight_time_min * 60.0

        for idx, chunk in enumerate(lane_chunks, start=1):
            survey_time = sum(l.length_m for l in chunk) / dprofile.survey_speed_m_s
            turns = max(0, len(chunk) - 1)
            turn_dist = 0.0
            for i in range(len(chunk) - 1):
                p1, p2 = chunk[i].end_metric, chunk[i + 1].start_metric
                turn_dist += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            turn_time = turns * dprofile.turn_time_penalty_s + turn_dist / dprofile.turn_speed_m_s

            entry = _haversine_m(v.pad.lat, v.pad.lon, chunk[0].start_wgs84[0], chunk[0].start_wgs84[1])
            exit_ = _haversine_m(chunk[-1].end_wgs84[0], chunk[-1].end_wgs84[1], v.pad.lat, v.pad.lon)
            transit_len = entry + exit_
            transit_time = transit_len / dprofile.transit_speed_m_s

            flight_time_s = survey_time + turn_time + transit_time
            battery_pct = round((flight_time_s / max_flight_sec) * 100.0, 1)
            is_last = idx == len(lane_chunks)
            charge_time_s = 0.0 if is_last else turnaround_s

            data_flow = compute_sortie_data_flow(
                chunk, base_plan.trigger_distance_m, dock.uplink_mbps, mb_per_frame_raw, edge_reduction_ratio
            )
            all_data_flow_reports.append(data_flow)

            sortie_takeoff_offset = takeoff_offset_s + cumulative_time_s
            sorties.append(
                Sortie(
                    sortie_index=idx,
                    system_id=v.system_id,
                    lane_ids=[l.lane_id for l in chunk],
                    flight_length_m=round(sum(l.length_m for l in chunk), 2),
                    survey_time_s=round(survey_time, 1),
                    turns_count=turns,
                    turn_time_s=round(turn_time, 1),
                    transit_length_m=round(transit_len, 2),
                    transit_time_s=round(transit_time, 1),
                    flight_time_s=round(flight_time_s, 1),
                    battery_consumed_pct=battery_pct,
                    battery_safe=battery_pct <= (100.0 - battery_reserve_pct),
                    charge_time_s=round(charge_time_s, 1),
                    takeoff_offset_s=round(sortie_takeoff_offset, 1),
                    transit_altitude_m=transit_altitude_m,
                    frames_count=data_flow.frames_count,
                    raw_data_mb=data_flow.raw_data_mb,
                    edge_output_mb=data_flow.edge_output_mb,
                    raw_uplink_time_s=data_flow.raw_uplink_time_s,
                    edge_uplink_time_s=data_flow.edge_uplink_time_s,
                )
            )
            cumulative_time_s += flight_time_s + charge_time_s

            if not sorties[-1].battery_safe:
                battery_unresolved.append(
                    f"Дрон #{v.system_id}, вылет #{idx}: одна полоса/группа полос ({battery_pct:.1f}% батареи) "
                    f"не помещается в безопасный бюджет одного вылета даже без дальнейшего дробления."
                )

        this_vehicle_reports = [
            DataFlowReport(
                frames_count=s.frames_count,
                raw_data_mb=s.raw_data_mb,
                edge_output_mb=s.edge_output_mb,
                raw_uplink_time_s=s.raw_uplink_time_s,
                edge_uplink_time_s=s.edge_uplink_time_s,
                mb_per_frame_raw=mb_per_frame_raw,
                edge_reduction_ratio=edge_reduction_ratio,
            )
            for s in sorties
        ]
        vehicle_data_flow = _aggregate_data_flow(this_vehicle_reports, dock.uplink_mbps)

        mission_time_with_charging_s = takeoff_offset_s + cumulative_time_s

        updated_vehicles.append(
            v.model_copy(
                update={
                    "transit_length_m": recomputed["transit_length_m"],
                    "transit_time_s": recomputed["transit_time_s"],
                    "total_estimated_time_s": round(sum(s.flight_time_s for s in sorties), 1),
                    "battery_consumed_pct": max((s.battery_consumed_pct for s in sorties), default=0.0),
                    "battery_safe": all(s.battery_safe for s in sorties),
                    "num_sorties": len(sorties),
                    "sorties": sorties,
                    "mission_time_with_charging_s": round(mission_time_with_charging_s, 1),
                    "takeoff_offset_s": round(takeoff_offset_s, 1),
                    "transit_altitude_m": transit_altitude_m,
                    "landing_offset_s": round(landing_offset_s, 1),
                    "data_flow": vehicle_data_flow,
                }
            )
        )

    # Пересчёт makespan с учётом зарядки между вылетами
    makespan_with_charging = max(
        (v.mission_time_with_charging_s for v in updated_vehicles), default=base_plan.makespan_s
    )

    # Многовылетное планирование разрешает батарейные нарушения из базового (однократного) плана —
    # убираем соответствующие блокировки и заменяем их пояснением про dock-режим.
    remaining_violations = [
        v for v in base_plan.safety_violations if "расчётное потребление" not in v
    ]
    advisories = list(base_plan.advisories)
    if max_sorties > 1:
        advisories.append(
            f"Dock-режим: миссия разбита на до {max_sorties} вылетов на дрон с зарядкой/заменой батареи "
            f"между ними ({dock.turnaround_time_min():.0f} мин на цикл)."
        )
    advisories.extend(battery_unresolved)

    data_flow_total = _aggregate_data_flow(all_data_flow_reports, dock.uplink_mbps)

    updated_plan = base_plan.model_copy(
        update={
            "vehicles": updated_vehicles,
            "dock": dock,
            "data_flow_total": data_flow_total,
            "makespan_with_charging_s": round(makespan_with_charging, 1),
            "makespan_s": round(makespan_with_charging, 1),
            "num_cycles": max_sorties,
            "safety_violations": remaining_violations,
            "is_safe": len(remaining_violations) == 0,
            "advisories": advisories,
        }
    )

    return updated_plan


def recommend_dock_position(
    field: FieldPolygon,
    num_drones: int = 4,
    altitude_m: float = 30.0,
    ground_speed_m_s: float = 5.0,
    camera: Optional[CameraProfile] = None,
    side_overlap: float = 0.70,
    front_overlap: float = 0.80,
    drone_profile: Optional[DroneProfile] = None,
) -> DockPositionRecommendation:
    """
    Перебирает кандидатов на позицию машины ПО ПЕРИМЕТРУ поля (машина стоит на границе
    поля/дороге — не заезжает внутрь) и выбирает точку с минимальным оценочным makespan
    (максимальный по дронам транзит площадка<->галсы). Возвращает выбранную точку,
    все кандидаты и текстовое объяснение выбора.
    """
    cam = camera or CameraProfile.default_research_camera()
    dprofile = drone_profile or DroneProfile(system_id=1, survey_speed_m_s=ground_speed_m_s)

    opt_params = calculate_optical_parameters(altitude_m, ground_speed_m_s, cam, side_overlap, front_overlap)
    center_lat, center_lon = calculate_polygon_center_wgs84(field.coordinates)
    transformer = LocalCoordinateTransformer(center_lat, center_lon)
    poly = build_field_polygon(field, transformer)

    _, best_lanes, _ = optimize_sweep_angle(
        poly=poly,
        line_spacing_m=opt_params["line_spacing_m"],
        survey_speed_m_s=ground_speed_m_s,
        turn_speed_m_s=dprofile.turn_speed_m_s,
        turn_penalty_s=dprofile.turn_time_penalty_s,
        transformer=transformer,
    )

    if not best_lanes:
        fallback = DockPositionCandidate(
            label="поле не содержит галсов", lat=center_lat, lon=center_lon,
            avg_transit_m=0.0, max_transit_m=0.0, estimated_makespan_s=0.0,
            explanation="Недостаточно данных для рекомендации позиции — поле слишком мало для галсов.",
        )
        return DockPositionRecommendation(chosen=fallback, candidates=[fallback])

    dummy_pad = LandingPad(pad_id="TMP", system_id=1, lat=center_lat, lon=center_lon)
    k = min(num_drones, len(best_lanes))
    splits = partition_lanes_dp(best_lanes, k, dprofile, dummy_pad)
    lane_groups = [best_lanes[s:e] for s, e in splits if e > s]

    # Кандидаты по периметру поля: углы (вершины) и середины сторон
    exterior = field.coordinates
    n_vertices = len(exterior)
    candidates_metric: List[Tuple[str, float, float]] = []
    for i in range(n_vertices):
        lat1, lon1 = exterior[i]
        lat2, lon2 = exterior[(i + 1) % n_vertices]
        x1, y1 = transformer.wgs84_to_metric(lat1, lon1)
        x2, y2 = transformer.wgs84_to_metric(lat2, lon2)
        candidates_metric.append((f"угол поля #{i + 1}", x1, y1))
        candidates_metric.append((f"середина стороны #{i + 1}", (x1 + x2) / 2.0, (y1 + y2) / 2.0))

    results: List[DockPositionCandidate] = []
    for label, mx, my in candidates_metric:
        lat_c, lon_c = transformer.metric_to_wgs84(mx, my)
        max_transit_m = 0.0
        total_transit_m = 0.0
        worst_time_s = 0.0
        groups_counted = 0

        for lanes_subset in lane_groups:
            if not lanes_subset:
                continue
            groups_counted += 1
            first, last = lanes_subset[0], lanes_subset[-1]
            entry = _haversine_m(lat_c, lon_c, first.start_wgs84[0], first.start_wgs84[1])
            exit_ = _haversine_m(last.end_wgs84[0], last.end_wgs84[1], lat_c, lon_c)
            transit_m = entry + exit_

            survey_time = sum(l.length_m for l in lanes_subset) / ground_speed_m_s
            turns = max(0, len(lanes_subset) - 1)
            turn_dist = sum(
                math.hypot(
                    lanes_subset[i + 1].start_metric[0] - lanes_subset[i].end_metric[0],
                    lanes_subset[i + 1].start_metric[1] - lanes_subset[i].end_metric[1],
                )
                for i in range(len(lanes_subset) - 1)
            )
            turn_time = turns * dprofile.turn_time_penalty_s + turn_dist / dprofile.turn_speed_m_s
            transit_time = transit_m / dprofile.transit_speed_m_s
            total_time = survey_time + turn_time + transit_time

            max_transit_m = max(max_transit_m, transit_m)
            total_transit_m += transit_m
            worst_time_s = max(worst_time_s, total_time)

        avg_transit_m = total_transit_m / max(1, groups_counted)
        results.append(
            DockPositionCandidate(
                label=label,
                lat=round(lat_c, 7),
                lon=round(lon_c, 7),
                avg_transit_m=round(avg_transit_m, 1),
                max_transit_m=round(max_transit_m, 1),
                estimated_makespan_s=round(worst_time_s, 1),
                explanation=(
                    f"{label}: средний транзит {avg_transit_m:.0f} м, макс. {max_transit_m:.0f} м, "
                    f"оценка makespan {worst_time_s:.0f} с"
                ),
            )
        )

    results.sort(key=lambda r: r.estimated_makespan_s)
    chosen = results[0]
    worst = results[-1]
    delta_s = worst.estimated_makespan_s - chosen.estimated_makespan_s
    chosen = chosen.model_copy(
        update={
            "explanation": (
                chosen.explanation
                + f". Для сравнения: «{worst.label}» дал бы makespan {worst.estimated_makespan_s:.0f} с "
                f"(хуже на {delta_s:.0f} с)."
            )
        }
    )

    return DockPositionRecommendation(chosen=chosen, candidates=results)
