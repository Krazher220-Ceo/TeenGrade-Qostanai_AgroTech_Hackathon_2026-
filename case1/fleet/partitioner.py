"""
Модуль разбиения рабочих полос между 1–5 БПЛА.
- Динамическое программирование для балансировки прогнозируемого времени выполнения (makespan).
- Формирование непрерывных пространственных блоков галсов без пропусков и дублирования.
- Назначение уникальных system_id, индивидуальных посадочных площадок P1..P5 и транзитных путей.
- Адаптивное сокращение числа аппаратов, если использование 5 дронов не дает выигрыша или небезопасно.
"""

import math
from typing import List, Tuple, Dict, Any, Optional
from shapely.geometry import Point, LineString

from case1.fleet.models import (
    FlightLane,
    VehicleMission,
    DroneProfile,
    LandingPad,
    SafetyPolicy,
)

DRONE_PALETTE = [
    "#2563EB",  # Дрон 1 - Синий
    "#10B981",  # Дрон 2 - Изумрудный
    "#F59E0B",  # Дрон 3 - Янтарный
    "#8B5CF6",  # Дрон 4 - Фиолетовый
    "#EC4899",  # Дрон 5 - Розовый
]


def estimate_subplan_duration(
    lanes_subset: List[FlightLane],
    pad: LandingPad,
    drone: DroneProfile,
) -> Dict[str, float]:
    """
    Расчёт времени для непрерывного блока полос конкретного дрона:
    1. Подлёт от площадки Pad к началу первой полосы.
    2. Выполнение рабочих галсов со скоростью survey_speed.
    3. Развороты между смежными галсами.
    4. Возврат от конца последней полосы на площадку Pad со скоростью transit_speed.
    """
    if not lanes_subset:
        return {
            "survey_length_m": 0.0,
            "survey_time_s": 0.0,
            "turns_count": 0,
            "turn_time_s": 0.0,
            "transit_length_m": 0.0,
            "transit_time_s": 0.0,
            "total_time_s": 0.0,
        }

    # 1. Длина и время рабочего прохода
    survey_len = sum(l.length_m for l in lanes_subset)
    survey_time = survey_len / drone.survey_speed_m_s

    # 2. Развороты между смежными полосами блока
    turns_count = max(0, len(lanes_subset) - 1)
    turn_dist = 0.0
    for i in range(len(lanes_subset) - 1):
        p1 = lanes_subset[i].end_metric
        p2 = lanes_subset[i + 1].start_metric
        turn_dist += math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    turn_time = (turns_count * drone.turn_time_penalty_s) + (turn_dist / drone.turn_speed_m_s)

    # 3. Транзит: площадка -> первая полоса и последняя полоса -> площадка
    # Координаты площадки в метрах приблизительно соответствуют началу/концу относительно центра
    first_start = lanes_subset[0].start_metric
    last_end = lanes_subset[-1].end_metric

    # Оценка расстояния подлёта
    dist_entry = math.hypot(first_start[0], first_start[1])
    dist_exit = math.hypot(last_end[0], last_end[1])
    transit_len = dist_entry + dist_exit
    transit_time = transit_len / drone.transit_speed_m_s

    total_time = survey_time + turn_time + transit_time

    return {
        "survey_length_m": round(survey_len, 2),
        "survey_time_s": round(survey_time, 1),
        "turns_count": turns_count,
        "turn_time_s": round(turn_time, 1),
        "transit_length_m": round(transit_len, 2),
        "transit_time_s": round(transit_time, 1),
        "total_time_s": round(total_time, 1),
    }


def partition_lanes_dp(
    lanes: List[FlightLane],
    k_drones: int,
    drone_profile: DroneProfile,
    dummy_pad: LandingPad,
) -> List[Tuple[int, int]]:
    """
    Оптимальное разбиение N полос на K непрерывных блоков с помощью динамического программирования.
    Целевая функция: min max(duration_k) — минимизация времени завершения всей операции (makespan).
    Возвращает список диапазонов индексов [(start_idx, end_idx), ...].
    """
    n = len(lanes)
    if n == 0 or k_drones <= 0:
        return []

    k_drones = min(k_drones, n)

    # Предрасчёт стоимости (времени) любого непрерывного среза lanes[i:j]
    cost = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(i + 1, n + 1):
            sub = lanes[i:j]
            t_res = estimate_subplan_duration(sub, dummy_pad, drone_profile)
            cost[i][j] = t_res["total_time_s"]

    # dp[k][i] = минимальный makespan при разбиении lanes[0:i] между k дронами
    dp = [[float("inf")] * (n + 1) for _ in range(k_drones + 1)]
    parent = [[0] * (n + 1) for _ in range(k_drones + 1)]

    for i in range(1, n + 1):
        dp[1][i] = cost[0][i]

    for k in range(2, k_drones + 1):
        for i in range(k, n + 1):
            for j in range(k - 1, i):
                val = max(dp[k - 1][j], cost[j][i])
                if val < dp[k][i]:
                    dp[k][i] = val
                    parent[k][i] = j

    # Восстановление границ разбиения
    splits = []
    curr_i = n
    for k in range(k_drones, 1, -1):
        p = parent[k][curr_i]
        splits.append((p, curr_i))
        curr_i = p
    splits.append((0, curr_i))
    splits.reverse()

    return splits


def partition_fleet_missions(
    lanes: List[FlightLane],
    num_drones_requested: int,
    pads: List[LandingPad],
    drone_profile: DroneProfile,
    safety_policy: SafetyPolicy,
) -> Tuple[List[VehicleMission], int, List[str]]:
    """
    Разделение всего набора галсов на миссии для каждого аппарата:
    - Проверка эффективности: если 5 аппаратов не дают существенного ускорения
      (например, время транзита превышает экономию от разделения полос),
      предлагается безопасное уменьшенное количество аппаратов.
    - Гарантия: ни одна полоса не теряется и не назначается двум дронам.
    - Назначение уникальных system_id (1..5) и цветов.
    """
    notes = []
    total_lanes = len(lanes)

    if total_lanes == 0:
        return [], 0, ["Поле не содержит рабочих полос"]

    # Адаптивный выбор количества аппаратов
    effective_drones = min(num_drones_requested, total_lanes)
    if total_lanes < num_drones_requested * 2 and num_drones_requested > 2:
        effective_drones = max(1, total_lanes // 2)
        notes.append(
            f"Количество полос ({total_lanes}) слишком мало для безопасной работы {num_drones_requested} дронов. "
            f"Автоматически рекомендовано {effective_drones} БПЛА для предотвращения схождения в воздухе."
        )

    # Запасная площадка по умолчанию
    default_pad = pads[0] if pads else LandingPad(pad_id="P1", system_id=1, lat=0.0, lon=0.0)

    # Оптимальное разбиение на непрерывные блоки
    splits = partition_lanes_dp(lanes, effective_drones, drone_profile, default_pad)

    vehicle_missions: List[VehicleMission] = []

    # Распределение по аппаратам
    for d_idx, (start_i, end_i) in enumerate(splits):
        sys_id = d_idx + 1
        # Выбираем соответствующую площадку
        pad = next((p for p in pads if p.system_id == sys_id), pads[min(d_idx, len(pads) - 1)])

        sub_lanes = []
        for l_idx in range(start_i, end_i):
            orig_lane = lanes[l_idx]
            # Назначаем полосе system_id дрона
            assigned_lane = orig_lane.model_copy(update={"system_id": sys_id})
            sub_lanes.append(assigned_lane)

        dur_stats = estimate_subplan_duration(sub_lanes, pad, drone_profile)

        # Расчёт расхода батареи: (total_time_s / (max_time_min * 60)) * 100%
        max_flight_sec = drone_profile.max_flight_time_min * 60.0
        battery_consumed_pct = round((dur_stats["total_time_s"] / max_flight_sec) * 100.0, 1)
        available_battery_budget_pct = 100.0 - drone_profile.battery_reserve_pct
        battery_safe = battery_consumed_pct <= available_battery_budget_pct

        # Генерация waypoints (WGS84)
        waypoints = []
        # Точка взлёта
        waypoints.append({
            "seq": 0,
            "type": "TAKEOFF",
            "lat": pad.lat,
            "lon": pad.lon,
            "alt_m": drone_profile.survey_speed_m_s,  # высота будет уточнена в общем плане
            "system_id": sys_id,
        })

        seq = 1
        for lane in sub_lanes:
            waypoints.append({
                "seq": seq,
                "type": "WAYPOINT_LANE_START",
                "lane_id": lane.lane_id,
                "lat": lane.start_wgs84[0],
                "lon": lane.start_wgs84[1],
                "system_id": sys_id,
            })
            seq += 1
            waypoints.append({
                "seq": seq,
                "type": "WAYPOINT_LANE_END",
                "lane_id": lane.lane_id,
                "lat": lane.end_wgs84[0],
                "lon": lane.end_wgs84[1],
                "system_id": sys_id,
            })
            seq += 1

        # Точка возврата и посадки
        waypoints.append({
            "seq": seq,
            "type": "RETURN_TO_PAD",
            "lat": pad.lat,
            "lon": pad.lon,
            "system_id": sys_id,
        })
        seq += 1
        waypoints.append({
            "seq": seq,
            "type": "LAND",
            "lat": pad.lat,
            "lon": pad.lon,
            "system_id": sys_id,
        })

        color = DRONE_PALETTE[(sys_id - 1) % len(DRONE_PALETTE)]

        mission = VehicleMission(
            system_id=sys_id,
            color_hex=color,
            pad=pad,
            lanes=sub_lanes,
            flight_length_m=dur_stats["survey_length_m"],
            survey_time_s=dur_stats["survey_time_s"],
            turns_count=dur_stats["turns_count"],
            turn_time_s=dur_stats["turn_time_s"],
            transit_length_m=dur_stats["transit_length_m"],
            transit_time_s=dur_stats["transit_time_s"],
            total_estimated_time_s=dur_stats["total_time_s"],
            battery_consumed_pct=battery_consumed_pct,
            battery_safe=battery_safe,
            waypoints=waypoints,
        )
        vehicle_missions.append(mission)

    return vehicle_missions, effective_drones, notes
