"""
Генератор покрытия поля параллельными галсами (Boustrophedon / Lawnmower).
- Построение рабочих полос с учётом сложной формы поля и внутренних зон отчуждения.
- Автоматический перебор и выбор оптимального угла с минимизацией общего времени съёмки и разворотов.
- Расчёт процента покрытия (synthetic swath coverage >= 99%).
"""

import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from shapely.geometry import (
    Polygon,
    MultiPolygon,
    LineString,
    MultiLineString,
    Point,
    box,
)
from shapely.affinity import rotate
from shapely.ops import unary_union

from case1.fleet.models import (
    CameraProfile,
    DroneProfile,
    FlightLane,
    FieldPolygon,
)
from case1.fleet.geometry import (
    LocalCoordinateTransformer,
    calculate_optical_parameters,
    build_field_polygon,
    get_principal_angles,
)


def generate_lanes_for_angle(
    poly: Polygon,
    angle_deg: float,
    line_spacing_m: float,
    transformer: LocalCoordinateTransformer,
    min_lane_length_m: float = 5.0,
) -> List[FlightLane]:
    """
    Построение параллельных полос (галсов) под заданным углом angle_deg:
    1. Поворот полигона на -angle_deg.
    2. Нарезка горизонтальными лучами с шагом line_spacing_m.
    3. Пересечение лучей с полигоном (автоматически отсекает выемки и запретные зоны).
    4. Обратный поворот отрезков на +angle_deg.
    5. Сортировка и чередование направлений (Boustrophedon snake pattern).
    """
    if poly.is_empty or line_spacing_m <= 0:
        return []

    centroid = poly.centroid
    cx, cy = centroid.x, centroid.y

    # Поворачиваем полигон так, чтобы галсы стали горизонтальными
    rotated_poly = rotate(poly, -angle_deg, origin=(cx, cy))
    minx, miny, maxx, maxy = rotated_poly.bounds

    # Генерируем y-координаты горизонтальных линий
    # Начинаем с отступа в половину шага для равномерного покрытия краёв
    y_coords = []
    curr_y = miny + line_spacing_m * 0.5
    while curr_y < maxy:
        y_coords.append(curr_y)
        curr_y += line_spacing_m

    if not y_coords:
        # Если полигон меньше line_spacing, проводим хотя бы одну осевую линию
        y_coords.append((miny + maxy) / 2.0)

    raw_segments: List[Tuple[float, float, float, float, float]] = []

    # Расширенный отрезок для надёжного пересечения
    x_pad = (maxx - minx) * 0.2 + 10.0
    line_x1 = minx - x_pad
    line_x2 = maxx + x_pad

    for y in y_coords:
        scan_line = LineString([(line_x1, y), (line_x2, y)])
        intersection = rotated_poly.intersection(scan_line)

        if intersection.is_empty:
            continue

        lines = []
        if intersection.geom_type == "LineString":
            lines = [intersection]
        elif intersection.geom_type == "MultiLineString":
            lines = list(intersection.geoms)
        elif intersection.geom_type == "GeometryCollection":
            lines = [g for g in intersection.geoms if g.geom_type == "LineString"]

        for l in lines:
            if l.length >= min_lane_length_m:
                coords = list(l.coords)
                sx, sy = coords[0]
                ex, ey = coords[-1]
                raw_segments.append((sx, sy, ex, ey, y))

    if not raw_segments:
        return []

    # Сортируем полосы по y (снизу вверх)
    raw_segments.sort(key=lambda s: s[4])

    lanes: List[FlightLane] = []
    # Чередуем направление (змейка: слева-направо, справа-налево)
    for idx, seg in enumerate(raw_segments):
        sx, sy, ex, ey, _ = seg
        if idx % 2 == 1:
            # Разворачиваем направление для змейки
            sx, sy, ex, ey = ex, ey, sx, sy

        seg_geom = LineString([(sx, sy), (ex, ey)])
        # Поворачиваем обратно на +angle_deg
        restored = rotate(seg_geom, angle_deg, origin=(cx, cy))
        r_coords = list(restored.coords)
        rx1, ry1 = r_coords[0]
        rx2, ry2 = r_coords[-1]

        # Переводим в WGS84
        wgs_start = transformer.metric_to_wgs84(rx1, ry1)
        wgs_end = transformer.metric_to_wgs84(rx2, ry2)

        length_m = float(restored.length)
        direction_deg = (math.degrees(math.atan2(ry2 - ry1, rx2 - rx1)) + 360.0) % 360.0

        lanes.append(
            FlightLane(
                lane_id=idx + 1,
                start_metric=(round(rx1, 2), round(ry1, 2)),
                end_metric=(round(rx2, 2), round(ry2, 2)),
                start_wgs84=(round(wgs_start[0], 7), round(wgs_start[1], 7)),
                end_wgs84=(round(wgs_end[0], 7), round(wgs_end[1], 7)),
                length_m=round(length_m, 2),
                direction_deg=round(direction_deg, 1),
            )
        )

    return lanes


def calculate_plan_time_metrics(
    lanes: List[FlightLane],
    survey_speed_m_s: float,
    turn_speed_m_s: float,
    turn_penalty_s: float,
) -> Dict[str, float]:
    """
    Расчёт длины и времени обследования:
    - Длина рабочих проходов
    - Длина и время межполосных переходов/разворотов
    - Число разворотов
    - Общее рабочее время
    """
    if not lanes:
        return {
            "total_survey_length_m": 0.0,
            "total_turn_length_m": 0.0,
            "turns_count": 0,
            "survey_time_s": 0.0,
            "turn_time_s": 0.0,
            "total_time_s": 0.0,
        }

    total_survey_length = sum(lane.length_m for lane in lanes)
    survey_time = total_survey_length / survey_speed_m_s

    total_turn_length = 0.0
    turns_count = len(lanes) - 1

    for i in range(len(lanes) - 1):
        prev_end = lanes[i].end_metric
        next_start = lanes[i + 1].start_metric
        dist = math.hypot(next_start[0] - prev_end[0], next_start[1] - prev_end[1])
        total_turn_length += dist

    turn_time = (turns_count * turn_penalty_s) + (total_turn_length / turn_speed_m_s)
    total_time = survey_time + turn_time

    return {
        "total_survey_length_m": round(total_survey_length, 2),
        "total_turn_length_m": round(total_turn_length, 2),
        "turns_count": turns_count,
        "survey_time_s": round(survey_time, 1),
        "turn_time_s": round(turn_time, 1),
        "total_time_s": round(total_time, 1),
    }


def calculate_coverage_ratio(
    poly: Polygon,
    lanes: List[FlightLane],
    line_spacing_m: float,
) -> float:
    """
    Расчёт коэффициента покрытия площади поля рабочими полосами:
    Буферизация каждой полосы на половину шага line_spacing_m и объединение.
    Возвращает долю [0.0, 1.0] покрытой площади поля.
    """
    if poly.is_empty or not lanes:
        return 0.0

    swath_half_width = line_spacing_m * 0.55  # С учётом минимального запаса перекрытия
    swaths = []

    for lane in lanes:
        line = LineString([lane.start_metric, lane.end_metric])
        # Плоский буфер полосы (прямоугольная полоса захвата)
        swaths.append(line.buffer(swath_half_width, cap_style="flat"))

    if not swaths:
        return 0.0

    merged_swaths = unary_union(swaths)
    intersection = poly.intersection(merged_swaths)

    coverage_ratio = intersection.area / poly.area
    return min(1.0, max(0.0, float(coverage_ratio)))


def optimize_sweep_angle(
    poly: Polygon,
    line_spacing_m: float,
    survey_speed_m_s: float,
    turn_speed_m_s: float,
    turn_penalty_s: float,
    transformer: LocalCoordinateTransformer,
) -> Tuple[float, List[FlightLane], Dict[str, Any]]:
    """
    Поиск оптимального угла рабочих проходов:
    Перебираются углы длинных сторон полигона, а также сетка углов с шагом 15°.
    Критерий выбора: минимальное общее время обследования (учёт длины и разворотов).
    """
    principal = get_principal_angles(poly)
    grid_angles = list(range(0, 180, 15))
    all_candidate_angles = sorted(list(set([float(a) for a in principal + grid_angles])))

    best_angle = 0.0
    best_lanes: List[FlightLane] = []
    best_time = float("inf")
    best_metrics: Dict[str, Any] = {}

    for angle in all_candidate_angles:
        lanes = generate_lanes_for_angle(poly, angle, line_spacing_m, transformer)
        if not lanes:
            continue

        metrics = calculate_plan_time_metrics(
            lanes, survey_speed_m_s, turn_speed_m_s, turn_penalty_s
        )
        total_time = metrics["total_time_s"]

        if total_time < best_time:
            best_time = total_time
            best_angle = angle
            best_lanes = lanes
            best_metrics = metrics

    if not best_lanes and all_candidate_angles:
        best_angle = 0.0
        best_lanes = generate_lanes_for_angle(poly, 0.0, line_spacing_m, transformer)
        best_metrics = calculate_plan_time_metrics(
            best_lanes, survey_speed_m_s, turn_speed_m_s, turn_penalty_s
        )

    return best_angle, best_lanes, best_metrics
