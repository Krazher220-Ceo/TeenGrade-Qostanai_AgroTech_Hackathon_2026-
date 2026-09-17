import sys
from pathlib import Path
import pytest
import math
from shapely.geometry import Polygon, Point

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from case1.fleet.models import CameraProfile, FieldPolygon, ExclusionZone
from case1.fleet.geometry import (
    LocalCoordinateTransformer,
    calculate_optical_parameters,
    build_field_polygon,
)
from case1.fleet.coverage_planner import (
    generate_lanes_for_angle,
    calculate_plan_time_metrics,
    calculate_coverage_ratio,
    optimize_sweep_angle,
)


def test_square_field_coverage_metrics():
    """
    Проверка покрытия квадратного поля 500x500 м (25 га):
    Параметры из исследования: line_spacing = 14.4 м, скорость 5 м/с.
    Ожидается ~35 полос, покрытие >= 99%, все точки внутри рабочей зоны.
    """
    center_lat, center_lon = 53.2000, 63.6000
    transformer = LocalCoordinateTransformer(center_lat, center_lon)

    # Поле 500x500 м
    corners_metric = [(-250, -250), (250, -250), (250, 250), (-250, 250)]
    corners_wgs84 = transformer.coords_to_wgs84(corners_metric)
    field = FieldPolygon(field_id="sq_500", name="Square 500m", coordinates=corners_wgs84)

    poly = build_field_polygon(field, transformer)

    line_spacing = 14.4
    lanes = generate_lanes_for_angle(poly, angle_deg=0.0, line_spacing_m=line_spacing, transformer=transformer)

    assert 33 <= len(lanes) <= 36, f"Ожидалось ~35 полос, получено {len(lanes)}"

    # Все начальные и конечные точки должны быть строго внутри полигона поля
    for lane in lanes:
        p_start = Point(lane.start_metric)
        p_end = Point(lane.end_metric)
        # Точки на границе или внутри
        assert poly.buffer(1.0).contains(p_start)
        assert poly.buffer(1.0).contains(p_end)

    # Проверка процента покрытия
    cov_ratio = calculate_coverage_ratio(poly, lanes, line_spacing)
    assert cov_ratio >= 0.99, f"Покрытие {cov_ratio*100:.2f}% меньше требуемых 99%"


def test_angle_optimizer_prefers_long_edge():
    """
    Для вытянутого поля 600x150 м оптимизатор должен выбрать угол вдоль длинной стороны (0°),
    так как это дает в ~4 раза меньше разворотов, чем поперек (90°).
    """
    center_lat, center_lon = 53.2000, 63.6000
    transformer = LocalCoordinateTransformer(center_lat, center_lon)

    # Прямоугольник 600 м по оси X и 150 м по оси Y
    corners_metric = [(-300, -75), (300, -75), (300, 75), (-300, 75)]
    corners_wgs84 = transformer.coords_to_wgs84(corners_metric)
    field = FieldPolygon(field_id="rect_600x150", name="Rect field", coordinates=corners_wgs84)

    poly = build_field_polygon(field, transformer)

    best_angle, best_lanes, best_metrics = optimize_sweep_angle(
        poly=poly,
        line_spacing_m=15.0,
        survey_speed_m_s=5.0,
        turn_speed_m_s=3.0,
        turn_penalty_s=4.0,
        transformer=transformer,
    )

    # Продольный угол 0.0° (или 180°) дает минимум разворотов
    assert best_angle in [0.0, 180.0]

    # Сравнение с поперечным углом 90°
    transverse_lanes = generate_lanes_for_angle(poly, 90.0, 15.0, transformer)
    transverse_metrics = calculate_plan_time_metrics(transverse_lanes, 5.0, 3.0, 4.0)

    assert best_metrics["total_time_s"] < transverse_metrics["total_time_s"]
    assert best_metrics["turns_count"] < transverse_metrics["turns_count"]


def test_coverage_skips_exclusion_zone():
    """
    Проверка того, что рабочие полосы не пересекают внутреннюю зону отчуждения.
    """
    center_lat, center_lon = 53.2000, 63.6000
    transformer = LocalCoordinateTransformer(center_lat, center_lon)

    corners_metric = [(-200, -200), (200, -200), (200, 200), (-200, 200)]
    corners_wgs84 = transformer.coords_to_wgs84(corners_metric)

    # Препятствие по центру 80x80 м
    hole_metric = [(-40, -40), (40, -40), (40, 40), (-40, 40)]
    hole_wgs84 = transformer.coords_to_wgs84(hole_metric)

    field = FieldPolygon(
        field_id="field_with_hole",
        name="Поле с препятствием",
        coordinates=corners_wgs84,
        exclusion_zones=[
            ExclusionZone(zone_id="ez1", name="Озеро", coordinates=hole_wgs84)
        ]
    )

    poly = build_field_polygon(field, transformer)
    hole_poly = Polygon(hole_metric)

    lanes = generate_lanes_for_angle(poly, angle_deg=0.0, line_spacing_m=15.0, transformer=transformer)

    # Ни один рабочий отрезок не должен заходить вглубь препятствия
    for lane in lanes:
        mid_x = (lane.start_metric[0] + lane.end_metric[0]) / 2.0
        mid_y = (lane.start_metric[1] + lane.end_metric[1]) / 2.0
        assert not hole_poly.buffer(-1.0).contains(Point(mid_x, mid_y))
