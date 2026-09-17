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
    calculate_optical_parameters,
    LocalCoordinateTransformer,
    build_field_polygon,
    calculate_polygon_center_wgs84,
    get_principal_angles,
)


def test_optical_parameters_research_match():
    """
    Проверка совпадения формул с официальным числовым примером из drone_fleet_offline_research.md:
    Высота 30 м, сенсор 6.4x4.8 мм, изображение 4000x3000 px, f=4 мм,
    боковое перекрытие 70%, продольное 80%, скорость 5 м/с.
    Ожидается:
      Footprint: 48 x 36 м
      GSD: 1.2 см/px
      Шаг полос: 14.4 м
      Интервал снимков: 1.44 с
    """
    camera = CameraProfile(
        name="Research Camera",
        sensor_width_mm=6.4,
        sensor_height_mm=4.8,
        focal_length_mm=4.0,
        image_width_px=4000,
        image_height_px=3000,
    )

    res = calculate_optical_parameters(
        altitude_m=30.0,
        ground_speed_m_s=5.0,
        camera=camera,
        side_overlap=0.70,
        front_overlap=0.80,
    )

    assert math.isclose(res["footprint_width_m"], 48.0, rel_tol=1e-3)
    assert math.isclose(res["footprint_height_m"], 36.0, rel_tol=1e-3)
    assert math.isclose(res["gsd_x_cm_px"], 1.2, rel_tol=1e-3)
    assert math.isclose(res["gsd_y_cm_px"], 1.2, rel_tol=1e-3)
    assert math.isclose(res["line_spacing_m"], 14.4, rel_tol=1e-3)
    assert math.isclose(res["trigger_distance_m"], 7.2, rel_tol=1e-3)
    assert math.isclose(res["trigger_interval_s"], 1.44, rel_tol=1e-3)


def test_optical_parameters_input_validation():
    camera = CameraProfile.default_research_camera()

    # Недопустимая высота
    with pytest.raises(ValueError):
        calculate_optical_parameters(0.0, 5.0, camera, 0.7, 0.8)

    # Недопустимая скорость
    with pytest.raises(ValueError):
        calculate_optical_parameters(30.0, -1.0, camera, 0.7, 0.8)

    # Недопустимый overlap (>= 1.0)
    with pytest.raises(ValueError):
        calculate_optical_parameters(30.0, 5.0, camera, 1.0, 0.8)

    with pytest.raises(ValueError):
        calculate_optical_parameters(30.0, 5.0, camera, 0.7, -0.1)


def test_coordinate_transformer_roundtrip():
    """Проверка точности прямого и обратного пересчёта координат WGS84 <-> Local Metric."""
    # Координаты Костанайской области (~53.2° N, 63.6° E)
    center_lat, center_lon = 53.2145, 63.6241
    transformer = LocalCoordinateTransformer(center_lat, center_lon)

    # Центр должен быть ровно в (0, 0)
    cx, cy = transformer.wgs84_to_metric(center_lat, center_lon)
    assert math.isclose(cx, 0.0, abs_tol=1e-4)
    assert math.isclose(cy, 0.0, abs_tol=1e-4)

    # Точка на удалении
    test_lat, test_lon = 53.2185, 63.6300
    mx, my = transformer.wgs84_to_metric(test_lat, test_lon)
    back_lat, back_lon = transformer.metric_to_wgs84(mx, my)

    assert math.isclose(back_lat, test_lat, abs_tol=1e-7)
    assert math.isclose(back_lon, test_lon, abs_tol=1e-7)


def test_build_polygon_with_exclusion_zone():
    """Проверка вычитания зоны отчуждения (hole) из полигона поля."""
    center_lat, center_lon = 53.2000, 63.6000
    transformer = LocalCoordinateTransformer(center_lat, center_lon)

    # Поле ~500x500 м
    # В метрах: от -250 до +250
    corners_metric = [(-250, -250), (250, -250), (250, 250), (-250, 250)]
    corners_wgs84 = transformer.coords_to_wgs84(corners_metric)

    # Препятствие в центре поля (100x100 м)
    hole_metric = [(-50, -50), (50, -50), (50, 50), (-50, 50)]
    hole_wgs84 = transformer.coords_to_wgs84(hole_metric)

    field = FieldPolygon(
        field_id="test_field_with_hole",
        name="Поле с препятствием",
        coordinates=corners_wgs84,
        exclusion_zones=[
            ExclusionZone(zone_id="ez1", name="Опора ЛЭП / Болото", coordinates=hole_wgs84)
        ],
    )

    poly = build_field_polygon(field, transformer, boundary_buffer_m=0.0)

    # Площадь: 500*500 - 100*100 = 250 000 - 10 000 = 240 000 м² (24 га)
    assert math.isclose(poly.area, 240000.0, rel_tol=1e-2)

    # Центр поля (0, 0) должен быть внутри выреза, то есть НЕ в полигоне
    assert not poly.contains(Point(0, 0))

    # Точка (150, 150) должна быть внутри полигона
    assert poly.contains(Point(150, 150))


def test_concave_field_polygon():
    """Проверка работы с L-образным (вогнутым) полигоном."""
    center_lat, center_lon = 53.2000, 63.6000
    transformer = LocalCoordinateTransformer(center_lat, center_lon)

    # L-образный контур
    l_shape_metric = [
        (0, 0), (200, 0), (200, 100), (100, 100), (100, 200), (0, 200)
    ]
    l_shape_wgs84 = transformer.coords_to_wgs84(l_shape_metric)

    field = FieldPolygon(
        field_id="l_field",
        name="L-образное поле",
        coordinates=l_shape_wgs84,
    )

    poly = build_field_polygon(field, transformer)
    # Площадь L-образного: 200*100 + 100*100 = 30 000 м²
    assert math.isclose(poly.area, 30000.0, rel_tol=1e-2)
    # Внутренняя выемка (150, 150) не должна принадлежать полигону
    assert not poly.contains(Point(150, 150))
    # Внутри L-контура (50, 50) должна принадлежать
    assert poly.contains(Point(50, 50))


def test_principal_angles():
    """Проверка определения углов сторон ориентированного ограничивающего прямоугольника."""
    poly = Polygon([(0, 0), (300, 0), (300, 100), (0, 100)])
    angles = get_principal_angles(poly)
    assert 0.0 in angles or 90.0 in angles or 180.0 in angles
