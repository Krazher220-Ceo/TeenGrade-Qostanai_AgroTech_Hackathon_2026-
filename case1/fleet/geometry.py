"""
Геометрическое ядро планировщика покрытия БПЛА.
- Преобразование WGS84 (lat/lon) <-> Local Metric (ENU / Azimuthal Equidistant в метрах) через PyProj.
- Расчёт оптических параметров (footprint, GSD, line_spacing, trigger_interval).
- Операции над полигонами полей и зонами отчуждения через Shapely.
- Расчёт эффективного покрытия.
"""

import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import (
    Polygon,
    MultiPolygon,
    LineString,
    MultiLineString,
    Point,
    box,
)
from shapely.affinity import rotate, translate
from shapely.ops import unary_union
from shapely.validation import make_valid

from case1.fleet.models import CameraProfile, FieldPolygon, ExclusionZone


def calculate_optical_parameters(
    altitude_m: float,
    ground_speed_m_s: float,
    camera: CameraProfile,
    side_overlap: float,
    front_overlap: float,
) -> Dict[str, float]:
    """
    Расчёт оптических параметров съёмки по формулам Pinhole-камеры:
    footprint_width = altitude × sensor_width / focal_length
    footprint_height = altitude × sensor_height / focal_length
    gsd_x = footprint_width / image_width_px
    gsd_y = footprint_height / image_height_px
    line_spacing = footprint_width × (1 - side_overlap)
    trigger_distance = footprint_height × (1 - front_overlap)
    trigger_interval = trigger_distance / ground_speed
    """
    if altitude_m <= 0:
        raise ValueError(f"Высота полёта должна быть > 0 (получено {altitude_m})")
    if ground_speed_m_s <= 0:
        raise ValueError(f"Скорость должна быть > 0 (получено {ground_speed_m_s})")
    if not (0.0 <= side_overlap < 1.0):
        raise ValueError(f"Боковое перекрытие должно быть в [0, 1) (получено {side_overlap})")
    if not (0.0 <= front_overlap < 1.0):
        raise ValueError(f"Продольное перекрытие должно быть в [0, 1) (получено {front_overlap})")

    # Footprint на земле (в метрах)
    footprint_width = altitude_m * (camera.sensor_width_mm / camera.focal_length_mm)
    footprint_height = altitude_m * (camera.sensor_height_mm / camera.focal_length_mm)

    # GSD (разрешение на земле, в см/пиксель)
    gsd_x_m = footprint_width / camera.image_width_px
    gsd_y_m = footprint_height / camera.image_height_px
    gsd_x_cm = gsd_x_m * 100.0
    gsd_y_cm = gsd_y_m * 100.0

    # Шаг между галсами и дистанция/интервал триггера камеры
    line_spacing = footprint_width * (1.0 - side_overlap)
    trigger_distance = footprint_height * (1.0 - front_overlap)
    trigger_interval = trigger_distance / ground_speed_m_s

    return {
        "footprint_width_m": round(footprint_width, 4),
        "footprint_height_m": round(footprint_height, 4),
        "gsd_x_cm_px": round(gsd_x_cm, 3),
        "gsd_y_cm_px": round(gsd_y_cm, 3),
        "line_spacing_m": round(line_spacing, 4),
        "trigger_distance_m": round(trigger_distance, 4),
        "trigger_interval_s": round(trigger_interval, 3),
    }


class LocalCoordinateTransformer:
    """
    Проекционный преобразователь WGS84 <-> Local Metric Coordinate System (в метрах).
    Использует Azimuthal Equidistant проекцию (aeqd), центрированную на контрольной точке поля,
    что минимизирует масштабные искажения расстояний в радиусе десятков километров.
    """

    def __init__(self, center_lat: float, center_lon: float):
        self.center_lat = center_lat
        self.center_lon = center_lon

        # Локальная метрическая проекция
        proj_string = (
            f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} "
            f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
        self.local_crs = CRS.from_string(proj_string)
        self.wgs84_crs = CRS.from_epsg(4326)

        # Трансформеры для быстрого пересчёта
        # В pyproj: для EPSG 4326 порядок (lat, lon) при always_xy=False или (lon, lat) при True.
        # Задаём always_xy=True, где x=lon, y=lat для WGS84.
        self._to_metric = Transformer.from_crs(self.wgs84_crs, self.local_crs, always_xy=True)
        self._to_wgs84 = Transformer.from_crs(self.local_crs, self.wgs84_crs, always_xy=True)

    def wgs84_to_metric(self, lat: float, lon: float) -> Tuple[float, float]:
        """Преобразование (lat, lon) -> (x_m, y_m)."""
        x, y = self._to_metric.transform(lon, lat)
        return float(x), float(y)

    def metric_to_wgs84(self, x_m: float, y_m: float) -> Tuple[float, float]:
        """Преобразование (x_m, y_m) -> (lat, lon)."""
        lon, lat = self._to_wgs84.transform(x_m, y_m)
        return float(lat), float(lon)

    def coords_to_metric(self, coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Преобразование списка [(lat, lon), ...] -> [(x, y), ...]."""
        return [self.wgs84_to_metric(lat, lon) for lat, lon in coords]

    def coords_to_wgs84(self, metric_coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Преобразование списка [(x, y), ...] -> [(lat, lon), ...]."""
        return [self.metric_to_wgs84(x, y) for x, y in metric_coords]


def build_field_polygon(
    field: FieldPolygon,
    transformer: LocalCoordinateTransformer,
    boundary_buffer_m: float = 0.0,
) -> Polygon:
    """
    Построение Shapely-полигона поля в метрических координатах:
    1. Перевод периметра в метры.
    2. Вычитание зон отчуждения (exclusion zones).
    3. Отступ внутрь (buffer) при необходимости.
    4. Исправление геометрии.
    """
    outer_metric = transformer.coords_to_metric(field.coordinates)
    if outer_metric[0] != outer_metric[-1]:
        outer_metric.append(outer_metric[0])

    poly = Polygon(outer_metric)
    if not poly.is_valid:
        poly = make_valid(poly)

    # Вычитание запретных зон
    if field.exclusion_zones:
        holes = []
        for ez in field.exclusion_zones:
            ez_metric = transformer.coords_to_metric(ez.coordinates)
            if ez_metric[0] != ez_metric[-1]:
                ez_metric.append(ez_metric[0])
            ez_poly = Polygon(ez_metric)
            if not ez_poly.is_valid:
                ez_poly = make_valid(ez_poly)
            holes.append(ez_poly)

        holes_union = unary_union(holes)
        poly = poly.difference(holes_union)

    if boundary_buffer_m > 0:
        buffered = poly.buffer(-boundary_buffer_m)
        if not buffered.is_empty:
            poly = buffered

    return poly


def calculate_polygon_center_wgs84(coordinates: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Центроид исходного полигона в WGS84."""
    lats = [c[0] for c in coordinates]
    lons = [c[1] for c in coordinates]
    return float(np.mean(lats)), float(np.mean(lons))


def get_principal_angles(polygon: Polygon) -> List[float]:
    """
    Определение основных характерных углов полигона:
    - Углы сторон минимального ориентированного ограничивающего прямоугольника.
    - Углы длинных сегментов границы.
    Возвращает нормализованные углы в диапазоне [0, 180).
    """
    angles = set()

    # Извлекаем сегменты внешнего периметра
    if polygon.is_empty:
        return [0.0, 90.0]

    coords = list(polygon.exterior.coords)
    longest_len = 0.0
    longest_angle = 0.0

    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length > 1e-3:
            deg = math.degrees(math.atan2(dy, dx)) % 180.0
            angles.add(round(deg, 1))
            if length > longest_len:
                longest_len = length
                longest_angle = deg

    # Обязательно включаем угол длиннейшей стороны
    angles.add(round(longest_angle, 1))

    # Стандартные углы для полноты поиска
    for base in [0.0, 45.0, 90.0, 135.0]:
        angles.add(base)

    return sorted(list(angles))

