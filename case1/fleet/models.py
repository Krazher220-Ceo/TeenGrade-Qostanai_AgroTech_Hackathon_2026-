"""
Pydantic-модели данных для offline-планировщика флота БПЛА (1–5 дронов).
Все физические величины явно содержат единицы измерения в наименованиях.
"""

from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class MissionState(str, Enum):
    DRAFT = "DRAFT"
    VALIDATED = "VALIDATED"
    EXPORTED = "EXPORTED"
    SIMULATION_READY = "SIMULATION_READY"


class CameraProfile(BaseModel):
    name: str = "Standard Ag Drone Camera"
    sensor_width_mm: float = Field(gt=0, description="Ширина сенсора в мм")
    sensor_height_mm: float = Field(gt=0, description="Высота сенсора в мм")
    focal_length_mm: float = Field(gt=0, description="Фокусное расстояние в мм")
    image_width_px: int = Field(gt=0, description="Ширина изображения в пикселях")
    image_height_px: int = Field(gt=0, description="Высота изображения в пикселях")

    @classmethod
    def default_research_camera(cls) -> "CameraProfile":
        """Камера из числового примера drone_fleet_offline_research.md (6.4x4.8 мм, f=4мм, 4000x3000)."""
        return cls(
            name="Research Compact Survey Camera",
            sensor_width_mm=6.4,
            sensor_height_mm=4.8,
            focal_length_mm=4.0,
            image_width_px=4000,
            image_height_px=3000,
        )

    @classmethod
    def dji_mavic_3e(cls) -> "CameraProfile":
        """DJI Mavic 3 Enterprise Wide Camera (4/3 CMOS, 17.3x13.0 мм, f=12.29 мм, 5280x3956 px)."""
        return cls(
            name="DJI Mavic 3E Wide (Mechanical Shutter)",
            sensor_width_mm=17.3,
            sensor_height_mm=13.0,
            focal_length_mm=12.29,
            image_width_px=5280,
            image_height_px=3956,
        )


class DroneProfile(BaseModel):
    system_id: int = Field(ge=1, le=5, description="Уникальный MAVLink/системный ID аппарата (1..5)")
    name: str = "AgroDrone"
    survey_speed_m_s: float = Field(ge=2.0, le=15.0, default=5.0, description="Рабочая скорость на галсе, м/с (18 км/ч = 5.0 м/с)")
    turn_speed_m_s: float = Field(ge=1.0, le=8.0, default=3.0, description="Скорость выполнения разворота, м/с")
    turn_time_penalty_s: float = Field(ge=0.0, default=4.0, description="Штрафное время на один разворот между полосами, с")
    transit_speed_m_s: float = Field(ge=2.0, le=20.0, default=7.0, description="Скорость подлёта от площадки к полосе и обратно, м/с")
    battery_reserve_pct: float = Field(ge=15.0, le=40.0, default=20.0, description="Аварийный резерв батареи, %")
    max_flight_time_min: float = Field(ge=5.0, le=60.0, default=25.0, description="Максимальное паспортное время полёта с полезной нагрузкой, мин")
    turn_radius_m: float = Field(ge=2.0, le=30.0, default=5.0, description="Радиус разворота, м")


class LandingPad(BaseModel):
    pad_id: str = Field(description="Идентификатор площадки (P1..P5)")
    system_id: int = Field(ge=1, le=5, description="Привязанный system_id БПЛА")
    lat: float = Field(ge=-90.0, le=90.0, description="Широта точки взлёта/посадки WGS84")
    lon: float = Field(ge=-180.0, le=180.0, description="Долгота точки взлёта/посадки WGS84")
    alt_m: float = Field(default=0.0, description="Абсолютная высота над уровнем моря, м")


class ExclusionZone(BaseModel):
    zone_id: str
    name: str
    coordinates: List[Tuple[float, float]] = Field(
        min_length=3, description="Координаты запретной зоны [(lat, lon), ...]"
    )


class FieldPolygon(BaseModel):
    field_id: str = "demo_field"
    name: str = "Демо-поле"
    coordinates: List[Tuple[float, float]] = Field(
        min_length=3, description="Координаты внешнего периметра поля [(lat, lon), ...]"
    )
    exclusion_zones: List[ExclusionZone] = Field(default_factory=list, description="Внутренние препятствия / зоны отчуждения")

    @field_validator("coordinates")
    @classmethod
    def validate_polygon_closed(cls, v: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(v) < 3:
            raise ValueError("Полигон должен содержать не менее 3 точек")
        return v


class SafetyPolicy(BaseModel):
    min_altitude_m: float = 10.0
    max_altitude_m: float = 120.0
    min_speed_m_s: float = 2.0
    max_speed_m_s: float = 15.0
    min_side_overlap: float = 0.40
    max_side_overlap: float = 0.85
    min_front_overlap: float = 0.40
    max_front_overlap: float = 0.90
    min_battery_reserve_pct: float = 15.0
    boundary_buffer_m: float = 5.0
    inter_drone_buffer_m: float = 5.0
    min_pad_separation_m: float = 5.0
    require_independent_pads: bool = True



class FlightLane(BaseModel):
    lane_id: int
    system_id: Optional[int] = None
    start_metric: Tuple[float, float] = Field(description="(x, y) в локальных метрах")
    end_metric: Tuple[float, float] = Field(description="(x, y) в локальных метрах")
    start_wgs84: Tuple[float, float] = Field(description="(lat, lon)")
    end_wgs84: Tuple[float, float] = Field(description="(lat, lon)")
    length_m: float = Field(ge=0.0)
    direction_deg: float = Field(description="Направление галса в градусах")


class VehicleMission(BaseModel):
    system_id: int
    color_hex: str
    pad: LandingPad
    lanes: List[FlightLane] = Field(default_factory=list)
    flight_length_m: float = 0.0
    survey_time_s: float = 0.0
    turns_count: int = 0
    turn_time_s: float = 0.0
    transit_length_m: float = 0.0
    transit_time_s: float = 0.0
    total_estimated_time_s: float = 0.0
    battery_consumed_pct: float = 0.0
    battery_safe: bool = True
    waypoints: List[Dict[str, Any]] = Field(default_factory=list)


class FleetPlan(BaseModel):
    plan_id: str
    created_at: str
    field_id: str
    state: MissionState = MissionState.DRAFT
    num_drones_requested: int = Field(ge=1, le=5)
    num_drones_assigned: int = Field(ge=1, le=5)

    altitude_m: float
    ground_speed_m_s: float
    side_overlap: float
    front_overlap: float

    # Расчётные оптические характеристики
    footprint_width_m: float
    footprint_height_m: float
    gsd_x_cm_px: float
    gsd_y_cm_px: float
    line_spacing_m: float
    trigger_distance_m: float
    trigger_interval_s: float

    # Геометрия галсов
    sweep_angle_deg: float
    total_lanes_count: int
    field_area_ha: float
    covered_area_ha: float
    coverage_percentage: float

    # Временные метрики
    makespan_s: float = Field(description="Время завершения всей операции (максимальное время среди аппаратов)")
    theoretical_single_drone_time_s: float
    speedup_factor: float

    # Назначенные миссии
    vehicles: List[VehicleMission] = Field(default_factory=list)

    # Валидация и безопасность
    is_safe: bool = False
    safety_violations: List[str] = Field(default_factory=list)
    safety_warnings: List[str] = Field(default_factory=list)
    advisories: List[str] = Field(default_factory=list)

    human_confirmation_required: bool = True
    rules_version: str = "2026.1-cheatsheet"
