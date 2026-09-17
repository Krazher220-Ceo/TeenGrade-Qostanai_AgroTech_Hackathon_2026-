"""
Экспорт полетных миссий и планов флота БПЛА.
- Экспорт в versioned JSON (mission-pack.json).
- Экспорт в RFC 7946 GeoJSON со стилизованными слоями (границы, запретные зоны, полосы по дронам, посадочные точки).
- Сохранение в локальную базу данных SQLite для полной автономности (offline-first).
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional

from case1.fleet.models import (
    FleetPlan,
    FieldPolygon,
    VehicleMission,
    MissionState,
)


def export_fleet_plan_json(plan: FleetPlan, output_path: Path) -> Path:
    """Экспорт плана флота в стандартный структурированный JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = plan.model_dump(mode="json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return output_path


def export_fleet_plan_geojson(
    plan: FleetPlan,
    field: FieldPolygon,
    output_path: Path,
) -> Path:
    """
    Экспорт в стандартный формат GeoJSON (RFC 7946):
    - Слой 1: Внешний периметр поля (Polygon)
    - Слой 2: Зоны отчуждения (Polygon)
    - Слой 3: Рабочие полосы съёмки с привязкой к system_id и цветовой маркировкой (LineString)
    - Слой 4: Посадочные площадки P1..P5 (Point)
    - Слой 5: Траектории подлёта (LineString)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features: List[Dict[str, Any]] = []

    # 1. Периметр поля
    # В GeoJSON порядок координат: [lon, lat]
    field_coords = [[lon, lat] for lat, lon in field.coordinates]
    if field_coords[0] != field_coords[-1]:
        field_coords.append(field_coords[0])

    features.append({
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [field_coords],
        },
        "properties": {
            "feature_type": "field_boundary",
            "field_id": field.field_id,
            "name": field.name,
            "area_ha": plan.field_area_ha,
            "stroke": "#1E293B",
            "stroke-width": 3,
            "fill": "#E2E8F0",
            "fill-opacity": 0.2,
        },
    })

    # 2. Зоны отчуждения
    for ez in field.exclusion_zones:
        ez_coords = [[lon, lat] for lat, lon in ez.coordinates]
        if ez_coords[0] != ez_coords[-1]:
            ez_coords.append(ez_coords[0])

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [ez_coords],
            },
            "properties": {
                "feature_type": "exclusion_zone",
                "zone_id": ez.zone_id,
                "name": ez.name,
                "stroke": "#EF4444",
                "stroke-width": 2,
                "fill": "#FCA5A5",
                "fill-opacity": 0.5,
            },
        })

    # 3. Рабочие полосы и траектории каждого дрона
    for v in plan.vehicles:
        for lane in v.lanes:
            line_coords = [
                [lane.start_wgs84[1], lane.start_wgs84[0]],
                [lane.end_wgs84[1], lane.end_wgs84[0]],
            ]
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": line_coords,
                },
                "properties": {
                    "feature_type": "survey_lane",
                    "system_id": v.system_id,
                    "lane_id": lane.lane_id,
                    "length_m": lane.length_m,
                    "stroke": v.color_hex,
                    "stroke-width": 3,
                },
            })

        # Посадочная площадка
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [v.pad.lon, v.pad.lat],
            },
            "properties": {
                "feature_type": "landing_pad",
                "system_id": v.system_id,
                "pad_id": v.pad.pad_id,
                "marker-color": v.color_hex,
                "marker-symbol": "heliport",
            },
        })

        # Линия транзита от площадки к первой полосе
        if v.lanes:
            first_lane = v.lanes[0]
            transit_coords = [
                [v.pad.lon, v.pad.lat],
                [first_lane.start_wgs84[1], first_lane.start_wgs84[0]],
            ]
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": transit_coords,
                },
                "properties": {
                    "feature_type": "transit_route",
                    "system_id": v.system_id,
                    "stroke": v.color_hex,
                    "stroke-width": 1.5,
                    "stroke-dasharray": "5,5",
                },
            })

    geojson_doc = {
        "type": "FeatureCollection",
        "name": f"FleetPlan_{plan.plan_id}",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "properties": {
            "plan_id": plan.plan_id,
            "created_at": plan.created_at,
            "drones_count": len(plan.vehicles),
            "makespan_s": plan.makespan_s,
            "coverage_percentage": plan.coverage_percentage,
            "state": plan.state.value,
        },
        "features": features,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson_doc, f, indent=2, ensure_ascii=False)

    return output_path


class OfflineFleetDatabase:
    """
    Локальное хранилище планов и миссий в SQLite (полностью офлайн).
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_tables(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fleet_plans (
                    plan_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    field_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    num_drones_requested INTEGER,
                    num_drones_assigned INTEGER,
                    altitude_m REAL,
                    ground_speed_m_s REAL,
                    total_lanes INTEGER,
                    coverage_percentage REAL,
                    makespan_s REAL,
                    raw_json TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vehicle_missions (
                    mission_id TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL,
                    system_id INTEGER NOT NULL,
                    total_time_s REAL,
                    flight_length_m REAL,
                    battery_consumed_pct REAL,
                    waypoints_count INTEGER,
                    raw_json TEXT NOT NULL,
                    FOREIGN KEY(plan_id) REFERENCES fleet_plans(plan_id)
                )
            """)
            conn.commit()

    def save_plan(self, plan: FleetPlan):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO fleet_plans (
                    plan_id, created_at, field_id, state,
                    num_drones_requested, num_drones_assigned,
                    altitude_m, ground_speed_m_s, total_lanes,
                    coverage_percentage, makespan_s, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                plan.plan_id,
                plan.created_at,
                plan.field_id,
                plan.state.value,
                plan.num_drones_requested,
                plan.num_drones_assigned,
                plan.altitude_m,
                plan.ground_speed_m_s,
                plan.total_lanes_count,
                plan.coverage_percentage,
                plan.makespan_s,
                plan.model_dump_json(),
            ))

            for v in plan.vehicles:
                mission_id = f"{plan.plan_id}_drone_{v.system_id}"
                cursor.execute("""
                    INSERT OR REPLACE INTO vehicle_missions (
                        mission_id, plan_id, system_id,
                        total_time_s, flight_length_m, battery_consumed_pct,
                        waypoints_count, raw_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    mission_id,
                    plan.plan_id,
                    v.system_id,
                    v.total_estimated_time_s,
                    v.flight_length_m,
                    v.battery_consumed_pct,
                    len(v.waypoints),
                    v.model_dump_json(),
                ))

            conn.commit()

    def get_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT raw_json FROM fleet_plans WHERE plan_id = ?", (plan_id,))
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return None
