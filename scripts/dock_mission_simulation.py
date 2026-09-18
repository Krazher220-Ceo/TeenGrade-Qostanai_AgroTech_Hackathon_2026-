#!/usr/bin/env python3
"""Симуляция полного цикла работы dock-станции для видео питча (кейс 1, Qostanai AgroTech 2026).

Концепция: машина агронома с раскладной крышей (dock-станция, 4 гнезда) приезжает
на границу поля (дорога), разворачивает станцию, 4 дрона поочерёдно взлетают из
ОДНОЙ точки, облетают свои сектора поля, возвращаются, станция их заряжает,
забирает фото, прогоняет edge-детектор (1-я модель) и отправляет через Starlink
на сервер только кропы-боксы + метаданные. На сервере HITL-очередь для агронома.

ВАЖНО — почему в этой версии другие цифры, чем в фотограмметрическом прогоне:
side_overlap=0.70 / front_overlap=0.80 (использовались изначально) — это стандарт
ФОТОГРАММЕТРИИ для построения бесшовной ортомозаики (нужен для точного картирования
контуров очагов). Для ФИТОМОНИТОРИНГА/РАЗВЕДКИ сорняков такое перекрытие избыточно:
снимки нужны для детекции объектов на кадре, а не для сшивки. Поэтому здесь
считаются 4 отдельных сценария (см. --scenario и секцию "scenarios" в отчёте):

  a) Сплошной обзор 30 м, перекрытие для скаутинга (0.25/0.25) — облёт всего поля,
     но с кадрами в разы реже, чем при фотограмметрии.
  b) Выборочные транссекты 30 м (~12% площади, каждый ~8-й галс) — стандартный приём
     фитомониторинга по выборке.
  c) Двухпроходный: b) + детальный облёт СИНТЕТИЧЕСКИХ очагов на 12 м (~4% площади).
  d) Фотограмметрический (0.70/0.80, сплошной) — оставлен для сравнения; ДЛЯ ОРТОМОЗАИКИ,
     НЕ используется для скаутинга (даёт нереалистичные для разведки 14 циклов/~11 ч/~600 ГБ).

Для видео выбран сценарий b) — см. обоснование в конце run(): у него ровно 2 цикла
(вылет -> зарядка -> вылет) для ВСЕХ 4 дронов одновременно, что позволяет честно
показать анимацией оба цикла целиком, без урезания/экстраполяции.

ВАЖНО (общие оговорки):
- Геометрия галсов, распределение между дронами, время полёта/зарядки и объёмы
  данных считаются РЕАЛЬНЫМ кодом планировщика (case1.fleet.dock, case1.fleet.planner).
- Анимация показывает ПОЛНОСТЬЮ выбранный сценарий (все его циклы) в сжатом масштабе
  времени.
- HITL-очередь (детекции с уверенностью < 75%) и очаги для сценария c) — синтетические
  оценки/маркеры, а не результат реального инференса модели.

Запуск:
    .venv/bin/python scripts/dock_mission_simulation.py --output case1/output/dock_simulation
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qostanai-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "qostanai-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from case1.fleet.models import (
    CameraProfile,
    DockStation,
    DroneProfile,
    FieldPolygon,
    FlightLane,
    LandingPad,
    SafetyPolicy,
    VehicleMission,
)
from case1.fleet.geometry import (
    LocalCoordinateTransformer,
    calculate_optical_parameters,
    calculate_polygon_center_wgs84,
)
from case1.fleet.planner import plan_fleet_coverage
from case1.fleet.dock import (
    plan_dock_fleet_mission,
    recommend_dock_position,
    scouting_safety_policy,
    compute_sortie_data_flow,
    recompute_transit_from_pad,
)

SEED = 20260918
HITL_REVIEW_RATE = 0.08  # синтетическая доля кадров, требующих ручной проверки агрономом (<75% увер.)
EDGE_DETECT_S_PER_FRAME = 0.05  # синтетическая пропускная способность edge-инференса на станции (кадр/с)

SCOUTING_SIDE_OVERLAP = 0.25
SCOUTING_FRONT_OVERLAP = 0.25
PHOTOGRAMMETRY_SIDE_OVERLAP = 0.70
PHOTOGRAMMETRY_FRONT_OVERLAP = 0.80
TRANSECT_STRIDE = 8  # каждый 8-й галс ~= 12.5% площади
HOTSPOT_ALTITUDE_M = 12.0
HOTSPOT_OVERLAP = 0.60
HOTSPOT_SPEED_M_S = 5.0  # детальный проход выполняется медленнее для качества кадров (допущение)


def build_400ha_field(center_lat: float = 53.2300, center_lon: float = 63.6200, half_side_m: float = 1000.0) -> FieldPolygon:
    """Квадратное демо-поле ровно 2x2 км (400 га) в Костанайской области."""
    lat_step = half_side_m / 111320.0
    lon_step = half_side_m / (111320.0 * math.cos(math.radians(center_lat)))
    coords = [
        (center_lat - lat_step, center_lon - lon_step),
        (center_lat - lat_step, center_lon + lon_step),
        (center_lat + lat_step, center_lon + lon_step),
        (center_lat + lat_step, center_lon - lon_step),
    ]
    return FieldPolygon(
        field_id="kostanay_400ha_pitch_demo",
        name="Костанай — демо-поле 400 га (2×2 км)",
        coordinates=coords,
    )


def fmt_hms(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Сценарии (для сравнительной таблицы)
# ---------------------------------------------------------------------------

def _scenario_row(label: str, notes: str, plan_charge, plan_swap, extra: Optional[Dict] = None) -> Dict:
    row = {
        "scenario": label,
        "notes": notes,
        "num_cycles": plan_charge.num_cycles,
        "area_ha": plan_charge.field_area_ha,
        "coverage_percentage": plan_charge.coverage_percentage,
        "total_lanes_count": plan_charge.total_lanes_count,
        "makespan_with_charging_s": plan_charge.makespan_with_charging_s,
        "makespan_with_charging_hms": fmt_hms(plan_charge.makespan_with_charging_s),
        "makespan_with_battery_swap_s": plan_swap.makespan_with_charging_s,
        "makespan_with_battery_swap_hms": fmt_hms(plan_swap.makespan_with_charging_s),
        "frames_count": plan_charge.data_flow_total.frames_count if plan_charge.data_flow_total else 0,
        "raw_data_gb": round((plan_charge.data_flow_total.raw_data_mb if plan_charge.data_flow_total else 0.0) / 1024.0, 2),
        "edge_output_mb": round(plan_charge.data_flow_total.edge_output_mb, 1) if plan_charge.data_flow_total else 0.0,
        "raw_uplink_time_hms": fmt_hms(plan_charge.data_flow_total.raw_uplink_time_s) if plan_charge.data_flow_total else "00:00:00",
        "edge_uplink_time_hms": fmt_hms(plan_charge.data_flow_total.edge_uplink_time_s) if plan_charge.data_flow_total else "00:00:00",
    }
    if extra:
        row.update(extra)
    return row


def compute_hotspot_pass(
    transformer: LocalCoordinateTransformer,
    dock: DockStation,
    camera: CameraProfile,
    drone_profile: DroneProfile,
    hotspot_specs: List[Tuple[Tuple[float, float], float]],
    mb_per_frame_raw: float,
    edge_reduction_ratio: float,
) -> Dict:
    """
    Детальный облёт СИНТЕТИЧЕСКИХ очагов (обнаруженных на предыдущем проходе) на низкой
    высоте. Геометрия РЕАЛЬНАЯ (полноценный plan_fleet_coverage на маленьком полигоне),
    но сами позиции очагов — синтетические маркеры для демонстрации сценария, а не
    результат реального обнаружения.
    """
    policy = scouting_safety_policy(dock, min_side_overlap=HOTSPOT_OVERLAP - 0.05, min_front_overlap=HOTSPOT_OVERLAP - 0.05)
    pad = LandingPad(pad_id=f"{dock.dock_id}-HOTSPOT", system_id=1, lat=dock.lat, lon=dock.lon, alt_m=dock.alt_m)

    hotspots = []
    total_area_ha = 0.0
    total_frames = 0
    total_raw_mb = 0.0
    total_edge_mb = 0.0
    max_flight_time_s = 0.0

    for i, (center_xy, half_side_m) in enumerate(hotspot_specs, start=1):
        corners_metric = [
            (center_xy[0] - half_side_m, center_xy[1] - half_side_m),
            (center_xy[0] + half_side_m, center_xy[1] - half_side_m),
            (center_xy[0] + half_side_m, center_xy[1] + half_side_m),
            (center_xy[0] - half_side_m, center_xy[1] + half_side_m),
        ]
        corners_wgs84 = [transformer.metric_to_wgs84(x, y) for x, y in corners_metric]
        hotspot_field = FieldPolygon(
            field_id=f"hotspot_{i}",
            name=f"Синтетический очаг #{i} (детальный облёт 12 м)",
            coordinates=corners_wgs84,
        )

        hs_plan = plan_fleet_coverage(
            field=hotspot_field,
            num_drones=1,
            altitude_m=HOTSPOT_ALTITUDE_M,
            ground_speed_m_s=HOTSPOT_SPEED_M_S,
            camera=camera,
            side_overlap=HOTSPOT_OVERLAP,
            front_overlap=HOTSPOT_OVERLAP,
            pads=[pad],
            safety_policy=policy,
        )
        if not hs_plan.vehicles or not hs_plan.vehicles[0].lanes:
            continue

        v = hs_plan.vehicles[0]
        transit = recompute_transit_from_pad(v, drone_profile)
        flight_time_s = v.survey_time_s + v.turn_time_s + transit["transit_time_s"]
        data_flow = compute_sortie_data_flow(
            v.lanes, hs_plan.trigger_distance_m, dock.uplink_mbps, mb_per_frame_raw, edge_reduction_ratio
        )

        hotspots.append(
            {
                "label": hotspot_field.name,
                "center_metric": center_xy,
                "half_side_m": half_side_m,
                "area_ha": hs_plan.field_area_ha,
                "coverage_percentage": hs_plan.coverage_percentage,
                "lanes": [l.model_dump() for l in v.lanes],
                "pad": pad,
                "flight_time_s": round(flight_time_s, 1),
                "frames_count": data_flow.frames_count,
                "raw_data_mb": data_flow.raw_data_mb,
                "edge_output_mb": data_flow.edge_output_mb,
                "edge_uplink_time_s": data_flow.edge_uplink_time_s,
            }
        )
        total_area_ha += hs_plan.field_area_ha
        total_frames += data_flow.frames_count
        total_raw_mb += data_flow.raw_data_mb
        total_edge_mb += data_flow.edge_output_mb
        max_flight_time_s = max(max_flight_time_s, flight_time_s)

    return {
        "hotspots": hotspots,
        "total_area_ha": round(total_area_ha, 2),
        "total_frames": total_frames,
        "total_raw_mb": round(total_raw_mb, 1),
        "total_edge_mb": round(total_edge_mb, 1),
        "max_flight_time_s": round(max_flight_time_s, 1),
    }


# ---------------------------------------------------------------------------
# Хронология для анимации (несколько циклов на дрона)
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    phase: str
    p0: Tuple[float, float]
    p1: Tuple[float, float]
    duration_s: float
    cycle_index: int = 1


@dataclass
class CycleArrival:
    system_id: int
    cycle_index: int
    land_time_s: float
    frames_count: int
    edge_mb: float
    edge_uplink_time_s: float
    hitl_added: int


@dataclass
class DroneTimeline:
    system_id: int
    color: str
    pad_metric: Tuple[float, float]
    segments: List[Segment]
    start_offset_s: float
    total_duration_s: float
    transit_altitude_m: float
    frames_total: int
    raw_mb_total: float
    edge_mb_total: float
    arrivals: List[CycleArrival] = field(default_factory=list)

    @property
    def end_time_s(self) -> float:
        return self.start_offset_s + self.total_duration_s


def build_drone_timeline_multi_cycle(
    v: VehicleMission, transformer: LocalCoordinateTransformer, drone_profile: DroneProfile, color: str
) -> DroneTimeline:
    """Строит хронологию ВСЕХ вылетов (sorties) дрона подряд, с зарядкой между ними."""
    pad_metric = transformer.wgs84_to_metric(v.pad.lat, v.pad.lon)
    lane_by_id: Dict[int, FlightLane] = {l.lane_id: l for l in v.lanes}

    segments: List[Segment] = []
    arrivals: List[CycleArrival] = []
    cumulative_before_cycle_s = 0.0

    for sortie in v.sorties:
        lanes_seq = [lane_by_id[lid] for lid in sortie.lane_ids if lid in lane_by_id]
        if not lanes_seq:
            continue
        cyc = sortie.sortie_index

        segments.append(Segment("ВЗЛЁТ", pad_metric, pad_metric, 6.0, cyc))
        entry_pt = lanes_seq[0].start_metric
        dist_entry = math.hypot(entry_pt[0] - pad_metric[0], entry_pt[1] - pad_metric[1])
        segments.append(Segment("ПОДЛЁТ К СЕКТОРУ", pad_metric, entry_pt, dist_entry / drone_profile.transit_speed_m_s, cyc))

        for i, lane in enumerate(lanes_seq):
            survey_time = lane.length_m / drone_profile.survey_speed_m_s
            segments.append(Segment("ОБЛЁТ СЕКТОРА", lane.start_metric, lane.end_metric, survey_time, cyc))
            if i < len(lanes_seq) - 1:
                nxt = lanes_seq[i + 1].start_metric
                turn_dist = math.hypot(nxt[0] - lane.end_metric[0], nxt[1] - lane.end_metric[1])
                turn_time = drone_profile.turn_time_penalty_s + turn_dist / drone_profile.turn_speed_m_s
                segments.append(Segment("РАЗВОРОТ", lane.end_metric, nxt, turn_time, cyc))

        last_end = lanes_seq[-1].end_metric
        dist_exit = math.hypot(pad_metric[0] - last_end[0], pad_metric[1] - last_end[1])
        segments.append(Segment("ВОЗВРАТ НА СТАНЦИЮ", last_end, pad_metric, dist_exit / drone_profile.transit_speed_m_s, cyc))
        segments.append(Segment("ПОСАДКА", pad_metric, pad_metric, 5.0, cyc))

        land_time_s = cumulative_before_cycle_s + sum(s.duration_s for s in segments if s.cycle_index == cyc)
        arrivals.append(
            CycleArrival(
                system_id=v.system_id,
                cycle_index=cyc,
                land_time_s=v.takeoff_offset_s + land_time_s,
                frames_count=sortie.frames_count,
                edge_mb=sortie.edge_output_mb,
                edge_uplink_time_s=sortie.edge_uplink_time_s,
                hitl_added=round(sortie.frames_count * HITL_REVIEW_RATE),
            )
        )

        charge_label = f"ЗАРЯДКА (цикл {cyc} -> {cyc + 1})" if sortie.charge_time_s > 0 else "ЗАРЯДКА"
        segments.append(Segment(charge_label, pad_metric, pad_metric, sortie.charge_time_s, cyc))
        # Накопленная длительность ЭТОГО цикла целиком (включая зарядку) — база для следующего цикла
        cumulative_before_cycle_s += sum(s.duration_s for s in segments if s.cycle_index == cyc)

    total_duration = sum(s.duration_s for s in segments)

    return DroneTimeline(
        system_id=v.system_id,
        color=color,
        pad_metric=pad_metric,
        segments=segments,
        start_offset_s=v.takeoff_offset_s,
        total_duration_s=total_duration,
        transit_altitude_m=v.transit_altitude_m,
        frames_total=sum(s.frames_count for s in v.sorties),
        raw_mb_total=sum(s.raw_data_mb for s in v.sorties),
        edge_mb_total=sum(s.edge_output_mb for s in v.sorties),
        arrivals=arrivals,
    )


def drone_state_at(tl: DroneTimeline, t: float) -> Tuple[Tuple[float, float], str, List[Tuple[float, float]], int]:
    """Позиция, фаза, пройденная траектория и номер текущего цикла дрона в момент t."""
    local_t = t - tl.start_offset_s
    trail: List[Tuple[float, float]] = [tl.pad_metric]

    if local_t <= 0.0:
        return tl.pad_metric, "ОЖИДАНИЕ В ОЧЕРЕДИ НА ВЗЛЁТ", trail, 1

    if local_t >= tl.total_duration_s:
        for s in tl.segments:
            trail.append(s.p1)
        last_cycle = tl.segments[-1].cycle_index if tl.segments else 1
        return tl.pad_metric, "МИССИЯ ЗАВЕРШЕНА / НА СТАНЦИИ", trail, last_cycle

    cum = 0.0
    for s in tl.segments:
        seg_end = cum + s.duration_s
        if local_t <= seg_end:
            frac = 0.0 if s.duration_s <= 1e-9 else (local_t - cum) / s.duration_s
            frac = min(1.0, max(0.0, frac))
            x = s.p0[0] + (s.p1[0] - s.p0[0]) * frac
            y = s.p0[1] + (s.p1[1] - s.p0[1]) * frac
            trail.append((x, y))
            return (x, y), s.phase, trail, s.cycle_index
        trail.append(s.p1)
        cum = seg_end

    return tl.pad_metric, "ЗАРЯДКА", trail, tl.segments[-1].cycle_index if tl.segments else 1


def build_station_queue(timelines: List[DroneTimeline]) -> List[Dict]:
    """Однопоточная очередь станции (edge-детектор + Starlink-канал), FIFO по времени посадки, по ВСЕМ циклам."""
    all_arrivals = [a for tl in timelines for a in tl.arrivals]
    all_arrivals.sort(key=lambda a: a.land_time_s)

    tasks = []
    station_free = 0.0
    for arrival in all_arrivals:
        edge_duration = arrival.frames_count * EDGE_DETECT_S_PER_FRAME
        start = max(station_free, arrival.land_time_s)
        edge_end = start + edge_duration
        upload_end = edge_end + arrival.edge_uplink_time_s
        station_free = upload_end
        tasks.append(
            {
                "system_id": arrival.system_id,
                "cycle_index": arrival.cycle_index,
                "land_time_s": arrival.land_time_s,
                "edge_start_s": start,
                "edge_end_s": edge_end,
                "upload_start_s": edge_end,
                "upload_end_s": upload_end,
                "frames_count": arrival.frames_count,
                "edge_mb": arrival.edge_mb,
                "hitl_added": arrival.hitl_added,
            }
        )
    return tasks


def watermark(ax) -> None:
    ax.text(
        0.5, 0.5, "СИНТЕТИЧЕСКАЯ СИМУЛЯЦИЯ", transform=ax.transAxes, ha="center", va="center",
        fontsize=16, color="crimson", alpha=0.12, rotation=25, weight="bold", zorder=0,
    )


def draw_static_map(
    output_dir: Path,
    field_metric: List[Tuple[float, float]],
    dock_metric: Tuple[float, float],
    timelines: List[DroneTimeline],
    plan,
    scenario_label: str,
) -> Path:
    """Итоговая статичная карта: контур поля, станция, ВСЕ галсы выбранного сценария (все дроны, все циклы)."""
    fig, ax = plt.subplots(figsize=(10, 9))
    poly_xy = field_metric + [field_metric[0]]
    xs, ys = zip(*poly_xy)
    ax.plot(xs, ys, color="#1E293B", linewidth=2)
    ax.fill(xs, ys, color="#E2E8F0", alpha=0.3)

    for v in plan.vehicles:
        color = next(tl.color for tl in timelines if tl.system_id == v.system_id)
        for lane in v.lanes:
            ax.plot(
                [lane.start_metric[0], lane.end_metric[0]],
                [lane.start_metric[1], lane.end_metric[1]],
                color=color, linewidth=1.8, alpha=0.9,
            )

    ax.scatter([dock_metric[0]], [dock_metric[1]], marker="s", s=260, color="#0F172A", zorder=5, label="Станция (машина)")
    for tl in timelines:
        ax.scatter([tl.pad_metric[0]], [tl.pad_metric[1]], s=70, color=tl.color, edgecolors="black", zorder=6)

    ax.set_title(
        f"{scenario_label}\nКостанай, {plan.field_area_ha:.0f} га: {plan.num_drones_assigned} дрона(ов), "
        f"{plan.total_lanes_count} галсов ({plan.coverage_percentage:.1f}% площади), {plan.num_cycles} цикл(ов)",
        fontsize=11,
    )
    ax.set_xlabel("Восток, м")
    ax.set_ylabel("Север, м")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=9)
    watermark(ax)
    fig.tight_layout()

    out_path = output_dir / "dock_simulation_map.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_animation(
    output_dir: Path,
    field_metric: List[Tuple[float, float]],
    dock_metric: Tuple[float, float],
    timelines: List[DroneTimeline],
    station_tasks: List[Dict],
    plan,
    scenario_table: List[Dict],
    fps: int = 24,
) -> Tuple[Path, str]:
    main_end_s = max(tl.end_time_s for tl in timelines)
    intro_s, deploy_s, outro_summary_s, outro_table_s = 2.5, 2.5, 4.0, 6.0
    anim_main_s = 28.0  # сжатая длительность показа ВСЕХ циклов выбранного сценария
    total_anim_s = intro_s + deploy_s + anim_main_s + outro_summary_s + outro_table_s
    n_frames = int(total_anim_s * fps)

    xs = [p[0] for p in field_metric]
    ys = [p[1] for p in field_metric]
    pad_x0, pad_x1 = min(xs) - 150, max(xs) + 150
    pad_y0, pad_y1 = min(ys) - 150, max(ys) + 150

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    gs = fig.add_gridspec(1, 3, width_ratios=[2.1, 1.0, 0.02])
    ax_map = fig.add_subplot(gs[0, 0])
    ax_status = fig.add_subplot(gs[0, 1])
    ax_status.axis("off")

    total_frames_all = sum(tl.frames_total for tl in timelines)
    total_raw_mb_all = sum(tl.raw_mb_total for tl in timelines)
    total_edge_mb_all = sum(tl.edge_mb_total for tl in timelines)
    total_num_cycles = plan.num_cycles

    def draw_frame(i: int):
        ax_map.clear()
        ax_status.clear()
        ax_status.axis("off")

        t_anim = i / fps

        poly_xy = field_metric + [field_metric[0]]
        pxs, pys = zip(*poly_xy)
        ax_map.plot(pxs, pys, color="#1E293B", linewidth=2)
        ax_map.fill(pxs, pys, color="#E9F3E6", alpha=0.6)
        ax_map.set_xlim(pad_x0, pad_x1)
        ax_map.set_ylim(pad_y0, pad_y1)
        ax_map.set_aspect("equal")
        ax_map.set_xlabel("Восток, м")
        ax_map.set_ylabel("Север, м")
        watermark(ax_map)

        car_w, car_h = 40, 20
        ax_map.add_patch(
            Rectangle(
                (dock_metric[0] - car_w / 2, dock_metric[1] - car_h / 2), car_w, car_h,
                facecolor="#334155", edgecolor="black", zorder=5,
            )
        )
        ax_map.plot([dock_metric[0], dock_metric[0]], [dock_metric[1] + car_h / 2, dock_metric[1] + car_h / 2 + 25],
                    color="#0EA5E9", linewidth=2, zorder=5)
        ax_map.scatter([dock_metric[0]], [dock_metric[1] + car_h / 2 + 25], marker="^", s=60, color="#0EA5E9", zorder=5)

        if t_anim < intro_s:
            ax_map.set_title("Машина агронома прибывает на границу поля (вдоль дороги)", fontsize=12)
            status_lines = ["ЭТАП: ПРИЕЗД", "", "Станция ещё не развёрнута.", "", "РЕЖИМ: скаутинг сорняков", "(выборочные транссекты, не ортомозаика)"]
        elif t_anim < intro_s + deploy_s:
            ax_map.set_title("Раскладная крыша открыта, станция развёрнута — дроны в гнёздах", fontsize=12)
            for tl in timelines:
                ax_map.scatter([tl.pad_metric[0]], [tl.pad_metric[1]], s=90, color=tl.color, edgecolors="black", zorder=6)
            status_lines = ["ЭТАП: РАЗВЁРТЫВАНИЕ СТАНЦИИ", "", f"Дронов в гнёздах: {len(timelines)}",
                             f"Сценарий: выборочные транссекты (~{plan.coverage_percentage:.0f}% площади)"]
        elif t_anim < intro_s + deploy_s + anim_main_s:
            t_main = t_anim - intro_s - deploy_s
            sim_t = (t_main / anim_main_s) * main_end_s

            states = [drone_state_at(tl, sim_t) for tl in timelines]
            cycles_now = [c for (_, _, _, c) in states]
            phases_now = [p for (_, p, _, _) in states]
            all_charging_or_done = all(("ЗАРЯДКА" in p or "ЗАВЕРШЕНА" in p or "ОЖИДАНИЕ" in p) for p in phases_now)
            current_cycle_display = max(1, min(cycles_now)) if cycles_now else 1

            if all_charging_or_done and t_main > 1.0:
                ax_map.set_title(
                    f"⚡ Станция заряжает дронов — T+{fmt_hms(sim_t)} (между циклами {current_cycle_display} и {current_cycle_display + 1})",
                    fontsize=12,
                )
            else:
                ax_map.set_title(
                    f"Облёт поля {plan.field_area_ha:.0f} га — T+{fmt_hms(sim_t)}  (цикл {current_cycle_display} из {total_num_cycles})",
                    fontsize=12,
                )
            status_lines = [f"Т+{fmt_hms(sim_t)}   Цикл {current_cycle_display} из {total_num_cycles}", ""]

            frames_so_far = 0.0
            raw_so_far = 0.0
            for tl, (pos, phase, trail, cyc) in zip(timelines, states):
                x, y = pos
                txs, tys = zip(*trail)
                ax_map.plot(txs, tys, color=tl.color, linewidth=2.2, alpha=0.85, zorder=3)
                ax_map.scatter([x], [y], s=110, color=tl.color, edgecolors="black", zorder=7)
                ax_map.annotate(f"Д{tl.system_id}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)

                local_t = sim_t - tl.start_offset_s
                frac_flown = 0.0 if local_t <= 0 else min(1.0, local_t / max(1e-6, tl.total_duration_s))
                frames_so_far += tl.frames_total * frac_flown
                raw_so_far += tl.raw_mb_total * frac_flown

                status_lines.append(f"Дрон {tl.system_id} [{tl.transit_altitude_m:.0f} м]: {phase.lower()}")

            uploaded_mb = 0.0
            hitl_count = 0
            for task in station_tasks:
                if sim_t >= task["upload_end_s"]:
                    uploaded_mb += task["edge_mb"]
                    hitl_count += task["hitl_added"]
                elif sim_t >= task["upload_start_s"]:
                    frac = (sim_t - task["upload_start_s"]) / max(1e-6, task["upload_end_s"] - task["upload_start_s"])
                    uploaded_mb += task["edge_mb"] * frac

            status_lines += [
                "",
                f"Кадров снято: {int(frames_so_far)} / {total_frames_all}",
                f"Сырые данные: {raw_so_far/1024.0:.2f} ГБ / {total_raw_mb_all/1024.0:.2f} ГБ",
                f"Выгружено (Starlink, кропы): {uploaded_mb:.1f}/{total_edge_mb_all:.1f} МБ",
                f"HITL-очередь агроному: {hitl_count} карточек",
            ]
        elif t_anim < intro_s + deploy_s + anim_main_s + outro_summary_s:
            for tl in timelines:
                (x, y), phase, trail, cyc = drone_state_at(tl, main_end_s)
                txs, tys = zip(*trail)
                ax_map.plot(txs, tys, color=tl.color, linewidth=2.2, alpha=0.85, zorder=3)
                ax_map.scatter([x], [y], s=110, color=tl.color, edgecolors="black", zorder=7)

            uploaded_mb = sum(t["edge_mb"] for t in station_tasks)
            hitl_count = sum(t["hitl_added"] for t in station_tasks)
            ax_map.set_title(f"Все {total_num_cycles} цикла(ов) сценария завершены — T+{fmt_hms(main_end_s)}", fontsize=12)
            status_lines = [
                f"СЦЕНАРИЙ ЗАВЕРШЁН ({total_num_cycles} цикл(ов))",
                "",
                f"Кадров снято: {total_frames_all}",
                f"Сырые данные: {total_raw_mb_all/1024.0:.2f} ГБ",
                f"Отправлено на сервер: {total_edge_mb_all:.1f} МБ",
                f"HITL-очередь агроному: {hitl_count} карточек",
                "",
                f"Покрытие: {plan.coverage_percentage:.1f}% площади поля",
                f"(выборочные транссекты — не сплошная ортомозаика)",
            ]
        else:
            ax_map.axis("off")
            ax_map.set_title("Сравнение сценариев разведки (см. simulation_report.json)", fontsize=12)
            header = f"{'Сценарий':50s}{'Циклы':>7s}{'T(заряд)':>11s}{'T(замена)':>11s}{'ГБ сырых':>10s}{'МБ кропов':>11s}"
            lines = [header, "-" * len(header)]
            for row in scenario_table:
                lines.append(
                    f"{row['scenario'][:50]:50s}{row['num_cycles']:>7d}{row['makespan_with_charging_hms']:>11s}"
                    f"{row['makespan_with_battery_swap_hms']:>11s}{row['raw_data_gb']:>10.2f}{row['edge_output_mb']:>11.1f}"
                )
            ax_map.text(
                0.02, 0.82, "\n".join(lines), transform=ax_map.transAxes, fontsize=8.6, va="top", ha="left",
                family="monospace",
            )
            status_lines = ["Главный сценарий для видео:", "b) Выборочные транссекты, 30 м", "", "(см. обоснование в отчёте)"]

        ax_status.text(
            0.02, 0.98, "\n".join(status_lines), transform=ax_status.transAxes,
            fontsize=10.5, va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round", facecolor="#F8FAFC", edgecolor="#CBD5E1"),
        )
        ax_status.set_title("Станция / статус", fontsize=11)

    writer_name = "ffmpeg" if shutil.which("ffmpeg") else "pillow"
    if writer_name == "ffmpeg":
        out_path = output_dir / "dock_mission_simulation.mp4"
        writer = animation.FFMpegWriter(fps=fps, bitrate=3200)
    else:
        out_path = output_dir / "dock_mission_simulation.gif"
        writer = animation.PillowWriter(fps=fps)

    anim = animation.FuncAnimation(fig, draw_frame, frames=n_frames, interval=1000 / fps)
    anim.save(str(out_path), writer=writer)
    plt.close(fig)
    return out_path, writer_name


def build_event_log(dock: DockStation, timelines: List[DroneTimeline], station_tasks: List[Dict], plan) -> List[Dict]:
    events: List[Dict] = []

    def add(t_s: float, text: str, **details):
        events.append({"t_s": round(t_s, 1), "t_hms": fmt_hms(max(0.0, t_s)), "event": text, **details})

    add(-150.0, "Машина агронома прибывает на границу поля (дорога)", dock_id=dock.dock_id)
    add(-30.0, "Раскладная крыша открыта, станция развёрнута", num_slots=dock.num_slots)

    for tl in timelines:
        add(tl.start_offset_s, f"Взлёт дрона #{tl.system_id} (цикл 1)", system_id=tl.system_id, transit_altitude_m=tl.transit_altitude_m)
        for a in tl.arrivals:
            add(a.land_time_s, f"Посадка дрона #{tl.system_id} (цикл {a.cycle_index})", system_id=tl.system_id, frames_count=a.frames_count)

    for task in station_tasks:
        add(task["edge_start_s"], f"Станция: edge-детектор обрабатывает партию дрона #{task['system_id']} (цикл {task['cycle_index']})",
            system_id=task["system_id"], frames_count=task["frames_count"])
        add(task["upload_end_s"], f"Сервер: получена партия дрона #{task['system_id']} (цикл {task['cycle_index']}), добавлено в HITL-очередь",
            system_id=task["system_id"], hitl_added=task["hitl_added"])

    events.sort(key=lambda e: e["t_s"])
    return events


def run(output: Path) -> Dict:
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    field = build_400ha_field()
    camera = CameraProfile.dji_mavic_3e()
    drone_profile = DroneProfile(system_id=1, survey_speed_m_s=8.0, battery_reserve_pct=20.0)
    mb_per_frame_raw, edge_reduction_ratio = 12.0, 0.02

    position_rec = recommend_dock_position(
        field=field, num_drones=4, altitude_m=30.0, ground_speed_m_s=8.0, camera=camera, drone_profile=drone_profile,
    )
    chosen = position_rec.chosen

    dock_charge = DockStation(
        dock_id="UAZ_Patriot_01", lat=chosen.lat, lon=chosen.lon, num_slots=4, slot_offset_m=2.0,
        charge_time_min=30.0, battery_swap_enabled=False, uplink_mbps=20.0, edge_detector_enabled=True,
        takeoff_interval_s=18.0, base_transit_altitude_m=30.0, transit_altitude_step_m=5.0,
    )
    dock_swap = dock_charge.model_copy(update={"battery_swap_enabled": True, "battery_swap_time_min": 2.0})

    scouting_policy = scouting_safety_policy(dock_charge, min_side_overlap=SCOUTING_SIDE_OVERLAP, min_front_overlap=SCOUTING_FRONT_OVERLAP)
    scouting_policy_swap = scouting_safety_policy(dock_swap, min_side_overlap=SCOUTING_SIDE_OVERLAP, min_front_overlap=SCOUTING_FRONT_OVERLAP)

    common_kwargs = dict(
        field=field, num_drones=4, altitude_m=30.0, ground_speed_m_s=8.0, camera=camera,
        drone_profile=drone_profile, battery_reserve_pct=20.0,
        mb_per_frame_raw=mb_per_frame_raw, edge_reduction_ratio=edge_reduction_ratio,
    )

    # --- Сценарий A: сплошной обзор, перекрытие для скаутинга ---
    plan_a_charge = plan_dock_fleet_mission(
        dock=dock_charge, side_overlap=SCOUTING_SIDE_OVERLAP, front_overlap=SCOUTING_FRONT_OVERLAP,
        safety_policy=scouting_policy, **common_kwargs,
    )
    plan_a_swap = plan_dock_fleet_mission(
        dock=dock_swap, side_overlap=SCOUTING_SIDE_OVERLAP, front_overlap=SCOUTING_FRONT_OVERLAP,
        safety_policy=scouting_policy_swap, **common_kwargs,
    )

    # --- Сценарий B: выборочные транссекты (~12% площади) ---
    plan_b_charge = plan_dock_fleet_mission(
        dock=dock_charge, side_overlap=SCOUTING_SIDE_OVERLAP, front_overlap=SCOUTING_FRONT_OVERLAP,
        safety_policy=scouting_policy, lane_sample_stride=TRANSECT_STRIDE, **common_kwargs,
    )
    plan_b_swap = plan_dock_fleet_mission(
        dock=dock_swap, side_overlap=SCOUTING_SIDE_OVERLAP, front_overlap=SCOUTING_FRONT_OVERLAP,
        safety_policy=scouting_policy_swap, lane_sample_stride=TRANSECT_STRIDE, **common_kwargs,
    )

    # --- Сценарий D: фотограмметрический (для сравнения; НЕ для скаутинга) ---
    plan_d_charge = plan_dock_fleet_mission(
        dock=dock_charge, side_overlap=PHOTOGRAMMETRY_SIDE_OVERLAP, front_overlap=PHOTOGRAMMETRY_FRONT_OVERLAP,
        **common_kwargs,
    )
    plan_d_swap = plan_dock_fleet_mission(
        dock=dock_swap, side_overlap=PHOTOGRAMMETRY_SIDE_OVERLAP, front_overlap=PHOTOGRAMMETRY_FRONT_OVERLAP,
        **common_kwargs,
    )

    # --- Сценарий C: B + детальный облёт синтетических очагов на 12 м (~4% площади) ---
    center_lat, center_lon = calculate_polygon_center_wgs84(field.coordinates)
    transformer = LocalCoordinateTransformer(center_lat, center_lon)
    hotspot_specs = [((-350.0, 250.0), 141.0), ((450.0, -300.0), 141.0)]  # 2 квадрата ~4 га каждый (~2% + ~2% = ~4% поля)

    hotspot_charge = compute_hotspot_pass(transformer, dock_charge, camera, drone_profile, hotspot_specs, mb_per_frame_raw, edge_reduction_ratio)
    hotspot_swap = compute_hotspot_pass(transformer, dock_swap, camera, drone_profile, hotspot_specs, mb_per_frame_raw, edge_reduction_ratio)

    c_charge_makespan = plan_b_charge.makespan_with_charging_s + dock_charge.turnaround_time_min() * 60.0 + hotspot_charge["max_flight_time_s"]
    c_swap_makespan = plan_b_swap.makespan_with_charging_s + dock_swap.turnaround_time_min() * 60.0 + hotspot_swap["max_flight_time_s"]

    # --- Сравнительная таблица сценариев ---
    scenario_table = [
        _scenario_row(
            "a) Сплошной обзор 30м, скаутинг 0.25/0.25", "Весь периметр, редкое перекрытие — только детекция, не ортомозаика.",
            plan_a_charge, plan_a_swap,
        ),
        _scenario_row(
            f"b) Выборочные транссекты 30м (каждый {TRANSECT_STRIDE}-й галс)",
            "Стандарт фитомониторинга по выборке — ГЛАВНЫЙ сценарий для видео.",
            plan_b_charge, plan_b_swap,
        ),
        {
            **_scenario_row(
                "c) Двухпроходный: b) + детальный облёт очагов 12м",
                "b) плюс синтетические очаги (~4% площади) на 12 м; числа очагов — оценка по формулам, не отдельный FleetPlan-прогон с сортиями.",
                plan_b_charge, plan_b_swap,
            ),
            "num_cycles": plan_b_charge.num_cycles + 1,
            "makespan_with_charging_s": round(c_charge_makespan, 1),
            "makespan_with_charging_hms": fmt_hms(c_charge_makespan),
            "makespan_with_battery_swap_s": round(c_swap_makespan, 1),
            "makespan_with_battery_swap_hms": fmt_hms(c_swap_makespan),
            "frames_count": (plan_b_charge.data_flow_total.frames_count if plan_b_charge.data_flow_total else 0) + hotspot_charge["total_frames"],
            "raw_data_gb": round(
                ((plan_b_charge.data_flow_total.raw_data_mb if plan_b_charge.data_flow_total else 0.0) + hotspot_charge["total_raw_mb"]) / 1024.0, 2
            ),
            "edge_output_mb": round(
                (plan_b_charge.data_flow_total.edge_output_mb if plan_b_charge.data_flow_total else 0.0) + hotspot_charge["total_edge_mb"], 1
            ),
            "hotspot_detail": {
                "altitude_m": HOTSPOT_ALTITUDE_M,
                "area_ha": hotspot_charge["total_area_ha"],
                "frames": hotspot_charge["total_frames"],
                "raw_mb": hotspot_charge["total_raw_mb"],
                "edge_mb": hotspot_charge["total_edge_mb"],
                "note": "Позиции очагов синтетические (демонстрационные маркеры), геометрия облёта — реальный расчёт planner'а.",
            },
        },
        _scenario_row(
            "d) Фотограмметрический 0.70/0.80 (сплошной)", "Для построения ОРТОМОЗАИКИ; НЕ используется для скаутинга сорняков.",
            plan_d_charge, plan_d_swap,
        ),
    ]

    # --- Выбор сценария для видео: b) — 2 цикла у ВСЕХ 4 дронов одновременно ---
    video_plan = plan_b_charge
    video_dock = dock_charge
    scenario_label_for_video = f"Сценарий b): выборочные транссекты 30 м (каждый {TRANSECT_STRIDE}-й галс, ~{video_plan.coverage_percentage:.0f}% площади)"
    video_selection_reason = (
        "Выбран сценарий b) (не c)): у него РОВНО "
        f"{video_plan.num_cycles} цикла(ов) взлёт-вылет-зарядка, ОДИНАКОВЫХ для всех 4 дронов, "
        "что позволяет честно анимировать оба цикла целиком без урезания. Сценарий c) добавляет "
        "гетерогенный второй проход (только 2 из 4 дронов летят на детальный облёт синтетических "
        "очагов на другой высоте) — числа для него посчитаны реальными формулами и приведены в "
        "таблице, но полная покадровая анимация такого гетерогенного цикла добавляет инженерный "
        "риск без большой дополнительной пользы для питча; вместо этого его сравнение показано "
        "в финальных титрах видео."
    )

    if not all(v.sorties for v in video_plan.vehicles):
        raise RuntimeError("Выбранный сценарий для видео оставил дрон(ов) без вылетов — проверьте параметры прореживания.")

    field_metric = transformer.coords_to_metric(field.coordinates)
    dock_metric = transformer.wgs84_to_metric(video_dock.lat, video_dock.lon)

    palette = ["#2563EB", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899"]
    timelines = [
        build_drone_timeline_multi_cycle(v, transformer, drone_profile, palette[(v.system_id - 1) % len(palette)])
        for v in video_plan.vehicles
    ]
    station_tasks = build_station_queue(timelines)

    map_png = draw_static_map(output, field_metric, dock_metric, timelines, video_plan, scenario_label_for_video)
    video_path, writer_used = render_animation(
        output, field_metric, dock_metric, timelines, station_tasks, video_plan, scenario_table, fps=24
    )

    events = build_event_log(video_dock, timelines, station_tasks, video_plan)

    optics_30m = calculate_optical_parameters(30.0, 8.0, camera, SCOUTING_SIDE_OVERLAP, SCOUTING_FRONT_OVERLAP)
    optics_12m = calculate_optical_parameters(HOTSPOT_ALTITUDE_M, HOTSPOT_SPEED_M_S, camera, HOTSPOT_OVERLAP, HOTSPOT_OVERLAP)
    optics_photogrammetry_30m = calculate_optical_parameters(30.0, 8.0, camera, PHOTOGRAMMETRY_SIDE_OVERLAP, PHOTOGRAMMETRY_FRONT_OVERLAP)

    report = {
        "provenance": "SYNTHETIC_SIMULATION_FOR_PITCH_VIDEO_NOT_MODEL_VALIDATION",
        "seed": SEED,
        "field": {"field_id": field.field_id, "area_ha": video_plan.field_area_ha, "dimensions_m": "2000 x 2000 (номинально)"},
        "dock_position_recommendation": {"chosen": chosen.model_dump(), "candidates_evaluated": len(position_rec.candidates)},
        "dock_station": dock_charge.model_dump(),
        "optics": {
            "scouting_30m_side025_front025": optics_30m,
            "hotspot_detail_12m": optics_12m,
            "photogrammetry_30m_side070_front080_FOR_COMPARISON_ONLY": optics_photogrammetry_30m,
        },
        "scenarios": scenario_table,
        "video_scenario": {
            "chosen": "b",
            "label": scenario_label_for_video,
            "reason": video_selection_reason,
        },
        "hitl_model": {
            "review_rate_assumption": HITL_REVIEW_RATE,
            "edge_detect_s_per_frame_assumption": EDGE_DETECT_S_PER_FRAME,
            "note": "Синтетическая оценка доли кадров с уверенностью <75%; без реального инференса модели.",
        },
        "timeline_events": events,
        "artifacts": {"map_png": str(map_png.resolve()), "video": str(video_path.resolve()), "video_writer": writer_used},
        "limitations": [
            "Сценарий c) (двухпроходный) посчитан по реальным формулам планировщика, но детальный проход "
            "на 12 м не проведён как полноценный dock-план с сортиями/очередью зарядки для 2 из 4 дронов "
            "и НЕ анимирован — только агрегированные цифры в таблице сценариев.",
            "Позиции очагов для сценария c) — синтетические демонстрационные маркеры, не результат "
            "реального обнаружения моделью.",
            "HITL-очередь и покадровая скорость edge-детектора — синтетические предположения для питча.",
            "Очередь станции (edge-детектор + Starlink-канал) — однопоточная FIFO модель по времени посадки.",
            "Позиции дронов в анимации интерполируются линейно по сегментам — не является replay реальной телеметрии.",
        ],
    }

    report_path = output / "simulation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("case1/output/dock_simulation"))
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps({"scenarios": report["scenarios"], "video_scenario": report["video_scenario"]}, ensure_ascii=False, indent=2))
    print(f"Артефакты: {args.output.resolve()}")


if __name__ == "__main__":
    main()
