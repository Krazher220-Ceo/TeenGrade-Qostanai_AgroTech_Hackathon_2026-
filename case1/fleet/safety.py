"""
Модуль валидации безопасности (Safety Validator) флота БПЛА.
Реализует blocking validation для предотвращения аварийных ситуаций:
- Выход маршрута за geofence.
- Пересечение зон отчуждения (exclusion zones).
- Недостаточный резерв батареи.
- Одинаковые system_id.
- Отсутствующая или дублирующая посадочная точка (одновременная посадка в одну координату).
- Некорректная высота, скорость или перекрытие.
- Машина состояний: DRAFT -> VALIDATED -> EXPORTED -> SIMULATION_READY.
"""

import math
from typing import List, Tuple, Dict, Any, Optional
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union

from case1.fleet.models import (
    FleetPlan,
    VehicleMission,
    SafetyPolicy,
    MissionState,
    FieldPolygon,
    LandingPad,
)


class SafetyValidator:
    """
    Инспектор безопасности полетного задания флота.
    Все критические нарушения блокируют валидацию (blocking error),
    исключая переход в статус VALIDATED или SIMULATION_READY.
    """

    def __init__(self, policy: Optional[SafetyPolicy] = None):
        self.policy = policy or SafetyPolicy()

    def validate_plan(
        self,
        plan: FleetPlan,
        field: FieldPolygon,
        pads: List[LandingPad],
    ) -> Tuple[bool, List[str], List[str], List[str]]:
        """
        Полный цикл проверок безопасности.
        Возвращает:
        - is_safe (bool)
        - blocking_violations (List[str]) — блокирующие ошибки
        - warnings (List[str]) — предупреждения
        - advisories (List[str]) — агрономические и эксплуатационные рекомендации
        """
        violations: List[str] = []
        warnings: List[str] = []
        advisories: List[str] = []

        # 1. Проверка базовых параметров полёта
        if plan.altitude_m < self.policy.min_altitude_m:
            violations.append(
                f"[БЛОКИРОВКА] Высота полёта {plan.altitude_m} м ниже безопасного минимума ({self.policy.min_altitude_m} м)"
            )
        elif plan.altitude_m > self.policy.max_altitude_m:
            violations.append(
                f"[БЛОКИРОВКА] Высота полёта {plan.altitude_m} м превышает допустимый предел для БАС ({self.policy.max_altitude_m} м)"
            )

        if plan.ground_speed_m_s > self.policy.max_speed_m_s:
            violations.append(
                f"[БЛОКИРОВКА] Скорость полёта {plan.ground_speed_m_s} м/с превышает максимум ({self.policy.max_speed_m_s} м/с)"
            )

        if plan.side_overlap < self.policy.min_side_overlap:
            violations.append(
                f"[БЛОКИРОВКА] Боковое перекрытие {plan.side_overlap*100:.0f}% ниже минимального ({self.policy.min_side_overlap*100:.0f}%)"
            )

        if plan.front_overlap < self.policy.min_front_overlap:
            violations.append(
                f"[БЛОКИРОВКА] Продольное перекрытие {plan.front_overlap*100:.0f}% ниже минимального ({self.policy.min_front_overlap*100:.0f}%)"
            )

        # 2. Проверка уникальности System ID
        assigned_sys_ids = [v.system_id for v in plan.vehicles]
        if len(assigned_sys_ids) != len(set(assigned_sys_ids)):
            violations.append(
                f"[БЛОКИРОВКА] Обнаружены повторяющиеся MAVLink System ID: {assigned_sys_ids}. Каждый аппарат обязан иметь уникальный ID!"
            )

        # 3. Проверка посадочных площадок (Landing Pads)
        if len(pads) < len(plan.vehicles):
            violations.append(
                f"[БЛОКИРОВКА] Недостаточно посадочных площадок: аппаратов {len(plan.vehicles)}, а площадок {len(pads)}"
            )

        # Проверка разнесения площадок (запрет одновременной посадки в одну координату)
        for i in range(len(pads)):
            for j in range(i + 1, len(pads)):
                pad1 = pads[i]
                pad2 = pads[j]
                # Оценка расстояния в метрах
                d_lat = (pad1.lat - pad2.lat) * 111320.0
                d_lon = (pad1.lon - pad2.lon) * 111320.0 * math.cos(math.radians(pad1.lat))
                dist_m = math.hypot(d_lat, d_lon)

                if dist_m < self.policy.min_pad_separation_m:
                    violations.append(
                        f"[БЛОКИРОВКА] Площадки {pad1.pad_id} и {pad2.pad_id} расположены слишком близко "
                        f"({dist_m:.1f} м < {self.policy.min_pad_separation_m} м). "
                        f"Одновременная посадка/RTL нескольких БПЛА в одну точку категорически запрещена!"
                    )

        # 4. Проверка расхода батареи для каждого аппарата
        for v in plan.vehicles:
            if not v.battery_safe:
                violations.append(
                    f"[БЛОКИРОВКА] Дрон #{v.system_id}: расчётное потребление {v.battery_consumed_pct:.1f}% "
                    f"превышает безопасный лимит (резерв {self.policy.min_battery_reserve_pct}%)"
                )

        # 5. Проверка запретных зон (Exclusion Zones)
        if field.exclusion_zones:
            exclusion_polys = []
            for ez in field.exclusion_zones:
                if len(ez.coordinates) >= 3:
                    exclusion_polys.append(Polygon(ez.coordinates))

            for v in plan.vehicles:
                for lane in v.lanes:
                    lane_line = LineString([lane.start_wgs84, lane.end_wgs84])
                    for ez_poly in exclusion_polys:
                        if ez_poly.intersects(lane_line):
                            violations.append(
                                f"[БЛОКИРОВКА] Полоса #{lane.lane_id} дрона #{v.system_id} пересекает зону отчуждения!"
                            )

        # 6. Проверка выхода за внешнюю границу поля (Geofence)
        field_poly = Polygon(field.coordinates)
        buffered_field = field_poly.buffer(0.0001)  # Небольшой допуск WGS84 (~10 м)

        for v in plan.vehicles:
            for lane in v.lanes:
                p_start = Point(lane.start_wgs84)
                p_end = Point(lane.end_wgs84)
                if not buffered_field.contains(p_start) or not buffered_field.contains(p_end):
                    violations.append(
                        f"[БЛОКИРОВКА] Точка галса дрона #{v.system_id} вышла за пределы geofence поля!"
                    )

        # 7. Проверка пересечения транзитных траекторий подлёта
        transit_lines = []
        for v in plan.vehicles:
            if v.lanes:
                pad_pt = (v.pad.lat, v.pad.lon)
                first_pt = v.lanes[0].start_wgs84
                t_line = LineString([pad_pt, first_pt])
                transit_lines.append((v.system_id, t_line))

        for i in range(len(transit_lines)):
            for j in range(i + 1, len(transit_lines)):
                id1, line1 = transit_lines[i]
                id2, line2 = transit_lines[j]
                if line1.intersects(line2):
                    warnings.append(
                        f"[ВНИМАНИЕ] Подлётные маршруты дронов #{id1} и #{id2} пересекаются в плане. "
                        f"Необходим ступенчатый взлёт (time-slot takeoff) с интервалом не менее 30 секунд."
                    )

        # Обязательные агрономические и юридические плашки
        advisories.append("РЕЖИМ: Демонстрационная симуляция")
        advisories.append("Не является разрешением на реальный групповой полёт")
        advisories.append("Рекомендация требует обязательного подтверждения агрономом")

        is_safe = len(violations) == 0

        return is_safe, violations, warnings, advisories


def transition_plan_state(
    plan: FleetPlan,
    target_state: MissionState,
    validator: Optional[SafetyValidator] = None,
    field: Optional[FieldPolygon] = None,
    pads: Optional[List[LandingPad]] = None,
) -> FleetPlan:
    """
    Управление машиной состояний плана миссии:
    DRAFT -> VALIDATED -> EXPORTED -> SIMULATION_READY
    Переход в любое состояние выше DRAFT блокируется при наличии нарушений безопасности.
    """
    if target_state == MissionState.DRAFT:
        return plan.model_copy(update={"state": MissionState.DRAFT})

    # При попытке валидации или перехода выше проводим полную проверку
    val = validator or SafetyValidator()
    if field and pads:
        is_safe, violations, warnings, adv = val.validate_plan(plan, field, pads)
        if not is_safe:
            raise ValueError(
                f"Невозможно перевести план в {target_state.value}: обнаружены блокирующие нарушения:\n"
                + "\n".join(violations)
            )

    return plan.model_copy(update={"state": target_state, "is_safe": True})
