"""
Демонстрационный полигон поля в Костанайской области для мгновенной офлайн-работы.
Содержит:
- Реалистичный контур поля (около 25 га, ~500x500 м) в WGS84 координатах Костаная (53.2° N, 63.6° E).
- Внутреннее препятствие (зона отчуждения: лесополоса / опора ЛЭП / водоём).
- 5 разнесенных безопасных посадочных площадок P1..P5 на наземной станции обслуживания.
"""

from typing import List, Tuple
from case1.fleet.models import FieldPolygon, ExclusionZone, LandingPad


def get_kostanay_demo_field() -> FieldPolygon:
    """
    Типовое пшеничное поле в окрестностях Костаная (~24.5 га)
    с внутренней зоной отчуждения (озеро / заболоченный овраг).
    """
    # Внешний периметр (~500x500 м)
    # Центр: 53.2150 N, 63.6250 E
    field_coords: List[Tuple[float, float]] = [
        (53.2172, 63.6210),
        (53.2175, 63.6285),
        (53.2130, 63.6290),
        (53.2127, 63.6215),
    ]

    # Препятствие (зона отчуждения) в центре-юге поля (~80x60 м)
    lake_coords: List[Tuple[float, float]] = [
        (53.2145, 63.6240),
        (53.2148, 63.6252),
        (53.2142, 63.6255),
        (53.2139, 63.6243),
    ]

    exclusion = ExclusionZone(
        zone_id="ez_lake_01",
        name="Внутренний водоем / заболоченный участок",
        coordinates=lake_coords,
    )

    return FieldPolygon(
        field_id="kostanay_field_demo_01",
        name="Костанай — Поле №1 (Демонстрационное)",
        coordinates=field_coords,
        exclusion_zones=[exclusion],
    )


def get_demo_landing_pads(field: FieldPolygon) -> List[LandingPad]:
    """
    Генерация 5 разнесенных физических площадок P1..P5 в сервисной наземной зоне
    к югу от границы поля с шагом 8 метров друг от друга (соответствует требованиям безопасности).
    """
    # Базовая точка сервисной зоны южнее поля
    min_lat = min(c[0] for c in field.coordinates)
    base_lon = (min(c[1] for c in field.coordinates) + max(c[1] for c in field.coordinates)) / 2.0
    base_lat = min_lat - 0.0003  # ~30 метров южнее границы поля

    pads = []
    # Разносим площадки по долготе с шагом ~8-10 метров (~0.00012 градуса)
    for i in range(5):
        sys_id = i + 1
        pad_lon = base_lon + (i - 2) * 0.00015
        pads.append(
            LandingPad(
                pad_id=f"P{sys_id}",
                system_id=sys_id,
                lat=round(base_lat, 7),
                lon=round(pad_lon, 7),
                alt_m=0.0,
            )
        )

    return pads
