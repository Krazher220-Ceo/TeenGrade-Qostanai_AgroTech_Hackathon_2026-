"""
Реестр многолетних сорняков (case1/output/perennial_registry.sqlite).

Идея питча: многолетники (бодяк, вьюнок, пырей и т.д. — см. единый список
PERENNIAL_SPECIES_MARKERS / PERENNIAL_SPECIES_CODES в case1.fleet.agronomy_rules)
сохраняются в отдельную SQLite-базу, чтобы можно было сравнивать одно и то же
поле от сезона к сезону: где появились новые очаги, где они исчезли, как
изменилась плотность заселения по видам.

Источник данных для облёта — отчёт детекций (case1/output/all_fields_report.json,
формат cmd_process из case1_main.py) или, эквивалентно, all_fields_detections.csv.
Источник='model' — автоматическая детекция; source='agronomist' — ручная
корректировка/подтверждение агронома (например, после верификации через
/api/v1/verify-item).

Каждая строка реестра — агрегат (поле, сезон, кадр, вид): количество объектов
этого вида на кадре и получившаяся плотность (шт/м²), а не отдельный бокс —
это совпадает с гранулярностью, в которой агрономическая шпаргалка ("Олжа Агро")
считает ЭПВ (экономический порог вредоносности) многолетников.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from case1.fleet.agronomy_rules import PERENNIAL_SPECIES_CODES, PERENNIAL_SPECIES_MARKERS
from case1.ml.multitask_model import SPECIES_RU_MAP

CASE1_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = CASE1_DIR / "output"
DEFAULT_DB_PATH = OUTPUT_DIR / "perennial_registry.sqlite"
DEFAULT_REPORT_PATH = OUTPUT_DIR / "all_fields_report.json"

DEFAULT_FIELD_ID = "kostanay_field_demo_01"


def _species_ru(species_code: str) -> str:
    return SPECIES_RU_MAP.get(species_code, species_code)


def is_perennial_species(species_code: str) -> bool:
    """True, если код/название вида относится к многолетникам (единый справочник)."""
    if not species_code:
        return False
    normalized = str(species_code).strip().lower()
    return normalized in PERENNIAL_SPECIES_CODES or normalized in PERENNIAL_SPECIES_MARKERS


class PerennialRegistry:
    """SQLite-реестр многолетних сорняков по полям и сезонам облёта."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS perennial_occurrences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    field TEXT NOT NULL,
                    season TEXT NOT NULL,
                    image_id TEXT NOT NULL,
                    species TEXT NOT NULL,
                    species_ru TEXT NOT NULL,
                    lat REAL,
                    lon REAL,
                    density_per_m2 REAL NOT NULL,
                    occurrence_count INTEGER NOT NULL,
                    field_area_m2 REAL NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_perennial_field_season
                ON perennial_occurrences(field, season)
            """)
            conn.commit()

    # --------------------------------------------------------------------
    # Запись
    # --------------------------------------------------------------------

    def clear_field_season(self, field: str, season: str, source: Optional[str] = None) -> int:
        """Удаляет ранее сохранённые строки для (field, season[, source]) — для переингеста."""
        with self._get_connection() as conn:
            if source:
                cur = conn.execute(
                    "DELETE FROM perennial_occurrences WHERE field=? AND season=? AND source=?",
                    (field, season, source),
                )
            else:
                cur = conn.execute(
                    "DELETE FROM perennial_occurrences WHERE field=? AND season=?",
                    (field, season),
                )
            conn.commit()
            return cur.rowcount

    def _insert_row(
        self, field: str, season: str, image_id: str, species: str, species_ru: str,
        lat: Optional[float], lon: Optional[float], density_per_m2: float,
        occurrence_count: int, field_area_m2: float, source: str,
    ) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO perennial_occurrences (
                    field, season, image_id, species, species_ru, lat, lon,
                    density_per_m2, occurrence_count, field_area_m2, source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                field, season, image_id, species, species_ru, lat, lon,
                round(float(density_per_m2), 4), int(occurrence_count), round(float(field_area_m2), 4),
                source, datetime.now(timezone.utc).isoformat(),
            ))
            conn.commit()

    def ingest_field_report(
        self,
        report_path: Path = DEFAULT_REPORT_PATH,
        field: str = DEFAULT_FIELD_ID,
        season: str = "unknown",
        source: str = "model",
        replace_existing: bool = True,
    ) -> Dict[str, Any]:
        """
        Загружает многолетников из отчёта детекций (all_fields_report.json) в
        реестр как облёт поля `field` за сезон `season`. Игнорирует культуру,
        сомнительные (review_required) детекции и малолетние сорняки — в
        реестр попадают только многолетники (единый справочник, см. модуль
        case1.fleet.agronomy_rules).
        """
        report_path = Path(report_path)
        if not report_path.exists():
            raise FileNotFoundError(f"Отчёт детекций не найден: {report_path}")
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        if not isinstance(report, list):
            raise ValueError(f"Ожидался список кадров в {report_path}")

        if replace_existing:
            self.clear_field_season(field, season, source=source)

        rows_inserted = 0
        species_totals: Counter = Counter()

        for image_entry in report:
            image_id = str(image_entry.get("filename", ""))
            lat = image_entry.get("lat")
            lon = image_entry.get("lon")
            area = float((image_entry.get("agronomy_evaluation") or {}).get("field_area_m2") or 1.0)
            detections = image_entry.get("detections", []) or []

            counts: Counter = Counter()
            for det in detections:
                if det.get("review_required"):
                    continue
                species_code = str(det.get("species") or "").strip().lower()
                if not is_perennial_species(species_code):
                    continue
                counts[species_code] += 1

            for species_code, cnt in counts.items():
                density = round(cnt / max(area, 0.1), 4)
                self._insert_row(
                    field=field, season=season, image_id=image_id,
                    species=species_code, species_ru=_species_ru(species_code),
                    lat=lat, lon=lon, density_per_m2=density,
                    occurrence_count=cnt, field_area_m2=area, source=source,
                )
                rows_inserted += 1
                species_totals[species_code] += cnt

        return {
            "field": field,
            "season": season,
            "source": source,
            "report_path": str(report_path),
            "images_processed": len(report),
            "rows_inserted": rows_inserted,
            "counts_by_species": dict(species_totals),
        }

    def seed_demo_season(
        self,
        field: str,
        demo_season: str,
        species_density_overrides: Dict[str, float],
        base_lat: Optional[float] = None,
        base_lon: Optional[float] = None,
        field_area_m2: float = 10.0,
        source: str = "demo_synthetic",
    ) -> Dict[str, Any]:
        """
        Явно помеченные демонстрационные данные для второго сезона, чтобы можно
        было показать сравнение "сезон к сезону" на дашборде без выдумывания
        реальных облётов. source ВСЕГДА "demo_synthetic" (или другое явное имя,
        начинающееся не с "model"/"agronomist"), чтобы такие строки нельзя было
        перепутать с реальными измерениями при отображении/экспорте.
        """
        if source in ("model", "agronomist"):
            raise ValueError(
                "seed_demo_season нельзя маскировать под реальный источник "
                "('model'/'agronomist') — используйте явную демо-метку."
            )
        self.clear_field_season(field, demo_season, source=source)
        rows_inserted = 0
        for species_code, density in species_density_overrides.items():
            occurrence_count = max(1, round(density * field_area_m2))
            self._insert_row(
                field=field, season=demo_season, image_id=f"demo_{field}_{demo_season}",
                species=species_code, species_ru=_species_ru(species_code),
                lat=base_lat, lon=base_lon, density_per_m2=density,
                occurrence_count=occurrence_count, field_area_m2=field_area_m2, source=source,
            )
            rows_inserted += 1
        return {
            "field": field, "season": demo_season, "source": source,
            "rows_inserted": rows_inserted, "is_demo": True,
        }

    # --------------------------------------------------------------------
    # Чтение
    # --------------------------------------------------------------------

    def list_perennials(self, field: Optional[str] = None, season: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM perennial_occurrences WHERE 1=1"
        params: List[Any] = []
        if field:
            query += " AND field = ?"
            params.append(field)
        if season:
            query += " AND season = ?"
            params.append(season)
        query += " ORDER BY season, species, image_id"
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def list_fields(self) -> List[str]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT DISTINCT field FROM perennial_occurrences ORDER BY field").fetchall()
        return [r["field"] for r in rows]

    def list_seasons(self, field: Optional[str] = None) -> List[str]:
        if field:
            query = "SELECT DISTINCT season FROM perennial_occurrences WHERE field = ? ORDER BY season"
            params = [field]
        else:
            query = "SELECT DISTINCT season FROM perennial_occurrences ORDER BY season"
            params = []
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [r["season"] for r in rows]

    def _aggregate_density_by_species(self, rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Плотность по виду = сумма occurrence_count / сумма field_area_m2 по всем кадрам."""
        agg: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            sp = row["species"]
            entry = agg.setdefault(sp, {"species_ru": row["species_ru"], "count": 0, "area": 0.0})
            entry["count"] += row["occurrence_count"]
            entry["area"] += row["field_area_m2"]
        for sp, entry in agg.items():
            entry["density"] = round(entry["count"] / max(entry["area"], 0.1), 4)
        return agg

    def compare_seasons(self, field: str, season_a: str, season_b: str) -> Dict[str, Any]:
        """
        Сравнивает плотность многолетников по видам между двумя сезонами облёта
        одного поля: новые очаги (появились в season_b, отсутствовали в
        season_a), исчезнувшие очаги (наоборот) и изменение плотности по
        видам, которые встречаются в обоих сезонах.
        """
        rows_a = self.list_perennials(field=field, season=season_a)
        rows_b = self.list_perennials(field=field, season=season_b)
        agg_a = self._aggregate_density_by_species(rows_a)
        agg_b = self._aggregate_density_by_species(rows_b)

        all_species = sorted(set(agg_a.keys()) | set(agg_b.keys()))
        comparisons = []
        new_foci: List[str] = []
        disappeared_foci: List[str] = []

        for sp in all_species:
            density_a = agg_a.get(sp, {}).get("density", 0.0)
            density_b = agg_b.get(sp, {}).get("density", 0.0)
            delta = round(density_b - density_a, 4)
            delta_pct = round((delta / density_a) * 100.0, 1) if density_a > 0 else None
            species_ru = (agg_b.get(sp) or agg_a.get(sp) or {}).get("species_ru", _species_ru(sp))

            if density_a <= 0 and density_b > 0:
                status = "new_focus"
                new_foci.append(sp)
            elif density_a > 0 and density_b <= 0:
                status = "disappeared"
                disappeared_foci.append(sp)
            elif density_b > density_a:
                status = "increased"
            elif density_b < density_a:
                status = "decreased"
            else:
                status = "stable"

            comparisons.append({
                "species": sp,
                "species_ru": species_ru,
                "density_season_a": density_a,
                "density_season_b": density_b,
                "delta_density": delta,
                "delta_pct": delta_pct,
                "status": status,
            })

        return {
            "field": field,
            "season_a": season_a,
            "season_b": season_b,
            "species_comparison": comparisons,
            "new_foci": new_foci,
            "disappeared_foci": disappeared_foci,
        }


def get_default_registry() -> PerennialRegistry:
    return PerennialRegistry(DEFAULT_DB_PATH)
