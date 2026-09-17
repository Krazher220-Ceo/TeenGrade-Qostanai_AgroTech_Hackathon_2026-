"""
Адаптер интеграции с PX4 SITL и QGroundControl / MAVLink (Этап 5).
Генерирует:
1. Полетные планы в формате QGroundControl Plan (.plan) для каждого аппарата.
2. Конфигурацию портов и координат для мульти-аппаратной симуляции SITL (1–5 аппаратов).
3. Готовый скрипт запуска SITL с разнесёнными точками старта P1..P5.

ВАЖНО: Адаптер формирует конфигурационные артефакты без автоматического запуска
тяжёлых процессов симуляции и без установки тяжёлых внешних пакетов.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from case1.fleet.models import FleetPlan, VehicleMission, LandingPad


class SITLFleetAdapter:
    """Адаптер подготовки полетных заданий для симуляторов PX4/ArduPilot и QGroundControl."""

    # Стандартные порты MAVLink для симулятора
    BASE_MAVSDK_PORT = 14540
    QGC_BROADCAST_PORT = 14550

    @classmethod
    def generate_qgc_mission_plan(
        cls,
        mission: VehicleMission,
        altitude_m: float,
    ) -> Dict[str, Any]:
        """
        Генерация файла QGroundControl Plan v1 для конкретного system_id:
        Включает взлёт на заданную высоту, проход всех точек галсов с включением камеры,
        и безопасный возврат на индивидуальную площадку (RTL / LAND).
        """
        items: List[Dict[str, Any]] = []

        # 1. Точка взлёта (Takeoff)
        items.append({
            "autoContinue": True,
            "command": 22,  # MAV_CMD_NAV_TAKEOFF
            "frame": 3,     # MAV_FRAME_GLOBAL_RELATIVE_ALT
            "params": [0, 0, 0, None, mission.pad.lat, mission.pad.lon, altitude_m],
            "type": "SimpleItem",
        })

        # 2. Навигационные точки рабочих полос
        for wp in mission.waypoints:
            if wp["type"] in ["WAYPOINT_LANE_START", "WAYPOINT_LANE_END"]:
                items.append({
                    "autoContinue": True,
                    "command": 16,  # MAV_CMD_NAV_WAYPOINT
                    "frame": 3,
                    "params": [0, 0, 0, None, wp["lat"], wp["lon"], altitude_m],
                    "type": "SimpleItem",
                })

        # 3. Возврат и посадка на свою площадку (Return to launch / Land)
        items.append({
            "autoContinue": True,
            "command": 21,  # MAV_CMD_NAV_LAND
            "frame": 3,
            "params": [0, 0, 0, None, mission.pad.lat, mission.pad.lon, 0],
            "type": "SimpleItem",
        })

        qgc_plan = {
            "fileType": "Plan",
            "version": 1,
            "groundStation": "QGroundControl",
            "mission": {
                "cruiseSpeed": 5.0,
                "hoverSpeed": 3.0,
                "firmwareType": 12,  # PX4 Pro
                "vehicleType": 2,    # Multi-Rotor
                "plannedHomePosition": [mission.pad.lat, mission.pad.lon, mission.pad.alt_m],
                "items": items,
            },
            "meta": {
                "system_id": mission.system_id,
                "pad_id": mission.pad.pad_id,
                "lanes_count": len(mission.lanes),
                "total_estimated_time_s": mission.total_estimated_time_s,
            },
        }

        return qgc_plan

    @classmethod
    def generate_sitl_bash_script(cls, plan: FleetPlan, output_path: Path) -> Path:
        """
        Генерация bash-скрипта для запуска симуляции 1–5 дронов в PX4 SITL
        с индивидуальными портами MAVLink и стартовыми координатами P1..P5.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "#!/usr/bin/env bash",
            "# Скрипт конфигурации запуска 1–5 БПЛА в PX4 SITL",
            "# Создан автоматически генератором AgroVision AI",
            "# ВНИМАНИЕ: Запуск симулятора требует установленного PX4-Autopilot",
            "",
            f"# Назначено аппаратов: {len(plan.vehicles)}",
            f"# Высота полёта: {plan.altitude_m} м",
            "",
            "echo '=== Запуск виртуального флота БПЛА в PX4 SITL ==='",
            "",
        ]

        for v in plan.vehicles:
            mavsdk_port = cls.BASE_MAVSDK_PORT + (v.system_id - 1)
            lines.append(f"# --- ДРОН #{v.system_id} (Площадка {v.pad.pad_id}) ---")
            lines.append(f"export PX4_SYS_ID={v.system_id}")
            lines.append(f"export PX4_HOME_LAT={v.pad.lat}")
            lines.append(f"export PX4_HOME_LON={v.pad.lon}")
            lines.append(f"export PX4_HOME_ALT={v.pad.alt_m}")
            lines.append(f"export MAVSDK_PORT_{v.system_id}={mavsdk_port}")
            lines.append(
                f"# Команда для отдельного окна: ./build/px4_sitl_default/bin/px4 -i {v.system_id - 1} "
                f"-d '$PX4_DIR/etc' -w sitl_drone_{v.system_id}"
            )
            lines.append("")

        lines.append("echo 'Конфигурация флота сформирована. Запуск выполняется по запросу оператора.'")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    @classmethod
    def export_simulation_pack(cls, plan: FleetPlan, output_dir: Path) -> Dict[str, Any]:
        """
        Экспорт полного пакета симуляции:
        - mission_drone_1.plan ... mission_drone_N.plan
        - sitl_fleet_setup.sh
        - fleet_sitl_manifest.json
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        plan_files = []
        for v in plan.vehicles:
            plan_data = cls.generate_qgc_mission_plan(v, plan.altitude_m)
            p_file = output_dir / f"mission_drone_{v.system_id}_{v.pad.pad_id}.plan"
            with open(p_file, "w", encoding="utf-8") as f:
                json.dump(plan_data, f, indent=2, ensure_ascii=False)
            plan_files.append(str(p_file))

        sh_file = output_dir / "sitl_fleet_setup.sh"
        cls.generate_sitl_bash_script(plan, sh_file)

        manifest = {
            "plan_id": plan.plan_id,
            "drones_count": len(plan.vehicles),
            "state": plan.state.value,
            "qgc_plan_files": plan_files,
            "setup_script": str(sh_file),
            "vehicles": [
                {
                    "system_id": v.system_id,
                    "pad_id": v.pad.pad_id,
                    "mavlink_port": cls.BASE_MAVSDK_PORT + (v.system_id - 1),
                    "lanes_count": len(v.lanes),
                    "flight_time_s": v.total_estimated_time_s,
                }
                for v in plan.vehicles
            ],
        }

        manifest_file = output_dir / "fleet_sitl_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return manifest
