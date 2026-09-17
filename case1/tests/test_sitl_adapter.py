import sys
import json
import tempfile
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from case1.fleet.demo_field import get_kostanay_demo_field
from case1.fleet.planner import plan_fleet_coverage
from case1.fleet.sitl_adapter import SITLFleetAdapter


def test_sitl_adapter_qgc_plan_generation():
    field = get_kostanay_demo_field()
    plan = plan_fleet_coverage(field=field, num_drones=3)

    assert len(plan.vehicles) == 3

    for v in plan.vehicles:
        qgc_plan = SITLFleetAdapter.generate_qgc_mission_plan(v, plan.altitude_m)

        assert qgc_plan["fileType"] == "Plan"
        assert qgc_plan["version"] == 1
        assert "mission" in qgc_plan
        items = qgc_plan["mission"]["items"]

        # Должен быть взлёт (cmd 22), навигационные точки (cmd 16) и посадка (cmd 21)
        cmds = [it["command"] for it in items]
        assert 22 in cmds, "Отсутствует команда TAKEOFF"
        assert 16 in cmds, "Отсутствуют навигационные точки WAYPOINT"
        assert 21 in cmds, "Отсутствует команда LAND"
        assert qgc_plan["meta"]["system_id"] == v.system_id


def test_sitl_simulation_pack_export():
    field = get_kostanay_demo_field()
    plan = plan_fleet_coverage(field=field, num_drones=5)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        manifest = SITLFleetAdapter.export_simulation_pack(plan, tmp_path)

        assert manifest["drones_count"] == len(plan.vehicles)
        assert len(manifest["qgc_plan_files"]) == len(plan.vehicles)

        setup_sh = Path(manifest["setup_script"])
        assert setup_sh.exists()
        sh_content = setup_sh.read_text(encoding="utf-8")
        assert "export PX4_SYS_ID=1" in sh_content
        assert "export MAVSDK_PORT_1=14540" in sh_content
        if len(plan.vehicles) >= 5:
            assert "export PX4_SYS_ID=5" in sh_content
            assert "export MAVSDK_PORT_5=14544" in sh_content
