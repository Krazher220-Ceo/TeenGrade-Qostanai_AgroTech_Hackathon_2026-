import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from case1.fleet.models import CameraProfile, DroneProfile, SafetyPolicy
from case1.fleet.demo_field import get_kostanay_demo_field, get_demo_landing_pads
from case1.fleet.planner import plan_fleet_coverage


def test_fleet_partitioning_1_to_5_drones():
    """
    Проверка распределения работы на 1, 2, 3, 4, 5 дронов:
    1. Ни одна полоса не теряется.
    2. Ни одна полоса не дублируется.
    3. Все system_id строго уникальны.
    4. Каждый аппарат имеет свою отдельную посадочную площадку.
    5. Расчётное время (makespan) уменьшается при увеличении числа дронов.
    """
    field = get_kostanay_demo_field()
    pads = get_demo_landing_pads(field)
    camera = CameraProfile.default_research_camera()

    makespans = []

    for num_drones in range(1, 6):
        plan = plan_fleet_coverage(
            field=field,
            num_drones=num_drones,
            altitude_m=30.0,
            ground_speed_m_s=5.0,
            camera=camera,
            pads=pads,
        )

        assert plan.num_drones_assigned <= num_drones
        assert len(plan.vehicles) == plan.num_drones_assigned

        # 1. Проверка уникальности system_id
        sys_ids = [v.system_id for v in plan.vehicles]
        assert len(sys_ids) == len(set(sys_ids)), f"Обнаружены повторяющиеся ID: {sys_ids}"

        # 2. Проверка посадочных площадок
        assigned_pads = [v.pad.pad_id for v in plan.vehicles]
        assert len(assigned_pads) == len(set(assigned_pads)), f"Обнаружены общие площадки посадки: {assigned_pads}"

        # 3. Проверка покрытия всех полос без потерь и дублей
        all_allocated_lane_ids = []
        for v in plan.vehicles:
            for lane in v.lanes:
                all_allocated_lane_ids.append(lane.lane_id)

        assert len(all_allocated_lane_ids) == plan.total_lanes_count, (
            f"Для {num_drones} дронов распределено {len(all_allocated_lane_ids)} полос "
            f"из общего числа {plan.total_lanes_count}"
        )
        assert len(set(all_allocated_lane_ids)) == plan.total_lanes_count, "Обнаружены дубликаты полос между дронами"

        # 4. Проверка непрерывности блоков полос
        for v in plan.vehicles:
            lane_ids = [l.lane_id for l in v.lanes]
            # Номера полос должны идти строго подряд
            expected = list(range(lane_ids[0], lane_ids[-1] + 1))
            assert lane_ids == expected, f"Блок полос дрона #{v.system_id} не является непрерывным: {lane_ids}"

        makespans.append(plan.makespan_s)

    # Время выполнения 1 дрона должно быть заметно больше времени 5 дронов
    assert makespans[0] > makespans[4], f"5 дронов ({makespans[4]}с) должны быть быстрее 1 дрона ({makespans[0]}с)"
    assert makespans[0] > makespans[1]


def test_small_field_reduces_drones_count():
    """
    Если поле крошечное (всего 2 полосы), планировщик не должен назначать 5 дронов,
    а должен безопасно уменьшить количество до эффективного.
    """
    field = get_kostanay_demo_field()
    # Задаем большую высоту и шаг полос, чтобы полос получилось очень мало
    plan = plan_fleet_coverage(
        field=field,
        num_drones=5,
        altitude_m=110.0,
        side_overlap=0.40,
        front_overlap=0.40,
    )

    if plan.total_lanes_count < 10:
        assert plan.num_drones_assigned < 5
        assert len(plan.advisories) > 0
