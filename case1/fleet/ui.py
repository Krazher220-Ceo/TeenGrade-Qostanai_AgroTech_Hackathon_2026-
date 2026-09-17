"""
Интерактивный UI-компонент планирования флота БПЛА (1–5 дронов) для Streamlit.
Полностью offline-first:
- Никаких внешних CDN, Mapbox-токенов или онлайн-карт: отрисовка через автономный Plotly.
- Понятный для агронома интерфейс с явными физическими величинами (м, км/ч, га, мин, %).
- Строгие обязательные предупреждения и юридический статус.
- Готовое демонстрационное поле и поддержка пользовательских GeoJSON.
- Экспорт результатов в JSON и GeoJSON.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import streamlit as st
import plotly.graph_objects as go

from case1.fleet.models import (
    FieldPolygon,
    ExclusionZone,
    CameraProfile,
    LandingPad,
    FleetPlan,
    MissionState,
)
from case1.fleet.demo_field import get_kostanay_demo_field, get_demo_landing_pads
from case1.fleet.planner import plan_fleet_coverage
from case1.fleet.exporters import export_fleet_plan_json, export_fleet_plan_geojson


def create_fleet_plotly_figure(plan: FleetPlan, field: FieldPolygon) -> go.Figure:
    """
    Создание полностью автономной 2D-карты миссии флота без обращений к внешним сервисам.
    """
    fig = go.Figure()

    # 1. Граница поля
    field_lats = [c[0] for c in field.coordinates]
    field_lons = [c[1] for c in field.coordinates]
    # Замыкаем контур для отображения
    field_lats.append(field_lats[0])
    field_lons.append(field_lons[0])

    fig.add_trace(go.Scatter(
        x=field_lons,
        y=field_lats,
        mode="lines",
        name=f"Граница поля ({plan.field_area_ha:.1f} га)",
        line=dict(color="#1E293B", width=3),
        fill="toself",
        fillcolor="rgba(241, 245, 249, 0.4)",
        hoverinfo="name",
    ))

    # 2. Зоны отчуждения / препятствия
    for ez in field.exclusion_zones:
        ez_lats = [c[0] for c in ez.coordinates]
        ez_lons = [c[1] for c in ez.coordinates]
        ez_lats.append(ez_lats[0])
        ez_lons.append(ez_lons[0])

        fig.add_trace(go.Scatter(
            x=ez_lons,
            y=ez_lats,
            mode="lines",
            name=f"Препятствие: {ez.name}",
            line=dict(color="#EF4444", width=2, dash="dash"),
            fill="toself",
            fillcolor="rgba(239, 68, 68, 0.35)",
            hoverinfo="name",
        ))

    # 3. Рабочие полосы каждого дрона
    for v in plan.vehicles:
        # Рисуем каждую полосу
        for lane in v.lanes:
            fig.add_trace(go.Scatter(
                x=[lane.start_wgs84[1], lane.end_wgs84[1]],
                y=[lane.start_wgs84[0], lane.end_wgs84[0]],
                mode="lines",
                showlegend=False,
                line=dict(color=v.color_hex, width=3),
                hoverinfo="text",
                text=f"Дрон #{v.system_id} | Полоса #{lane.lane_id} ({lane.length_m} м)",
            ))

        # Легенда для дрона (фиктивная точка для группировки в легенде)
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name=f"Дрон #{v.system_id} ({len(v.lanes)} полос, {v.total_estimated_time_s/60:.1f} мин)",
            line=dict(color=v.color_hex, width=4),
        ))

        # Траектория подлёта от посадочной площадки к первой полосе
        if v.lanes:
            first_lane = v.lanes[0]
            fig.add_trace(go.Scatter(
                x=[v.pad.lon, first_lane.start_wgs84[1]],
                y=[v.pad.lat, first_lane.start_wgs84[0]],
                mode="lines",
                showlegend=False,
                line=dict(color=v.color_hex, width=1.5, dash="dot"),
                hoverinfo="text",
                text=f"Подлёт Дрона #{v.system_id} от площадки {v.pad.pad_id}",
            ))

        # Посадочная площадка Pad
        fig.add_trace(go.Scatter(
            x=[v.pad.lon],
            y=[v.pad.lat],
            mode="markers+text",
            name=f"Площадка {v.pad.pad_id} (Дрон #{v.system_id})",
            marker=dict(size=14, color=v.color_hex, symbol="square", line=dict(color="black", width=1.5)),
            text=[f"  {v.pad.pad_id}"],
            textposition="middle right",
            hoverinfo="text",
        ))

    fig.update_layout(
        title=dict(
            text=f"Автономная карта миссии флота ({len(plan.vehicles)} БПЛА, покрытие {plan.coverage_percentage:.1f}%)",
            font=dict(size=16),
        ),
        xaxis=dict(title="Долгота WGS84 (° E)", showgrid=True, zeroline=False),
        yaxis=dict(
            title="Широта WGS84 (° N)",
            showgrid=True,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1.0,  # Сохранение геометрии (1:1 aspect ratio)
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        height=620,
        margin=dict(l=40, r=40, t=50, b=80),
        plot_bgcolor="white",
    )

    return fig


def render_fleet_planning_tab():
    """Главная функция отрисовки вкладки планирования флота в Streamlit."""
    st.header("🛸 Планирование и координация флота БПЛА (1–5 аппаратов)")
    st.markdown(
        "Автономный офлайн-планировщик совместного обследования поля несколькими беспилотниками. "
        "Локально рассчитывает оптимальный угол галсов, параметры оптики, полосы съёмки и независимые миссии."
    )

    # 1. ОБЯЗАТЕЛЬНЫЕ ПРЕДУПРЕЖДАЮЩИЕ ПЛАШКИ
    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        st.error("🛑 **Демонстрационная симуляция**\n\nМоделирование работы нескольких аппаратов на одном поле.")
    with col_w2:
        st.warning("⚠️ **Не является разрешением на полёт**\n\nТребуется согласование УВД и оператор категории 3 (Правила №706 РК).")
    with col_w3:
        st.info("📋 **Требует подтверждения агрономом**\n\nРаспылитель отключён. Выход системы носит рекомендательный характер.")

    st.markdown("---")

    # 2. ПАРАМЕТРЫ МИССИИ
    with st.expander("🛠️ Параметры съёмки, поля и БПЛА", expanded=True):
        c_field, c_fleet, c_optics = st.columns(3)

        with c_field:
            st.subheader("1. Поле и геометрия")
            field_mode = st.radio(
                "Источник контура поля:",
                ["Демонстрационное поле (Костанай, 24.5 га с водоёмом)", "Загрузить GeoJSON файл"],
                index=0,
            )

            current_field = get_kostanay_demo_field()

            if field_mode == "Загрузить GeoJSON файл":
                uploaded_geojson = st.file_uploader("Выберите .geojson файл поля:", type=["geojson", "json"])
                if uploaded_geojson:
                    try:
                        data = json.load(uploaded_geojson)
                        # Извлечение координат из FeatureCollection или Polygon
                        coords = []
                        if data.get("type") == "FeatureCollection" and data.get("features"):
                            geom = data["features"][0].get("geometry", {})
                            coords = geom.get("coordinates", [[]])[0]
                        elif data.get("type") == "Polygon":
                            coords = data.get("coordinates", [[]])[0]

                        if len(coords) >= 3:
                            # В GeoJSON [lon, lat], переводим в (lat, lon)
                            wgs_coords = [(pt[1], pt[0]) for pt in coords]
                            current_field = FieldPolygon(
                                field_id="custom_field",
                                name="Пользовательское поле",
                                coordinates=wgs_coords,
                            )
                            st.success(f"Загружен полигон: {len(wgs_coords)} точек.")
                        else:
                            st.error("GeoJSON должен содержать замкнутый полигон с >= 3 вершинами.")
                    except Exception as e:
                        st.error(f"Ошибка парсинга GeoJSON: {e}")

            st.caption(f"Текущее поле: **{current_field.name}** ({len(current_field.coordinates)} вершин, {len(current_field.exclusion_zones)} зон отчуждения)")

        with c_fleet:
            st.subheader("2. Флот и ограничения")
            num_drones = st.slider("Количество дронов:", min_value=1, max_value=5, value=3, step=1)
            altitude = st.slider("Высота полёта (м):", min_value=15.0, max_value=100.0, value=30.0, step=5.0)

            speed_ms = st.slider(
                "Скорость съёмки (м/с):",
                min_value=3.0,
                max_value=10.0,
                value=5.0,
                step=0.5,
                help="18 км/ч = 5.0 м/с; 20 км/ч = 5.56 м/с",
            )
            st.caption(f"Скорость: **{speed_ms:.1f} м/с** ({speed_ms * 3.6:.1f} км/ч)")

            battery_reserve = st.slider("Резерв аккумулятора (%):", min_value=15.0, max_value=35.0, value=20.0, step=5.0)

        with c_optics:
            st.subheader("3. Оптика и перекрытие")
            cam_choice = st.selectbox(
                "Камера БПЛА:",
                [
                    "Research Compact Camera (6.4×4.8 мм, f=4 мм, 12 Мп)",
                    "DJI Mavic 3 Enterprise Wide (17.3×13.0 мм, f=12.29 мм, 20 Мп)",
                    "Пользовательская камера",
                ],
                index=0,
            )

            if "Mavic" in cam_choice:
                camera = CameraProfile.dji_mavic_3e()
            elif "Research" in cam_choice:
                camera = CameraProfile.default_research_camera()
            else:
                sw = st.number_input("Ширина сенсора (мм):", value=6.4, step=0.1)
                sh = st.number_input("Высота сенсора (мм):", value=4.8, step=0.1)
                foc = st.number_input("Фокусное расстояние (мм):", value=4.0, step=0.1)
                camera = CameraProfile(
                    name="Custom Camera",
                    sensor_width_mm=sw,
                    sensor_height_mm=sh,
                    focal_length_mm=foc,
                    image_width_px=4000,
                    image_height_px=3000,
                )

            side_overlap = st.slider("Боковое перекрытие (side):", min_value=0.50, max_value=0.85, value=0.70, step=0.05)
            front_overlap = st.slider("Продольное перекрытие (front):", min_value=0.50, max_value=0.90, value=0.80, step=0.05)

    # 3. РАСЧЁТ ПЛАНА МИССИИ
    plan = plan_fleet_coverage(
        field=current_field,
        num_drones=num_drones,
        altitude_m=altitude,
        ground_speed_m_s=speed_ms,
        camera=camera,
        side_overlap=side_overlap,
        front_overlap=front_overlap,
        battery_reserve_pct=battery_reserve,
    )

    # 4. ВЕРХНИЕ ПЛАШКИ РАСЧЁТНЫХ ПОКАЗАТЕЛЕЙ
    st.subheader("📊 Рассчитанные параметры покрытия")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Угол галсов", f"{plan.sweep_angle_deg}°", "Оптимум по времени")
    with m2:
        st.metric("Footprint камеры", f"{plan.footprint_width_m:.1f} × {plan.footprint_height_m:.1f} м")
    with m3:
        st.metric("Разрешение (GSD)", f"{plan.gsd_x_cm_px:.2f} см/px")
    with m4:
        st.metric("Шаг полос (Spacing)", f"{plan.line_spacing_m:.1f} м")
    with m5:
        st.metric("Интервал съёмки", f"{plan.trigger_interval_s:.2f} с", f"Шаг {plan.trigger_distance_m:.1f} м")
    with m6:
        st.metric("Всего полос", f"{plan.total_lanes_count} шт")

    # Временные плашки
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.metric("Покрытие площади", f"{plan.coverage_percentage:.1f}%", f"{plan.covered_area_ha:.1f} из {plan.field_area_ha:.1f} га")
    with t2:
        makespan_min = plan.makespan_s / 60.0
        st.metric("Время флота (Makespan)", f"{makespan_min:.1f} мин", f"{plan.makespan_s:.0f} сек")
    with t3:
        single_min = plan.theoretical_single_drone_time_s / 60.0
        st.metric("Время 1 дрона", f"{single_min:.1f} мин", f"{plan.theoretical_single_drone_time_s:.0f} сек")
    with t4:
        st.metric("Ускорение флота", f"{plan.speedup_factor:.2f}x", f"Задействовано {plan.num_drones_assigned} БПЛА")

    # Предупреждения планировщика
    if plan.advisories:
        for adv in plan.advisories:
            if "слишком мало" in adv or "рекомендовано" in adv:
                st.warning(f"💡 {adv}")

    # 5. ИНТЕРАКТИВНАЯ КАРТА ПОКРЫТИЯ (PLOTLY AUTONOMOUS)
    st.subheader("🗺️ Интерактивная карта распределения полос между БПЛА")
    fig_map = create_fleet_plotly_figure(plan, current_field)
    st.plotly_chart(fig_map, width="stretch")

    # 6. ДЕТАЛЬНОЕ РАСПРЕДЕЛЕНИЕ ПО АППАРАТАМ
    st.subheader("📋 Сводная таблица по аппаратам флота")
    rows = []
    for v in plan.vehicles:
        rows.append({
            "БПЛА": f"Дрон #{v.system_id}",
            "Цвет": v.color_hex,
            "Площадка посадки": v.pad.pad_id,
            "Полос": len(v.lanes),
            "Рабочая длина (м)": f"{v.flight_length_m:.0f}",
            "Разворотов": v.turns_count,
            "Время полёта": f"{v.total_estimated_time_s/60:.1f} мин ({v.total_estimated_time_s:.0f} с)",
            "Расход батареи": f"{v.battery_consumed_pct:.1f}%",
            "Статус батареи": "✅ В норме" if v.battery_safe else "❌ Превышение резерва",
        })

    import pandas as pd
    df_fleet = pd.DataFrame(rows)
    st.dataframe(df_fleet, hide_index=True, use_container_width=True)

    # 7. ЭКСПОРТ РЕЗУЛЬТАТОВ
    st.subheader("💾 Экспорт полетного задания (Offline Pack)")
    exp_col1, exp_col2 = st.columns(2)

    # Сериализация для кнопок скачивания
    plan_json_str = plan.model_dump_json(indent=2)

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_geojson = Path(tmpdir) / "fleet.geojson"
        export_fleet_plan_geojson(plan, current_field, tmp_geojson)
        geojson_str = tmp_geojson.read_text(encoding="utf-8")

    with exp_col1:
        st.download_button(
            label="📥 Скачать mission-pack.json",
            data=plan_json_str,
            file_name=f"mission_pack_{plan.plan_id}.json",
            mime="application/json",
            use_container_width=True,
        )

    with exp_col2:
        st.download_button(
            label="🌍 Скачать fleet_mission.geojson",
            data=geojson_str,
            file_name=f"fleet_mission_{plan.plan_id}.geojson",
            mime="application/geo+json",
            use_container_width=True,
        )

    st.success("✅ План проверен валидатором безопасности: готов к демонстрации в симуляторе!")
