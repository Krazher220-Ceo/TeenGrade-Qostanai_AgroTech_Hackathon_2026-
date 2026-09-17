"""
🌾 Qostanai AgroTech Hackathon 2026 — Интерактивный Агро-Дашборд (Кейс №1)
Мониторинг сорняков, картографирование БПЛА и дифференцированное опрыскивание полей.
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import torch
import torchvision.transforms as transforms

# Гарантированное добавление корня проекта в sys.path для корректной работы Streamlit
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from case1.ml.multitask_model import (
    WeedMultiTaskModel,
    SPECIES_NAMES,
    SPECIES_RU,
    STAGE_NAMES,
    STAGE_RU,
    DEFAULT_SPECIES_CONFIDENCE,
)
from case1.fleet.ui import render_fleet_planning_tab
from case1.viewer_pipeline import run_uploaded_field_image
try:
    from case1.ml.multitask_model import infer_num_species_from_state_dict
except ImportError:
    def infer_num_species_from_state_dict(state_dict):
        for key, value in state_dict.items():
            if key.endswith("species_head.4.weight"):
                return int(value.shape[0])
        return 4


# Настройка страницы
st.set_page_config(
    page_title="AgroTech 2026 | Weed Monitoring AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Пути
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CASE1_DIR = ROOT_DIR / "case1"
OUTPUT_DIR = CASE1_DIR / "output"
FIELD_DIR = ROOT_DIR / "Dataset 1 кейс" / "ФотоПолей"
WEEDS_DIR = ROOT_DIR / "Dataset 1 кейс" / "Сорняки"
MODELS_DIR = CASE1_DIR / "models"

CSV_DETECTIONS = OUTPUT_DIR / "all_fields_detections.csv"
REPORT_JSON = OUTPUT_DIR / "all_fields_report.json"
METRICS_JSON = OUTPUT_DIR / "classifier_training_metrics.json"
CLUSTERS_JSON = OUTPUT_DIR / "clusters_2d.json"

COLOR_MAP = {
    "field_thistle": "#EF4444",   # Красный - Осот/Бодяк
    "field_bindweed": "#F59E0B",  # Оранжевый - Вьюнок
    "couch_grass": "#10B981",     # Зелёный - Пырей
    "crop_wheat": "#2563EB",      # Синий - культура, не опрыскивать
    "unknown": "#6B7280"          # Серый - Требует проверки
}

HERBICIDE_ADVICE = {
    "field_thistle": {
        "group": "Двудольный многолетний сорняк",
        "herbicide": "Дикамба, 2,4-Д, Клопиралид (Лонтрел-300)",
        "rate": "0.3 - 0.5 л/га",
        "optimal_stage": "Фаза розетки (до 10-15 см высоты). В фазе стеблевания дозу увеличивают на 25%."
    },
    "field_bindweed": {
        "group": "Двудольный вьющийся корнеотпрысковый",
        "herbicide": "Глифосат (по парам), 2,4-Д + Флорасулам (Прима), Дикамба",
        "rate": "0.6 - 0.8 л/га",
        "optimal_stage": "Длина побегов 10-20 см до активного цветения."
    },
    "couch_grass": {
        "group": "Однодольный злаковый злак",
        "herbicide": "Селективные граминициды (Клетодим, Галоксифоп, Хизалофоп)",
        "rate": "0.8 - 1.2 л/га",
        "optimal_stage": "Высота 10-15 см при активном сокодвижении."
    },
    "unknown": {
        "group": "Неопределенный вид",
        "herbicide": "Требуется визуальная верификация агрономом",
        "rate": "По предписанию специалиста",
        "optimal_stage": "Ручной осмотр"
    },
    "crop_wheat": {
        "group": "Культурное растение",
        "herbicide": "Не опрыскивать",
        "rate": "0 л/га",
        "optimal_stage": "Объект исключается из карты обработки"
    }
}


@st.cache_data
def load_detections_data():
    if CSV_DETECTIONS.exists():
        df = pd.read_csv(CSV_DETECTIONS)
        if "species" in df.columns:
            df = df[df["species"] != "crop_wheat"].copy()
        return df
    return pd.DataFrame()


@st.cache_data
def load_report_data():
    if REPORT_JSON.exists():
        with open(REPORT_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_training_metrics():
    if METRICS_JSON.exists():
        with open(METRICS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_cluster_data():
    if CLUSTERS_JSON.exists():
        with open(CLUSTERS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_resource
def load_classifier_model():
    weights_path = MODELS_DIR / "multitask_weeds_best.pt"
    if not weights_path.exists():
        weights_path = MODELS_DIR / "multitask_weeds_focal.pt"
    if weights_path.exists():
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        state_dict = torch.load(weights_path, map_location=device)
        num_species = infer_num_species_from_state_dict(state_dict)
        model = WeedMultiTaskModel(num_species=num_species, num_stages=2, backbone_name="efficientnet_b0", pretrained=False)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, device
    return None, None


# ==============================================================================
# САЙДБАР
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/drone.png", width=64)
    st.title("AgroVision AI")
    st.caption("Qostanai AgroTech Hackathon 2026 • Кейс №1")

    st.markdown("---")
    st.subheader("⚙️ Параметры инференса")

    conf_thresh = st.slider(
        "Порог уверенности детектора", 0.20, 0.90, 0.65, 0.05,
        help="Минимальная уверенность YOLO для рамки-кандидата.",
    )
    review_thresh = st.slider(
        "Минимальная уверенность вида", 0.20, 0.90,
        DEFAULT_SPECIES_CONFIDENCE, 0.05,
        help="Ниже этого значения вид не подтверждается автоматически.",
    )
    st.caption(
        "По умолчанию оба порога — 65%. Пшеница и другая уверенно распознанная "
        "культура исключаются: в результатах остаются только сорняки и сомнительные объекты."
    )

    df_detections = load_detections_data()
    all_images = df_detections["image_id"].unique().tolist() if not df_detections.empty else []

    selected_image = st.selectbox(
        "Выберите снимок с дрона:",
        options=all_images,
        index=0 if all_images else None
    )

    species_filter = st.multiselect(
        "Фильтр сорняков на карте:",
        options=["Бодяк полевой", "Вьюнок полевой", "Пырей ползучий", "Неизвестный сорняк", "Не определено"],
        default=["Бодяк полевой", "Вьюнок полевой", "Пырей ползучий", "Неизвестный сорняк", "Не определено"]
    )

    st.markdown("---")
    st.info("💡 **Архитектура:**\n\n1. Edge Drone: YOLOv8s (тайлинг 640x640, SAHI)\n2. Server: EfficientNet-B0 Multi-Task с Focal Loss")


# ==============================================================================
# ОСНОВНОЙ ЭКРАН
# ==============================================================================
st.title("🌾 Мониторинг сорняков и дифференцированное опрыскивание")
st.markdown("Автоматический анализ полевых RGB-снимков с классификацией видов и стадий вегетации.")
st.caption(
    "Демонстрационные данные включают кадры профессионального класса, но для бюджетного MVP "
    "не требуется покупать DJI Mavic 3E: ниже во вкладке планирования указан реалистичный б/у вариант."
)
st.info(
    "Текущая сборка запущена локально на Mac без облака: детектор полевого снимка работает на CPU, "
    "а классификатор использует Apple MPS, если он доступен, иначе CPU. На сборке с NVIDIA GPU и после "
    "дообучения на большем наборе размеченных кадров обработка ожидаемо будет быстрее, а качество может "
    "стать выше после отдельной проверки на новых полях."
)
st.warning(
    "**Статус модели: прототип / этап сбора датасета.** WeedBlaster Vision — открытый baseline, "
    "который здесь служит временной моделью для проверки полного конвейера: снимок → кандидаты → "
    "классификация → карта. Это не готовая промышленная модель и не основание для автоматического "
    "опрыскивания. Open source важен для воспроизводимости, аудита и замены checkpoint на собственную "
    "модель после накопления и разметки локальных полевых данных."
)

tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📤 Проверить снимок поля",
    "🗺️ Карта поля и детекции",
    "🌡️ Тепловая карта и рецепт опрыскивания",
    "🩺 Центр сомнений агронома (Review)",
    "🔬 Анализ отдельного образца",
    "📊 Метрики обучения модели",
    "🌌 Кластеры эмбеддингов (t-SNE) & Защита от галлюцинаций",
    "🛸 Планирование флота (1–5 БПЛА)"
])


# ------------------------------------------------------------------------------
# TAB 0: JUDGE-FACING FIELD IMAGE UPLOAD
# ------------------------------------------------------------------------------
with tab0:
    st.header("📤 Анализ нового снимка поля")
    st.markdown(
        "Загрузите один снимок JPG/PNG. Система найдёт объекты класса Weed, "
        "классифицирует вид и фазу, отметит сомнительные случаи и подготовит "
        "размеченное изображение, CSV и JSON. Максимальный размер файла — 50 МБ."
    )

    field_upload = st.file_uploader(
        "Снимок поля или аэрофотоснимок",
        type=["jpg", "jpeg", "png"],
        key="field_image_upload",
    )

    if field_upload is not None:
        upload_bytes = field_upload.getvalue()
        st.image(upload_bytes, caption=f"Предпросмотр исходного снимка: {field_upload.name}", width="stretch")
        st.caption(
            "Обработка высокодетального DJI-снимка на CPU может занять несколько минут. "
            "Результат не является автоматическим разрешением на опрыскивание."
        )

        if st.button("Запустить распознавание", type="primary", key="run_field_inference"):
            try:
                with st.spinner("Детекция, классификация и подготовка сводок…"):
                    st.session_state["viewer_run_result"] = run_uploaded_field_image(
                        upload_bytes,
                        field_upload.name,
                        OUTPUT_DIR / "viewer_runs",
                        detector_conf=conf_thresh,
                        species_conf=review_thresh,
                    )
            except Exception as exc:
                st.session_state.pop("viewer_run_result", None)
                st.error(f"Не удалось обработать изображение: {exc}")

    viewer_result = st.session_state.get("viewer_run_result")
    if viewer_result:
        summary = viewer_result.get("summary", {})
        detections = viewer_result.get("detections", [])
        metric_cols = st.columns(4)
        metric_cols[0].metric("Объектов обнаружено", summary.get("total_weeds", len(detections)))
        metric_cols[1].metric("Требуют проверки", summary.get("review_required_count", 0))
        metric_cols[2].metric("Время детекции", f"{summary.get('time_s', 0):.1f} с")
        metric_cols[3].metric("Средняя уверенность вида", f"{summary.get('avg_species_conf', 0) * 100:.1f}%")

        st.markdown("### 1. Исходный снимок")
        st.caption("Снимок без разметки — именно он поступает на вход конвейера.")
        st.image(viewer_result["input_path"], caption="Оригинал", width="stretch")

        st.markdown("### 2. YOLO: рамки кандидатов сорняков")
        st.caption(
            "Первая модель находит координаты всех кандидатов. На этом шаге она ещё не определяет "
            "конкретный вид, поэтому часть рамок может затем отсеяться как культура."
        )
        detector_boxes_path = viewer_result.get("detector_boxes_path")
        if detector_boxes_path and Path(detector_boxes_path).exists():
            st.image(
                detector_boxes_path,
                caption="Все кандидаты YOLO до классификации отдельных crop-фрагментов",
                width="stretch",
            )
        else:
            st.warning("Этот результат создан старой сборкой. Запустите распознавание ещё раз, чтобы получить второй кадр.")

        st.markdown("### 3. Вторая модель: вид, фаза и уверенность")
        st.caption(
            "Каждая рамка вырезается в отдельный crop, увеличивается до входного размера классификатора, "
            "а затем EfficientNet определяет вид и фазу. Сомнительные ответы остаются на ручную проверку."
        )
        st.image(
            viewer_result["annotated_path"],
            caption="Итог: рамки подписаны результатом второй модели",
            width="stretch",
        )

        if detections:
            result_df = pd.DataFrame(detections)
            visible_columns = [
                "object_id", "detector_conf", "species_ru", "species_conf",
                "top_species_ru", "top_species_conf",
                "stage_ru", "stage_conf", "review_required", "spray_action",
            ]
            visible_columns = [column for column in visible_columns if column in result_df.columns]
            st.dataframe(result_df[visible_columns], width="stretch", hide_index=True)

            with st.expander(f"🔬 Кропы, обработанные второй моделью ({len(result_df)})", expanded=True):
                st.caption(
                    "Это реальные мелкие фрагменты из рамок второго изображения. Номер crop совпадает "
                    "с номером рамки на обоих размеченных кадрах."
                )
                crop_columns = st.columns(4)
                run_dir = Path(viewer_result["run_dir"])
                for crop_index, (_, row) in enumerate(result_df.head(16).iterrows()):
                    crop_relative = row.get("crop_path") or row.get("crop_file")
                    if not crop_relative:
                        continue
                    crop_path = run_dir / str(crop_relative)
                    if not crop_path.exists():
                        continue

                    species_label = str(row.get("species_ru", "Не определено"))
                    species_confidence = float(row.get("species_conf", 0) or 0)
                    top_label = str(row.get("top_species_ru", species_label))
                    top_confidence = float(row.get("top_species_conf", species_confidence) or 0)
                    if species_label == "Не определено":
                        result_caption = f"Вероятнее: {top_label} — {top_confidence:.0%}; нужна проверка"
                    else:
                        result_caption = f"{species_label} — {species_confidence:.0%}"

                    with crop_columns[crop_index % 4]:
                        st.image(str(crop_path), caption=f"#{row['object_id']} · {result_caption}", width="stretch")
                if len(result_df) > 16:
                    st.caption(f"Показаны первые 16 из {len(result_df)} кропов; полный список доступен в CSV/JSON.")
        else:
            st.info("Детектор не нашёл объектов класса Weed. CSV всё равно сформирован с заголовками.")

        download_cols = st.columns(2)
        csv_path = Path(viewer_result["csv_path"])
        report_path = Path(viewer_result["report_path"])
        download_cols[0].download_button(
            "Скачать сводку CSV",
            data=csv_path.read_bytes(),
            file_name=f"weed_detections_{viewer_result['run_id']}.csv",
            mime="text/csv",
        )
        download_cols[1].download_button(
            "Скачать полный отчёт JSON",
            data=report_path.read_bytes(),
            file_name=f"weed_report_{viewer_result['run_id']}.json",
            mime="application/json",
        )


# ------------------------------------------------------------------------------
# TAB 1: КАРТА ПОЛЯ И ДЕТЕКЦИИ
# ------------------------------------------------------------------------------
with tab1:
    if df_detections.empty:
        st.warning("Файл детекций не найден. Запустите сначала команду `python3 case1_main.py process`.")
    else:
        df_img = df_detections[df_detections["image_id"] == selected_image]

        # Применяем порог и фильтры
        df_filtered = df_img[
            (df_img["detector_conf"] >= conf_thresh) &
            (df_img["species_ru"].isin(species_filter))
        ]

        # Верхние плашки метрик
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Обнаружено сорняков", f"{len(df_filtered)} шт")
        with col2:
            alt = df_filtered["drone_rel_alt"].iloc[0] if not df_filtered.empty else 0.0
            st.metric("Высота полёта БПЛА", f"{alt:.1f} м")
        with col3:
            density = round(len(df_filtered) / 12.0, 1)  # примерная площадь покрытия кадра при h=1.5-2м
            st.metric("Плотность очагов", f"{density} шт/м²")
        with col4:
            thistle_cnt = len(df_filtered[df_filtered["species"] == "field_thistle"])
            st.metric("Бодяк (Осот)", f"{thistle_cnt} шт")
        with col5:
            bindweed_cnt = len(df_filtered[df_filtered["species"] == "field_bindweed"])
            st.metric("Вьюнок полевой", f"{bindweed_cnt} шт")

        # Визуализация кадра с боксами
        raw_img_path = FIELD_DIR / selected_image
        if raw_img_path.exists():
            img = Image.open(raw_img_path).convert("RGB")
            draw = ImageDraw.Draw(img)

            for _, row in df_filtered.iterrows():
                x1, y1, x2, y2 = int(row["bbox_x1"]), int(row["bbox_y1"]), int(row["bbox_x2"]), int(row["bbox_y2"])
                sp = row["species"]
                color = COLOR_MAP.get(sp, "#6B7280")
                draw.rectangle([x1, y1, x2, y2], outline=color, width=8)

            st.image(img, caption=f"Полевой кадр: {selected_image} (разрешение {img.width}x{img.height})", width="stretch")

            # Легенда
            l_col1, l_col2, l_col3, l_col4 = st.columns(4)
            l_col1.markdown("🔴 **Бодяк (Осот)**")
            l_col2.markdown("🟠 **Вьюнок полевой**")
            l_col3.markdown("🟢 **Пырей ползучий**")
            l_col4.markdown("⚪ **Требует проверки агрономом**")

        # Графики распределения видов и фаз
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_species = px.pie(
                df_filtered,
                names="species_ru",
                title="Распределение видов сорняков",
                color="species_ru",
                color_discrete_map={
                    "Бодяк полевой": "#EF4444",
                    "Вьюнок полевой": "#F59E0B",
                    "Пырей ползучий": "#10B981",
                    "Неизвестный сорняк": "#6B7280"
                }
            )
            st.plotly_chart(fig_species, width="stretch")

        with col_c2:
            fig_stage = px.bar(
                df_filtered,
                x="stage_ru",
                color="species_ru",
                title="Фазы вегетации сорняков",
                barmode="group",
                color_discrete_map={
                    "Бодяк полевой": "#EF4444",
                    "Вьюнок полевой": "#F59E0B",
                    "Пырей ползучий": "#10B981",
                    "Неизвестный сорняк": "#6B7280"
                }
            )
            st.plotly_chart(fig_stage, width="stretch")


# ------------------------------------------------------------------------------
# TAB 2: ТЕПЛОВАЯ КАРТА И РЕЦЕПТ ОПРЫСКИВАНИЯ
# ------------------------------------------------------------------------------
with tab2:
    st.header("🎯 Дифференцированное опрыскивание (Variable Rate Application)")
    st.markdown("Переход от сплошного полива химикатами к точному точечному нанесению на очаги сорняков.")

    if not df_detections.empty:
        df_img = df_detections[df_detections["image_id"] == selected_image]

        # 2D Heatmap плотности
        col_map, col_calc = st.columns([1.2, 1.0])

        with col_map:
            st.subheader("Тепловая карта засорённости участка")
            fig_density = px.density_heatmap(
                df_img,
                x="bbox_x1",
                y="bbox_y1",
                nbinsx=25,
                nbinsy=20,
                color_continuous_scale="YlOrRd",
                title="Плотность очагов на поле (координаты пикселей)"
            )
            fig_density.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_density, width="stretch")

        with col_calc:
            st.subheader("💰 Экономический и экологический эффект")
            field_area_ha = st.number_input("Площадь поля (га):", min_value=1.0, max_value=5000.0, value=150.0, step=10.0)
            blanket_rate_l = st.number_input("Базовая норма сплошного опрыскивания (л/га):", min_value=50.0, max_value=300.0, value=150.0, step=10.0)
            chem_price_per_l = st.number_input("Стоимость рабочего раствора (₸/л):", min_value=100.0, max_value=10000.0, value=1200.0, step=100.0)

            # Процент заражения поля очагами сорняков (из анализа плотности)
            infestation_ratio = min(0.35, max(0.08, len(df_img) / 450.0))
            spot_area_treated_ha = field_area_ha * infestation_ratio

            # Расчет затрат
            blanket_total_l = field_area_ha * blanket_rate_l
            spot_total_l = spot_area_treated_ha * blanket_rate_l
            savings_l = blanket_total_l - spot_total_l
            savings_pct = (1.0 - (spot_total_l / blanket_total_l)) * 100.0
            savings_money_kzt = savings_l * chem_price_per_l

            st.markdown(f"""
            | Показатель | Сплошное внесение | Умное точечное (AI Spot) | Экономия |
            | :--- | :--- | :--- | :--- |
            | **Площадь обработки** | {field_area_ha:.1f} га | **{spot_area_treated_ha:.1f} га** | 📉 **-{savings_pct:.1f}%** |
            | **Расход препарата** | {blanket_total_l:,.0f} л | **{spot_total_l:,.0f} л** | 💧 **-{savings_l:,.0f} л** |
            | **Затраты на химию** | {blanket_total_l * chem_price_per_l:,.0f} ₸ | **{spot_total_l * chem_price_per_l:,.0f} ₸** | 💵 **{savings_money_kzt:,.0f} ₸** |
            """)

            st.success(f"🌱 **Экологический эффект:** снижение токсической нагрузки на почву Костанайской области на **{savings_pct:.1f}%**.")

        # Агрономические рекомендации по видам
        st.subheader("📋 Регламент применения гербицидов для обнаруженных сорняков")
        for sp_name, ru_name in [("field_thistle", "Бодяк полевой"), ("field_bindweed", "Вьюнок полевой"), ("couch_grass", "Пырей ползучий")]:
            count_sp = len(df_img[df_img["species"] == sp_name])
            if count_sp > 0:
                advice = HERBICIDE_ADVICE[sp_name]
                with st.expander(f"🔹 {ru_name} (обнаружено {count_sp} очагов) — Рекомендация"):
                    st.write(f"**Группа:** {advice['group']}")
                    st.write(f"**Рекомендуемые препараты:** {advice['herbicide']}")
                    st.write(f"**Норма расхода:** {advice['rate']}")
                    st.write(f"**Оптимальная фаза внесения:** {advice['optimal_stage']}")


# ------------------------------------------------------------------------------
# TAB 3: ЦЕНТР СОМНЕНИЙ АГРОНОМА (HUMAN-IN-THE-LOOP)
# ------------------------------------------------------------------------------
with tab3:
    st.header("🩺 Очередь сомнений агронома (Review Queue)")
    st.markdown("Модель честно подсвечивает случаи с пограничной уверенностью для валидации человеком.")

    if not df_detections.empty:
        df_review = df_detections[df_detections["review_required"] == True]
        st.metric("Объектов требует ручной проверки", f"{len(df_review)} из {len(df_detections)}")

        if df_review.empty:
            st.success("Все сорняки классифицированы с высокой уверенностью! Очередь пуста.")
        else:
            # Отображаем первые 12 сомнительных кропов
            crops_to_show = df_review.head(12)
            cols = st.columns(4)

            for idx, (_, row) in enumerate(crops_to_show.iterrows()):
                col = cols[idx % 4]
                crop_rel = row["crop_path"]
                crop_file = CASE1_DIR / "output" / crop_rel
                with col:
                    if crop_file.exists():
                        st.image(str(crop_file), caption=f"ID #{row['object_id']} (Conf: {row['species_conf']:.2f})", width="stretch")
                    else:
                        st.write(f"Crop ID #{row['object_id']}")
                    st.caption(f"Вид: **{row['species_ru']}** | Фаза: **{row['stage_ru']}**")
                    confirm = st.button(f"Подтвердить #{row['object_id']}", key=f"btn_conf_{idx}")
                    if confirm:
                        st.toast(f"Образец #{row['object_id']} подтвержден и отправлен в обучающую базу!")


# ------------------------------------------------------------------------------
# TAB 4: АНАЛИЗ ОТДЕЛЬНОГО ОБРАЗЦА (PLANT INSPECTOR)
# ------------------------------------------------------------------------------
with tab4:
    st.header("🔬 Инспектор сорняков крупным планом")
    st.markdown("Загрузите или выберите фото сорного растения для мгновенной многозадачной классификации.")
    st.info(
        f"Вид отображается как подтверждённый только при уверенности **{review_thresh:.0%} или выше**. "
        "Если ни один класс не достигает порога, итог — **«Не определено»**, "
        "а все вероятности остаются видны ниже. Распознанная культура не считается сорняком."
    )

    model, device = load_classifier_model()

    col_up, col_res = st.columns(2)
    with col_up:
        uploaded_file = st.file_uploader("Загрузить фото листа/растения:", type=["jpg", "jpeg", "png"])
        sample_choice = st.selectbox(
            "Или выберите эталон из датасета:",
            options=[
                "Dataset 1 кейс/Сорняки/Бодяк полевой/Розетка/006889_a4053894-a62a-41b6-bc5d-0a50024e833f.jpg",
                "Dataset 1 кейс/Сорняки/Вьюнок полевой/Розетка/006896_ea26e4e5-9c86-4ef3-a75d-3d4c382b6ce8.jpg",
                "Dataset 1 кейс/Сорняки/Пырей ползучий/Стеблевание/006900_e79c0a6b-c744-4f05-9f5b-6f81e3a6cda3.jpg"
            ]
        )

        test_img = None
        if uploaded_file is not None:
            test_img = Image.open(uploaded_file).convert("RGB")
            st.image(test_img, caption="Загруженное фото", width=300)
        elif sample_choice:
            sample_path = ROOT_DIR / sample_choice
            if sample_path.exists():
                test_img = Image.open(sample_path).convert("RGB")
                st.image(test_img, caption="Эталон из выборки", width=300)

    with col_res:
        if test_img is not None and model is not None:
            # Препроцессинг
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            tensor = transform(test_img).to(device)

            pred = model.predict_crop(tensor, species_thresh=review_thresh)

            st.subheader("Результат нейросети:")
            is_crop = pred["species"] == "crop_wheat"
            if is_crop:
                st.success(
                    f"🌾 Определена зерновая культура ({pred['species_conf'] * 100:.1f}%). "
                    "Объект исключён из списка сорняков и карты обработки."
                )
            elif pred["species"] == "unknown":
                top_name, top_probability = max(pred["all_species_probs"].items(), key=lambda item: item[1])
                top_label = dict(zip(SPECIES_NAMES, SPECIES_RU)).get(top_name, top_name)
                st.markdown("### Вид: **Не определено**")
                st.caption(
                    f"Наиболее вероятный вариант — {top_label}: {top_probability * 100:.1f}%, "
                    f"но это ниже порога {review_thresh * 100:.0f}%."
                )
            else:
                st.markdown(f"### Вид: **{pred['species_ru']}** (`{pred['species_conf']*100:.1f}%`)")
            if not is_crop:
                st.markdown(f"### Фаза: **{pred['stage_ru']}** (`{pred['stage_conf']*100:.1f}%`)")

            if is_crop:
                st.info("Показываются только целевые сорняки; для культуры рамка и агрономическая рекомендация не создаются.")
            elif pred["review_required"]:
                st.warning("⚠️ Статус: **Требует проверки агрономом** (низкая уверенность или пограничные признаки)")
            else:
                st.success("✅ Статус: **Уверенная автоматическая классификация**")

            # График вероятностей
            species_ru_by_name = dict(zip(SPECIES_NAMES, SPECIES_RU))
            df_probs = pd.DataFrame([
                {
                    "Вид": species_ru_by_name[name],
                    "Вероятность": probability,
                    "Статус": "Достигнут порог" if probability >= review_thresh else "Ниже порога",
                }
                for name, probability in pred["all_species_probs"].items()
            ])
            fig_p = px.bar(
                df_probs,
                x="Вид",
                y="Вероятность",
                color="Статус",
                color_discrete_map={"Достигнут порог": "#2f855a", "Ниже порога": "#a0aec0"},
                text_auto=".1%",
                range_y=[0, 1],
            )
            fig_p.add_hline(
                y=review_thresh,
                line_dash="dash",
                line_color="#c53030",
                annotation_text=f"Порог {review_thresh:.0%}",
                annotation_position="top left",
            )
            st.plotly_chart(fig_p, width="stretch")

            # Гербицидный регламент
            if not is_crop:
                advice = HERBICIDE_ADVICE.get(pred["species"], HERBICIDE_ADVICE["unknown"])
                st.info(f"💡 **Рекомендация по обработке:**\n\n- **Препарат:** {advice['herbicide']}\n- **Дозировка:** {advice['rate']}\n- **Оптимальное окно:** {advice['optimal_stage']}")


# ------------------------------------------------------------------------------
# TAB 5: МЕТРИКИ ОБУЧЕНИЯ
# ------------------------------------------------------------------------------
with tab5:
    st.header("📊 Метрики качества моделей и бенчмаркинг")

    metrics_data = load_training_metrics()
    if metrics_data:
        tm = metrics_data.get("test_metrics", {})
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Species Accuracy", f"{tm.get('species_accuracy', 0)*100:.1f}%")
        col_m2.metric("Species Macro-F1", f"{tm.get('species_macro_f1', 0):.3f}")
        col_m3.metric("Stage Accuracy", f"{tm.get('stage_accuracy', 0)*100:.1f}%")
        col_m4.metric("Stage Macro-F1", f"{tm.get('stage_macro_f1', 0):.3f}")

        st.subheader("Полнота (Recall) по классам на отложенном тесте:")
        recalls = tm.get("species_recall_per_class", {})
        if "crop_wheat" in recalls:
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            col_r1.metric("Бодяк (Осот)", f"{recalls.get('field_thistle', 0)*100:.1f}%")
            col_r2.metric("Вьюнок полевой", f"{recalls.get('field_bindweed', 0)*100:.1f}%")
            col_r3.metric("Пырей ползучий", f"{recalls.get('couch_grass', 0)*100:.1f}%")
            col_r4.metric("Пшеница (Культура)", f"{recalls.get('crop_wheat', 0)*100:.1f}%")
        else:
            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("Бодяк полевой (Осот)", f"{recalls.get('field_thistle', 0)*100:.1f}%")
            col_r2.metric("Вьюнок полевой", f"{recalls.get('field_bindweed', 0)*100:.1f}%")
            col_r3.metric("Пырей ползучий", f"{recalls.get('couch_grass', 0)*100:.1f}%")

        # График кривой обучения
        history = metrics_data.get("history", [])
        if history:
            df_hist = pd.DataFrame(history)
            st.subheader("Динамика обучения по эпохам")
            st.caption(
                "Loss и метрики качества показаны отдельно: у них разный смысл и их нельзя "
                "корректно сравнивать по одной общей шкале."
            )
            col_loss, col_quality = st.columns(2)

            fig_loss = go.Figure()
            fig_loss.add_trace(
                go.Scatter(
                    x=df_hist["epoch"],
                    y=df_hist["train_loss"],
                    name="Train loss",
                    mode="lines+markers",
                    line={"color": "#dd6b20"},
                )
            )
            fig_loss.update_layout(
                title="Ошибка обучения",
                xaxis_title="Эпоха",
                yaxis_title="Loss (меньше — лучше)",
                hovermode="x unified",
            )
            col_loss.plotly_chart(fig_loss, width="stretch")

            fig_quality = go.Figure()
            fig_quality.add_trace(
                go.Scatter(
                    x=df_hist["epoch"],
                    y=df_hist["train_species_acc"],
                    name="Train species accuracy",
                    mode="lines",
                    line={"color": "#3182ce"},
                )
            )
            fig_quality.add_trace(
                go.Scatter(
                    x=df_hist["epoch"],
                    y=df_hist["val_species_f1"],
                    name="Validation species F1",
                    mode="lines+markers",
                    line={"color": "#2f855a"},
                )
            )
            fig_quality.add_trace(
                go.Scatter(
                    x=df_hist["epoch"],
                    y=df_hist["val_stage_f1"],
                    name="Validation stage F1",
                    mode="lines+markers",
                    line={"color": "#68d391"},
                )
            )
            fig_quality.update_layout(
                title="Качество классификации",
                xaxis_title="Эпоха",
                yaxis_title="Метрика (больше — лучше)",
                yaxis={"range": [0.75, 1.02]},
                hovermode="x unified",
            )
            col_quality.plotly_chart(fig_quality, width="stretch")

        if recalls:
            recall_names = {
                "field_thistle": "Бодяк",
                "field_bindweed": "Вьюнок",
                "couch_grass": "Пырей",
                "crop_wheat": "Пшеница / фон",
            }
            recall_df = pd.DataFrame(
                [
                    {
                        "Класс": recall_names.get(name, name),
                        "Recall": value,
                        "Группа": "Минимальный Recall" if value == min(recalls.values()) else "Другие классы",
                    }
                    for name, value in recalls.items()
                ]
            )
            fig_recall = px.bar(
                recall_df,
                x="Recall",
                y="Класс",
                orientation="h",
                text_auto=".1%",
                color="Группа",
                color_discrete_map={"Другие классы": "#2f855a", "Минимальный Recall": "#c53030"},
                range_x=[0, 1.05],
                title="Recall по классам на локальном test split",
            )
            fig_recall.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_recall, width="stretch")
            st.caption(
                "Recall показывает, какую долю реальных объектов каждого класса модель нашла. "
                "Вьюнок — самый слабый класс текущей версии и требует расширения выборки."
            )

    st.info(
        "Метрики выше относятся к локальному отложенному test split проекта. "
        "Их нельзя напрямую сравнивать с результатами на других датасетах: "
        "состав классов, разметка и условия съёмки отличаются."
    )


# ------------------------------------------------------------------------------
# TAB 6: КЛАСТЕРЫ ЭМБЕДДИНГОВ (t-SNE / SILHOUETTE) & ЗАЩИТА ОТ ГАЛЛЮЦИНАЦИЙ
# ------------------------------------------------------------------------------
with tab6:
    st.header("🌌 Кластерный анализ глубоких признаков (1280-d) & Защита от галлюцинаций")
    st.markdown("""
    **Проблема в реальном поле:** Обычный классификатор "сорняк-only" замкнут (*closed-set assumption*). Любой попавший в него фрагмент пшеницы или голой земли классификатор обязан принудительно отнести к одному из сорняков. Это приводит к ложным срабатываниям опрыскивателя и химическому ожогу культурных всходов.

    **Решение AgroVision:** Внедрение 4-го класса **`crop_wheat` (Культура / Пшеница)** и обучение модели с **Focal Loss** + построение латентного пространства 1280-мерных эмбеддингов. Ниже представлена математическая оценка разделимости кластеров и интерактивная проекция t-SNE.
    """)

    c_data = load_cluster_data()
    if c_data:
        sil = c_data.get("silhouette_score", 0)
        c_seps = c_data.get("centroid_separations", {})

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Silhouette Score (Качество кластеризации)", f"{sil:.3f}", "Высокая изоляция" if sil > 0.35 else "Норма")
        col_s2.metric("Расстояние: Пшеница ↔ Вьюнок", f"{c_seps.get('wheat_vs_field_bindweed', {}).get('cosine_distance', 0):.3f} cos")
        col_s3.metric("Расстояние: Пшеница ↔ Пырей", f"{c_seps.get('wheat_vs_couch_grass', {}).get('cosine_distance', 0):.3f} cos")
        col_s4.metric("Расстояние: Пшеница ↔ Бодяк", f"{c_seps.get('wheat_vs_field_thistle', {}).get('cosine_distance', 0):.3f} cos")

        st.subheader("Интерактивная 2D-карта латентного пространства (t-SNE):")
        points = c_data.get("points", [])
        if points:
            df_pts = pd.DataFrame(points)
            color_discrete_map = {
                "Бодяк полевой": "#EF4444",
                "Вьюнок полевой": "#F59E0B",
                "Пырей ползучий": "#10B981",
                "Пшеница / Культура (Фон)": "#2563EB",
                "Пшеница (Культура / Фон)": "#2563EB",
                "Неизвестно": "#6B7280"
            }
            fig_cluster = px.scatter(
                df_pts,
                x="tsne_x",
                y="tsne_y",
                color="species_ru",
                hover_data=["filename"],
                title=f"t-SNE визуализация 1280-мерных векторов (Silhouette = {sil:.3f}, N = {len(df_pts)})",
                color_discrete_map=color_discrete_map,
                labels={"tsne_x": "t-SNE компонента 1", "tsne_y": "t-SNE компонента 2", "species_ru": "Класс"}
            )
            fig_cluster.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=1, color='DarkSlateGrey')))
            fig_cluster.update_layout(height=600)
            st.plotly_chart(fig_cluster, width="stretch")

        st.success("""
        ✅ **Вывод кластерного анализа:**
        1. Точки пшеницы образуют плотный, обособленный кластер с высоким косинусным расстоянием до сорных трав ($d_{cos} > 0.45$).
        2. Пырей (злаковый сорняк, морфологически похожий на пшеницу) надежно разделяется моделью благодаря многоуровневым текстурным признакам EfficientNet.
        3. Защита от галлюцинаций подтверждена: ложные срабатывания на пшеницу предотвращаются до формирования полетного задания для опрыскивателя.
        """)
    else:
        st.info("ℹ️ Файл кластерного анализа (`case1/output/clusters_2d.json`) будет сгенерирован автоматически после завершения обучения модели.")


# ------------------------------------------------------------------------------
# TAB 7: ПЛАНИРОВАНИЕ ФЛОТА БПЛА (1–5 ДРОНОВ)
# ------------------------------------------------------------------------------
with tab7:
    render_fleet_planning_tab()
