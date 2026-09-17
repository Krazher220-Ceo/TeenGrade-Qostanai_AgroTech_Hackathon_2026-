# AgroVision AI — мониторинг сорняков и планирование флота БПЛА

AgroVision AI — исследовательский прототип для кейса №1 Qostanai AgroTech Hackathon 2026. Система принимает полевой RGB-снимок, обнаруживает объекты-кандидаты, классифицирует вид сорняка и фазу развития, направляет сомнительные случаи на ручную проверку, формирует CSV/JSON-сводки и рассчитывает офлайн-миссию для 1–5 БПЛА.

Проект предназначен для демонстрации воспроизводимого ML- и planning-конвейера. Он не является сертифицированным автопилотом, агрономическим назначением или разрешением на автоматическое опрыскивание.

## Быстрая оценка проекта проверяющим

Из корня репозитория:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r case1/requirements.txt
python case1_main.py status
python case1_main.py test
python case1_main.py dashboard
```

Откройте `http://localhost:8501` и начните с вкладки **«Проверить снимок поля»**:

1. загрузите JPG/PNG размером до 50 МБ;
2. нажмите **«Запустить распознавание»**;
3. посмотрите размеченное изображение, количество объектов и очередь ручной проверки;
4. скачайте таблицу CSV и полный JSON-отчёт;
5. откройте вкладку **«Планирование флота»** и рассчитайте миссию для 1–5 аппаратов.

Первый анализ большого DJI-кадра на CPU может занять несколько минут из-за тайлинга и покадровой классификации объектов.

Текущая демонстрационная сборка проверена локально на Apple Silicon Mac без облака: полевой YOLO-детектор запускается на CPU, классификатор использует Apple MPS при наличии и иначе переключается на CPU. Сборка с NVIDIA GPU ожидаемо ускорит инференс; дообучение на большей и разнообразной разметке может улучшить качество, но это необходимо подтвердить отдельным тестом на новых полях.

## Реалистичный вариант дрона для MVP

**Рекомендуемый бюджетный носитель камеры — б/у DJI Mini 2 SE.** По объявлениям вторичного рынка на 17 сентября 2026 года рабочие комплекты встречаются примерно за **120 000–150 000 ₸**. Это не фиксированная цена: она зависит от состояния, количества аккумуляторов и комплектации. У модели есть GPS/GLONASS/Galileo, 12-мегапиксельная камера, видео 2.7K, трёхосевой подвес и паспортное время полёта до 31 минуты.

| Вариант | Примерная цена | Роль в проекте |
|---|---:|---|
| Б/у DJI Mini 2 SE | 120 000–150 000 ₸ | Рекомендуемый доступный носитель камеры для пилотной съёмки |
| Б/у DJI Spark | 75 000–105 000 ₸ | Запасной устаревший вариант; паспортный полёт около 16 минут |
| DJI Mavic 3E | примерно от 2 млн ₸ | Профессиональный референс; не входит в бюджетный MVP |

Проверять предложения можно на [OLX.kz](https://www.olx.kz/elektronika/foto-video/q-%D0%B1-%D1%83-dji-mini-se/) и [Kaspi Объявлениях](https://obyavleniya.kaspi.kz/elektronika/fototehnika/kvadrokoptery/k--%D0%B4%D1%80%D0%BE%D0%BD%D1%8B/). Технические характеристики Mini 2 SE сверяются по [официальной странице DJI](https://www.dji.com/mini-2-se/specs). Перед покупкой б/у аппарата нужно проверить аккумуляторы и число циклов, GPS/RTH, камеру и подвес, пульт, зарядку, историю падений и выполнить тестовый полёт.

DJI Mavic 3E остаётся в планировщике только как оптический профиль для сравнения покрытия. Наличие такого профиля не означает, что проект требует покупки этой модели.

## Проверенный статус

Состояние зафиксировано 17 сентября 2026 года.

| Контур | Статус | Фактическое подтверждение |
|---|---|---|
| Python-окружение | Проверено | Python 3.11.16; `pip check`: конфликтов зависимостей нет |
| Автоматические тесты | Проверено | 37/37 тестов проходят; остаются 4 предупреждения о deprecated API |
| Синтаксис Python | Проверено | `compileall` для `case1`, `case1_main.py` и скрипта WeedBlaster |
| Детектор WeedBlaster | Проверено | Checkpoint 22 600 106 байт, SHA-256 совпадает, класс `Weed` загружается |
| Классификатор вида/фазы | Проверено | Оба checkpoint загружаются; forward и feature extraction проходят |
| Обработка 5 DJI-снимков | Проверено | 660 строк CSV = 660 объектов JSON; 5 размеченных изображений |
| Загрузка нового фото в Streamlit | Проверено | Реальный прогон: 4096×3072, 84 объекта, размеченный JPG, CSV и JSON |
| FastAPI | Проверено | GET `/health`, GET `/model-info`, POST `/classify`, POST `/process-batch` |
| Streamlit | Проверено | Сервер запущен локально, HTTP `200 OK`, восемь вкладок |
| Планировщик 1–5 БПЛА | Проверено расчётно | JSON, GeoJSON и SQLite; геометрия, partitioning и safety-тесты |
| Одиночный PX4/Gazebo SITL | Проверено локально | QGroundControl heartbeat, x500 взлетел примерно на 2,23 м, сел и разоружился |
| Пять PX4/Gazebo аппаратов | Не проверено | Не выполнен синхронный multi-vehicle прогон с пятью ID/портами |
| Реальный групповой полёт | Не реализован | Нужны испытания, связь, failsafe, разрешения и отдельный safety case |

## Основной пользовательский сценарий

### Один снимок поля → готовая сводка

Первая вкладка Streamlit реализует сценарий, который проверяющий выполняет без командной строки:

```text
JPG/PNG
  → проверка типа и размера
  → тайлинг 1280×1280 с overlap 20%
  → YOLOv8 Weed detection
  → Global NMS, IoU 0.45
  → crop каждого объекта
  → EfficientNet-B0: вид + фаза
  → human-in-the-loop safety gate: вид подтверждается только при confidence ≥ 65%
  → crop_wheat исключается из рамок и карты: показываются только сорняки и сомнительные объекты
  → 1) исходный кадр
  → 2) кадр с рамками кандидатов YOLO
  → 3) кадр с подписями вида/фазы от EfficientNet
  → crop-галерея + CSV + JSON
```

Каждый запуск получает стабильный ID по SHA-256 загруженного файла и сохраняется в `case1/output/viewer_runs/<run_id>/`.

| Результат | Содержимое | Назначение |
|---|---|---|
| `input.*` | Исходный загруженный кадр | Сравнение с результатом |
| `detector_boxes/detector_boxes_input.*` | Все рамки кандидатов без названия вида | Проверка работы YOLO до классификации |
| `annotated/annotated_input.*` | Оставшиеся рамки с видом, фазой, уверенностью и флагом проверки | Проверка второй модели и отсева культуры |
| `crops/input/` | Отдельные мелкие вырезки каждой рамки | Прозрачность классификации EfficientNet |
| `all_fields_detections.csv` | Одна строка на обнаруженный объект | Excel, BI, агрегация |
| `all_fields_report.json` | Сводка снимка и вложенные детекции | API, интеграция, аудит |
| `crops/input/` | Вырезки найденных объектов | Ручная проверка и доразметка |

Даже при нуле детекций CSV создаётся с полным заголовком, а JSON содержит сводку обработанного кадра.

## Что показывает Streamlit

Dashboard содержит восемь вкладок.

| Вкладка | Что проверяется |
|---|---|
| Проверить снимок поля | Новый кадр, полный inference, annotated image, CSV/JSON download |
| Карта поля и детекции | Рамки, confidence-фильтры, распределение видов и фаз |
| Тепловая карта и рецепт | Пространственная плотность и расчётный сценарий обработки |
| Центр сомнений | Объекты с `review_required=true` |
| Анализ образца | Классификация крупного плана растения |
| Метрики обучения | Accuracy, Macro-F1, Recall и история обучения |
| Кластеры эмбеддингов | t-SNE/Silhouette и разделимость классов |
| Планирование флота | Полосы, площадки, KPI, safety-отчёт, JSON/GeoJSON |

Агрономические рекомендации — демонстрационные правила. Они не заменяют этикетку препарата, локальный регламент и решение агронома.

## Архитектура

```text
Полевой снимок
    ├── YOLOv8 + tiling + NMS → bounding boxes + crops
    ├── EfficientNet-B0 multi-task → species + stage + uncertainty
    ├── Reporting → annotated image + CSV + JSON
    └── Fleet planning → lanes + partition + safety + exports
```

| Слой | Основные файлы | Ответственность |
|---|---|---|
| CLI | `case1_main.py` | status, audit, process, training, tests, dashboard, fleet-plan |
| Viewer workflow | `case1/viewer_pipeline.py` | Валидация загрузки и сбор артефактов |
| Web UI | `case1/dashboard/app.py` | Streamlit и загрузка нового снимка |
| ML | `case1/ml/` | Dataset, model, training, clustering, detector preparation |
| API | `case1/server/` | FastAPI classification endpoints |
| Fleet | `case1/fleet/` | Геометрия, coverage, partitioning, safety, exporters, SITL |
| Tests | `case1/tests/` | 34 автоматические проверки |

## Данные и воспроизводимость

| Набор | Количество | Назначение | Ограничение |
|---|---:|---|---|
| Эталонные фото сорняков | 552 | Вид и фаза | Крупный план, не UAV detection ground truth |
| DJI-снимки полей | 5 | End-to-end demo | Нет ручных bounding boxes и истинных видов объектов |
| `manifest.csv` | 552 | Исходный classification manifest | Только три сорняка |
| `manifest_v2.csv` | 897 | Итоговый multi-task manifest | Добавлены 345 crop/background примеров |

Проверены все пути обоих manifest: отсутствующих файлов нет. Все 552 эталонных изображения уникальны по SHA-256.

### Распределение `manifest_v2.csv`

| Split | Объектов |
|---|---:|
| Train | 610 |
| Validation | 134 |
| Test | 153 |
| Итого | 897 |

| Класс | Объектов |
|---|---:|
| Бодяк полевой | 215 |
| Вьюнок полевой | 222 |
| Пырей ползучий | 115 |
| Пшеница / фон | 345 |

| Источник | Объектов |
|---|---:|
| Reference camera | 552 |
| Aerial crops | 245 |
| DJI field crops | 100 |

В данных отсутствует фаза «Розетка» для пырея. Модель принудительно не выдаёт эту неподтверждённую комбинацию.

## Модели

### Основной demo-path

| Этап | Модель | Роль |
|---|---|---|
| Детекция | WeedBlaster YOLOv8s | Восемь культур + общий `Weed`; pipeline использует только `Weed` |
| Классификация | EfficientNet-B0 multi-task | Четыре species-класса и фаза |
| Safety gate | Confidence thresholds + rules | `unknown`, `manual_review`, `do_not_spray` |

**Статус: прототип, идёт сбор и очистка локального датасета.** Открытая WeedBlaster Vision сейчас используется как временный baseline-заглушка для проверки полного конвейера, а не как готовая промышленная модель. Она позволяет воспроизводимо продемонстрировать путь «снимок → кандидаты → классификация → карта», но её результаты нельзя использовать как самостоятельное основание для автоматического опрыскивания.

Open source здесь принципиален: веса и код можно проверить, воспроизвести, аудировать и затем заменить checkpoint собственной моделью, дообученной на размеченных кадрах целевых полей. Class ID не зашит в код: pipeline находит `Weed` по имени.

### Экспериментальный aerial detector

`case1/models/weed_detector_finetuned.pt` загружается и содержит классы `crop`/`weed`. В сохранённом пятиэпоховом run:

| Метрика последней эпохи | Значение |
|---|---:|
| Precision(B) | 0,627 |
| Recall(B) | 0,666 |
| mAP50(B) | 0,637 |
| mAP50–95(B) | 0,327 |

Полевой smoke-test не подтвердил пригодность этой версии как default: 464 candidate-boxes были классифицированы downstream-моделью как 457 `crop_wheat` и 7 `unknown`, без уверенного целевого сорняка. Поэтому checkpoint доступен только явно:

```bash
python case1_main.py process --detector finetuned --output /tmp/agrovision-finetuned
```

Это сигнал domain shift/false positives, а не повод выбирать модель по имени checkpoint.

## Метрики классификатора

Источник: `case1/output/classifier_training_metrics.json`, локальный test split из 153 объектов.

| Метрика | Значение |
|---|---:|
| Species Accuracy | 0,915 |
| Species Macro-F1 | 0,902 |
| Stage Accuracy | 1,000 |
| Stage Macro-F1 | 1,000 |

| Класс | Recall |
|---|---:|
| Бодяк полевой | 0,971 |
| Вьюнок полевой | 0,677 |
| Пырей ползучий | 1,000 |
| Пшеница / фон | 0,985 |

### Графики обучения и качества

![Кривые обучения EfficientNet-B0](docs/assets/training_curves.png)

Слева видно снижение `train loss` по 35 эпохам. Справа отдельно показаны accuracy на обучающей выборке и F1 на validation, поэтому разные по смыслу метрики не смешиваются с loss на одной шкале.

![Recall по классам](docs/assets/class_recall.png)

Красным выделен класс с минимальным Recall. В текущем локальном тесте это вьюнок: модель находит 67,7% его примеров, поэтому именно этот класс требует расширения данных и дополнительной проверки.

Оба графика воспроизводятся из JSON без ручного ввода значений:

```bash
python case1/ml/generate_readme_charts.py
```

Stage-метрика `1,000` относится только к текущему локальному split и упрощённой разметке. Она не доказывает качество на другом поле, высоте или камере.

## Фактический прогон пяти DJI-снимков

Источники: `case1/output/all_fields_report.json` и `case1/output/all_fields_detections.csv` после повторного запуска default pipeline.

| Снимок | Candidate detections | Manual review | Detector time, с | Avg species confidence |
|---|---:|---:|---:|---:|
| `DJI_20260609122641_0177_D.JPG` | 84 | 21 | 6,70 | 0,649 |
| `DJI_20260615115515_0164_D.JPG` | 231 | 12 | 5,54 | 0,711 |
| `DJI_20260615134123_0339_D.JPG` | 93 | 15 | 5,72 | 0,649 |
| `DJI_20260615134520_0343_D.JPG` | 221 | 30 | 5,46 | 0,688 |
| `DJI_20260615134710_0347_D.JPG` | 31 | 6 | 5,33 | 0,663 |
| **Итого** | **660** | **84** | **28,75** | — |

В текущем CSV: 558 `crop_wheat`, 63 `couch_grass`, 5 `field_bindweed`, 1 `field_thistle`, 33 `unknown`. `review_required=true` у 84 объектов, то есть 12,7%.

![Результаты полевого demo-run](docs/assets/field_demo_overview.png)

В рабочем интерфейсе класс `crop_wheat` используется как защитный фон и не показывается рамками. На карте и в выгрузках остаются только распознанные сорняки и объекты, требующие ручной проверки.

## Форматы сводок

### CSV: одна строка на объект

| Группа | Поля |
|---|---|
| Идентификация | `image_id`, `object_id` |
| Детектор | `detector_conf`, `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`, `bbox_width`, `bbox_height` |
| Вид | `species`, `species_ru`, `species_conf` |
| Фаза | `stage`, `stage_ru`, `stage_conf` |
| Safety | `spray_action`, `review_required` |
| Метаданные | `drone_lat`, `drone_lon`, `drone_abs_alt`, `drone_rel_alt`, `timestamp` |
| Аудит | `crop_path` |

### JSON: одна запись на снимок

| Поле | Значение |
|---|---|
| `filename` | Имя кадра |
| `time_s` | Detector inference без полного UI overhead |
| `total_weeds` | Candidate boxes класса Weed после NMS |
| `review_required_count` | Объекты для ручной проверки |
| `counts_by_species`, `counts_by_stage` | Агрегированные сводки |
| `avg_species_conf` | Средняя уверенность классификатора |
| `annotated_image` | Имя размеченного кадра |
| `detections` | Полный список объектов |

## Планировщик флота

### Алгоритм

1. WGS84-полигон переводится в локальные метры через `pyproj`.
2. Из рабочей области вычитаются exclusion zones.
3. Рассчитываются footprint, GSD, шаг полос и trigger interval.
4. Проверяются кандидаты угла покрытия.
5. Полосы пересекаются с допустимой геометрией.
6. Dynamic programming распределяет смежные полосы между 1–5 аппаратами.
7. Назначаются уникальные `system_id`, UDP-порт и площадка.
8. Выполняется статическая safety-валидация.
9. План экспортируется в JSON, GeoJSON, SQLite и QGroundControl `.plan`.

### KPI демонстрационного поля

Параметры: высота 30 м, скорость 5 м/с, пять аппаратов.

| KPI | Значение | Тип доказательства |
|---|---:|---|
| Рабочая площадь | 23,4 га | Расчёт геометрии |
| Покрытие | 23,3 га / 99,5% | Расчёт геометрии |
| Полосы | 41 | Планировщик |
| Время флота | 1044 с / 17,4 мин | Модель миссии |
| Время одного аппарата | 4848,7 с / 80,8 мин | Модель сравнения |
| Ускорение | 4,64× | Расчётное, не flight benchmark |
| GSD | 1,20 см/px | Оптическая модель |
| Межполосный шаг | 14,4 м | Оптическая модель |
| Trigger interval | 1,44 с | Оптическая модель |

| Аппарат | Полос | Длина, м | Время, мин | Расчёт батареи | Площадка |
|---|---:|---:|---:|---:|---|
| Drone 1 | 9 | 4427 | 17,4 | 69,6% | P1 |
| Drone 2 | 8 | 3398 | 16,4 | 65,6% | P2 |
| Drone 3 | 7 | 1342 | 15,4 | 61,7% | P3 |
| Drone 4 | 8 | 2603 | 16,4 | 65,8% | P4 |
| Drone 5 | 9 | 4427 | 17,4 | 69,6% | P5 |

`99,5%` — покрытие полосами в 2D-модели, а не процент уничтоженных сорняков и не доказательство безопасного группового полёта.

## PX4/Gazebo/QGroundControl

На тестовом Apple Silicon Mac проверены QGroundControl 5.1.4, PX4 commit `fe87271`, Gazebo Sim 8.15.0 и один x500: heartbeat, взлёт примерно на 2,23 м, посадка, разоружение и `.ulg`-лог размером 26 675 364 байта.

```bash
cd /path/to/PX4-Autopilot
make px4_sitl gz_x500
```

В `pxh>`:

```bash
mavlink status
commander takeoff
listener vehicle_local_position -n 1
commander land
```

Загрузка проектного `.plan` и одновременный запуск пяти экземпляров ещё не проверены. Для документированного PX4 multi-vehicle Gazebo-сценария используйте Ubuntu и актуальную инструкцию PX4.

## API

```bash
uvicorn case1.server.app:app --host 127.0.0.1 --port 8000
```

| Метод | Endpoint | Назначение |
|---|---|---|
| GET | `/health` | Состояние модели и device |
| GET | `/model-info` | Классы, фазы и ограничения |
| POST | `/classify` | Один crop через Base64 или путь |
| POST | `/process-batch` | Batch классификации crops |

FastAPI классифицирует готовые crops. Полный detector pipeline нового кадра реализован в Streamlit/CLI.

## Автоматические тесты

```bash
python case1_main.py test
# или
python -m pytest -q case1/tests
```

| Файл | Тестов | Покрываемый риск |
|---|---:|---|
| `test_agronomy_rules.py` | 2 | Thresholds и фазы |
| `test_coverage_planner.py` | 3 | Покрытие, угол, exclusion zone |
| `test_fleet_exporters.py` | 2 | JSON/GeoJSON и SQLite |
| `test_fleet_geometry.py` | 6 | Оптика, координаты, polygons |
| `test_fleet_partitioning.py` | 2 | Разбиение 1–5 и малое поле |
| `test_fleet_safety.py` | 5 | ID, площадки, высота, state machine |
| `test_sitl_adapter.py` | 2 | QGC `.plan` и simulation pack |
| `test_smoke.py` | 8 | Веса, модели, API, артефакты |
| `test_viewer_pipeline.py` | 4 | Upload validation и CSV/JSON/annotated |
| **Всего** | **34** | **34 passed** |

Известны четыре предупреждения: `FastAPI.on_event`, AnyIO alias и совместимость Starlette TestClient/httpx. Они не ломают текущий прогон, но требуют миграции на lifespan и актуальный test client.

## Аудит файлов

| Проверка | Результат |
|---|---|
| JSON/YAML parse | Успешно |
| CSV shape | Согласованное число колонок |
| SQLite integrity | `ok` |
| DOCX container | Корректный ZIP-контейнер |
| PDF | 5 страниц, не зашифрован |
| Изображения | 5 801 корректно прочитано; 9 LFS pointers ниже |
| Checkpoints | 4 YOLO + 2 classifier загружаются |

Девять файлов в `weedblaster-vision-yolov8s/results/` — 131-байтные Git LFS pointers, а не изображения:

- `F1_curve.png`, `PR_curve.png`, `P_curve.png`, `R_curve.png`;
- `confusion_matrix.png`, `confusion_matrix_normalized.png`, `results.png`;
- `val_batch0_labels.jpg`, `val_batch0_pred.jpg`.

Они не участвуют в inference. Перед публикацией выполните `git lfs pull` либо исключите pointers и не показывайте их как локальные графики.

## Принципы разработки

| Принцип | Как применён |
|---|---|
| Evidence before claims | Метрики только из JSON/CSV/test logs |
| TDD | Geometry, partitioning, safety, exports и upload adapter покрыты тестами |
| Human-in-the-loop | Низкая уверенность → `manual_review` |
| Offline-first | Fleet plan и результаты сохраняются локально |
| Reproducibility | Команды, manifests и checkpoints зафиксированы |
| Domain-shift awareness | Fine-tuned detector не стал default после слабого smoke-test |
| Honest scope | Расчётный KPI не называется полевым KPI |

## Roadmap

### Этап 0 — выполнено

- [x] аудит данных и checkpoints;
- [x] detector + multi-task classifier;
- [x] обработка пяти DJI-снимков;
- [x] Streamlit upload → annotated + CSV + JSON;
- [x] планировщик 1–5 аппаратов;
- [x] 37 тестов;
- [x] одиночный PX4/Gazebo smoke-flight.

### Этап 1 — до отбора во второй этап

- [ ] clean clone/install test на отдельной машине;
- [ ] standalone packaging `case1` или публикация всего runtime root;
- [ ] `LICENSE` и `THIRD_PARTY_NOTICES`;
- [ ] восстановить/удалить 9 LFS pointers;
- [ ] видео: upload → annotated → CSV/JSON → fleet plan;
- [ ] вручную разметить DJI holdout для field precision/recall;
- [ ] откалибровать confidence/review thresholds.

### Этап 2 — подтверждение ML

- [ ] UAV dataset с совпадающими культурой, высотой и видами;
- [ ] group-aware split по полю/дню;
- [ ] сравнить baseline, fine-tuned и segmentation baseline;
- [ ] mAP50–95, per-class metrics, false spray on crop;
- [ ] calibration error и latency CPU/MPS/GPU;
- [ ] dataset manifest, hashes и licenses.

### Этап 3 — подтверждение миссии

- [ ] импорт одного `.plan` в QGroundControl;
- [ ] полная миссия с `.ulg`;
- [ ] 2, затем 5 SITL-аппаратов;
- [ ] link-loss, low-battery, RTL и geofence;
- [ ] minimum separation по telemetry.

### Этап 4 — полевой пилот

- [ ] экспертная разметка и protocol approval;
- [ ] shadow mode без опрыскивания;
- [ ] controlled plot с ручным подтверждением;
- [ ] журналирование и rollback;
- [ ] отдельный aviation/agronomy safety case.

## KPI следующего этапа

| KPI | Сейчас | Следующее доказательство |
|---|---:|---|
| Classifier Macro-F1 | 0,902, local split | Независимое поле/день и CI |
| Вьюнок Recall | 0,677 | Независимый holdout |
| Field detector mAP | Не измерено | Ручная bbox-разметка DJI |
| False spray on crop | Не измерено end-to-end | Отдельный safety KPI |
| Manual review rate | 12,7% demo-run | Калибровка после ground truth |
| Fleet coverage | 99,5% расчётно | Сравнение с SITL track |
| Five-vehicle completion | Не проверено | 5/5 mission + land + logs |
| Minimum separation | Не моделируется | Telemetry > safety threshold |

## Демонстрация для жюри

| Время | Действие | Что доказывает |
|---:|---|---|
| 0:00–0:20 | Открыть вкладку загрузки | Понятный вход |
| 0:20–0:55 | Загрузить DJI-снимок | End-to-end inference |
| 0:55–1:20 | Annotated image и review queue | Объяснимость |
| 1:20–1:35 | Скачать CSV и JSON | Проверяемые артефакты |
| 1:35–2:10 | Рассчитать 5 БПЛА | Coverage и partitioning |
| 2:10–2:35 | Показать KPI | Измеримый расчётный эффект |
| 2:35–3:00 | Тесты и ограничения | Инженерная зрелость |

## Структура и публикация

```text
repository-root/
├── case1_main.py
├── weedblaster-vision-yolov8s/best.pt
└── case1/
    ├── README.md
    ├── dashboard/
    ├── fleet/
    ├── ml/
    ├── models/
    ├── output/
    ├── server/
    ├── tests/
    └── viewer_pipeline.py
```

Главный README находится в корне репозитория рядом с `case1_main.py`, а код приложения — в каталоге `case1/`. Для воспроизводимого запуска публикуйте корень репозитория целиком: runtime также использует baseline checkpoint `weedblaster-vision-yolov8s/best.pt` уровнем выше пакета `case1`.

## Основные команды

```bash
python case1_main.py status
python case1_main.py audit
python case1_main.py test
python case1_main.py process --detector baseline --output case1/output
python case1_main.py process --image /path/to/field.jpg --output /tmp/agrovision-one
python case1_main.py process --detector finetuned --output /tmp/agrovision-finetuned
python case1_main.py dashboard --port 8501
uvicorn case1.server.app:app --host 127.0.0.1 --port 8000
python case1_main.py fleet-plan --drones 5 --altitude 30 --speed 5 --output /tmp/agrovision-fleet
```

## Лицензии и безопасность

- WeedBlaster заявлен как AGPL-3.0; проверьте совместимость перед публикацией и коммерческим использованием.
- Условия datasets должны храниться вместе с URL, версией и hash.
- `human_confirmation_required` остаётся обязательным.
- Реальный многодроновый полёт требует отдельного допуска.
- Agronomy output — decision support, а не рецепт без специалиста.

Главный критерий: **каждая цифра связана с воспроизводимым файлом, тестом или логом; всё остальное явно помечено как план или гипотеза**.
