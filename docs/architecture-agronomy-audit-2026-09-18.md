# Архитектурный и агрономический аудит Case 1

Дата проверки: 2026-09-18. Scope: текущий checkout проекта и синтетический integration harness. Числа из мастер-отчёта не считаются подтверждёнными, если они не воспроизведены текущим прогоном или первичным источником.

## Executive verdict

Концепция промышленно разумна как **post-flight decision-support pipeline**: БПЛА → детекция/классификация → human-in-the-loop → карта-задание → импорт в Task Controller. Для автономного real-time spot spraying решение пока не готово. Главные блокеры: фактический checkpoint на 4 вида и 2 фазы вместо заявленных 26+3; отсутствие объектной геопривязки; демонстрационный, не XSD-валидированный ISOXML; возможная потеря частично не синхронизированных действий PWA; неподтверждённые экономические KPI.

## Что подтверждено в текущем checkout

- Реализованы API `/api/v1/review-queue`, `/api/v1/verify-item`, `/api/v1/sync`, `/api/v1/executive-stats` и PWA offline queue.
- Код имеет full-frame + tiled inference, raw-RGB ExG, IoU/IoM merge, species/stage heads и safety actions.
- Свежий реальный прогон одного DJI-кадра завершился: 37 объединённых объектов, 2.31 s detection stage на CPU, checkpoint сообщил 4 species classes и 2 stage classes.
- Текущий тестовый запуск: 48 passed, 1 failed. Падает contract test viewer pipeline: mock processor не принимает новый `detector_path` keyword.
- Синтетический harness прошёл contract checks: crop sprayed = 0, low-confidence auto-actioned = 0, offline queue after sync = 0, replay распознан как duplicate; GeoJSON читается, XML well-formed.

## Несостыковки и риски

| Приоритет | Наблюдение | Риск | Требуемое действие |
|---|---|---|---|
| P0 | Мастер-отчёт говорит 26 species + 3 stages; загруженный checkpoint фактически 4 + 2. 26 названий есть лишь в catalogue/map. | Жюри обнаружит подмену справочника фактическим покрытием модели. | Показывать `supported_by_model` и `catalog_only`; либо обучить/валидировать 26+3 checkpoint. |
| P0 | Один `drone_lat/lon` кадра копируется всем объектам. | Все prescription points кадра могут наложиться; пиксельный bbox не становится координатой растения. | Ортофото/DSM, camera intrinsics+pose, RTK/GCP, pixel-to-ground transform и измеренная CE90/RMSE. |
| P0 | `TASKDATA.XML` в dashboard — вручную собранный фрагмент; нет XSD validation/package references/DDI/device allocation. | Импорт TC может быть отвергнут или неверно интерпретировать норму. | Зафиксировать версию, валидировать официальными XSD и провести acceptance test на каждой целевой модели терминала. |
| P0 | PWA очищает весь `offlineActions` при любом HTTP `res.ok`; API возвращает HTTP 200 даже при `failed_count > 0`. | Частичный batch failure способен безвозвратно удалить неотправленные решения агронома. | ACK на каждый immutable event ID; удалять только подтверждённые IDs; retry/dead-letter queue. |
| P0 | Нет auth/RBAC/signature/audit revisions; JSON/CSV — shared mutable persistence. | Подмена решений, lost update, гонки при нескольких устройствах. | OIDC/device identity, append-only event log, optimistic version, transactional DB, immutable audit trail. |
| P1 | Инвариант отчёта — `<60%`; код по умолчанию использует 65%. | Не unsafe (65% консервативнее), но разные queue/coverage и неповторимость метрик. | Один versioned policy config; график selective risk/coverage для выбора порога. |
| P1 | ExG-комментарий обещает «100% обнаружение»; ExG выделяет также пшеницу и зависит от света. | Ложная гарантия и рост FP при смыкании рядов. | Переименовать в candidate generator; ROC/PR по lighting strata, auto white-balance lock/normalization. |
| P1 | Merge объединяет рамки по геометрии без семантики/масштаба и зависит от порядка. | Слипание соседних растений или раздробление одной розетки и неверная плотность. | Connected-components/weighted-box fusion с physical-size gates; объектная ground truth и count MAE. |
| P1 | `82.4%` и KZT savings в API hard-coded/эвристические. | Экономический тезис нельзя защищать как измеренный. | Расчёт из treated area, baseline rate, фактической цены, overlap, buffer и manual-review coverage; диапазон чувствительности. |
| P1 | 2.1 ms не воспроизведено текущим E2E: один кадр занял 2.31 s на CPU. | Смешение model-only latency и end-to-end wall time. | Указывать hardware, precision, batch, warm-up, image/tile count; p50/p95/p99 отдельно по стадиям. |
| P2 | Human-wins реализован как overwrite записи, но без server revision/vector clock. | Offline edits разных агрономов разрешаются неявно. | Deterministic policy: human > AI; затем role priority; затем server revision; конфликт сохранять, не тереть. |

## Соответствие промышленному Precision Ag

Архитектура соответствует типовой логике FMIS/Task Controller и human-in-the-loop, но статус следует формулировать так:

- **сейчас:** decision-support prototype, post-flight advisory mapping;
- **после XSD + terminal tests:** interoperable prescription export candidate;
- **после spatial calibration + field validation + functional safety:** production pilot;
- **не заявлять:** AEF-certified, ISO-conformant product, autonomous closed-loop spraying или гарантированную совместимость с Amazone/John Deere/Trimble.

Minimum production controls: dataset/model card, field-level split, calibration/ECE, OOD detector, signed model/rules versions, observability, retry/idempotency, geospatial uncertainty buffer, agronomist approval, rollback and as-applied reconciliation.

## E2E simulation harness

Скрипт: `scripts/e2e_pipeline_simulation.py`.

Он детерминированно проверяет:

1. RGB frame и synthetic truth;
2. overlapping 1024 px tiling;
3. ExG candidate generation;
4. detector fragments и micro-noise;
5. containment-aware IoM merge;
6. 27-way species space (26 weeds + crop), 3 stages и abstention `<0.60`;
7. crop `do_not_spray` invariant;
8. durable offline SQLite queue как контрактный аналог IndexedDB;
9. agronomist override;
10. idempotent batch sync с human-wins;
11. EPSG:4326 GeoJSON и well-formed ISOXML-like `TASKDATA.XML`;
12. four README plots, SHA-256 manifests and limitations.

Запуск:

```bash
.venv/bin/python scripts/e2e_pipeline_simulation.py --output /private/tmp/qostanai-e2e
```

Это **не** inference обученных весов и **не** источник полевых accuracy/latency. Такая маркировка встроена в JSON и watermark графиков.

## README evidence pack

### Обязательные графики

1. **Latency distribution:** stacked per-stage + total; warm/cold; p50/p95/p99; hardware/precision/tile count. Отдельно model-only и E2E.
2. **Species confusion matrix:** raw + row-normalized; 26 classes + wheat + unknown; support рядом с каждой строкой.
3. **Stage confusion matrix:** 3 stages, но только на истинных weed crops; не считать stage для wheat/background.
4. **One-vs-rest PR curves:** предпочтительнее ROC при редких классах. Отдельная ROC/PR для wheat safety.
5. **Risk-coverage curve:** error rate против доли auto-decisions при threshold sweep; отметить 0.60/0.65.
6. **SAHI/merge ablation:** full frame vs sliced; tile sizes; raw fragments → merged objects; AP_small, recall, count MAE, latency.
7. **Field heatmap:** reviewed prescription и uncertainty/manual-review layer; scale bar, CRS, CE90/RMSE, no-spray buffers.
8. **Economics waterfall:** total area → candidate area → reviewed spray area → litres avoided → KZT range; показать assumptions.

### Safety metrics

- false-spray-on-wheat rate и upper 95% confidence bound;
- manual-review coverage, review precision и median review time;
- field-level macro-F1/per-class recall; mAP50-95, AP_small;
- Expected Calibration Error/Brier score;
- geolocation error and nozzle-footprint overlap;
- sync loss/duplicate/conflict counts under airplane-mode and partial failure tests.

## Опасные edge cases

1. **Стерня, закат, длинные тени.** Raw ExG меняет распределение, стерня/блики дают FP. Парирование: exposure/white-balance policy, color constancy, illumination strata, ExG abstention, hard negatives; ночью/при flare — manual review.
2. **Смыкание рядов пшеницы.** ExG видит сплошную зелёную массу, IoM может склеить культуру и сорняк. Парирование: row model/segmentation, crop class safety gate, physical-size/row-alignment priors, запрет spray при смешанном crop probability.
3. **Ранний вьюнок под листьями культуры.** Малая площадь и сходная текстура дают FN/низкую confidence. Парирование: lower-altitude reflight, SAHI recall mode, temporal revisit, class-specific threshold и manual-review hotspot; отдельно контролировать recall вьюнка.
4. **GPS/pose jump или rolling shutter на развороте.** Верная классификация превращается в неверную точку обработки. Парирование: reject frames by GNSS/IMU quality and blur, orthorectification, time synchronization, uncertainty ellipse + no-spray buffer, контрольный импорт/as-applied comparison.

## Три тезиса для жюри

1. **Экономит не модель, а замкнутый доказуемый workflow.** БПЛА локализует неоднородность, edge processing сохраняет работу без связи, агроном снимает самые дорогие ошибки, а TC-GEO переносит подтверждённое решение в машину без ручной перерисовки.
2. **Мы оптимизируем стоимость ошибки, а не только mAP.** Пшеница — hard no-spray; низкая уверенность — abstain; rare weeds получают focal/MTL treatment. Поэтому система может увеличивать coverage постепенно, не превращая false positive в повреждение культуры.
3. **Offline-first — операционное преимущество поля.** Облако не является single point of failure: очередь, версии и idempotent sync позволяют завершить обход при отсутствии LTE. Экономию в миллионах тенге следует показывать сценарным расчётом по площади/цене/норме, а не hard-coded процентом.

## Научная база

Полная citation matrix и точные claim boundaries находятся в `docs/citation-matrix-research.md`. Кратко: ISO 11783-10/AEF обосновывают TC-GEO и prescription maps; Woebbecke — vegetation/background ExG; Akyon et al. — slicing; WeedSense — weed-specific MTL precedent; Lin et al. — focal mechanism. Ни один источник не подтверждает автоматически наши конкретные пороги, метрики или совместимость терминалов.
