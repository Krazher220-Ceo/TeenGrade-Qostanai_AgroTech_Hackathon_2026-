# Citation matrix: ISO 11783, ExG, SAHI, Multi-Task Learning и Focal Loss

Дата проверки источников: 2026-09-18. В матрицу включены только первичные или принадлежащие организации-владельцу источники: ISO, AEF, официальный ISOBUS Data Dictionary, оригинальные статьи/материалы конференций и официальные репозитории авторов.

## Краткая матрица для защиты

| Компонент проекта | Что первичный источник действительно подтверждает | Что допустимо сказать жюри | Что источник **не** доказывает |
|---|---|---|---|
| ISO 11783-10 / ISOBUS Task Controller | ISO 11783-10:2015 определяет прикладной уровень Task Controller, обмен TC–ECU, формат обмена с farm-management computer, необходимые вычисления управления и формат сообщений к control function. AEF отдельно описывает TC-GEO: location-dependent documentation и variable-rate prescription maps для management zones. | «Экспортируем задание в экосистему ISO 11783-10/ISO-XML и проектируем его для передачи prescription map в TC-GEO». | Наличие файла с именем `TASKDATA.XML` само по себе не подтверждает совместимость с конкретным терминалом Amazone/John Deere/Trimble. Нужны XSD-валидация, корректные DDI/единицы/масштабы и испытание импорта на целевом TC; AEF-сертификацию продукта нельзя заявлять без сертификации. |
| TASKDATA.XML | Официальный ISOBUS Data Dictionary публикует XSD основного task-data file, включая `ISO11783_TaskFile_V4-3.xsd`, а также Common, ExternalFile, LinkListFile и TimeLog schemas. | «Генерируем ISOXML task-data package и валидируем основной XML и связанные файлы по опубликованным схемам соответствующей версии». | GeoJSON или произвольный XML, просто переименованный в `TASKDATA.XML`, не является ISOXML. XSD-valid также не гарантирует семантическую или машинную совместимость. |
| ExG / `2G - R - B` | Woebbecke et al. (1995) исследовали цветовые индексы для разделения растительного материала и почвы/остатков. Индекс `2g-r-b`, modified hue и green chromatic coordinate лучше других рассмотренных индексов отделяли растения от нерастительного фона при уровне значимости 0.05; работа сообщает работоспособность при затенённых и незатенённых солнечных условиях и связь со spot spraying. | «ExG — быстрый RGB-признак растительность/фон, исторически обоснованный для weed sensing и spot spraying; используем его как предфильтр». | Он не отличает пшеницу от сорняка, не определяет вид/фазу и не обосновывает универсальный порог `>30`. Порог обязан быть откалиброван для конкретной камеры, экспозиции, цветового пространства и полевых условий. В статье формула записана для chromatic coordinates (`2g-r-b`); raw-RGB `2G-R-B` — реализация, которую следует валидировать отдельно. |
| SAHI | Akyon, Altinuc, Temizel предлагают slicing-aided inference/fine-tuning: изображение режется на перекрывающиеся патчи, предсказания агрегируются. На VisDrone и xView авторы показывают прирост AP для конкретных детекторов/настроек и публикуют открытый фреймворк. | «SAHI — воспроизводимый способ повысить пиксельный масштаб мелких объектов на больших аэрофото; используем перекрывающийся тайлинг и объединяем предсказания в глобальных координатах». | Публикация не доказывает, что размер 1024 px оптимален для сорняков, не подтверждает наши mAP/latency и не описывает наш Containment-Aware IoM merge. Эти параметры и merging должны подтверждаться собственной абляцией. |
| Weed-specific Multi-Task Learning | WeedSense (ICCV Workshops 2025) совместно решает semantic segmentation, height estimation и growth-stage classification на данных 16 видов/11 недель и использует общие признаки с task-specific heads. Это прямой пример MTL для анализа сорняков и фаз роста. | «Совместное обучение взаимосвязанных признаков сорняка и развития имеет прямой precedent в peer-reviewed weed vision; общая backbone + отдельные heads — научно разумная архитектура». | WeedSense не является прямой валидацией нашей пары heads «26 species + 3 phases», EfficientNet-B0, Focal Loss или работы на полях Костанайской области. Его данные собраны в контролируемых условиях, а stage labels связаны с неделями/развитием; переносимость требуется доказать независимым field hold-out. |
| Focal Loss | Lin et al. вводят focal loss как модификацию cross-entropy, уменьшающую вклад хорошо классифицированных примеров и фокусирующую обучение на трудных примерах; исходная демонстрация выполнена на сильно несбалансированной foreground/background задаче dense object detection с RetinaNet. | «Focal Loss — обоснованный механизм снижения доминирования лёгких/частых примеров; `gamma=2` — известная исходная конфигурация, но её эффект мы подтверждаем абляцией на собственных классах». | Статья не доказывает автоматически улучшение 26-классовой species classification и не выбирает `gamma=2` за нас для этого датасета. Нужны сравнения CE vs weighted CE vs focal, per-class recall/F1, calibration и confidence-threshold analysis. |

## Источники и точные границы цитирования

### 1. ISO 11783-10, TC-GEO и ISOXML

1. **ISO 11783-10:2015, Part 10: Task controller and management information system data interchange.** Официальная карточка ISO указывает, что стандарт описывает application layer Task Controller, требования и сервисы связи TC с ECU, а также формат данных для farm-management computer, расчёты управления и сообщения control function. Версия 2015 года повторно подтверждена ISO в 2026 году и остаётся текущей на дату проверки.  
   Официальный источник: [ISO 11783-10:2015](https://www.iso.org/standard/61581.html)

2. **AEF ISOBUS functionality overview.** AEF разделяет Task Controller на TC-BAS, TC-GEO и TC-SC. Для TC-GEO прямо указаны location-dependent documentation/as-applied map и использование variable-rate prescription map для management zones с разными application rates.  
   Официальный источник: [AEF ISOBUS hand fan, pp. 5–6](https://www.aef-online.org/fileadmin/MEDIA/downloads/2020/AEF_Handfan_en_02.pdf)

3. **Официальные XSD ISOXML.** Раздел Supporting Documents ISOBUS Data Dictionary публикует схемы `ISO11783_TaskFile_V2-1.xsd`, `V3-3.xsd`, `V4-3.xsd` и связанные схемы версии 4.3. Это надёжная точка для машинной XSD-валидации экспортируемого набора.  
   Официальный источник: [ISOBUS Data Dictionary — Supporting Documents](https://www.isobus.net/isobus/file/supportingDocuments)

Практический критерий готовности экспорта: фиксировать целевую версию ISOXML; валидировать XML по соответствующему XSD; проверять ссылки ID/IDREF, геометрию/систему координат, treatment zones/grid, DDI, units/scales и device allocation; затем импортировать пакет на реальный или вендорский тестовый TC. Формулировка «совместимо с Amazone, John Deere и Trimble» до этих испытаний должна быть заменена на «предназначено для ISOXML-совместимых TC; совместимость конкретных моделей требует acceptance test».

### 2. Excess Green (ExG)

**Woebbecke, D. M.; Meyer, G. E.; Von Bargen, K.; Mortensen, D. A. (1995). “Color indices for weed identification under various soil, residue, and lighting conditions.” Transactions of the ASAE, 38(1), 259–269. DOI: 10.13031/2013.27838.** Авторы сравнивали индексы chromatic coordinates; `2g-r-b` входил в лучшие рассмотренные способы отделения живого растения от почвы/остатков и рассматривался в контексте sensing для spot spraying.  
Первоисточник/DOI: [ASABE DOI 10.13031/2013.27838](https://doi.org/10.13031/2013.27838)  
Авторская институциональная запись с исходным abstract: [Penn State Research Portal](https://pure.psu.edu/en/publications/color-indices-for-weed-identification-under-various-soil-residue-/)

Защитная формулировка: ExG — не классификатор сорняков, а дешёвый candidate mask / hard-negative filter. Порог `30` является проектной гипотезой и должен сопровождаться ROC/PR-анализом vegetation-vs-background на кадрах Mavic 3E, включая закат, тени, стерню и разные настройки экспозиции.

### 3. SAHI

**Akyon, F. C.; Altinuc, S. O.; Temizel, A. (2022). “Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection.” IEEE ICIP 2022, pp. 966–970. DOI: 10.1109/ICIP46576.2022.9897990.** Статья описывает overlapping slicing при inference и fine-tuning, интеграцию поверх детекторов и эксперименты на VisDrone/xView. Авторы сообщают прирост AP в своих конфигурациях; это evidence механизма, а не обещание такого же прироста на сорняках.  
Первоисточник: [IEEE DOI](https://doi.org/10.1109/ICIP46576.2022.9897990) · [author preprint](https://arxiv.org/abs/2202.06934) · [официальный репозиторий SAHI](https://github.com/obss/sahi)

Для README следует показывать собственную абляцию: full-frame vs SAHI; 640/1024/1536; overlap; AP_small/recall; число дублей до/после merge; ошибки на границах тайлов; latency/VRAM. Нельзя приписывать статье собственный иерархический режим или IoM-порог `0.35`.

### 4. Multi-Task Learning для сорняков и фаз

**Sarker, T. T.; Ahmed, K. R.; Islam, T.; Rankrape, C. B.; Gage, K. (2025). “WeedSense: Multi-Task Learning for Weed Segmentation, Height Estimation, and Growth Stage Classification.” ICCV Workshops 2025.** Это наиболее прямой найденный peer-reviewed первичный источник по weed-specific MTL: общая модель совместно предсказывает segmentation, height и growth stage на 16 weed species в течение 11-недельного цикла.  
Первоисточник: [CVF Open Access paper](https://openaccess.thecvf.com/content/ICCV2025W/CVPPA/html/Sarker_WeedSense_Multi-Task_Learning_for_Weed_Segmentation_Height_Estimation_and_Growth_ICCVW_2025_paper.html) · [PDF](https://openaccess.thecvf.com/content/ICCV2025W/CVPPA/papers/Sarker_WeedSense_Multi-Task_Learning_for_Weed_Segmentation_Height_Estimation_and_Growth_ICCVW_2025_paper.pdf) · [официальный код авторов](https://github.com/toqitahamid/WeedSense)

**Evidence gap:** прямой первичный источник, который валидирует именно один EfficientNet-B0 с двумя classification heads «вид сорняка (26) + агрономическая фаза (3)» и Focal Loss на зерновом поле, не найден. WeedSense подтверждает общую идею shared representation + task-specific heads, но не конкретную архитектуру проекта. Дополнительный независимый baseline по фазам — Teimouri et al. (2018), но это single-task CNN, не MTL: [Sensors, DOI 10.3390/s18051580](https://doi.org/10.3390/s18051580).

### 5. Focal Loss

**Lin, T.-Y.; Goyal, P.; Girshick, R.; He, K.; Dollár, P. (2017). “Focal Loss for Dense Object Detection.” ICCV 2017. DOI: 10.1109/ICCV.2017.324.** Focal Loss добавляет modulating factor к cross-entropy, подавляя вклад хорошо классифицированных примеров и концентрируя оптимизацию на трудных. Первичная демонстрация — RetinaNet и extreme foreground/background imbalance dense detection.  
Первоисточник: [CVF Open Access](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html) · [PDF](https://openaccess.thecvf.com/content_iccv_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf) · [DOI](https://doi.org/10.1109/ICCV.2017.324)

Корректная связь с проектом является инженерной экстраполяцией: редкие виды могут теряться за частыми, поэтому focal loss разумно тестировать. Для доказательства на проекте требуются stratified field split и таблица per-class support/precision/recall/F1, macro-F1, balanced accuracy, ECE/Brier score, а также абляция `gamma ∈ {0,1,2,3}` и class weights. `gamma=0` эквивалентен обычной cross-entropy и даёт чистый baseline.

## Рекомендуемый блок для README / слайда

> Мы не утверждаем, что одна публикация подтверждает весь конвейер. ISO 11783-10 и AEF обосновывают интерфейс Task Controller и variable-rate prescription maps; Woebbecke et al. — быстрый RGB vegetation filter; Akyon et al. — slicing для мелких объектов; WeedSense — shared multi-task representation для анализа сорняков и стадий; Lin et al. — focal mechanism для дисбаланса. Конкретные пороги, 26 видов, три фазы, скорость и совместимость оборудования подтверждаются только нашими полевыми тестами, абляциями, XSD-валидацией и импортом на целевой Task Controller.

## Минимальный пакет доказательств перед публичными заявлениями

1. XSD validation log для точной версии ISOXML и три acceptance-test отчёта импорта на целевые терминалы либо осторожная маркировка «не проверено на устройстве».
2. ExG ROC/PR и confusion table vegetation/background по сценариям освещения; отдельно crop-vs-weed, чтобы показать, что safety invariant обеспечивает классификатор/ручная проверка, а не ExG.
3. SAHI ablation с AP_small, recall, tile-boundary FN, duplicate count и latency; отдельная абляция NMS/IoM merge.
4. MTL ablation: shared two-head model против двух независимых моделей; per-species и per-stage metrics на field-level hold-out без соседних кадров одного растения в разных splits.
5. Focal ablation и calibration при production threshold 0.60; отдельно false-spray rate для wheat и coverage/manual-review rate.
