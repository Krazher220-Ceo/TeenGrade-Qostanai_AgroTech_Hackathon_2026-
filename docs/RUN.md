# AgroVision AI (Case 1) — Руководство по запуску (DevOps)

Этот документ — практическая, проверяемая инструкция «с нуля» для запуска
Кейса №1: FastAPI-бэкенд (`case1/server/app.py`), Streamlit-дашборд
(`case1/dashboard/app.py`) и офлайн PWA «АгроСкаут» (`case1/mobile/`).
Для архитектуры, метрик и продуктовой презентации см. основной
[`README.md`](../README.md) и [`case1/README.md`](../case1/README.md).

---

## 0. Требования

- Python 3.11 (проверено; должно работать и на 3.10+).
- [Git LFS](https://git-lfs.com/) — веса моделей (`*.pt`, `*.torchscript`) и часть
  изображений хранятся через LFS. Без него вы получите текстовые
  LFS-«указатели» вместо бинарных файлов, и тесты/сервер, загружающие модель,
  не заработают.
- (Опционально) Docker + Docker Compose v2 — для контейнерного запуска.
- (Опционально, для HTTPS/офлайн-теста PWA на телефоне) [mkcert](https://github.com/FiloSottile/mkcert)
  **или** [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) /
  [ngrok](https://ngrok.com/download).

---

## 1. Запуск с нуля (без Docker)

```bash
# 1. Клонирование репозитория
git clone <URL_РЕПОЗИТОРИЯ>
cd Qostanai_AgroTech_Hackathon_2026

# 2. Загрузка реальных весов моделей и крупных изображений из Git LFS
#    (без этого шага case1/models/*.pt останутся ~130-байтными указателями!)
git lfs install
git lfs pull

# 3. Создание виртуального окружения и установка зависимостей
make setup
source .venv/bin/activate   # опционально, если хотите работать вне make

# 4. Быстрая проверка тестами (модель-зависимые тесты сами
#    пропускаются, если веса ещё не подтянуты — см. раздел 6)
make test

# 5. Запуск сервера + дашборда одновременно
make demo
```

После `make demo`:
- FastAPI сервер: <http://localhost:8000> (Swagger — `/docs`, здоровье — `/health`).
- Мобильный PWA-клиент (раздаётся тем же сервером): <http://localhost:8000/mobile/>.
- Streamlit-дашборд: <http://localhost:8501>.

Остановить оба процесса — `Ctrl+C` (Makefile-цель `demo` перехватывает выход
и гасит фоновый uvicorn).

### Запуск сервисов по отдельности

```bash
make server      # только FastAPI, http://localhost:8000
make dashboard   # только Streamlit, http://localhost:8501
```

Переменные `SERVER_HOST`, `SERVER_PORT`, `DASHBOARD_PORT` можно переопределить,
например: `make server SERVER_PORT=8010`.

### Проверка предписаний и ISO-XML

```bash
make verify
```

Запускает `scripts/verify_prescription_and_taskdata.py`: адверсариальную
проверку GeoJSON (RFC 7946) и `TASKDATA.XML` (ISO 11783-10), плюс
регенерацию пайплайна в песочнице для проверки воспроизводимости.

---

## 2. Запуск через Docker

```bash
# ВАЖНО: сначала подтяните реальные веса из LFS на хосте — Docker COPY
# копирует файлы как есть, включая LFS-указатели, если их не подтянуть заранее.
git lfs pull

docker compose build
docker compose up -d

# Проверка
curl http://localhost:8000/health
open http://localhost:8501   # или просто откройте в браузере
```

Состав `docker-compose.yml`:
- **`api`** — `uvicorn case1.server.app:app` на порту `8000`, раздаёт `/mobile`
  (PWA) и REST API. Health-check дергает `GET /health` каждые 15 секунд.
- **`dashboard`** — `streamlit run case1/dashboard/app.py` на порту `8501`.

Оба сервиса монтируют `./case1/output` и `./case1/models` как volume'ы, чтобы:
- результаты инференса/синхронизации из PWA сохранялись на хосте и переживали
  пересоздание контейнера;
- можно было положить/обновить веса модели без пересборки образа.

Образ собирается на `python:3.11-slim` с **CPU-only** сборкой PyTorch
(`--index-url https://download.pytorch.org/whl/cpu`) — без CUDA, компактнее
и не требует GPU.

```bash
docker compose down          # остановить и удалить контейнеры
docker compose logs -f api   # логи сервера
```

> **Статус проверки в этой среде:** Docker CLI не был установлен на машине,
> где готовился этот набор инструментов, поэтому `docker compose build/up`
> здесь **не запускался и не проверялся живьём**. `Dockerfile` и
> `docker-compose.yml` проверены статически (`docker compose config`
> эквивалент — ручной разбор YAML, синтаксис валиден) и по внимательному
> ревью путей/портов/команд. Перед боевым использованием прогоните
> `docker compose build && docker compose up` и `curl /health` на машине
> с установленным Docker.

---

## 3. HTTPS для теста PWA на телефоне (обязательно для Android!)

**Проблема:** Service Worker (офлайн-кэш PWA) браузеры регистрируют только
в «безопасном контексте» — `https://` или `http://localhost`. Открытие PWA
по локальному IP ноутбука (`http://192.168.x.x:8000/mobile/`, как в разделе
3 основного README) **не даст Service Worker'у зарегистрироваться** на
Android/Chrome, и офлайн-режим не заработает, даже если весь остальной
функционал выглядит рабочим.

Решение — `scripts/serve_https.sh`, два варианта:

```bash
make https
# или явно:
scripts/serve_https.sh mkcert     # локальный доверенный сертификат
scripts/serve_https.sh tunnel     # публичный туннель (cloudflared/ngrok)
```

### Вариант A — mkcert (рекомендуется для повторных полевых тестов)

1. Установите mkcert: `brew install mkcert nss` (macOS) или см.
   [инструкцию для Linux/Windows](https://github.com/FiloSottile/mkcert#installation).
2. `scripts/serve_https.sh mkcert` — скрипт сам:
   - определит ваш LAN IP,
   - выпустит сертификат для `localhost`, `127.0.0.1` и вашего LAN IP в
     `.certs/` (не коммитится — в `.gitignore`),
   - запустит `uvicorn ... --ssl-keyfile --ssl-certfile`,
   - выведет инструкцию по установке корневого сертификата mkcert на Android.

**Установка корневого сертификата mkcert на Android (один раз на устройство):**
1. На компьютере: `mkcert -CAROOT` — там лежит `rootCA.pem`.
2. Перекиньте `rootCA.pem` на телефон и переименуйте в `rootCA.crt`.
3. Android: *Настройки → Безопасность → Шифрование и учётные данные →
   Установить сертификат → Сертификат ЦС* → выберите `rootCA.crt`.
4. Откройте `https://<LAN_IP>:8000/mobile/` в браузере телефона — замок
   должен быть закрытым без предупреждений.

### Вариант B — туннель (cloudflared/ngrok), без установки сертификата

1. Установите `cloudflared` (`brew install cloudflared`) или `ngrok`.
2. `scripts/serve_https.sh tunnel` — поднимет локальный сервер и туннель,
   выведет публичный `https://*.trycloudflare.com` (или `*.ngrok-free.app`)
   URL.
3. Откройте `<URL>/mobile/` на телефоне — сразу HTTPS, сертификат не нужен.
4. Минусы: URL новый при каждом запуске (без платного плана), трафик идёт
   через внешний туннель — не используйте с чувствительными данными.

---

## 4. Проверка офлайн-режима PWA (авиарежим)

1. Откройте `https://<LAN_IP или tunnel>/mobile/` на телефоне (см. раздел 3).
2. Нажмите **«Установить приложение» / «Добавить на главный экран»**.
3. Дождитесь тоста об успешном кэшировании (или проверьте в
   Chrome DevTools → *Application → Service Workers*, что воркер `activated
   and running`, и в *Application → Cache Storage* появился
   `agrovision-field-v12` со всеми файлами — `index.html`, `styles.css`,
   `app.js`, `api.js`, `db.js`, `store.js`, `types.js`, `catalog.js`,
   `manifest.json`, `icon.svg`).
4. Включите **режим полёта** на телефоне (или Chrome DevTools → *Network* →
   *Offline*).
5. Приложение должно продолжать открываться и показывать очередь карточек
   (данные, загруженные до отключения, лежат в IndexedDB `AgroScout_DB`).
   Свайпы/подтверждения сохраняются локально в очередь `offlineActions`.
6. Выключите авиарежим — приложение должно само отправить накопленные
   решения через `POST /api/v1/sync` и очистить локальную очередь.

Что было реально проверено при подготовке этого набора инструментов
(автоматизированным браузером в этой среде, без физического Android-устройства):
- сервер отдаёт все файлы PWA (`sw.js`, `manifest.json`, `types.js`,
  `catalog.js` и т.д.) с кодом `200` и корректным `Content-Type`, без
  редиректов;
- `manifest.json` валиден (корректный `start_url`, иконка, `display:
  standalone`);
- `case1/mobile/app.js` корректно перехватывает ошибку регистрации Service
  Worker (`.catch(...)`) и не ломает остальной функционал приложения;
- сам список кэшируемых файлов в `case1/mobile/sw.js` (`STATIC_ASSETS`)
  включает все нужные ассеты, версия кэша (`agrovision-field-v12`)
  согласована между списком файлов и query-параметрами `?v=12`.

Регистрация Service Worker **не была подтверждена сквозным тестом на
реальном Android-устройстве** — используемый в этой сессии
автоматизированный браузер-инструмент не подключён к живому телефону.
Пройдите разделы 3–4 на реальном устройстве перед демонстрацией жюри.

---

## 5. Тесты

```bash
make test
# эквивалент:
.venv/bin/python -m pytest case1/tests -q
```

Полный набор (`62 passed`) зелёный «из коробки» после `git lfs pull`.
Некоторые тесты **самостоятельно пропускаются (`skip`)**, если нужные
локальные артефакты недоступны — это осознанное поведение, а не баг:

| Тест | Условие пропуска |
|---|---|
| `test_smoke.py::test_weights_exist`, `test_detector_loads`, `test_classifier_loads`, `test_fastapi_server` | Веса модели — LFS-указатель, а не бинарник (не выполнен `git lfs pull`) |
| `test_latency_benchmark.py::test_run_latency_benchmark_smoke` | То же (веса классификатора) |
| `test_dataset_manager.py::test_splits_and_image_label_pairing` | Локальные YOLO-датасеты (`case1/data/yolo_*`) не распространяются через git (см. `.gitignore`) — соберите их через `case1/ml/build_full_dataset.py` / `prepare_yolo_aerial.py` |

`case1/tests/conftest.py` автоматически регенерирует
`case1/output/e2e_simulation/*` (используется адверсариальной проверкой
GeoJSON/ISO-XML) перед сессией тестов, если этих файлов ещё нет —
дополнительных шагов не требуется.

Про CI: `.github/workflows/tests.yml` гоняет тот же `pytest` на Ubuntu с
`actions/checkout@v4` (`lfs: false` — веса намеренно не подтягиваются, чтобы
не тратить LFS-квоту на каждый прогон) — модель-зависимые тесты там
пропускаются автоматически по той же логике.

---

## 6. Частые ошибки

**`_pickle.UnpicklingError: Weights only load failed` / модель не грузится**
Веса — это текстовый LFS-указатель (~130 байт), а не настоящий `.pt`.
Решение: `git lfs install && git lfs pull` (или `make lfs-pull`). Проверить:
`file case1/models/multitask_weeds_best.pt` должно показать `Zip archive
data...`, а не `ASCII text`.

**`ModuleNotFoundError` при `python -m uvicorn ...` / `pytest`**
Запускайте команды из корня репозитория (не из `case1/`) — код использует
абсолютные импорты `from case1....`. `make server`/`make test` уже это
учитывают.

**PWA «не устанавливается» / офлайн не работает на телефоне**
Почти всегда причина — открыт `http://192.168.x.x:...` вместо HTTPS. См.
раздел 3.

**`address already in use` на порту 8000/8501**
Другой процесс уже слушает порт (в том числе другой ваш терминал). Смените
порт: `make server SERVER_PORT=8010` или найдите и остановите процесс
(`lsof -nP -iTCP:8000 -sTCP:LISTEN`).

**Docker: `curl: (7) Failed to connect` на `/health` сразу после `docker compose up`**
Дайте контейнеру время на старт (`start_period: 20s` в health-check уже
учитывает загрузку модели) — команда `docker compose ps` должна показать
`healthy` через 15-30 секунд.

**`git lfs pull` ничего не скачивает / файлы всё ещё маленькие**
Убедитесь, что `git-lfs` установлен (`git lfs version`) и что репозиторий
инициализирован (`git lfs install` один раз на машину).

**Тест `test_splits_and_image_label_pairing` падает (не пропускается)**
Значит частично собраны каталоги `case1/data/yolo_*`, но не все нужные
сплиты (`train/val/test`) — либо доберите датасет полностью, либо удалите
частично собранную папку, чтобы тест снова корректно пропустился.
