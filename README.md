# Wildberries Storage

Прототип сервиса автоматического вызова транспорта на склады на основе прогноза отгрузок для командного трека WB Contests.

Сервис собран как связка:

- `Go API` — orchestration layer, бизнес-логика, batch pipeline, метрики, UI;
- `Python ML service` — inference layer, выбор модели, адаптация конкурсных датасетов;
- `Web UI` — демонстрационный контур для защиты и ручной проверки сценариев.

## Что уже реализовано

- онлайн-контур `predict -> decision -> transport call`;
- offline-контур `dataset -> submission.csv`;
- расчёт конкурсной метрики `WAPE + |Relative Bias|`;
- health checks и telemetry для Go API и Python ML service;
- встроенный UI для демонстрации и ручной проверки API;
- upload-флоу для датасетов и CSV-файлов метрик;
- Docker и `docker compose` для быстрого старта.

## Архитектура

```text
client / ui
  -> Go API
    -> Python ML service
    -> decision engine
    -> submission generator
    -> metrics service
```

Основная идея:

- `/predict` отвечает за прогноз;
- `/decision` переводит прогноз в количество машин;
- `/batch` формирует submission;
- `/metrics` считает качество на реальных `y_true`;
- `/model/select` показывает слой model governance;
- `/files/upload` решает безопасную загрузку файлов в Docker-сценарии.

## Бизнес-логика

Сервис работает с временным шагом `30 минут` и горизонтом `2 часа`.

Decision layer использует формулу:

```text
adjusted_load = y_pred * (1 + safety_factor)
trucks_needed = ceil(adjusted_load / truck_capacity)
```

Где:

- `safety_factor` защищает от недовызова транспорта;
- `truck_capacity` задаёт вместимость машины;
- `max_trucks_per_route` ограничивает число машин на маршрут.

## Метрика

Сервис считает ту же метрику, что и в соревновании:

```text
score = WAPE + |Relative Bias|
```

Важно:

- честная оценка качества считается только на реальных `y_true`;
- UI больше не подставляет `y_true = y_pred` после `/batch`;
- demo chain запускает `/metrics` только если загружен `actual.csv`;
- для файлового сценария в Docker нужно использовать workspace-relative paths или `POST /files/upload`.

## API

### `POST /predict`

Принимает точки прогноза и возвращает `id + y_pred + model`.

Пример:

```json
{
  "request_id": "req-2026-03-30-0001",
  "points": [
    {
      "id": 100001,
      "route_id": 12345,
      "timestamp": "2026-03-30T12:30:00Z"
    }
  ]
}
```

### `POST /decision`

Поддерживает два режима:

1. передать только `points` и дать сервису самому вызвать `/predict`;
2. передать `predictions` и исходные `points`.

Возвращает готовые решения по вызову транспорта.

### `POST /batch`

Формирует `submission.csv` формата `id,y_pred`.

Поддерживает:

- `points` в body;
- `input_path` на `.json`, `.csv`, `.parquet`.

Пример:

```json
{
  "request_id": "req-2026-03-30-0004",
  "input_path": "test_team_track.parquet",
  "output_path": "artifacts/submissions/team_submission.csv",
  "return_predictions": true
}
```

Примечания:

- `output_path` возвращается в ответе;
- при `return_predictions=true` сервис по-прежнему возвращает все строки предсказаний;
- после `/batch` UI автоматически подставляет только `prediction_path`, а не фейковые метрики.

### `POST /metrics`

Считает `WAPE`, `Relative Bias` и итоговый `score`.

Поддерживает два режима:

1. inline `observations`;
2. файловый сценарий через `actual_path` и `prediction_path`.

Пример через файлы:

```json
{
  "request_id": "req-metrics-001",
  "actual_path": "artifacts/uploads/actual.csv",
  "prediction_path": "artifacts/uploads/prediction.csv",
  "actual_column": "y_true",
  "prediction_column": "y_pred"
}
```

Ожидаемые CSV:

- `actual.csv`: `id,y_true`
- `prediction.csv`: `id,y_pred`

Важно:

- абсолютные Windows-пути вида `D:\...` внутри Docker не работают напрямую;
- для таких файлов используйте `POST /files/upload`, после чего передавайте `relative_path`.

### `POST /files/upload`

Загружает файл в workspace сервиса и возвращает:

- `filename`
- `path`
- `relative_path`
- `size`

Пример ответа:

```json
{
  "filename": "20260407_172746_actual.csv",
  "path": "/app/artifacts/uploads/20260407_172746_actual.csv",
  "relative_path": "artifacts/uploads/20260407_172746_actual.csv",
  "size": 26
}
```

### `POST /model/select`

Проксирует выбор лучшей модели по целевой метрике.

### `GET /healthz`

Проверяет состояние всего контура.

Пример ответа ML service:

```json
{
  "status": "ok",
  "service": "python-ml-service",
  "model": "lgbm_v1",
  "mode": "model"
}
```

### `GET /logs/go`

Возвращает буфер логов Go API.

### `GET /logs/ml`

Возвращает буфер логов Python ML service.

### `GET /stream/logs/go`

SSE-поток логов Go API.

### `GET /stream/logs/ml`

SSE-поток логов Python ML service.

### `GET /`

Открывает встроенный UI.

## UI

UI доступен на:

```text
http://localhost:8080/
```

В UI есть:

- презентационный dashboard;
- готовые payload'ы по основным endpoints;
- upload-кнопки для batch dataset, `actual.csv`, `prediction.csv`;
- demo flow `predict -> decision -> batch`;
- честный запуск `/metrics`, если загружен `actual.csv`;
- live telemetry Go API;
- live telemetry Python ML service.

### Что загружать в UI

#### 1. `Batch dataset`

Сюда загружается файл точек для прогноза.

Что подходит:

- конкурсный `test_team_track.parquet`;
- свой `.csv` или `.json` с полями `id`, `route_id`, `timestamp`.

Откуда берётся:

- для submission — из конкурсного тестового файла;
- для локальной проверки — из вашего validation/holdout набора без `y_true`, только с точками прогноза.

Что происходит дальше:

- UI загружает файл через `POST /files/upload`;
- `relative_path` автоматически подставляется в `batch.input_path`;
- потом вы запускаете `/batch`.

#### 2. `Metrics actual.csv`

Сюда загружается файл с истинными ответами.

Формат:

```csv
id,y_true
1,10.0
2,20.0
3,15.0
```

Откуда берётся:

- этот файл сервис не генерирует;
- вы готовите его сами из validation/holdout на основе `train_team_track.parquet` или другой размеченной выборки.

Что происходит дальше:

- UI загружает файл через `POST /files/upload`;
- `relative_path` автоматически подставляется в `metrics.actual_path`.

#### 3. `Metrics prediction.csv`

Это файл с предсказаниями.

Формат:

```csv
id,y_pred
1,9.2
2,22.1
3,14.7
```

Обычно его руками загружать не нужно.

Почему:

- после успешного `/batch` UI сам подставляет путь к сгенерированному `submission.csv` в `metrics.prediction_path`.

Когда эта кнопка нужна:

- если у вас уже есть отдельный CSV с предсказаниями;
- если вы хотите сравнить старую submission или внешний prediction-файл без запуска `/batch`.

### Типовой сценарий в UI

#### Для submission

1. Загрузить `test_team_track.parquet` в `Batch dataset`.
2. Нажать `/batch`.
3. Забрать `output_path` или скачать/использовать полученный CSV.

#### Для честной локальной оценки

1. Загрузить validation dataset в `Batch dataset`.
2. Нажать `/batch`, чтобы получить `prediction_path`.
3. Загрузить `actual.csv` в `Metrics actual.csv`.
4. Нажать `/metrics`.

Если у вас уже есть готовый файл `prediction.csv`, шаг с `/batch` можно пропустить и загрузить его через `Metrics prediction.csv`.

## Локальный запуск

### Go API

```powershell
go run ./cmd/server
```

### Python ML service

```powershell
python -m pip install -r requirements.txt
python ml_service/server.py
```

Общий пример переменных окружения:

- [.env.example](.env.example)

## Запуск через Docker

```powershell
docker compose up --build
```

После старта:

- Go API: `http://localhost:8080`
- Python ML service: `http://localhost:8090`
- UI: `http://localhost:8080/`

## Структура проекта

```text
cmd/server              точка входа Go API
internal/app            сборка зависимостей
internal/config         env-конфиг
internal/domain         модели запросов и ответов
internal/httpapi        handlers, router, streaming logs
internal/ml             HTTP-клиент внешнего ML-сервиса
internal/observability  логирование и streaming
internal/service        бизнес-логика
internal/storage        файловые адаптеры, upload, submission writer
internal/ui             встроенный frontend
ml_service              Python inference service
docs                    архитектура и контракты
```

## Проверка

```powershell
$env:GOCACHE="D:\hackatones\Wildberries-Storage\artifacts\gocache"
$env:GOTELEMETRY="off"
go test ./...
go build ./...
python -m py_compile ml_service/server.py
```

## PowerShell shortcuts

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\up.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\status.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\logs.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\down.ps1
```

`up.ps1` поднимает стек в detached mode и не блокирует терминал.

## Бизнес-допущения

- все машины в базовом сценарии считаются одинаковой вместимости;
- основной сигнал для вызова транспорта — прогноз на `2 часа`;
- отрицательные прогнозы обрезаются до нуля;
- решение ограничивается `max_trucks_per_route`;
- если модель недоступна, Python ML service может перейти в fallback-режим и не ломать интеграционный контур.
