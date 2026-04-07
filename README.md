# Wildberries Storage

Прототип системы автоматического вызова транспорта на склады на основе прогноза отгрузок для командного трека WB Contests.

Решение построено как связка сервисов:

- `Go API` — orchestration layer, бизнес-логика, batch pipeline, метрики, UI;
- `Python ML service` — inference layer, выбор модели, адаптация конкурсных датасетов;
- `Web UI` — демонстрационный контур для защиты и ручной проверки сценариев.

## Что уже закрыто по условию

- есть сервисный контур `прогноз -> решение -> вызов транспорта`;
- есть офлайн-контур `test_team_track.parquet -> submission.csv`;
- есть расчёт соревновательной метрики `WAPE + |Relative Bias|`;
- есть архитектурное описание, бизнес-допущения и запуск;
- есть демонстрационный интерфейс для показа работы сервиса;
- есть Docker и `docker-compose` для запуска из коробки.

## Архитектура

Основной сценарий:

```text
client / ui
  -> Go API
    -> Python ML service
    -> decision engine
    -> submission generator
```

Маршруты:

- `POST /predict` — получить прогноз;
- `POST /decision` — перевести прогноз в действие;
- `POST /batch` — сформировать submission;
- `POST /metrics` — посчитать `WAPE + |Relative Bias|`;
- `POST /model/select` — выбрать модель;
- `GET /healthz` — проверить состояние контура;
- `GET /` — открыть демонстрационный UI.

Подробности:

- [Архитектура](docs/architecture.md)
- [Контракт ML-сервиса](docs/ml-contract.md)
- [ML service notes](ml_service/README.md)

## Продуктовая логика

### Как используется прогноз

Сервис работает с временным шагом 30 минут и прогнозирует ожидаемый объём отгрузки по маршруту на горизонт 2 часа.

Прогноз применяется в decision layer:

```text
adjusted_load = y_pred * (1 + safety_factor)
trucks_needed = ceil(adjusted_load / truck_capacity)
```

Где:

- `safety_factor` нужен для защиты от недовызова транспорта;
- `truck_capacity` — бизнес-допущение о вместимости машины;
- `max_trucks_per_route` — эксплуатационное ограничение по вызову машин.

### На какой горизонт мы опираемся

Для MVP выбран горизонт 2 часа, потому что он соответствует целевой переменной `target_2h` и естественному операционному циклу вызова транспорта.

### Какие преобразования данных заложены

На стороне ML service поддержаны:

- нормализация входных точек в единый контракт `id + route_id + timestamp`;
- извлечение календарных признаков;
- адаптация конкурсных файлов `.parquet`, `.csv`, `.json` в поток `points`;
- поддержка fallback-режима, если модель недоступна.

## Оценка качества

### Соревновательная метрика

Сервис считает метрику, совпадающую с условием конкурса:

```text
score = WAPE + |Relative Bias|
```

Для этого используется endpoint `POST /metrics`.

### Бизнес-метрики сервиса

Помимо leaderboard score, в репозитории заложена логика для оценки продуктовой части:

- доля недовызова транспорта;
- доля перевызова транспорта;
- средняя загрузка машины;
- latency API;
- стабильность внешнего ML-контура.

### С чем сравниваемся

В репозитории уже есть baseline-артефакты:

- `baseline_ridge`
- `baseline_lgbm`
- `train.py`

Для защиты можно показывать сравнение как минимум с ridge baseline и объяснять выигрыш более сильной модели по метрике и по операционному качеству решений.

## Пути развития

- использовать реальные статусы обработки и лаговые признаки по маршрутам;
- учитывать разные типы машин и стоимость вызова;
- учитывать ограничения складов, смен и доступного парка;
- переводить decision engine на оптимизацию стоимости, а не только на `ceil`;
- вынести orchestration в event-driven контур;
- масштабировать Go API и ML service независимо;
- добавлять кэш последних прогнозов и feature store;
- подключать внешние данные: календарь распродаж, погоду, праздники, SLA складов.

## API

## Зачем Нужны Роуты

### `POST /predict`

Нужен, чтобы получить прогноз объёма отгрузки на ближайшие 2 часа для набора точек `id + route_id + timestamp`.

Что делает сервис:

- принимает точки прогноза;
- отправляет их в Python ML service;
- получает `y_pred` по каждой точке;
- возвращает компактный ответ `id + y_pred + model`.

Зачем нужен в продукте:

- это вход в основной контур системы;
- без него нельзя ни принять решение по транспорту, ни собрать submission для соревнования.

Что идёт дальше:

- ответ `/predict` используется в `/decision`;
- та же логика вызывается внутри `/batch`.

### `POST /decision`

Нужен, чтобы превратить прогноз в операционное действие: сколько машин вызвать на маршрут.

Что делает сервис:

- берёт `y_pred`;
- добавляет `safety_factor`;
- делит скорректированный объём на `truck_capacity`;
- округляет результат вверх до целого числа машин;
- ограничивает результат через `max_trucks_per_route`.

Зачем нужен в продукте:

- это основная бизнес-ценность решения;
- сервис отвечает не только на вопрос "что будет", но и на вопрос "что делать".

Что идёт дальше:

- после `/decision` можно формировать заявку на вызов транспорта;
- это финальный шаг онлайн-сценария.

### `POST /batch`

Нужен для offline-сценария соревнования: обработать весь тестовый набор и сформировать `submission.csv`.

Что делает сервис:

- берёт `points` из body или `input_path`;
- если передан `.parquet`, через Python adapter превращает его в нормализованные точки;
- прогоняет точки через прогнозный контур;
- сохраняет результат в CSV формата `id,y_pred`.

Зачем нужен в продукте:

- это мост между сервисной архитектурой и конкурсным лидербордом;
- именно этот маршрут даёт готовый файл для загрузки на платформу.

Что идёт дальше:

- на выходе появляется `submission.csv`;
- при `return_predictions=true` результаты можно использовать для технической проверки через `/metrics`.

### `POST /metrics`

Нужен, чтобы считать конкурсную метрику `WAPE + |Relative Bias|`.

Что делает сервис:

- принимает `y_true` и `y_pred` напрямую или через CSV-файлы;
- считает WAPE;
- считает Relative Bias;
- возвращает итоговый score.

Зачем нужен в продукте:

- это инструмент оценки качества модели и сервиса;
- позволяет сравнивать версии решения до отправки на платформу.

Что идёт дальше:

- используется для валидации модели;
- используется для сравнения с baseline;
- помогает объяснить жюри, как контролируется качество.

### `POST /model/select`

Нужен, чтобы выбрать модель из нескольких кандидатов.

Что делает сервис:

- отправляет список моделей в ML service;
- получает выбранную модель и ranking.

Зачем нужен в продукте:

- показывает, что inference-слой можно развивать независимо;
- позволяет переключать модели без переписывания decision layer.

### `GET /healthz`

Нужен, чтобы быстро проверить состояние всего контура.

Что делает сервис:

- проверяет доступность Go API;
- пингует Python ML service;
- возвращает общий статус.

Зачем нужен в продукте:

- полезен для локальной разработки, Docker-стенда и демонстрации на защите;
- показывает, что решение не только считает прогноз, но и готово к эксплуатации.

### `POST /predict`

Request:

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

Response:

```json
{
  "request_id": "req-2026-03-30-0001",
  "predictions": [
    {
      "id": 100001,
      "y_pred": 19.4
    }
  ],
  "model": "lgbm_v1"
}
```

### `POST /decision`

Поддерживает два режима:

1. передать только `points` и дать сервису самому вызвать `/predict`;
2. передать `predictions` из `/predict` и исходные `points`, чтобы decision layer восстановил контекст по `id`.

### `POST /batch`

Поддерживает:

- `points` в body;
- `input_path` на `.json`, `.csv` или `.parquet`.

Пример конкурсного сценария:

```json
{
  "request_id": "req-2026-03-30-0004",
  "input_path": "test_team_track.parquet",
  "output_path": "artifacts/submissions/team_submission.csv"
}
```

### `POST /metrics`

Принимает либо `observations`, либо пути к CSV и считает итоговую метрику.

### `POST /model/select`

Проксирует выбор лучшей модели по `wape_plus_rbias`.

## UI

Встроенный интерфейс доступен на:

```text
http://localhost:8080/
```

В UI есть:

- презентационный dashboard;
- готовые payload'ы по всем основным endpoints;
- demo flow `predict -> decision`;
- live telemetry Go API;
- live telemetry Python ML service.

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
internal/storage        файловые адаптеры и submission writer
internal/ui             встроенный frontend
ml_service              Python inference service
docs                    архитектура и контракты
```

## Проверка

```powershell
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

`up.ps1` starts the full stack in detached mode, so the terminal is not blocked.

## Бизнес-допущения

- все машины в базовом сценарии считаются одинаковой вместимости;
- прогноз на 2 часа используется как основной сигнал для вызова транспорта;
- отрицательные прогнозы обрезаются до нуля;
- решение ограничивается `max_trucks_per_route`;
- если модель недоступна, Python-сервис может перейти в fallback-режим, чтобы не ломать интеграционный контур.
