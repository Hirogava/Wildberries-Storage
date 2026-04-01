# Wildberries-Storage Backend

Прототип backend-системы автоматического вызова транспорта на склады для командного трека соревнования Wildberries.

Основной сервис написан на Go и отвечает за:

- приём точек прогнозирования;
- интеграцию с внешним ML/AI-сервисом по HTTP;
- преобразование прогноза в операционное решение по вызову машин;
- генерацию `submission.csv`;
- расчёт метрики `WAPE + |Relative Bias|`;
- встроенную web-панель для ручной проверки сценариев.

AI-часть в этом репозитории не реализуется и считается внешней зависимостью.

## Архитектура

- [Архитектура](/D:/hackatones/Wildberries-Storage/docs/architecture.md)
- [Контракт с ML-сервисом](/D:/hackatones/Wildberries-Storage/docs/ml-contract.md)

Схема взаимодействия:

```text
client
  -> Go backend (/predict, /decision, /batch, /metrics, /model/select)
  -> external ML service (/predict, /model/select)
```

## API

### `POST /predict`

Получает список точек и возвращает прогнозы.

Request:

```json
{
  "request_id": "req-2026-03-30-0001",
  "points": [
    {
      "id": 100001,
      "route_id": 12345,
      "timestamp": "2026-03-30T12:30:00Z"
    },
    {
      "id": 100002,
      "route_id": 12345,
      "timestamp": "2026-03-30T13:00:00Z"
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
    },
    {
      "id": 100002,
      "y_pred": 20.1
    }
  ],
  "model": "ridge_v1"
}
```

### `POST /decision`

Принимает либо только `points`, либо связку из компактных `predictions` и исходных `points`, а затем рассчитывает, сколько машин нужно вызвать.

Это сделано специально под цепочку:

```text
/predict -> compact predictions
/decision -> predictions + original points
```

Пример запроса с ответом `/predict` и исходными точками:

```json
{
  "request_id": "req-2026-03-30-0003",
  "points": [
    {
      "id": 100001,
      "route_id": 12345,
      "timestamp": "2026-03-30T12:30:00Z"
    }
  ],
  "predictions": [
    {
      "id": 100001,
      "y_pred": 19.4
    }
  ],
  "safety_factor": 0.1,
  "truck_capacity": 20,
  "max_trucks_per_route": 50
}
```

### `POST /batch`

Генерирует submission CSV.

Поддерживает 2 режима:

- передать `points` прямо в body;
- передать `input_path` на JSON или CSV-файл внутри workspace.

Пример:

```json
{
  "request_id": "req-2026-03-30-0004",
  "input_path": "data/test_points.json",
  "output_path": "artifacts/submissions/team_submission.csv"
}
```

Результат:

- создаётся CSV с колонками `id,y_pred`;
- возвращается абсолютный путь к файлу.

### `POST /metrics`

Считает метрику соревнования:

```text
score = WAPE + |Relative Bias|
```

Можно передать:

- `observations` прямо в JSON;
- либо пути к CSV-файлам `actual_path` и `prediction_path`.

### `POST /model/select`

Проксирует запрос на внешний ML-сервис для выбора лучшей модели по метрике `wape_plus_rbias`.

### `GET /healthz`

Проверяет состояние API и доступность ML-сервиса.

### `GET /`

Открывает встроенную UI-панель для ручной проверки всех маршрутов.

### `GET /logs/go`

Возвращает последние логи Go backend в JSON-формате.

### `GET /logs/ml`

Проксирует последние логи Python ML-stub через Go backend.

## Бизнес-логика

В MVP приняты такие допущения:

- прогноз строится на горизонт 2 часа;
- все машины одного типа и одной вместимости;
- чтобы уменьшить риск недогруза, к прогнозу применяется safety factor;
- отрицательные предсказания запрещены и обрезаются до 0;
- заявка на машины ограничивается `max_trucks_per_route`.

Формула решения:

```text
adjusted_load = y_pred * (1 + safety_factor)
trucks_needed = ceil(adjusted_load / truck_capacity)
```

## Структура проекта

```text
cmd/server            entrypoint
internal/app          wiring
internal/config       env config
internal/domain       request/response models
internal/httpapi      handlers and router
internal/ml           HTTP client for external ML service
internal/service      business logic
internal/storage      CSV/JSON file IO
docs                  architecture and contracts
```

## Запуск

1. Поднимите внешний ML-сервис, совместимый с контрактом из [docs/ml-contract.md](/D:/hackatones/Wildberries-Storage/docs/ml-contract.md).
2. При необходимости используйте переменные окружения из [.env.example](/D:/hackatones/Wildberries-Storage/.env.example).
3. Запустите API:

```powershell
go run ./cmd/server
```

По умолчанию сервер стартует на `:8080`.

UI будет доступен на:

```text
http://localhost:8080/
```

В UI есть две отдельные live-панели:

- логи Go API;
- логи Python ML Stub.

Для локального стенда можно также поднять минимальный ML-stub:

```powershell
python ml_service/server.py
```

Инструкции для ML-команды лежат в [ml_service/README.md](/D:/hackatones/Wildberries-Storage/ml_service/README.md).

## Проверка

```powershell
go test ./...
```

## Почему это решение подходит под задачу

- отделяет ML inference от бизнес-логики вызова транспорта;
- поддерживает онлайн-сценарий `/predict -> /decision`;
- поддерживает офлайн-сценарий `/batch -> submission.csv`;
- даёт прозрачную оценку качества через `/metrics`;
- легко масштабируется: Go API можно горизонтально масштабировать отдельно от ML-сервиса.
