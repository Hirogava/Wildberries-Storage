# HTTP-контракт с внешним ML-сервисом

Этот репозиторий не реализует AI-часть. Команда ML должна поднять отдельный сервис, совместимый с контрактом ниже.

## `GET /healthz`

Ответ `200 OK`, если сервис готов принимать запросы.

## `GET /logs`

Необязательный endpoint для локального стенда и UI-диагностики. Возвращает последние логи ML-сервиса.

## `POST /predict`

### Request

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

### Response

```json
{
  "request_id": "req-2026-03-30-0001",
  "predictions": [
    {
      "id": 100001,
      "y_pred": 19.4
    }
  ],
  "model": "ridge_v1"
}
```

`route_id` и `timestamp` намеренно не возвращаются: Go backend восстанавливает этот контекст из исходного `points` по `id`, когда передаёт результат в `/decision`.

## `POST /model/select`

### Request

```json
{
  "request_id": "req-2026-03-30-0002",
  "candidates": ["ridge_v1", "lgbm_v2"],
  "objective": "wape_plus_rbias",
  "context": {
    "dataset": "team_track_validation"
  }
}
```

### Response

```json
{
  "request_id": "req-2026-03-30-0002",
  "selected_model": "lgbm_v2",
  "ranking": [
    {
      "model": "lgbm_v2",
      "score": 0.1842
    },
    {
      "model": "ridge_v1",
      "score": 0.1911
    }
  ]
}
```
