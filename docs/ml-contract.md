# HTTP-контракт внешнего ML-сервиса

Go API ожидает внешний ML-сервис, совместимый с контрактом ниже.

## `GET /healthz`

Проверка состояния ML-сервиса.

## `GET /logs`

JSON-снимок последних логов ML-сервиса для UI и диагностики.

## `GET /stream/logs`

SSE-поток логов ML-сервиса.

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
  "model": "lgbm_v1"
}
```

`route_id` и `timestamp` не обязаны возвращаться: Go backend умеет восстановить этот контекст по исходным `points`.

## `POST /dataset/points`

Используется для batch-сценария и преобразует конкурсный файл в нормализованный список `points`.

### Request

```json
{
  "input_path": "test_team_track.parquet"
}
```

### Response

```json
{
  "points": [
    {
      "id": 4900,
      "route_id": 12345,
      "timestamp": "2026-03-30T12:30:00Z"
    }
  ]
}
```

## `POST /model/select`

### Request

```json
{
  "request_id": "req-2026-03-30-0002",
  "candidates": ["ridge_v1", "lgbm_v1", "lgbm_v2"],
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
  "selected_model": "lgbm_v1",
  "ranking": [
    {
      "model": "lgbm_v1",
      "score": 0.1842
    }
  ]
}
```
