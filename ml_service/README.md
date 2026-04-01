# Minimal ML Stub

Минимальный HTTP-сервис для локальной интеграции с Go backend.

Он нужен, чтобы:

- поднять рабочий стенд уже сейчас;
- дать ML-команде стабильный контракт;
- потом заменить только scoring-логику без переписывания API.

## Что уже есть

- `GET /healthz`
- `GET /logs`
- `POST /predict`
- `POST /model/select`

## Что нужно заменить ML-команде

В файле [server.py](/D:/hackatones/Wildberries-Storage/ml_service/server.py) есть функция `build_prediction(point)`.

Именно туда можно вставить:

- загрузку фичей;
- вызов модели;
- постобработку предсказаний;
- bias correction.

Если появится несколько моделей, можно расширить `select_model(candidates)`.

## Запуск

```powershell
python ml_service/server.py
```

По умолчанию сервис слушает `http://localhost:8090`.

Общий пример переменных окружения лежит в [.env.example](/D:/hackatones/Wildberries-Storage/.env.example).

## Переменные окружения

- `ML_SERVICE_HOST`, по умолчанию `0.0.0.0`
- `ML_SERVICE_PORT`, по умолчанию `8090`
- `ML_DEFAULT_MODEL`, по умолчанию `stub_v1`
- `ML_LOG_BUFFER_SIZE`, по умолчанию `300`

## Контракт

Сервис должен сохранить совместимость с [docs/ml-contract.md](/D:/hackatones/Wildberries-Storage/docs/ml-contract.md).
