# Python ML Service

Python-сервис отвечает за inference и поддержку batch-сценариев для Go API.

## Возможности

- `POST /predict`
- `POST /model/select`
- `POST /dataset/points`
- `GET /healthz`
- `GET /logs`
- `GET /stream/logs`

## Режимы работы

### Model mode

Если модель успешно загружена из `ML_MODEL_PATH`, сервис использует реальный inference.

### Fallback mode

Если модель недоступна, а `ML_FALLBACK_TO_RULES=true`, сервис остаётся рабочим и выдаёт детерминированный прогноз по упрощённой формуле. Это нужно, чтобы интеграционный контур и UI не ломались.

## Что заменяет ML-команда

Основная точка расширения:

- функция `build_prediction(point)` в [server.py](/D:/hackatones/Wildberries-Storage/ml_service/server.py)

Также можно расширять:

- `select_model(candidates)` для реального ranking моделей;
- `load_points_from_dataset(input_path)` для более сложной адаптации входных датасетов.

## Запуск

```powershell
python -m pip install -r requirements.txt
python ml_service/server.py
```

## Переменные окружения

См. [.env.example](/D:/hackatones/Wildberries-Storage/.env.example).

Наиболее важные:

- `WORKSPACE_DIR`
- `ML_MODEL_PATH`
- `ML_USE_MODEL_PREDICTION`
- `ML_FALLBACK_TO_RULES`
- `ML_SERVICE_HOST`
- `ML_SERVICE_PORT`

## Рекомендация по Python

Для контейнерного запуска в проекте используется Python 3.12 как более безопасная среда для `lightgbm + scikit-learn`.

## Контракт

Сервис должен быть совместим с [docs/ml-contract.md](/D:/hackatones/Wildberries-Storage/docs/ml-contract.md).
