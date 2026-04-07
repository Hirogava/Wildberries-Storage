import json
import math
import os
import threading
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.abspath(os.getenv("WORKSPACE_DIR", os.path.join(SCRIPT_DIR, "..")))
MODEL_PATH = os.path.abspath(os.getenv("ML_MODEL_PATH", os.path.join(WORKSPACE_DIR, "baseline_lgbm")))
USE_MODEL_PREDICTION = os.getenv("ML_USE_MODEL_PREDICTION", "true").lower() in {"1", "true", "yes", "on"}
FALLBACK_TO_RULES = os.getenv("ML_FALLBACK_TO_RULES", "true").lower() in {"1", "true", "yes", "on"}

HOST = os.getenv("ML_SERVICE_HOST", "0.0.0.0")
PORT = int(os.getenv("ML_SERVICE_PORT", "8090"))
DEFAULT_MODEL = os.getenv("ML_DEFAULT_MODEL", "lgbm_v1")
LOG_BUFFER_SIZE = int(os.getenv("ML_LOG_BUFFER_SIZE", "300"))

LOGS = deque(maxlen=LOG_BUFFER_SIZE)
LOGS_CONDITION = threading.Condition()
LOG_SEQUENCE = 0
MODELS = None
MODEL_LOAD_ERROR = None


if USE_MODEL_PREDICTION:
    try:
        MODELS = joblib.load(MODEL_PATH)
    except Exception as exc:
        MODEL_LOAD_ERROR = str(exc)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(level: str, component: str, message: str):
    global LOG_SEQUENCE

    entry = {
        "timestamp": utc_now_iso(),
        "level": level,
        "component": component,
        "message": message,
    }

    with LOGS_CONDITION:
        LOG_SEQUENCE += 1
        entry["seq"] = LOG_SEQUENCE
        LOGS.append(entry)
        LOGS_CONDITION.notify_all()

    print(f"[{entry['level']}] [{entry['component']}] {entry['message']}", flush=True)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    minutes_of_day = df["hour"] * 60 + df["minute"]
    df["tod_sin"] = np.sin(2 * np.pi * minutes_of_day / 1440)
    df["tod_cos"] = np.cos(2 * np.pi * minutes_of_day / 1440)

    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def build_predictions(points: list[dict]) -> list[dict]:
    df = pd.DataFrame(points)
    required = {"id", "route_id", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"points payload is missing columns: {sorted(missing)}")

    df = df.copy()
    df["id"] = pd.to_numeric(df["id"], errors="raise").astype("int64")
    df["route_id"] = pd.to_numeric(df["route_id"], errors="raise").astype("int64")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = add_calendar_features(df)

    if MODELS is not None and "target_step_1" in MODELS:
        feature_cols = ["route_id", "hour", "minute", "dayofweek", "is_weekend", "tod_sin", "tod_cos", "dow_sin", "dow_cos"]
        feature_cols = [column for column in feature_cols if column in df.columns]

        features = df[feature_cols].copy()
        if "route_id" in features.columns:
            features["route_id"] = features["route_id"].astype("category")

        values = np.asarray(MODELS["target_step_1"].predict(features), dtype=float)
    else:
        if not FALLBACK_TO_RULES:
            raise RuntimeError(f"model is unavailable: {MODEL_LOAD_ERROR or 'target_step_1 is missing'}")

        hour_factor = (df["hour"] + df["minute"] / 60.0) * 0.55
        route_factor = df["route_id"].mod(17) * 1.15
        values = np.asarray(8.0 + hour_factor + route_factor, dtype=float)

    values = np.round(np.maximum(0.0, values), 4)
    return [
        {
            "id": int(point_id),
            "y_pred": float(prediction),
        }
        for point_id, prediction in zip(df["id"], values, strict=False)
    ]


def select_model(candidates: list[str]) -> dict:
    if not candidates:
        candidates = [DEFAULT_MODEL]

    ranking = []
    for index, model in enumerate(candidates):
        ranking.append(
            {
                "model": model,
                "score": round(0.18 + index * 0.005, 6),
            }
        )

    ranking.sort(key=lambda item: item["score"])
    return {"selected_model": ranking[0]["model"], "ranking": ranking}


def resolve_workspace_path(path: str) -> str:
    if not path:
        raise ValueError("input_path must not be empty")

    resolved = path if os.path.isabs(path) else os.path.join(WORKSPACE_DIR, path)
    resolved = os.path.abspath(resolved)
    if os.path.commonpath([WORKSPACE_DIR, resolved]) != WORKSPACE_DIR:
        raise ValueError(f"path {path} is outside workspace")
    return resolved


def normalize_points_frame(df: pd.DataFrame, source_name: str) -> list[dict]:
    required = {"id", "route_id", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{source_name} is missing columns: {sorted(missing)}")

    timestamps = pd.to_datetime(df["timestamp"], utc=True)
    return [
        {
            "id": int(row_id),
            "route_id": int(route_id),
            "timestamp": ts.isoformat().replace("+00:00", "Z"),
        }
        for row_id, route_id, ts in zip(df["id"], df["route_id"], timestamps, strict=False)
    ]


def load_points_from_dataset(input_path: str) -> list[dict]:
    resolved = resolve_workspace_path(input_path)
    extension = os.path.splitext(resolved)[1].lower()

    if extension == ".parquet":
        return normalize_points_frame(pd.read_parquet(resolved), "parquet")

    if extension == ".csv":
        return normalize_points_frame(pd.read_csv(resolved), "csv")

    if extension == ".json":
        with open(resolved, "r", encoding="utf-8") as file:
            payload = json.load(file)

        if isinstance(payload, dict) and isinstance(payload.get("points"), list):
            return payload["points"]
        if isinstance(payload, list):
            return payload
        raise ValueError("json dataset must be an array of points or an object with points")

    raise ValueError(f"unsupported dataset format: {extension or resolved}")


class Handler(BaseHTTPRequestHandler):
    server_version = "WBPythonMLService/0.2"

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/healthz":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "service": "python-ml-service",
                    "model": DEFAULT_MODEL,
                    "mode": "model" if MODELS is not None else "fallback",
                    "timestamp": utc_now_iso(),
                },
            )
            return

        if path == "/logs":
            limit = self._parse_limit(parsed.query, 80)
            self._send_json(
                200,
                {
                    "service": "python-ml-service",
                    "entries": list(LOGS)[-limit:],
                },
            )
            return

        if path == "/stream/logs":
            limit = self._parse_limit(parsed.query, 80)
            self._handle_logs_stream(limit)
            return

        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path

        try:
            payload = self._read_json()
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return

        if path == "/predict":
            self._handle_predict(payload)
            return

        if path == "/dataset/points":
            self._handle_dataset_points(payload)
            return

        if path == "/model/select":
            self._handle_model_select(payload)
            return

        self._send_json(404, {"error": "not found"})

    def _handle_predict(self, payload: dict):
        request_id = payload.get("request_id", f"req-{math.floor(datetime.now().timestamp() * 1000)}")
        points = payload.get("points", [])
        log_event("INFO", "http", f"POST /predict request_id={request_id} points={len(points) if isinstance(points, list) else 'invalid'}")

        if not isinstance(points, list) or not points:
            self._send_json(400, {"error": "points must be a non-empty array"})
            return

        try:
            predictions = build_predictions(points)
        except (KeyError, TypeError, ValueError) as exc:
            self._send_json(400, {"error": f"invalid point payload: {exc}"})
            return
        except Exception as exc:
            log_event("ERROR", "predict", f"prediction failed request_id={request_id}: {exc}")
            self._send_json(500, {"error": f"prediction failed: {exc}"})
            return

        log_event("INFO", "predict", f"predictions built request_id={request_id} count={len(predictions)} model={DEFAULT_MODEL}")
        self._send_json(
            200,
            {
                "request_id": request_id,
                "predictions": predictions,
                "model": DEFAULT_MODEL,
            },
        )

    def _handle_dataset_points(self, payload: dict):
        input_path = payload.get("input_path", "")
        log_event("INFO", "http", f"POST /dataset/points input_path={input_path}")

        try:
            points = load_points_from_dataset(input_path)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except Exception as exc:
            log_event("ERROR", "dataset", f"dataset conversion failed input_path={input_path}: {exc}")
            self._send_json(500, {"error": f"dataset conversion failed: {exc}"})
            return

        self._send_json(200, {"points": points})

    def _handle_model_select(self, payload: dict):
        request_id = payload.get("request_id", f"req-{math.floor(datetime.now().timestamp() * 1000)}")
        candidates = payload.get("candidates", [])
        log_event("INFO", "http", f"POST /model/select request_id={request_id} candidates={len(candidates) if isinstance(candidates, list) else 'invalid'}")

        if candidates is not None and not isinstance(candidates, list):
            self._send_json(400, {"error": "candidates must be an array"})
            return

        result = select_model(candidates or [])
        log_event("INFO", "model-select", f"selected_model={result['selected_model']} request_id={request_id}")
        self._send_json(
            200,
            {
                "request_id": request_id,
                "selected_model": result["selected_model"],
                "ranking": result["ranking"],
            },
        )

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        if not raw:
            return {}

        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid json: {exc.msg}") from exc

    def _send_json(self, status_code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_logs_stream(self, limit: int):
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            with LOGS_CONDITION:
                initial_entries = list(LOGS)[-limit:]
                last_seq = initial_entries[-1]["seq"] if initial_entries else LOG_SEQUENCE

            for entry in initial_entries:
                self._write_sse("log", entry)

            self._write_raw(": connected\n\n")

            while True:
                with LOGS_CONDITION:
                    LOGS_CONDITION.wait(timeout=15)
                    current_entries = list(LOGS)

                new_entries = [entry for entry in current_entries if entry["seq"] > last_seq]
                if new_entries:
                    last_seq = new_entries[-1]["seq"]
                    for entry in new_entries:
                        self._write_sse("log", entry)
                    continue

                self._write_raw(": keepalive\n\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def _write_sse(self, event_name: str, payload: dict):
        data = json.dumps(payload, ensure_ascii=False)
        self._write_raw(f"event: {event_name}\ndata: {data}\n\n")

    def _write_raw(self, data: str):
        try:
            self.wfile.write(data.encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            raise

    def _parse_limit(self, query: str, fallback: int) -> int:
        params = parse_qs(query)
        raw = params.get("limit", [str(fallback)])[0]
        try:
            value = int(raw)
        except ValueError:
            return fallback
        return value if value > 0 else fallback

    def log_message(self, format, *args):
        return


def main():
    mode = "model" if MODELS is not None else "fallback"
    if MODELS is not None:
        log_event("INFO", "bootstrap", f"starting python ml service host={HOST} port={PORT} model={DEFAULT_MODEL} mode={mode}")
    else:
        log_event("WARN", "bootstrap", f"starting python ml service host={HOST} port={PORT} model={DEFAULT_MODEL} mode={mode} error={MODEL_LOAD_ERROR}")

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"python-ml-service listening on http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
