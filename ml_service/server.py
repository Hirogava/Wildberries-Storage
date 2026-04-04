import json
import math
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, '..', 'baseline_lgbm')
models = joblib.load(model_path)

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Циклическое кодирование времени суток
    minutes_of_day = df["hour"] * 60 + df["minute"]
    df["tod_sin"] = np.sin(2 * np.pi * minutes_of_day / 1440)
    df["tod_cos"] = np.cos(2 * np.pi * minutes_of_day / 1440)

    # Циклическое кодирование дня недели
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    return df


HOST = os.getenv("ML_SERVICE_HOST", "0.0.0.0")
PORT = int(os.getenv("ML_SERVICE_PORT", "8090"))
DEFAULT_MODEL = os.getenv("ML_DEFAULT_MODEL", "lgbm_v1")
LOG_BUFFER_SIZE = int(os.getenv("ML_LOG_BUFFER_SIZE", "300"))
LOGS = deque(maxlen=LOG_BUFFER_SIZE)
LOGS_CONDITION = threading.Condition()
LOG_SEQUENCE = 0


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


def build_prediction(point: dict) -> float:
    """
    ML prediction using loaded LGBM model.
    """
    df = pd.DataFrame([point])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = add_calendar_features(df)

    feature_cols = ["route_id"] + [
        "hour", "minute", "dayofweek", "is_weekend", "tod_sin", "tod_cos", "dow_sin", "dow_cos"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    categorical_features = ["route_id"]
    for c in categorical_features:
        if c in X.columns:
            X[c] = X[c].astype("category")

    pred = models["target_step_1"].predict(X)[0]
    return round(max(0.0, pred), 4)


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
    return {
        "selected_model": ranking[0]["model"],
        "ranking": ranking,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "WBMLStub/0.1"

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/healthz":
            log_event("INFO", "http", "GET /healthz")
            self._send_json(
                200,
                {
                    "status": "ok",
                    "service": "ml-stub",
                    "model": DEFAULT_MODEL,
                    "timestamp": utc_now_iso(),
                },
            )
            return

        if path == "/logs":
            limit = self._parse_limit(parsed.query, 80)
            log_event("INFO", "http", f"GET /logs limit={limit}")
            self._send_json(
                200,
                {
                    "service": "ml-stub",
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

        predictions = []
        for point in points:
            try:
                prediction = {
                    "id": int(point["id"]),
                    "y_pred": build_prediction(point),
                }
            except (KeyError, ValueError, TypeError) as exc:
                self._send_json(400, {"error": f"invalid point payload: {exc}"})
                return

            predictions.append(prediction)

        log_event("INFO", "predict", f"predictions built request_id={request_id} count={len(predictions)} model={DEFAULT_MODEL}")
        self._send_json(
            200,
            {
                "request_id": request_id,
                "predictions": predictions,
                "model": DEFAULT_MODEL,
            },
        )

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
        except (BrokenPipeError, ConnectionResetError):
            return

    def _write_sse(self, event_name: str, payload: dict):
        data = json.dumps(payload, ensure_ascii=False)
        self._write_raw(f"event: {event_name}\ndata: {data}\n\n")

    def _write_raw(self, data: str):
        try:
            self.wfile.write(data.encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            raise

    def _parse_limit(self, query: str, fallback: int) -> int:
        params = parse_qs(query)
        raw = params.get("limit", [str(fallback)])[0]
        try:
            value = int(raw)
            return value if value > 0 else fallback
        except ValueError:
            return fallback

    def log_message(self, format, *args):
        return


def main():
    log_event("INFO", "bootstrap", f"starting ml stub host={HOST} port={PORT} model={DEFAULT_MODEL}")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"ml-stub listening on http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
