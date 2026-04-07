const samples = {
  predict: {
    request_id: "req-2026-03-30-0001",
    points: [
      { id: 100001, route_id: 12345, timestamp: "2026-03-30T12:30:00Z" },
      { id: 100002, route_id: 12345, timestamp: "2026-03-30T13:00:00Z" },
      { id: 100003, route_id: 67890, timestamp: "2026-03-30T12:30:00Z" }
    ]
  },
  decision: {
    request_id: "req-2026-03-30-0003",
    points: [
      { id: 100001, route_id: 12345, timestamp: "2026-03-30T12:30:00Z" },
      { id: 100002, route_id: 12345, timestamp: "2026-03-30T13:00:00Z" },
      { id: 100003, route_id: 67890, timestamp: "2026-03-30T12:30:00Z" }
    ],
    predictions: [
      { id: 100001, y_pred: 19.4 },
      { id: 100002, y_pred: 20.1 },
      { id: 100003, y_pred: 11.7 }
    ],
    safety_factor: 0.1,
    truck_capacity: 20,
    max_trucks_per_route: 50
  },
  batch: {
    request_id: "req-2026-03-30-0004",
    input_path: "test_team_track.parquet",
    output_path: "artifacts/submissions/team_submission.csv",
    return_predictions: true
  },
  metrics: {
    request_id: "req-2026-03-30-0005",
    observations: [
      { id: 1, y_true: 10.0, y_pred: 9.2 },
      { id: 2, y_true: 20.0, y_pred: 22.1 },
      { id: 3, y_true: 15.0, y_pred: 14.7 }
    ]
  },
  model: {
    request_id: "req-2026-03-30-0006",
    candidates: ["ridge_v1", "lgbm_v1", "lgbm_v2"],
    objective: "wape_plus_rbias",
    context: {
      dataset: "team_track_validation",
      cadence: "30m"
    }
  }
};

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function byId(id) {
  return document.getElementById(id);
}

function setResponseMeta(text) {
  byId("response-meta").textContent = text;
}

function setOutput(value) {
  byId("response-output").textContent = typeof value === "string" ? value : pretty(value);
}

function parseEditor(id, fallback = {}) {
  const raw = byId(id).value.trim();
  if (!raw) {
    return fallback;
  }
  return JSON.parse(raw);
}

function parseEditorSafe(id, fallback = {}) {
  try {
    return parseEditor(id, fallback);
  } catch {
    return fallback;
  }
}

function writeEditor(id, payload) {
  byId(id).value = pretty(payload);
}

function setUploadStatus(target, text, kind = "idle") {
  const node = byId(`upload-status-${target}`);
  if (!node) {
    return;
  }

  node.className = "upload-status";
  if (kind === "ready") {
    node.classList.add("upload-status--ready");
  } else if (kind === "error") {
    node.classList.add("upload-status--error");
  }

  node.textContent = text;
}

function renderLogs(targetId, payload) {
  const entries = Array.isArray(payload?.entries) ? payload.entries : [];
  const target = byId(targetId);
  const shouldStickToBottom = target.scrollHeight - target.scrollTop - target.clientHeight < 32;

  target.textContent = entries.length
    ? entries
        .map((entry) => `[${entry.timestamp}] [${entry.level}] [${entry.component}] ${entry.message}`)
        .join("\n")
    : "No logs yet.";

  if (shouldStickToBottom) {
    target.scrollTop = target.scrollHeight;
  }
}

function fillAll() {
  writeEditor("predict-body", samples.predict);
  writeEditor("decision-body", samples.decision);
  writeEditor("batch-body", samples.batch);
  writeEditor("metrics-body", samples.metrics);
  writeEditor("model-body", samples.model);
  setUploadStatus("batch-input", "Файл не выбран");
  setUploadStatus("metrics-actual", "Файл не выбран");
  setUploadStatus("metrics-prediction", "Файл не выбран");
  setResponseMeta("Scenarios loaded");
}

function setStatusChip(id, label, status) {
  const node = byId(id);
  node.className = "status-chip";

  if (status === "up") {
    node.classList.add("status-chip--up");
  } else if (status === "down") {
    node.classList.add("status-chip--down");
  } else {
    node.classList.add("status-chip--muted");
  }

  node.textContent = label;
}

function syncDecisionFromPredict(request, response) {
  const current = parseEditorSafe("decision-body", {});
  writeEditor("decision-body", {
    request_id: current.request_id || `${response.request_id || request.request_id || "req"}-decision`,
    points: Array.isArray(request.points) ? request.points : [],
    predictions: Array.isArray(response.predictions) ? response.predictions : [],
    safety_factor: current.safety_factor ?? 0.1,
    truck_capacity: current.truck_capacity ?? 20,
    max_trucks_per_route: current.max_trucks_per_route ?? 50
  });
}

function normalizeWorkspacePath(path) {
  if (typeof path !== "string") {
    return path;
  }
  if (path.startsWith("/app/")) {
    return path.slice(5);
  }
  return path;
}

function syncMetricsFromBatch(request, response) {
  if (!response.output_path) {
    return;
  }

  const current = parseEditorSafe("metrics-body", {});
  if (!current.actual_path) {
    return;
  }

  writeEditor("metrics-body", {
    request_id: current.request_id || `${response.request_id || request.request_id || "req"}-metrics`,
    actual_path: current.actual_path,
    prediction_path: normalizeWorkspacePath(response.output_path),
    actual_column: current.actual_column || "y_true",
    prediction_column: current.prediction_column || "y_pred"
  });
}

function syncDownstream(endpoint, request, response) {
  if (endpoint === "/predict") {
    syncDecisionFromPredict(request, response);
    return;
  }

  if (endpoint === "/batch") {
    syncMetricsFromBatch(request, response);
  }
}

async function sendRequest(endpoint, source) {
  const startedAt = performance.now();

  try {
    const requestBody = parseEditor(source, {});
    setResponseMeta(`Running ${endpoint}...`);
    setOutput(`POST ${endpoint}`);

    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody)
    });

    const text = await response.text();
    const latency = Math.round(performance.now() - startedAt);
    let parsedBody;

    try {
      parsedBody = JSON.parse(text);
    } catch {
      parsedBody = text;
    }

    setOutput({
      status: response.status,
      latency_ms: latency,
      body: parsedBody
    });
    setResponseMeta(`${endpoint} • ${response.status} • ${latency} ms`);

    if (response.ok && parsedBody && typeof parsedBody === "object") {
      syncDownstream(endpoint, requestBody, parsedBody);
    }
  } catch (error) {
    setOutput({ error: error.message });
    setResponseMeta(`Request failed • ${endpoint}`);
  }
}

function buildMetricsPayloadForFileUpdate(current, updates = {}) {
  return {
    request_id: current.request_id || "req-2026-03-30-0005",
    actual_path: updates.actual_path ?? current.actual_path ?? "",
    prediction_path: updates.prediction_path ?? current.prediction_path ?? "",
    actual_column: current.actual_column || "y_true",
    prediction_column: current.prediction_column || "y_pred"
  };
}

function updateEditorFromUpload(target, uploadResponse) {
  const relativePath = uploadResponse?.relative_path || uploadResponse?.path || "";

  if (target === "batch-input") {
    const current = parseEditorSafe("batch-body", samples.batch);
    writeEditor("batch-body", {
      ...current,
      points: [],
      input_path: relativePath
    });
    setUploadStatus(target, `Загружено и подставлено в /batch: ${relativePath}`, "ready");
    return;
  }

  if (target === "metrics-actual") {
    const current = parseEditorSafe("metrics-body", samples.metrics);
    writeEditor("metrics-body", buildMetricsPayloadForFileUpdate(current, { actual_path: relativePath }));
    setUploadStatus(target, `Загружено и подставлено в actual_path: ${relativePath}`, "ready");
    return;
  }

  if (target === "metrics-prediction") {
    const current = parseEditorSafe("metrics-body", samples.metrics);
    writeEditor("metrics-body", buildMetricsPayloadForFileUpdate(current, { prediction_path: relativePath }));
    setUploadStatus(target, `Загружено и подставлено в prediction_path: ${relativePath}`, "ready");
  }
}

async function uploadFile(target) {
  const input = byId(`upload-${target}`);
  const file = input?.files?.[0];
  if (!file) {
    setResponseMeta("Choose a file first");
    setUploadStatus(target, "Сначала выберите файл через серую кнопку", "error");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  try {
    setResponseMeta(`Uploading ${file.name}...`);
    setUploadStatus(target, `Загружаю ${file.name}...`);
    const response = await fetch("/files/upload", {
      method: "POST",
      body: formData
    });
    const text = await response.text();
    const payload = text ? JSON.parse(text) : {};

    setOutput({
      status: response.status,
      body: payload
    });

    if (!response.ok) {
      setResponseMeta(`Upload failed • ${response.status}`);
      setUploadStatus(target, payload?.error || `Ошибка загрузки: ${response.status}`, "error");
      return;
    }

    updateEditorFromUpload(target, payload);
    setResponseMeta(`Uploaded ${payload.filename} • ${payload.relative_path}`);
    input.value = "";
  } catch (error) {
    setOutput({ error: error.message });
    setResponseMeta("Upload failed");
    setUploadStatus(target, `Ошибка загрузки: ${error.message}`, "error");
  }
}

async function healthCheckInternal(forceOutput) {
  try {
    const response = await fetch("/healthz");
    const payload = await response.json();

    const apiStatus = payload?.services?.api === "up" ? "up" : "down";
    const mlStatus = payload?.services?.ml === "up" ? "up" : "down";

    setStatusChip("api-status", `API: ${payload.services.api}`, apiStatus);
    setStatusChip("ml-status", `ML: ${payload.services.ml}`, mlStatus);
    byId("topology-mode").textContent = mlStatus === "up" ? "Go API + Python ML" : "Go API + fallback visibility";

    if (forceOutput) {
      setOutput({
        status: response.status,
        body: payload
      });
      setResponseMeta(`/healthz • ${response.status}`);
    }
  } catch (error) {
    setStatusChip("api-status", "API: down", "down");
    setStatusChip("ml-status", "ML: unknown", "down");

    if (forceOutput) {
      setOutput({ error: error.message });
      setResponseMeta("Health check failed");
    }
  }
}

function canRunDemoMetrics(request) {
  return Boolean(request?.actual_path && request?.prediction_path);
}

async function runDemoChain() {
  try {
    const predictRequest = parseEditorSafe("predict-body", samples.predict);
    setResponseMeta("Demo chain • predict");

    const predictResponse = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(predictRequest)
    });
    const predictPayload = await predictResponse.json();
    syncDecisionFromPredict(predictRequest, predictPayload);

    const decisionRequest = parseEditorSafe("decision-body", {});
    setResponseMeta("Demo chain • decision");
    const decisionResponse = await fetch("/decision", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(decisionRequest)
    });
    const decisionPayload = await decisionResponse.json();

    const batchRequest = {
      ...parseEditorSafe("batch-body", samples.batch),
      return_predictions: true
    };
    writeEditor("batch-body", batchRequest);

    setResponseMeta("Demo chain • batch");
    const batchResponse = await fetch("/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(batchRequest)
    });
    const batchPayload = await batchResponse.json();
    syncMetricsFromBatch(batchRequest, batchPayload);

    const metricsRequest = parseEditorSafe("metrics-body", samples.metrics);
    let metricsStatus = null;
    let metricsPayload = {
      skipped: true,
      reason: "Load actual.csv first to run an honest metrics check"
    };

    if (canRunDemoMetrics(metricsRequest)) {
      setResponseMeta("Demo chain • metrics");
      const metricsResponse = await fetch("/metrics", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(metricsRequest)
      });
      metricsStatus = metricsResponse.status;
      metricsPayload = await metricsResponse.json();
    }

    setOutput({
      predict: { status: predictResponse.status, body: predictPayload },
      decision: { status: decisionResponse.status, body: decisionPayload },
      batch: { status: batchResponse.status, body: batchPayload },
      metrics: { status: metricsStatus, body: metricsPayload }
    });
    setResponseMeta("Demo chain completed");
  } catch (error) {
    setOutput({ error: error.message });
    setResponseMeta("Demo chain failed");
  }
}

function connectLogStream(endpoint, targetId) {
  const target = byId(targetId);
  const entries = [];
  const maxEntries = 120;
  const source = new EventSource(endpoint);

  source.addEventListener("log", (event) => {
    try {
      const payload = JSON.parse(event.data);
      entries.push(payload);
      if (entries.length > maxEntries) {
        entries.shift();
      }
      renderLogs(targetId, { entries });
    } catch (error) {
      target.textContent = `Stream decode error: ${error.message}`;
    }
  });

  source.onerror = () => {
    target.textContent = "Telemetry stream is reconnecting...";
  };
}

document.querySelectorAll("[data-endpoint]").forEach((button) => {
  button.addEventListener("click", () => sendRequest(button.dataset.endpoint, button.dataset.source));
});
document.querySelectorAll("[data-upload-target]").forEach((button) => {
  button.addEventListener("click", () => uploadFile(button.dataset.uploadTarget));
});
document.querySelectorAll('input[type="file"][id^="upload-"]').forEach((input) => {
  input.addEventListener("change", () => {
    const suffix = input.id.replace("upload-", "");
    const file = input.files?.[0];
    setUploadStatus(suffix, file ? `Выбран файл: ${file.name}` : "Файл не выбран");
  });
});

document.querySelector('[data-action="fill-all"]').addEventListener("click", fillAll);
document.querySelector('[data-action="health"]').addEventListener("click", () => healthCheckInternal(true));
document.querySelector('[data-action="demo"]').addEventListener("click", runDemoChain);

fillAll();
healthCheckInternal(false);
setInterval(() => healthCheckInternal(false), 30000);
connectLogStream("/stream/logs/go?limit=80", "go-logs-output");
connectLogStream("/stream/logs/ml?limit=80", "ml-logs-output");
