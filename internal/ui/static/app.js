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
      { id: 100002, route_id: 12345, timestamp: "2026-03-30T13:00:00Z" }
    ],
    predictions: [
      { id: 100001, y_pred: 19.4 },
      { id: 100002, y_pred: 20.1 }
    ],
    safety_factor: 0.1,
    truck_capacity: 20,
    max_trucks_per_route: 50
  },
  batch: {
    request_id: "req-2026-03-30-0004",
    points: [
      { id: 100001, route_id: 12345, timestamp: "2026-03-30T12:30:00Z" },
      { id: 100002, route_id: 12345, timestamp: "2026-03-30T13:00:00Z" }
    ],
    output_path: "artifacts/submissions/demo_submission.csv",
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
    candidates: ["stub_v1", "ridge_v1", "lgbm_v2"],
    objective: "wape_plus_rbias",
    context: {
      dataset: "team_track_validation",
      warehouse: "all"
    }
  }
};

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function setOutput(value) {
  document.getElementById("response-output").textContent =
    typeof value === "string" ? value : pretty(value);
}

function renderLogs(targetId, payload) {
  const entries = Array.isArray(payload?.entries) ? payload.entries : [];
  const text = entries.length
    ? entries
        .map((entry) => `[${entry.timestamp}] [${entry.level}] [${entry.component}] ${entry.message}`)
        .join("\n")
    : "Логи пока пусты.";

  document.getElementById(targetId).textContent = text;
}

function fillAll() {
  document.getElementById("predict-body").value = pretty(samples.predict);
  document.getElementById("decision-body").value = pretty(samples.decision);
  document.getElementById("batch-body").value = pretty(samples.batch);
  document.getElementById("metrics-body").value = pretty(samples.metrics);
  document.getElementById("model-body").value = pretty(samples.model);
}

async function sendRequest(endpoint, source) {
  try {
    const raw = document.getElementById(source).value.trim();
    const body = raw ? JSON.parse(raw) : {};

    setOutput(`Отправка ${endpoint}...`);

    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });

    const text = await response.text();
    try {
      setOutput({
        status: response.status,
        body: JSON.parse(text)
      });
    } catch {
      setOutput({
        status: response.status,
        body: text
      });
    }
  } catch (error) {
    setOutput({ error: error.message });
  }
}

async function healthCheck() {
  try {
    setOutput("Проверка /healthz...");
    const response = await fetch("/healthz");
    setOutput({
      status: response.status,
      body: await response.json()
    });
  } catch (error) {
    setOutput({ error: error.message });
  }
}

function connectLogStream(endpoint, targetId) {
  const target = document.getElementById(targetId);
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
      target.textContent = `Ошибка парсинга лога: ${error.message}`;
    }
  });

  source.onerror = () => {
    target.textContent = "Поток логов переподключается...";
  };
}

document.querySelectorAll("[data-endpoint]").forEach((button) => {
  button.addEventListener("click", () => {
    sendRequest(button.dataset.endpoint, button.dataset.source);
  });
});

document.querySelector('[data-action="fill-all"]').addEventListener("click", fillAll);
document.querySelector('[data-action="health"]').addEventListener("click", healthCheck);

fillAll();
connectLogStream("/stream/logs/go?limit=80", "go-logs-output");
connectLogStream("/stream/logs/ml?limit=80", "ml-logs-output");
