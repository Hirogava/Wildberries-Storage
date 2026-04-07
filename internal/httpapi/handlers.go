package httpapi

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"wildberries-storage/internal/domain"
	"wildberries-storage/internal/observability"
	"wildberries-storage/internal/service"
)

type MLHealthChecker interface {
	Ping(ctx context.Context) error
}

type MLLogsReader interface {
	Logs(ctx context.Context, limit int) ([]observability.Entry, error)
}

type MLLogsStreamer interface {
	OpenLogsStream(ctx context.Context, limit int) (*http.Response, error)
}

type Handlers struct {
	forecast    *service.ForecastService
	decision    *service.DecisionService
	batch       *service.BatchService
	metrics     *service.MetricsService
	modelSelect *service.ModelSelectService
	mlHealth    MLHealthChecker
	mlLogs      MLLogsReader
	mlStream    MLLogsStreamer
	logger      *observability.Logger
}

func NewHandlers(
	forecast *service.ForecastService,
	decision *service.DecisionService,
	batch *service.BatchService,
	metrics *service.MetricsService,
	modelSelect *service.ModelSelectService,
	mlHealth MLHealthChecker,
	logger *observability.Logger,
) *Handlers {
	var mlLogs MLLogsReader
	if typed, ok := mlHealth.(MLLogsReader); ok {
		mlLogs = typed
	}
	var mlStream MLLogsStreamer
	if typed, ok := mlHealth.(MLLogsStreamer); ok {
		mlStream = typed
	}

	return &Handlers{
		forecast:    forecast,
		decision:    decision,
		batch:       batch,
		metrics:     metrics,
		modelSelect: modelSelect,
		mlHealth:    mlHealth,
		mlLogs:      mlLogs,
		mlStream:    mlStream,
		logger:      logger,
	}
}

func (h *Handlers) Health(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	status := domain.HealthResponse{
		Status: "ok",
		Services: map[string]string{
			"api": "up",
			"ml":  "unknown",
		},
		Timestamp: time.Now().UTC(),
	}

	if h.mlHealth != nil {
		ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
		defer cancel()

		if err := h.mlHealth.Ping(ctx); err != nil {
			status.Status = "degraded"
			status.Services["ml"] = "down"
		} else {
			status.Services["ml"] = "up"
		}
	}

	writeJSON(w, http.StatusOK, status)
}

func (h *Handlers) Predict(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var request domain.ForecastRequest
	if err := decodeJSON(r, &request); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if h.logger != nil {
		h.logger.Info("http", "POST /predict request_id=%s", request.RequestID)
	}

	response, err := h.forecast.Predict(r.Context(), request)
	if err != nil {
		if h.logger != nil {
			h.logger.Error("http", "POST /predict failed: %v", err)
		}
		writeServiceError(w, err)
		return
	}
	if h.logger != nil {
		h.logger.Info("http", "POST /predict completed request_id=%s predictions=%d", response.RequestID, len(response.Predictions))
	}

	writeJSON(w, http.StatusOK, response)
}

func (h *Handlers) Decide(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var request domain.DecisionRequest
	if err := decodeJSON(r, &request); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if h.logger != nil {
		h.logger.Info("http", "POST /decision request_id=%s points=%d predictions=%d", request.RequestID, len(request.Points), len(request.Predictions))
	}

	response, err := h.decision.Decide(r.Context(), request)
	if err != nil {
		if h.logger != nil {
			h.logger.Error("http", "POST /decision failed: %v", err)
		}
		writeServiceError(w, err)
		return
	}
	if h.logger != nil {
		h.logger.Info("http", "POST /decision completed request_id=%s decisions=%d", response.RequestID, len(response.Decisions))
	}

	writeJSON(w, http.StatusOK, response)
}

func (h *Handlers) Batch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var request domain.BatchRequest
	if err := decodeJSON(r, &request); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if h.logger != nil {
		h.logger.Info("http", "POST /batch request_id=%s points=%d input_path=%s", request.RequestID, len(request.Points), request.InputPath)
	}

	response, err := h.batch.Run(r.Context(), request)
	if err != nil {
		if h.logger != nil {
			h.logger.Error("http", "POST /batch failed: %v", err)
		}
		writeServiceError(w, err)
		return
	}
	if h.logger != nil {
		h.logger.Info("http", "POST /batch completed request_id=%s output_path=%s", response.RequestID, response.OutputPath)
	}

	writeJSON(w, http.StatusOK, response)
}

func (h *Handlers) Metrics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var request domain.MetricsRequest
	if err := decodeJSON(r, &request); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if h.logger != nil {
		h.logger.Info("http", "POST /metrics request_id=%s observations=%d", request.RequestID, len(request.Observations))
	}

	response, err := h.metrics.Calculate(request)
	if err != nil {
		if h.logger != nil {
			h.logger.Error("http", "POST /metrics failed: %v", err)
		}
		writeServiceError(w, err)
		return
	}
	if h.logger != nil {
		h.logger.Info("http", "POST /metrics completed request_id=%s score=%.6f", response.RequestID, response.Score)
	}

	writeJSON(w, http.StatusOK, response)
}

func (h *Handlers) ModelSelect(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var request domain.ModelSelectRequest
	if err := decodeJSON(r, &request); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if h.logger != nil {
		h.logger.Info("http", "POST /model/select request_id=%s candidates=%d", request.RequestID, len(request.Candidates))
	}

	response, err := h.modelSelect.Select(r.Context(), request)
	if err != nil {
		if h.logger != nil {
			h.logger.Error("http", "POST /model/select failed: %v", err)
		}
		writeServiceError(w, err)
		return
	}
	if h.logger != nil {
		h.logger.Info("http", "POST /model/select completed request_id=%s selected_model=%s", response.RequestID, response.SelectedModel)
	}

	writeJSON(w, http.StatusOK, response)
}

func (h *Handlers) GoLogs(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	limit := parseLimit(r, 80)
	if h.logger == nil {
		writeJSON(w, http.StatusOK, map[string]any{
			"service": "go-api",
			"entries": []observability.Entry{},
		})
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"service": "go-api",
		"entries": h.logger.Entries(limit),
	})
}

func (h *Handlers) MLLogs(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	limit := parseLimit(r, 80)
	if h.mlLogs == nil {
		writeError(w, http.StatusBadGateway, "ml logs reader is not configured")
		return
	}

	entries, err := h.mlLogs.Logs(r.Context(), limit)
	if err != nil {
		writeServiceError(w, err)
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"service": "ml-stub",
		"entries": entries,
	})
}

func (h *Handlers) GoLogsStream(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if h.logger == nil {
		writeError(w, http.StatusServiceUnavailable, "logger is not configured")
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming is not supported")
		return
	}

	limit := parseLimit(r, 80)
	setSSEHeaders(w)

	for _, entry := range h.logger.Entries(limit) {
		if err := writeSSEEntry(w, entry); err != nil {
			return
		}
	}
	flusher.Flush()

	entriesCh, cancel := h.logger.Subscribe()
	defer cancel()

	keepAlive := time.NewTicker(15 * time.Second)
	defer keepAlive.Stop()

	for {
		select {
		case <-r.Context().Done():
			return
		case entry, ok := <-entriesCh:
			if !ok {
				return
			}
			if err := writeSSEEntry(w, entry); err != nil {
				return
			}
			flusher.Flush()
		case <-keepAlive.C:
			if _, err := fmt.Fprint(w, ": keepalive\n\n"); err != nil {
				return
			}
			flusher.Flush()
		}
	}
}

func (h *Handlers) MLLogsStream(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if h.mlStream == nil {
		writeError(w, http.StatusBadGateway, "ml stream is not configured")
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming is not supported")
		return
	}

	limit := parseLimit(r, 80)
	upstream, err := h.mlStream.OpenLogsStream(r.Context(), limit)
	if err != nil {
		writeServiceError(w, err)
		return
	}
	defer upstream.Body.Close()

	setSSEHeaders(w)
	reader := bufio.NewReader(upstream.Body)

	for {
		line, err := reader.ReadBytes('\n')
		if len(line) > 0 {
			if _, writeErr := w.Write(line); writeErr != nil {
				return
			}
			flusher.Flush()
		}
		if err != nil {
			if errors.Is(err, io.EOF) {
				return
			}
			return
		}
	}
}

func decodeJSON(r *http.Request, target any) error {
	defer r.Body.Close()

	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()

	if err := decoder.Decode(target); err != nil {
		return err
	}
	if decoder.More() {
		return errors.New("request body must contain a single json object")
	}

	return nil
}

func writeJSON(w http.ResponseWriter, statusCode int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	_ = json.NewEncoder(w).Encode(payload)
}

func writeError(w http.ResponseWriter, statusCode int, message string) {
	writeJSON(w, statusCode, map[string]string{
		"error": message,
	})
}

func writeServiceError(w http.ResponseWriter, err error) {
	statusCode := http.StatusBadRequest
	if strings.Contains(err.Error(), "ml service") || strings.Contains(err.Error(), "ml health") || strings.Contains(err.Error(), "call ml") {
		statusCode = http.StatusBadGateway
	}
	writeError(w, statusCode, err.Error())
}

func parseLimit(r *http.Request, fallback int) int {
	raw := r.URL.Query().Get("limit")
	if raw == "" {
		return fallback
	}

	value, err := strconv.Atoi(raw)
	if err != nil || value <= 0 {
		return fallback
	}
	return value
}

func setSSEHeaders(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
}

func writeSSEEntry(w http.ResponseWriter, entry observability.Entry) error {
	payload, err := observability.MarshalEntry(entry)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "event: log\ndata: %s\n\n", payload)
	return err
}
