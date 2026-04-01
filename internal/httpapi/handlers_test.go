package httpapi

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"wildberries-storage/internal/domain"
	"wildberries-storage/internal/observability"
	"wildberries-storage/internal/service"
	"wildberries-storage/internal/storage"
)

type fakePredictor struct{}

func (f fakePredictor) Predict(_ context.Context, req domain.ForecastRequest) (domain.ForecastResponse, error) {
	predictions := make([]domain.Prediction, 0, len(req.Points))
	for _, point := range req.Points {
		predictions = append(predictions, domain.Prediction{
			ID:    point.ID,
			YPred: 12.5,
		})
	}

	return domain.ForecastResponse{
		RequestID:   req.RequestID,
		Predictions: predictions,
		Model:       "stub-model",
	}, nil
}

type fakeSelector struct{}

func (f fakeSelector) SelectModel(_ context.Context, req domain.ModelSelectRequest) (domain.ModelSelectResponse, error) {
	return domain.ModelSelectResponse{
		RequestID:     req.RequestID,
		SelectedModel: "stub-model",
	}, nil
}

type fakeHealth struct{}

func (f fakeHealth) Ping(_ context.Context) error { return nil }
func (f fakeHealth) Logs(_ context.Context, _ int) ([]observability.Entry, error) {
	return []observability.Entry{}, nil
}

func TestPredictHandler(t *testing.T) {
	forecastService := service.NewForecastService(fakePredictor{}, 0, 4)
	decisionService := service.NewDecisionService(forecastService, 0.1, 20, 50)
	batchService := service.NewBatchService(forecastService, storage.NewFileStore(t.TempDir()), t.TempDir())
	metricsService := service.NewMetricsService(storage.NewFileStore(t.TempDir()))
	modelSelectService := service.NewModelSelectService(fakeSelector{})

	handlers := NewHandlers(
		forecastService,
		decisionService,
		batchService,
		metricsService,
		modelSelectService,
		fakeHealth{},
		observability.NewLogger(50),
	)

	body, err := json.Marshal(domain.ForecastRequest{
		RequestID: "req-1",
		Points: []domain.ForecastPoint{
			{
				ID:        1,
				RouteID:   11,
				Timestamp: time.Date(2026, 4, 1, 12, 0, 0, 0, time.UTC),
			},
		},
	})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	request := httptest.NewRequest(http.MethodPost, "/predict", bytes.NewReader(body))
	recorder := httptest.NewRecorder()

	handlers.Predict(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", recorder.Code)
	}

	var response domain.ForecastResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}

	if len(response.Predictions) != 1 || response.Predictions[0].YPred != 12.5 {
		t.Fatalf("unexpected predictions: %+v", response.Predictions)
	}
}
