package service

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"wildberries-storage/internal/domain"
	"wildberries-storage/internal/storage"
)

type fakeBatchPredictor struct{}

func (f fakeBatchPredictor) Predict(_ context.Context, req domain.ForecastRequest) (domain.ForecastResponse, error) {
	predictions := make([]domain.Prediction, 0, len(req.Points))
	for _, point := range req.Points {
		predictions = append(predictions, domain.Prediction{
			ID:    point.ID,
			YPred: 15.5,
		})
	}

	return domain.ForecastResponse{
		RequestID:   req.RequestID,
		Predictions: predictions,
		Model:       "fake-model",
	}, nil
}

type fakeDatasetLoader struct{}

func (f fakeDatasetLoader) LoadPointsFromDataset(_ context.Context, inputPath string) ([]domain.ForecastPoint, error) {
	return []domain.ForecastPoint{
		{
			ID:        0,
			RouteID:   0,
			Timestamp: time.Date(2026, 3, 30, 12, 30, 0, 0, time.UTC),
		},
	}, nil
}

func TestBatchServiceRunLoadsParquetViaDatasetLoader(t *testing.T) {
	tempDir := t.TempDir()
	fileStore := storage.NewFileStore(tempDir)
	forecast := NewForecastService(fakeBatchPredictor{}, 0, 4)
	service := NewBatchService(forecast, fileStore, filepath.Join(tempDir, "artifacts"), fakeDatasetLoader{})

	response, err := service.Run(context.Background(), domain.BatchRequest{
		RequestID:  "req-batch-1",
		InputPath:  "test_team_track.parquet",
		OutputPath: filepath.Join("artifacts", "submission.csv"),
	})
	if err != nil {
		t.Fatalf("Run returned error: %v", err)
	}

	if response.TotalPoints != 1 {
		t.Fatalf("expected 1 prediction, got %d", response.TotalPoints)
	}

	content, err := os.ReadFile(response.OutputPath)
	if err != nil {
		t.Fatalf("read output file: %v", err)
	}

	expected := "id,y_pred\n0,15.5\n"
	if string(content) != expected {
		t.Fatalf("unexpected submission content: %q", string(content))
	}
}
