package service

import (
	"context"
	"fmt"
	"path/filepath"
	"time"

	"wildberries-storage/internal/domain"
	"wildberries-storage/internal/storage"
)

type BatchService struct {
	forecast      *ForecastService
	fileStore     *storage.FileStore
	submissionDir string
}

func NewBatchService(forecast *ForecastService, fileStore *storage.FileStore, submissionDir string) *BatchService {
	return &BatchService{
		forecast:      forecast,
		fileStore:     fileStore,
		submissionDir: submissionDir,
	}
}

func (s *BatchService) Run(ctx context.Context, req domain.BatchRequest) (domain.BatchResponse, error) {
	req.RequestID = ensureRequestID(req.RequestID)

	points := req.Points
	if len(points) == 0 {
		if req.InputPath == "" {
			return domain.BatchResponse{}, fmt.Errorf("either points or input_path must be provided")
		}

		loaded, err := s.fileStore.LoadPoints(req.InputPath)
		if err != nil {
			return domain.BatchResponse{}, err
		}
		points = loaded
	}

	response, err := s.forecast.Predict(ctx, domain.ForecastRequest{
		RequestID: req.RequestID,
		Points:    points,
	})
	if err != nil {
		return domain.BatchResponse{}, err
	}

	outputPath := req.OutputPath
	if outputPath == "" {
		filename := fmt.Sprintf("submission_%s.csv", time.Now().Format("20060102_150405"))
		outputPath = filepath.Join(s.submissionDir, filename)
	}

	resolvedOutput, err := s.fileStore.WriteSubmission(outputPath, response.Predictions)
	if err != nil {
		return domain.BatchResponse{}, err
	}

	batchResponse := domain.BatchResponse{
		RequestID:   req.RequestID,
		OutputPath:  resolvedOutput,
		TotalPoints: len(response.Predictions),
	}
	if req.ReturnPredictions {
		batchResponse.Predictions = response.Predictions
	}

	return batchResponse, nil
}
