package service

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"wildberries-storage/internal/domain"
	"wildberries-storage/internal/storage"
)

const batchPredictChunkSize = 1000

type DatasetPointsLoader interface {
	LoadPointsFromDataset(ctx context.Context, inputPath string) ([]domain.ForecastPoint, error)
}

type BatchService struct {
	forecast      *ForecastService
	fileStore     *storage.FileStore
	submissionDir string
	datasetLoader DatasetPointsLoader
}

func NewBatchService(
	forecast *ForecastService,
	fileStore *storage.FileStore,
	submissionDir string,
	datasetLoader DatasetPointsLoader,
) *BatchService {
	return &BatchService{
		forecast:      forecast,
		fileStore:     fileStore,
		submissionDir: submissionDir,
		datasetLoader: datasetLoader,
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
			if strings.EqualFold(filepath.Ext(req.InputPath), ".parquet") && s.datasetLoader != nil {
				loaded, err = s.datasetLoader.LoadPointsFromDataset(ctx, req.InputPath)
			}
			if err != nil {
				return domain.BatchResponse{}, err
			}
		}
		points = loaded
	}

	response, err := s.predictInChunks(ctx, req.RequestID, points)
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

func (s *BatchService) predictInChunks(ctx context.Context, requestID string, points []domain.ForecastPoint) (domain.ForecastResponse, error) {
	if len(points) <= batchPredictChunkSize {
		return s.forecast.Predict(ctx, domain.ForecastRequest{
			RequestID: requestID,
			Points:    points,
		})
	}

	predictions := make([]domain.Prediction, 0, len(points))
	for start := 0; start < len(points); start += batchPredictChunkSize {
		end := start + batchPredictChunkSize
		if end > len(points) {
			end = len(points)
		}

		chunkRequestID := fmt.Sprintf("%s-chunk-%d", requestID, start/batchPredictChunkSize+1)
		chunkResponse, err := s.forecast.Predict(ctx, domain.ForecastRequest{
			RequestID: chunkRequestID,
			Points:    points[start:end],
		})
		if err != nil {
			return domain.ForecastResponse{}, err
		}

		predictions = append(predictions, chunkResponse.Predictions...)
	}

	return domain.ForecastResponse{
		RequestID:   requestID,
		Predictions: predictions,
	}, nil
}
