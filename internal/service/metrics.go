package service

import (
	"fmt"

	"wildberries-storage/internal/domain"
	"wildberries-storage/internal/storage"
)

type MetricsService struct {
	loader *storage.FileStore
}

func NewMetricsService(loader *storage.FileStore) *MetricsService {
	return &MetricsService{loader: loader}
}

func (s *MetricsService) Calculate(req domain.MetricsRequest) (domain.MetricsResponse, error) {
	req.RequestID = ensureRequestID(req.RequestID)

	observations := req.Observations
	if len(observations) == 0 {
		if s.loader == nil || req.ActualPath == "" || req.PredictionPath == "" {
			return domain.MetricsResponse{}, fmt.Errorf("either observations or both file paths must be provided")
		}

		loaded, err := s.loader.LoadObservations(req.ActualPath, req.PredictionPath, req.ActualColumn, req.PredictionColumn)
		if err != nil {
			return domain.MetricsResponse{}, err
		}
		observations = loaded
	}

	if len(observations) == 0 {
		return domain.MetricsResponse{}, fmt.Errorf("observations are empty")
	}

	var sumAbsError, sumTrue, sumPred float64
	for _, observation := range observations {
		sumAbsError += absFloat(observation.YPred - observation.YTrue)
		sumTrue += observation.YTrue
		sumPred += observation.YPred
	}

	if sumTrue == 0 {
		return domain.MetricsResponse{}, fmt.Errorf("sum of y_true must be positive")
	}

	wape := sumAbsError / sumTrue
	relativeBias := absFloat(sumPred/sumTrue - 1)

	return domain.MetricsResponse{
		RequestID:    req.RequestID,
		Count:        len(observations),
		WAPE:         round(wape, 6),
		RelativeBias: round(relativeBias, 6),
		Score:        round(wape+relativeBias, 6),
	}, nil
}

func absFloat(value float64) float64 {
	if value < 0 {
		return -value
	}
	return value
}
