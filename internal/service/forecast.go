package service

import (
	"context"
	"fmt"
	"math"
	"time"

	"wildberries-storage/internal/domain"
)

type Predictor interface {
	Predict(ctx context.Context, req domain.ForecastRequest) (domain.ForecastResponse, error)
}

type ForecastService struct {
	predictor Predictor
	floor     float64
	precision int
}

func NewForecastService(predictor Predictor, floor float64, precision int) *ForecastService {
	return &ForecastService{
		predictor: predictor,
		floor:     floor,
		precision: precision,
	}
}

func (s *ForecastService) Predict(ctx context.Context, req domain.ForecastRequest) (domain.ForecastResponse, error) {
	if len(req.Points) == 0 {
		return domain.ForecastResponse{}, fmt.Errorf("points are required")
	}

	req.RequestID = ensureRequestID(req.RequestID)
	if err := validatePoints(req.Points); err != nil {
		return domain.ForecastResponse{}, err
	}

	response, err := s.predictor.Predict(ctx, req)
	if err != nil {
		return domain.ForecastResponse{}, err
	}

	if response.RequestID == "" {
		response.RequestID = req.RequestID
	}

	ordered, err := orderPredictions(req.Points, response.Predictions)
	if err != nil {
		return domain.ForecastResponse{}, err
	}

	for index := range ordered {
		ordered[index].YPred = round(maxFloat(s.floor, ordered[index].YPred), s.precision)
	}

	response.Predictions = ordered
	return response, nil
}

func validatePoints(points []domain.ForecastPoint) error {
	seen := make(map[int64]struct{}, len(points))
	for _, point := range points {
		if point.ID <= 0 {
			return fmt.Errorf("point id must be positive")
		}
		if point.RouteID <= 0 {
			return fmt.Errorf("route_id must be positive for point %d", point.ID)
		}
		if point.Timestamp.IsZero() {
			return fmt.Errorf("timestamp is required for point %d", point.ID)
		}
		if _, exists := seen[point.ID]; exists {
			return fmt.Errorf("duplicate point id %d", point.ID)
		}
		seen[point.ID] = struct{}{}
	}
	return nil
}

func orderPredictions(points []domain.ForecastPoint, predictions []domain.Prediction) ([]domain.Prediction, error) {
	if len(predictions) != len(points) {
		return nil, fmt.Errorf("ml service returned %d predictions for %d points", len(predictions), len(points))
	}

	byID := make(map[int64]domain.Prediction, len(predictions))
	for _, prediction := range predictions {
		byID[prediction.ID] = prediction
	}

	ordered := make([]domain.Prediction, 0, len(points))
	for _, point := range points {
		prediction, ok := byID[point.ID]
		if !ok {
			return nil, fmt.Errorf("missing prediction for point id %d", point.ID)
		}
		ordered = append(ordered, prediction)
	}

	return ordered, nil
}

func ensureRequestID(requestID string) string {
	if requestID != "" {
		return requestID
	}
	return fmt.Sprintf("req-%d", time.Now().UnixNano())
}

func round(value float64, precision int) float64 {
	if precision < 0 {
		return value
	}
	multiplier := math.Pow(10, float64(precision))
	return math.Round(value*multiplier) / multiplier
}

func maxFloat(left, right float64) float64 {
	if left > right {
		return left
	}
	return right
}
