package service

import (
	"context"
	"fmt"
	"math"

	"wildberries-storage/internal/domain"
)

type DecisionService struct {
	forecast            *ForecastService
	defaultSafetyFactor float64
	defaultTruckCap     float64
	maxTrucksPerRoute   int
}

func NewDecisionService(
	forecast *ForecastService,
	defaultSafetyFactor float64,
	defaultTruckCap float64,
	maxTrucksPerRoute int,
) *DecisionService {
	return &DecisionService{
		forecast:            forecast,
		defaultSafetyFactor: defaultSafetyFactor,
		defaultTruckCap:     defaultTruckCap,
		maxTrucksPerRoute:   maxTrucksPerRoute,
	}
}

func (s *DecisionService) Decide(ctx context.Context, req domain.DecisionRequest) (domain.DecisionResponse, error) {
	requestID := ensureRequestID(req.RequestID)
	safetyFactor := valueOrDefaultFloat(req.SafetyFactor, s.defaultSafetyFactor)
	truckCapacity := valueOrDefaultFloat(req.TruckCapacity, s.defaultTruckCap)
	maxTrucks := valueOrDefaultInt(req.MaxTrucksPerRoute, s.maxTrucksPerRoute)

	if safetyFactor < 0 {
		return domain.DecisionResponse{}, fmt.Errorf("safety_factor must be non-negative")
	}
	if truckCapacity <= 0 {
		return domain.DecisionResponse{}, fmt.Errorf("truck_capacity must be positive")
	}
	if maxTrucks <= 0 {
		return domain.DecisionResponse{}, fmt.Errorf("max_trucks_per_route must be positive")
	}

	predictions, err := s.resolvePredictions(ctx, req)
	if err != nil {
		return domain.DecisionResponse{}, err
	}

	decisions := make([]domain.Decision, 0, len(predictions))
	for _, prediction := range predictions {
		adjustedLoad := prediction.YPred * (1 + safetyFactor)
		trucksNeeded := 0
		action := "NO_ACTION"
		reason := "Forecasted load is zero, no truck request is created."

		if adjustedLoad > 0 {
			trucksNeeded = int(math.Ceil(adjustedLoad / truckCapacity))
			action = "CALL_TRUCKS"
			reason = "Truck request is created from forecast, safety buffer and standard truck capacity."
		}

		if trucksNeeded > maxTrucks {
			trucksNeeded = maxTrucks
			reason = "Truck request was capped by max_trucks_per_route."
		}

		decisions = append(decisions, domain.Decision{
			ID:             prediction.ID,
			RouteID:        prediction.RouteID,
			Timestamp:      prediction.Timestamp,
			YPred:          prediction.YPred,
			AdjustedLoad:   round(adjustedLoad, 4),
			TrucksNeeded:   trucksNeeded,
			TruckCapacity:  truckCapacity,
			SafetyFactor:   safetyFactor,
			Action:         action,
			DecisionReason: reason,
		})
	}

	return domain.DecisionResponse{
		RequestID: requestID,
		Decisions: decisions,
	}, nil
}

func (s *DecisionService) resolvePredictions(ctx context.Context, req domain.DecisionRequest) ([]domain.DecisionPrediction, error) {
	if len(req.Predictions) > 0 {
		return enrichDecisionPredictions(req.Predictions, req.Points)
	}

	if len(req.Points) == 0 {
		return nil, fmt.Errorf("either predictions or points must be provided")
	}

	forecastResponse, err := s.forecast.Predict(ctx, domain.ForecastRequest{
		RequestID: req.RequestID,
		Points:    req.Points,
	})
	if err != nil {
		return nil, err
	}

	byID := make(map[int64]float64, len(forecastResponse.Predictions))
	for _, prediction := range forecastResponse.Predictions {
		byID[prediction.ID] = prediction.YPred
	}

	predictions := make([]domain.DecisionPrediction, 0, len(req.Points))
	for _, point := range req.Points {
		predictions = append(predictions, domain.DecisionPrediction{
			ID:        point.ID,
			RouteID:   point.RouteID,
			Timestamp: point.Timestamp,
			YPred:     byID[point.ID],
		})
	}

	return predictions, nil
}

func enrichDecisionPredictions(predictions []domain.DecisionPrediction, points []domain.ForecastPoint) ([]domain.DecisionPrediction, error) {
	pointByID := make(map[int64]domain.ForecastPoint, len(points))
	for _, point := range points {
		pointByID[point.ID] = point
	}

	seen := make(map[int64]struct{}, len(predictions))
	normalized := make([]domain.DecisionPrediction, 0, len(predictions))

	for _, prediction := range predictions {
		if prediction.ID < 0 {
			return nil, fmt.Errorf("prediction id must be non-negative")
		}
		if prediction.YPred < 0 {
			prediction.YPred = 0
		}
		if _, exists := seen[prediction.ID]; exists {
			return nil, fmt.Errorf("duplicate prediction id %d", prediction.ID)
		}

		if (prediction.RouteID <= 0 || prediction.Timestamp.IsZero()) && len(pointByID) > 0 {
			point, ok := pointByID[prediction.ID]
			if !ok {
				return nil, fmt.Errorf("missing point metadata for prediction %d", prediction.ID)
			}

			if prediction.RouteID <= 0 {
				prediction.RouteID = point.RouteID
			}
			if prediction.Timestamp.IsZero() {
				prediction.Timestamp = point.Timestamp
			}
		}

		if prediction.RouteID < 0 {
			return nil, fmt.Errorf("route_id must be non-negative for prediction %d", prediction.ID)
		}
		if prediction.Timestamp.IsZero() {
			return nil, fmt.Errorf("timestamp is required for prediction %d", prediction.ID)
		}

		seen[prediction.ID] = struct{}{}
		normalized = append(normalized, prediction)
	}

	return normalized, nil
}

func valueOrDefaultFloat(value *float64, fallback float64) float64 {
	if value == nil {
		return fallback
	}
	return *value
}

func valueOrDefaultInt(value *int, fallback int) int {
	if value == nil {
		return fallback
	}
	return *value
}
