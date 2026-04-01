package service

import (
	"context"
	"testing"
	"time"

	"wildberries-storage/internal/domain"
)

func TestDecisionServiceDecide(t *testing.T) {
	service := NewDecisionService(nil, 0.10, 20, 50)

	response, err := service.Decide(context.Background(), domain.DecisionRequest{
		RequestID: "req-1",
		Predictions: []domain.DecisionPrediction{
			{
				ID:        1,
				RouteID:   101,
				Timestamp: time.Date(2026, 4, 1, 10, 0, 0, 0, time.UTC),
				YPred:     19.4,
			},
		},
	})
	if err != nil {
		t.Fatalf("Decide returned error: %v", err)
	}

	if len(response.Decisions) != 1 {
		t.Fatalf("expected 1 decision, got %d", len(response.Decisions))
	}

	decision := response.Decisions[0]
	if decision.TrucksNeeded != 2 {
		t.Fatalf("expected 2 trucks, got %d", decision.TrucksNeeded)
	}
	if decision.Action != "CALL_TRUCKS" {
		t.Fatalf("expected CALL_TRUCKS action, got %s", decision.Action)
	}
}

func TestDecisionServiceDecideEnrichesPredictResponseWithPoints(t *testing.T) {
	service := NewDecisionService(nil, 0.10, 20, 50)

	response, err := service.Decide(context.Background(), domain.DecisionRequest{
		RequestID: "req-2",
		Points: []domain.ForecastPoint{
			{
				ID:        100001,
				RouteID:   12345,
				Timestamp: time.Date(2026, 3, 30, 12, 30, 0, 0, time.UTC),
			},
		},
		Predictions: []domain.DecisionPrediction{
			{
				ID:    100001,
				YPred: 19.4,
			},
		},
	})
	if err != nil {
		t.Fatalf("Decide returned error: %v", err)
	}

	decision := response.Decisions[0]
	if decision.RouteID != 12345 {
		t.Fatalf("expected route_id 12345, got %d", decision.RouteID)
	}
	if !decision.Timestamp.Equal(time.Date(2026, 3, 30, 12, 30, 0, 0, time.UTC)) {
		t.Fatalf("unexpected timestamp: %s", decision.Timestamp)
	}
}
