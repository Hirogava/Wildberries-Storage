package service

import (
	"testing"

	"wildberries-storage/internal/domain"
)

func TestMetricsServiceCalculate(t *testing.T) {
	service := NewMetricsService(nil)

	response, err := service.Calculate(domain.MetricsRequest{
		RequestID: "req-1",
		Observations: []domain.MetricObservation{
			{ID: 1, YTrue: 10, YPred: 9},
			{ID: 2, YTrue: 20, YPred: 22},
		},
	})
	if err != nil {
		t.Fatalf("Calculate returned error: %v", err)
	}

	if response.Score != 0.133333 {
		t.Fatalf("expected score 0.133333, got %f", response.Score)
	}
}
