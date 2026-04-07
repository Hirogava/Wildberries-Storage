package domain

import "time"

type ForecastPoint struct {
	ID        int64     `json:"id"`
	RouteID   int64     `json:"route_id"`
	Timestamp time.Time `json:"timestamp"`
}

type ForecastRequest struct {
	RequestID string          `json:"request_id"`
	Points    []ForecastPoint `json:"points"`
}

type Prediction struct {
	ID    int64   `json:"id"`
	YPred float64 `json:"y_pred"`
}

type ForecastResponse struct {
	RequestID   string       `json:"request_id"`
	Predictions []Prediction `json:"predictions"`
	Model       string       `json:"model,omitempty"`
}

type DecisionPrediction struct {
	ID        int64     `json:"id"`
	RouteID   int64     `json:"route_id,omitempty"`
	Timestamp time.Time `json:"timestamp,omitempty"`
	YPred     float64   `json:"y_pred"`
}

type DecisionRequest struct {
	RequestID         string               `json:"request_id"`
	Points            []ForecastPoint      `json:"points,omitempty"`
	Predictions       []DecisionPrediction `json:"predictions,omitempty"`
	SafetyFactor      *float64             `json:"safety_factor,omitempty"`
	TruckCapacity     *float64             `json:"truck_capacity,omitempty"`
	MaxTrucksPerRoute *int                 `json:"max_trucks_per_route,omitempty"`
}

type Decision struct {
	ID             int64     `json:"id"`
	RouteID        int64     `json:"route_id"`
	Timestamp      time.Time `json:"timestamp"`
	YPred          float64   `json:"y_pred"`
	AdjustedLoad   float64   `json:"adjusted_load"`
	TrucksNeeded   int       `json:"trucks_needed"`
	TruckCapacity  float64   `json:"truck_capacity"`
	SafetyFactor   float64   `json:"safety_factor"`
	Action         string    `json:"action"`
	DecisionReason string    `json:"decision_reason"`
}

type DecisionResponse struct {
	RequestID string     `json:"request_id"`
	Decisions []Decision `json:"decisions"`
}

type BatchRequest struct {
	RequestID         string          `json:"request_id"`
	Points            []ForecastPoint `json:"points,omitempty"`
	InputPath         string          `json:"input_path,omitempty"`
	OutputPath        string          `json:"output_path,omitempty"`
	ReturnPredictions bool            `json:"return_predictions,omitempty"`
}

type BatchResponse struct {
	RequestID   string       `json:"request_id"`
	OutputPath  string       `json:"output_path"`
	TotalPoints int          `json:"total_points"`
	Predictions []Prediction `json:"predictions,omitempty"`
}

type MetricObservation struct {
	ID    int64   `json:"id"`
	YTrue float64 `json:"y_true"`
	YPred float64 `json:"y_pred"`
}

type MetricsRequest struct {
	RequestID        string              `json:"request_id"`
	Observations     []MetricObservation `json:"observations,omitempty"`
	ActualPath       string              `json:"actual_path,omitempty"`
	PredictionPath   string              `json:"prediction_path,omitempty"`
	ActualColumn     string              `json:"actual_column,omitempty"`
	PredictionColumn string              `json:"prediction_column,omitempty"`
}

type MetricsResponse struct {
	RequestID    string  `json:"request_id"`
	Count        int     `json:"count"`
	WAPE         float64 `json:"wape"`
	RelativeBias float64 `json:"relative_bias"`
	Score        float64 `json:"score"`
}

type ModelScore struct {
	Model string  `json:"model"`
	Score float64 `json:"score"`
}

type ModelSelectRequest struct {
	RequestID  string            `json:"request_id"`
	Candidates []string          `json:"candidates,omitempty"`
	Objective  string            `json:"objective,omitempty"`
	Context    map[string]string `json:"context,omitempty"`
}

type ModelSelectResponse struct {
	RequestID     string       `json:"request_id"`
	SelectedModel string       `json:"selected_model"`
	Ranking       []ModelScore `json:"ranking,omitempty"`
}

type HealthResponse struct {
	Status    string            `json:"status"`
	Services  map[string]string `json:"services"`
	Timestamp time.Time         `json:"timestamp"`
}

type FileUploadResponse struct {
	Filename     string `json:"filename"`
	Path         string `json:"path"`
	RelativePath string `json:"relative_path"`
	Size         int64  `json:"size"`
}
