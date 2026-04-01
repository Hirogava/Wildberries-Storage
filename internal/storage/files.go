package storage

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"wildberries-storage/internal/domain"
)

type FileStore struct {
	workspaceDir string
}

func NewFileStore(workspaceDir string) *FileStore {
	return &FileStore{workspaceDir: filepath.Clean(workspaceDir)}
}

func (s *FileStore) LoadPoints(path string) ([]domain.ForecastPoint, error) {
	resolvedPath, err := s.resolvePath(path)
	if err != nil {
		return nil, err
	}

	switch strings.ToLower(filepath.Ext(resolvedPath)) {
	case ".json":
		return s.loadPointsFromJSON(resolvedPath)
	case ".csv":
		return s.loadPointsFromCSV(resolvedPath)
	default:
		return nil, fmt.Errorf("unsupported points file format: %s", resolvedPath)
	}
}

func (s *FileStore) LoadObservations(actualPath, predictionPath, actualColumn, predictionColumn string) ([]domain.MetricObservation, error) {
	actualColumn = defaultString(actualColumn, "y_true")
	predictionColumn = defaultString(predictionColumn, "y_pred")

	actualRows, err := s.loadCSVByID(actualPath, actualColumn)
	if err != nil {
		return nil, err
	}
	predictionRows, err := s.loadCSVByID(predictionPath, predictionColumn)
	if err != nil {
		return nil, err
	}

	observations := make([]domain.MetricObservation, 0, len(actualRows))
	for id, yTrue := range actualRows {
		yPred, ok := predictionRows[id]
		if !ok {
			return nil, fmt.Errorf("missing prediction for id %d", id)
		}
		observations = append(observations, domain.MetricObservation{
			ID:    id,
			YTrue: yTrue,
			YPred: yPred,
		})
	}

	return observations, nil
}

func (s *FileStore) WriteSubmission(path string, predictions []domain.Prediction) (string, error) {
	resolvedPath, err := s.resolvePath(path)
	if err != nil {
		return "", err
	}

	if err := os.MkdirAll(filepath.Dir(resolvedPath), 0o755); err != nil {
		return "", fmt.Errorf("create output directory: %w", err)
	}

	file, err := os.Create(resolvedPath)
	if err != nil {
		return "", fmt.Errorf("create output file: %w", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	if err := writer.Write([]string{"id", "y_pred"}); err != nil {
		return "", fmt.Errorf("write csv header: %w", err)
	}

	for _, prediction := range predictions {
		row := []string{
			strconv.FormatInt(prediction.ID, 10),
			strconv.FormatFloat(prediction.YPred, 'f', -1, 64),
		}
		if err := writer.Write(row); err != nil {
			return "", fmt.Errorf("write csv row: %w", err)
		}
	}

	writer.Flush()
	if err := writer.Error(); err != nil {
		return "", fmt.Errorf("flush csv writer: %w", err)
	}

	return resolvedPath, nil
}

func (s *FileStore) loadPointsFromJSON(path string) ([]domain.ForecastPoint, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read points json: %w", err)
	}

	var request domain.ForecastRequest
	if err := json.Unmarshal(data, &request); err == nil && len(request.Points) > 0 {
		return request.Points, nil
	}

	var points []domain.ForecastPoint
	if err := json.Unmarshal(data, &points); err != nil {
		return nil, fmt.Errorf("decode points json: %w", err)
	}

	return points, nil
}

func (s *FileStore) loadPointsFromCSV(path string) ([]domain.ForecastPoint, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open points csv: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	rows, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("read points csv: %w", err)
	}
	if len(rows) < 2 {
		return nil, fmt.Errorf("points csv is empty")
	}

	header := indexHeader(rows[0])
	required := []string{"id", "route_id", "timestamp"}
	for _, column := range required {
		if _, ok := header[column]; !ok {
			return nil, fmt.Errorf("points csv must contain column %q", column)
		}
	}

	points := make([]domain.ForecastPoint, 0, len(rows)-1)
	for _, row := range rows[1:] {
		id, err := strconv.ParseInt(row[header["id"]], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("parse id: %w", err)
		}
		routeID, err := strconv.ParseInt(row[header["route_id"]], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("parse route_id: %w", err)
		}
		timestamp, err := time.Parse(time.RFC3339, row[header["timestamp"]])
		if err != nil {
			return nil, fmt.Errorf("parse timestamp: %w", err)
		}

		points = append(points, domain.ForecastPoint{
			ID:        id,
			RouteID:   routeID,
			Timestamp: timestamp,
		})
	}

	return points, nil
}

func (s *FileStore) loadCSVByID(path, valueColumn string) (map[int64]float64, error) {
	resolvedPath, err := s.resolvePath(path)
	if err != nil {
		return nil, err
	}

	file, err := os.Open(resolvedPath)
	if err != nil {
		return nil, fmt.Errorf("open csv %s: %w", resolvedPath, err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	rows, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("read csv %s: %w", resolvedPath, err)
	}
	if len(rows) < 2 {
		return nil, fmt.Errorf("csv %s is empty", resolvedPath)
	}

	header := indexHeader(rows[0])
	if _, ok := header["id"]; !ok {
		return nil, fmt.Errorf("csv %s must contain id column", resolvedPath)
	}
	if _, ok := header[valueColumn]; !ok {
		return nil, fmt.Errorf("csv %s must contain %s column", resolvedPath, valueColumn)
	}

	result := make(map[int64]float64, len(rows)-1)
	for _, row := range rows[1:] {
		id, err := strconv.ParseInt(row[header["id"]], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("parse id in %s: %w", resolvedPath, err)
		}
		value, err := strconv.ParseFloat(row[header[valueColumn]], 64)
		if err != nil {
			return nil, fmt.Errorf("parse %s in %s: %w", valueColumn, resolvedPath, err)
		}
		result[id] = value
	}

	return result, nil
}

func (s *FileStore) resolvePath(path string) (string, error) {
	if path == "" {
		return "", fmt.Errorf("path must not be empty")
	}

	var resolved string
	if filepath.IsAbs(path) {
		resolved = filepath.Clean(path)
	} else {
		resolved = filepath.Join(s.workspaceDir, path)
	}

	relative, err := filepath.Rel(s.workspaceDir, resolved)
	if err != nil {
		return "", fmt.Errorf("resolve path %s: %w", path, err)
	}
	if strings.HasPrefix(relative, "..") {
		return "", fmt.Errorf("path %s is outside workspace", path)
	}

	return resolved, nil
}

func indexHeader(columns []string) map[string]int {
	header := make(map[string]int, len(columns))
	for index, column := range columns {
		header[strings.TrimSpace(strings.ToLower(column))] = index
	}
	return header
}

func defaultString(value, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}
