package config

import (
	"os"
	"path/filepath"
	"strconv"
	"time"
)

type Config struct {
	HTTPAddress          string
	HTTPReadTimeout      time.Duration
	HTTPWriteTimeout     time.Duration
	HTTPIdleTimeout      time.Duration
	LogBufferSize        int
	MLServiceBaseURL     string
	MLServiceTimeout     time.Duration
	DefaultTruckCapacity float64
	DefaultSafetyFactor  float64
	MaxTrucksPerRoute    int
	PredictionFloor      float64
	PredictionPrecision  int
	SubmissionDir        string
	UploadDir            string
	WorkspaceDir         string
}

func Load() Config {
	wd, err := os.Getwd()
	if err != nil {
		wd = "."
	}

	cfg := Config{
		HTTPAddress:          getString("APP_ADDR", ":8080"),
		HTTPReadTimeout:      getDuration("APP_READ_TIMEOUT", 10*time.Second),
		HTTPWriteTimeout:     getDuration("APP_WRITE_TIMEOUT", 30*time.Second),
		HTTPIdleTimeout:      getDuration("APP_IDLE_TIMEOUT", 60*time.Second),
		LogBufferSize:        getInt("LOG_BUFFER_SIZE", 300),
		MLServiceBaseURL:     getString("ML_SERVICE_BASE_URL", "http://localhost:8090"),
		MLServiceTimeout:     getDuration("ML_SERVICE_TIMEOUT", 15*time.Second),
		DefaultTruckCapacity: getFloat("DEFAULT_TRUCK_CAPACITY", 20),
		DefaultSafetyFactor:  getFloat("DEFAULT_SAFETY_FACTOR", 0.10),
		MaxTrucksPerRoute:    getInt("MAX_TRUCKS_PER_ROUTE", 50),
		PredictionFloor:      getFloat("PREDICTION_FLOOR", 0),
		PredictionPrecision:  getInt("PREDICTION_PRECISION", 4),
		WorkspaceDir:         wd,
	}

	submissionDir := getString("SUBMISSION_DIR", filepath.Join("artifacts", "submissions"))
	cfg.SubmissionDir = resolvePath(wd, submissionDir)
	uploadDir := getString("UPLOAD_DIR", filepath.Join("artifacts", "uploads"))
	cfg.UploadDir = resolvePath(wd, uploadDir)

	return cfg
}

func getString(name, fallback string) string {
	value := os.Getenv(name)
	if value == "" {
		return fallback
	}
	return value
}

func getInt(name string, fallback int) int {
	value := os.Getenv(name)
	if value == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return fallback
	}
	return parsed
}

func getFloat(name string, fallback float64) float64 {
	value := os.Getenv(name)
	if value == "" {
		return fallback
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return fallback
	}
	return parsed
}

func getDuration(name string, fallback time.Duration) time.Duration {
	value := os.Getenv(name)
	if value == "" {
		return fallback
	}
	parsed, err := time.ParseDuration(value)
	if err != nil {
		return fallback
	}
	return parsed
}

func resolvePath(baseDir, value string) string {
	if filepath.IsAbs(value) {
		return filepath.Clean(value)
	}
	return filepath.Join(baseDir, value)
}
