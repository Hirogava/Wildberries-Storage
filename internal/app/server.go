package app

import (
	"net/http"

	"wildberries-storage/internal/config"
	"wildberries-storage/internal/httpapi"
	"wildberries-storage/internal/ml"
	"wildberries-storage/internal/observability"
	"wildberries-storage/internal/service"
	"wildberries-storage/internal/storage"
	"wildberries-storage/internal/ui"
)

func NewServer(cfg config.Config) (*http.Server, error) {
	fileStore := storage.NewFileStore(cfg.WorkspaceDir)
	logger := observability.NewLogger(cfg.LogBufferSize)
	logger.Info("bootstrap", "initializing go api addr=%s ml_base_url=%s", cfg.HTTPAddress, cfg.MLServiceBaseURL)
	mlClient := ml.NewClient(cfg.MLServiceBaseURL, cfg.MLServiceTimeout, logger)
	uiHandler, err := ui.Handler()
	if err != nil {
		return nil, err
	}

	forecastService := service.NewForecastService(mlClient, cfg.PredictionFloor, cfg.PredictionPrecision)
	decisionService := service.NewDecisionService(
		forecastService,
		cfg.DefaultSafetyFactor,
		cfg.DefaultTruckCapacity,
		cfg.MaxTrucksPerRoute,
	)
	batchService := service.NewBatchService(forecastService, fileStore, cfg.SubmissionDir, mlClient)
	metricsService := service.NewMetricsService(fileStore)
	modelSelectService := service.NewModelSelectService(mlClient)

	handlers := httpapi.NewHandlers(
		forecastService,
		decisionService,
		batchService,
		metricsService,
		modelSelectService,
		mlClient,
		fileStore,
		cfg.UploadDir,
		logger,
	)

	server := &http.Server{
		Addr:         cfg.HTTPAddress,
		Handler:      httpapi.NewRouter(handlers, uiHandler),
		ReadTimeout:  cfg.HTTPReadTimeout,
		WriteTimeout: cfg.HTTPWriteTimeout,
		IdleTimeout:  cfg.HTTPIdleTimeout,
	}

	return server, nil
}
