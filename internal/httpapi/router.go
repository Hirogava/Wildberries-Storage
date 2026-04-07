package httpapi

import "net/http"

func NewRouter(handlers *Handlers, uiHandler http.Handler) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", handlers.Health)
	mux.HandleFunc("/predict", handlers.Predict)
	mux.HandleFunc("/decision", handlers.Decide)
	mux.HandleFunc("/batch", handlers.Batch)
	mux.HandleFunc("/metrics", handlers.Metrics)
	mux.HandleFunc("/model/select", handlers.ModelSelect)
	mux.HandleFunc("/files/upload", handlers.UploadFile)
	mux.HandleFunc("/logs/go", handlers.GoLogs)
	mux.HandleFunc("/logs/ml", handlers.MLLogs)
	mux.HandleFunc("/stream/logs/go", handlers.GoLogsStream)
	mux.HandleFunc("/stream/logs/ml", handlers.MLLogsStream)
	if uiHandler != nil {
		mux.Handle("/", uiHandler)
	}
	return mux
}
