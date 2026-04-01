package main

import (
	"log"

	"wildberries-storage/internal/app"
	"wildberries-storage/internal/config"
)

func main() {
	cfg := config.Load()

	server, err := app.NewServer(cfg)
	if err != nil {
		log.Fatalf("build server: %v", err)
	}

	log.Printf("wb-storage backend is listening on %s", cfg.HTTPAddress)
	if err := server.ListenAndServe(); err != nil {
		log.Fatalf("serve http: %v", err)
	}
}
