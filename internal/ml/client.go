package ml

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"wildberries-storage/internal/domain"
	"wildberries-storage/internal/observability"
)

type Client struct {
	baseURL    string
	httpClient *http.Client
	logger     *observability.Logger
}

func NewClient(baseURL string, timeout time.Duration, logger *observability.Logger) *Client {
	return &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		httpClient: &http.Client{
			Timeout: timeout,
		},
		logger: logger,
	}
}

func (c *Client) Predict(ctx context.Context, req domain.ForecastRequest) (domain.ForecastResponse, error) {
	var response domain.ForecastResponse
	c.logInfo("predict request_id=%s points=%d", req.RequestID, len(req.Points))
	err := c.postJSON(ctx, "/predict", req, &response)
	if err != nil {
		c.logError("predict request failed: %v", err)
	} else {
		c.logInfo("predict response request_id=%s predictions=%d model=%s", response.RequestID, len(response.Predictions), response.Model)
	}
	return response, err
}

func (c *Client) SelectModel(ctx context.Context, req domain.ModelSelectRequest) (domain.ModelSelectResponse, error) {
	var response domain.ModelSelectResponse
	c.logInfo("model_select request_id=%s candidates=%d objective=%s", req.RequestID, len(req.Candidates), req.Objective)
	err := c.postJSON(ctx, "/model/select", req, &response)
	if err != nil {
		c.logError("model_select request failed: %v", err)
	} else {
		c.logInfo("model_select response request_id=%s selected_model=%s", response.RequestID, response.SelectedModel)
	}
	return response, err
}

func (c *Client) Ping(ctx context.Context) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/healthz", nil)
	if err != nil {
		return fmt.Errorf("build ml health request: %w", err)
	}

	response, err := c.httpClient.Do(request)
	if err != nil {
		return fmt.Errorf("call ml health endpoint: %w", err)
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 2048))
		return fmt.Errorf("ml health status %d: %s", response.StatusCode, strings.TrimSpace(string(body)))
	}

	return nil
}

func (c *Client) postJSON(ctx context.Context, path string, requestBody any, target any) error {
	body, err := json.Marshal(requestBody)
	if err != nil {
		return fmt.Errorf("marshal request body: %w", err)
	}

	request, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	request.Header.Set("Content-Type", "application/json")

	response, err := c.httpClient.Do(request)
	if err != nil {
		return fmt.Errorf("call ml service: %w", err)
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		respBody, _ := io.ReadAll(io.LimitReader(response.Body, 4096))
		return fmt.Errorf("ml service returned %d: %s", response.StatusCode, strings.TrimSpace(string(respBody)))
	}

	if err := json.NewDecoder(response.Body).Decode(target); err != nil {
		return fmt.Errorf("decode ml response: %w", err)
	}

	return nil
}

func (c *Client) Logs(ctx context.Context, limit int) ([]observability.Entry, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("%s/logs?limit=%d", c.baseURL, limit), nil)
	if err != nil {
		return nil, fmt.Errorf("build ml logs request: %w", err)
	}

	response, err := c.httpClient.Do(request)
	if err != nil {
		return nil, fmt.Errorf("call ml logs endpoint: %w", err)
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 2048))
		return nil, fmt.Errorf("ml logs status %d: %s", response.StatusCode, strings.TrimSpace(string(body)))
	}

	var payload struct {
		Entries []observability.Entry `json:"entries"`
	}
	if err := json.NewDecoder(response.Body).Decode(&payload); err != nil {
		return nil, fmt.Errorf("decode ml logs response: %w", err)
	}

	return payload.Entries, nil
}

func (c *Client) OpenLogsStream(ctx context.Context, limit int) (*http.Response, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("%s/stream/logs?limit=%d", c.baseURL, limit), nil)
	if err != nil {
		return nil, fmt.Errorf("build ml stream request: %w", err)
	}
	request.Header.Set("Accept", "text/event-stream")

	response, err := c.httpClient.Do(request)
	if err != nil {
		return nil, fmt.Errorf("call ml stream endpoint: %w", err)
	}

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 2048))
		response.Body.Close()
		return nil, fmt.Errorf("ml stream status %d: %s", response.StatusCode, strings.TrimSpace(string(body)))
	}

	return response, nil
}

func (c *Client) logInfo(format string, args ...any) {
	if c.logger != nil {
		c.logger.Info("ml-client", format, args...)
	}
}

func (c *Client) logError(format string, args ...any) {
	if c.logger != nil {
		c.logger.Error("ml-client", format, args...)
	}
}
