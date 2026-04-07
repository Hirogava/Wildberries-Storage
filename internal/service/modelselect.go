package service

import (
	"context"
	"fmt"

	"wildberries-storage/internal/domain"
)

type Selector interface {
	SelectModel(ctx context.Context, req domain.ModelSelectRequest) (domain.ModelSelectResponse, error)
}

type ModelSelectService struct {
	selector Selector
}

func NewModelSelectService(selector Selector) *ModelSelectService {
	return &ModelSelectService{selector: selector}
}

func (s *ModelSelectService) Select(ctx context.Context, req domain.ModelSelectRequest) (domain.ModelSelectResponse, error) {
	req.RequestID = ensureRequestID(req.RequestID)
	if req.Objective == "" {
		req.Objective = "wape_plus_rbias"
	}
	if req.Objective != "wape_plus_rbias" {
		return domain.ModelSelectResponse{}, fmt.Errorf("unsupported objective %q", req.Objective)
	}

	response, err := s.selector.SelectModel(ctx, req)
	if err != nil {
		return domain.ModelSelectResponse{}, err
	}
	if response.RequestID == "" {
		response.RequestID = req.RequestID
	}

	return response, nil
}
