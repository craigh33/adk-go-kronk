// Package kronk provides a Kronk-backed implementation of the adk-go
// [model.LLM] interface so ADK agents can run against local GGUF models served
// by the github.com/ardanlabs/kronk SDK with the same APIs used for Gemini or
// Bedrock providers.
package kronk

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"time"

	krnk "github.com/ardanlabs/kronk/sdk/kronk"
	krnkmodel "github.com/ardanlabs/kronk/sdk/kronk/model"
	"google.golang.org/adk/model"

	"github.com/craigh33/adk-go-kronk/internal/mappers"
)

var _ model.LLM = (*Model)(nil)

// defaultRequestTimeout is applied when the caller's context does not carry a
// deadline. Kronk's Chat / ChatStreaming functions require a deadline, so we
// attach a sensible default rather than failing the request.
const defaultRequestTimeout = 2 * time.Minute

// Config configures [New]. Use [NewWithKronk] if you already have a
// constructed *kronk.Kronk instance you want to reuse.
type Config struct {
	// ModelFiles are the GGUF file paths for the model to load. Typically
	// obtained from the Kronk models/catalog tools (see
	// github.com/ardanlabs/kronk/sdk/tools/models).
	ModelFiles []string

	// ModelOptions are forwarded to kronk.New when the provider constructs
	// its own Kronk instance. Optional.
	ModelOptions []krnkmodel.Option

	// InitOptions are forwarded to kronk.Init. Optional.
	InitOptions []krnk.InitOption
}

// Model implements [model.LLM] on top of a github.com/ardanlabs/kronk engine.
type Model struct {
	krn   *krnk.Kronk
	owned bool
}

// New constructs a [Model] that owns its Kronk engine. It ensures the Kronk
// runtime has been initialized (kronk.Init is guarded by a [sync.Once]
// internally), then loads the model described by cfg. Call [Model.Close] when
// you are done to unload the model.
func New(_ context.Context, cfg Config) (*Model, error) {
	if len(cfg.ModelFiles) == 0 {
		return nil, errors.New("kronk: Config.ModelFiles is required")
	}
	if err := krnk.Init(cfg.InitOptions...); err != nil {
		return nil, fmt.Errorf("kronk: init: %w", err)
	}
	opts := append(
		[]krnkmodel.Option{krnkmodel.WithModelFiles(cfg.ModelFiles)},
		cfg.ModelOptions...,
	)
	krn, err := krnk.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("kronk: new engine: %w", err)
	}
	return &Model{krn: krn, owned: true}, nil
}

// NewWithKronk wraps an existing *kronk.Kronk engine. The caller retains
// lifecycle ownership and is responsible for calling krn.Unload; Close on the
// returned Model is a no-op.
func NewWithKronk(krn *krnk.Kronk) (*Model, error) {
	if krn == nil {
		return nil, errors.New("kronk: nil *kronk.Kronk")
	}
	return &Model{krn: krn, owned: false}, nil
}

// Name returns the model identifier reported by Kronk (derived from the GGUF
// filename). This satisfies [model.LLM].
func (m *Model) Name() string {
	if m == nil || m.krn == nil {
		return ""
	}
	return m.krn.ModelInfo().ID
}

// Engine returns the underlying Kronk engine so callers can access Kronk
// features not exposed through the [model.LLM] surface (embeddings, rerank,
// direct HTTP streaming, etc.).
func (m *Model) Engine() *krnk.Kronk {
	if m == nil {
		return nil
	}
	return m.krn
}

// Close unloads the underlying model when this Model was constructed via
// [New]. When constructed via [NewWithKronk] the caller owns lifecycle and
// Close is a no-op.
func (m *Model) Close(ctx context.Context) error {
	if m == nil || !m.owned || m.krn == nil {
		return nil
	}
	if err := m.krn.Unload(ctx); err != nil {
		return fmt.Errorf("kronk: unload: %w", err)
	}
	return nil
}

// GenerateContent implements [model.LLM]. It maps the ADK LLMRequest into a
// Kronk chat document and streams results back as ADK LLMResponses. When
// stream is true, intermediate text deltas are emitted with Partial:true and
// a final TurnComplete:true response is emitted with aggregated tool calls,
// usage, and finish reason.
func (m *Model) GenerateContent(
	ctx context.Context,
	req *model.LLMRequest,
	stream bool,
) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if m == nil || m.krn == nil {
			yield(nil, errors.New("kronk: nil Model"))
			return
		}
		if req == nil {
			yield(nil, errors.New("kronk: nil LLMRequest"))
			return
		}

		ctx, cancel := ensureDeadline(ctx)
		if cancel != nil {
			defer cancel()
		}

		d, err := mappers.RequestFromLLMRequest(req, stream)
		if err != nil {
			yield(nil, fmt.Errorf("kronk: build request: %w", err))
			return
		}

		if stream {
			m.generateStream(ctx, d)(yield)
			return
		}
		m.generateUnary(ctx, d)(yield)
	}
}

func (m *Model) generateUnary(
	ctx context.Context,
	d krnkmodel.D,
) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		resp, err := m.krn.Chat(ctx, d)
		if err != nil {
			yield(nil, fmt.Errorf("kronk: chat: %w", err))
			return
		}
		out, err := mappers.LLMResponseFromChatResponse(resp)
		if err != nil {
			yield(nil, fmt.Errorf("kronk: map response: %w", err))
			return
		}
		out.TurnComplete = true
		if !yield(out, nil) {
			return
		}
	}
}

func (m *Model) generateStream(
	ctx context.Context,
	d krnkmodel.D,
) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		ch, err := m.krn.ChatStreaming(ctx, d)
		if err != nil {
			yield(nil, fmt.Errorf("kronk: chat stream: %w", err))
			return
		}

		state := mappers.NewStreamState()
		for resp := range ch {
			partial, err := mappers.StreamChunkToLLMResponse(state, resp)
			if err != nil {
				// Surface model-side errors through the iterator so callers
				// can distinguish them from transport failures via the error
				// message.
				yield(nil, err)
				return
			}
			if partial != nil && !yield(partial, nil) {
				return
			}
		}

		final := mappers.FinalStreamResponse(state)
		if !yield(final, nil) {
			return
		}
	}
}

// ensureDeadline guarantees ctx has a deadline. Kronk requires one; when the
// caller omits one we attach [defaultRequestTimeout]. Returns a non-nil cancel
// when a new deadline was attached.
func ensureDeadline(ctx context.Context) (context.Context, context.CancelFunc) {
	if _, ok := ctx.Deadline(); ok {
		return ctx, nil
	}
	return context.WithTimeout(ctx, defaultRequestTimeout)
}
