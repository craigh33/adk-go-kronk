package mappers

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"

	krnkmodel "github.com/ardanlabs/kronk/sdk/kronk/model"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// rawFunctionArgsJSONKey stores the original JSON string when tool call
// arguments cannot be parsed into a map. Callers can reach into
// FunctionCall.Args[rawFunctionArgsJSONKey] as a fallback.
const rawFunctionArgsJSONKey = "rawArgsJson"

// customMetadataKeyModel carries the Kronk model ID on the LLMResponse so
// callers that aggregate responses across providers can still disambiguate.
const customMetadataKeyModel = "kronk_model_id"

// LLMResponseFromChatResponse converts a non-streaming Kronk ChatResponse
// into an ADK LLMResponse. Errors are surfaced via the returned error when
// the Kronk response carries FinishReasonError.
func LLMResponseFromChatResponse(resp krnkmodel.ChatResponse) (*model.LLMResponse, error) {
	if len(resp.Choices) == 0 {
		return &model.LLMResponse{
			Content:        &genai.Content{Role: "model", Parts: []*genai.Part{{Text: ""}}},
			FinishReason:   genai.FinishReasonOther,
			UsageMetadata:  usageToGenai(resp.Usage),
			CustomMetadata: customMetadataFromResponse(resp),
		}, nil
	}

	choice := resp.Choices[0]
	reason := choice.FinishReason()

	if reason == krnkmodel.FinishReasonError {
		msg := ""
		switch {
		case choice.Message != nil:
			msg = choice.Message.Content
		case choice.Delta != nil:
			msg = choice.Delta.Content
		}
		return nil, fmt.Errorf("kronk model error: %s", msg)
	}

	msg := choice.Message
	if msg == nil {
		msg = choice.Delta
	}
	parts := partsFromResponseMessage(msg)
	if len(parts) == 0 {
		parts = []*genai.Part{{Text: ""}}
	}

	return &model.LLMResponse{
		Content:        &genai.Content{Role: "model", Parts: parts},
		FinishReason:   finishReasonToGenai(reason),
		UsageMetadata:  usageToGenai(resp.Usage),
		CustomMetadata: customMetadataFromResponse(resp),
	}, nil
}

// =============================================================================
// Streaming

// StreamState accumulates streamed Kronk chunks so a single TurnComplete
// LLMResponse can be synthesized at the end of the stream.
type StreamState struct {
	modelID        string
	text           strings.Builder
	lastYieldedLen int

	reasoning       strings.Builder
	reasoningActive bool

	toolCallsByID  map[string]*streamToolCall
	toolCallsOrder []string

	finishReason string
	lastUsage    *krnkmodel.Usage
}

type streamToolCall struct {
	id   string
	name string
	args strings.Builder
}

// NewStreamState constructs an empty [StreamState].
func NewStreamState() *StreamState {
	return &StreamState{
		toolCallsByID: make(map[string]*streamToolCall),
	}
}

// StreamChunkToLLMResponse advances state with a Kronk streaming chunk and
// returns an intermediate LLMResponse (with Partial:true) when the chunk
// carries a new text delta. A nil response with a nil error means the chunk
// was absorbed without producing observable output (tool-call accumulation,
// usage updates, etc.).
func StreamChunkToLLMResponse(state *StreamState, resp krnkmodel.ChatResponse) (*model.LLMResponse, error) {
	if state == nil {
		return nil, errors.New("nil StreamState")
	}
	if state.modelID == "" && resp.Model != "" {
		state.modelID = resp.Model
	}
	if resp.Usage != nil {
		state.lastUsage = resp.Usage
	}
	if len(resp.Choices) == 0 {
		return nil, nil //nolint:nilnil // Nothing to emit for this chunk.
	}

	choice := resp.Choices[0]
	reason := choice.FinishReason()
	if reason != "" {
		state.finishReason = reason
	}

	if reason == krnkmodel.FinishReasonError {
		msg := ""
		if choice.Delta != nil {
			msg = choice.Delta.Content
		} else if choice.Message != nil {
			msg = choice.Message.Content
		}
		return nil, fmt.Errorf("kronk model error: %s", msg)
	}

	delta := choice.Delta
	if delta == nil {
		return nil, nil //nolint:nilnil // Choice carries no delta to emit.
	}

	for _, tc := range delta.ToolCalls {
		accumulateToolCall(state, tc)
	}

	if delta.Reasoning != "" {
		state.reasoningActive = true
		state.reasoning.WriteString(delta.Reasoning)
	}

	if delta.Content != "" {
		state.text.WriteString(delta.Content)
		out := state.text.String()[state.lastYieldedLen:]
		state.lastYieldedLen = state.text.Len()
		return &model.LLMResponse{
			Content: &genai.Content{
				Role:  "model",
				Parts: []*genai.Part{{Text: out}},
			},
			Partial: true,
		}, nil
	}

	return nil, nil //nolint:nilnil // Chunk was absorbed without producing observable output.
}

// FinalStreamResponse builds the terminal TurnComplete LLMResponse for a
// completed Kronk stream. Usage, reasoning, tool calls, and finish reason are
// all aggregated from state.
func FinalStreamResponse(state *StreamState) *model.LLMResponse {
	if state == nil {
		return &model.LLMResponse{
			Content:      &genai.Content{Role: "model", Parts: []*genai.Part{{Text: ""}}},
			FinishReason: genai.FinishReasonOther,
			TurnComplete: true,
		}
	}

	parts := make([]*genai.Part, 0, 2+len(state.toolCallsOrder))
	if state.reasoning.Len() > 0 {
		parts = append(parts, &genai.Part{Text: state.reasoning.String(), Thought: true})
	}
	if state.text.Len() > 0 {
		parts = append(parts, &genai.Part{Text: state.text.String()})
	}
	for _, id := range state.toolCallsOrder {
		tc := state.toolCallsByID[id]
		if tc == nil {
			continue
		}
		parts = append(parts, &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   tc.id,
				Name: tc.name,
				Args: functionArgsFromRawJSON(tc.args.String()),
			},
		})
	}
	if len(parts) == 0 {
		parts = []*genai.Part{{Text: ""}}
	}

	var custom map[string]any
	if state.modelID != "" {
		custom = map[string]any{customMetadataKeyModel: state.modelID}
	}

	return &model.LLMResponse{
		Content:        &genai.Content{Role: "model", Parts: parts},
		FinishReason:   finishReasonToGenai(state.finishReason),
		UsageMetadata:  usageToGenai(state.lastUsage),
		CustomMetadata: custom,
		TurnComplete:   true,
	}
}

// =============================================================================
// Helpers

func partsFromResponseMessage(m *krnkmodel.ResponseMessage) []*genai.Part {
	if m == nil {
		return nil
	}
	var parts []*genai.Part
	if m.Reasoning != "" {
		parts = append(parts, &genai.Part{Text: m.Reasoning, Thought: true})
	}
	if m.Content != "" {
		parts = append(parts, &genai.Part{Text: m.Content})
	}
	for _, tc := range m.ToolCalls {
		parts = append(parts, &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   tc.ID,
				Name: tc.Function.Name,
				Args: argsToMap(tc.Function.Arguments),
			},
		})
	}
	return parts
}

func accumulateToolCall(state *StreamState, tc krnkmodel.ResponseToolCall) {
	id := tc.ID
	if id == "" {
		id = fmt.Sprintf("tool_%d", tc.Index)
	}
	existing, ok := state.toolCallsByID[id]
	if !ok {
		existing = &streamToolCall{id: id}
		state.toolCallsByID[id] = existing
		state.toolCallsOrder = append(state.toolCallsOrder, id)
	}
	if tc.Function.Name != "" {
		existing.name = tc.Function.Name
	}
	if len(tc.Function.Arguments) > 0 {
		b, err := json.Marshal(map[string]any(tc.Function.Arguments))
		if err == nil {
			existing.args.Reset()
			existing.args.Write(b)
		}
	} else if tc.Raw != "" {
		existing.args.WriteString(tc.Raw)
	}
}

func argsToMap(args krnkmodel.ToolCallArguments) map[string]any {
	if len(args) == 0 {
		return map[string]any{}
	}
	return map[string]any(args)
}

func functionArgsFromRawJSON(raw string) map[string]any {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return map[string]any{}
	}
	var parsed map[string]any
	if err := json.Unmarshal([]byte(trimmed), &parsed); err == nil {
		return parsed
	}
	return map[string]any{rawFunctionArgsJSONKey: raw}
}

func finishReasonToGenai(reason string) genai.FinishReason {
	switch reason {
	case krnkmodel.FinishReasonStop, krnkmodel.FinishReasonTool:
		return genai.FinishReasonStop
	case "length":
		return genai.FinishReasonMaxTokens
	case krnkmodel.FinishReasonError:
		return genai.FinishReasonOther
	case "":
		return ""
	default:
		return genai.FinishReasonOther
	}
}

func usageToGenai(u *krnkmodel.Usage) *genai.GenerateContentResponseUsageMetadata {
	if u == nil {
		return nil
	}
	return &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     clampInt32(u.PromptTokens),
		CandidatesTokenCount: clampInt32(u.CompletionTokens),
		TotalTokenCount:      clampInt32(u.TotalTokens),
	}
}

// clampInt32 converts an int token count to int32, saturating at
// [math.MaxInt32] so excessively large (or negative) token counts never
// overflow silently.
func clampInt32(v int) int32 {
	switch {
	case v > math.MaxInt32:
		return math.MaxInt32
	case v < math.MinInt32:
		return math.MinInt32
	default:
		return int32(v)
	}
}

func customMetadataFromResponse(resp krnkmodel.ChatResponse) map[string]any {
	if resp.Model == "" {
		return nil
	}
	return map[string]any{customMetadataKeyModel: resp.Model}
}
