package mappers

import (
	"math"
	"strings"
	"testing"

	krnkmodel "github.com/ardanlabs/kronk/sdk/kronk/model"
	"google.golang.org/genai"
)

func TestLLMResponseFromChatResponse_TextAndReasoning(t *testing.T) {
	t.Parallel()

	stop := krnkmodel.FinishReasonStop
	resp := krnkmodel.ChatResponse{
		Model: "Qwen3",
		Choices: []krnkmodel.Choice{{
			Message: &krnkmodel.ResponseMessage{
				Role:      krnkmodel.RoleAssistant,
				Content:   "hello!",
				Reasoning: "thinking...",
			},
			FinishReasonPtr: &stop,
		}},
		Usage: &krnkmodel.Usage{PromptTokens: 3, CompletionTokens: 5, TotalTokens: 8},
	}

	out, err := LLMResponseFromChatResponse(resp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.FinishReason != genai.FinishReasonStop {
		t.Fatalf("expected FinishReasonStop, got %v", out.FinishReason)
	}
	if out.Content == nil || out.Content.Role != "model" {
		t.Fatalf("expected model-role content, got %#v", out.Content)
	}
	if len(out.Content.Parts) != 2 {
		t.Fatalf("expected reasoning + text parts, got %d", len(out.Content.Parts))
	}
	if !out.Content.Parts[0].Thought || out.Content.Parts[0].Text != "thinking..." {
		t.Fatalf("unexpected reasoning part: %#v", out.Content.Parts[0])
	}
	if out.Content.Parts[1].Text != "hello!" {
		t.Fatalf("unexpected text part: %#v", out.Content.Parts[1])
	}
	if out.UsageMetadata == nil || out.UsageMetadata.TotalTokenCount != 8 {
		t.Fatalf("unexpected usage: %#v", out.UsageMetadata)
	}
	if out.CustomMetadata[customMetadataKeyModel] != "Qwen3" {
		t.Fatalf("expected model ID in custom metadata, got %#v", out.CustomMetadata)
	}
}

func TestLLMResponseFromChatResponse_ToolCall(t *testing.T) {
	t.Parallel()

	tool := krnkmodel.FinishReasonTool
	resp := krnkmodel.ChatResponse{
		Choices: []krnkmodel.Choice{{
			Message: &krnkmodel.ResponseMessage{
				Role: krnkmodel.RoleAssistant,
				ToolCalls: []krnkmodel.ResponseToolCall{{
					ID:    "call_1",
					Index: 0,
					Type:  "function",
					Function: krnkmodel.ResponseToolCallFunction{
						Name:      "get_weather",
						Arguments: krnkmodel.ToolCallArguments{"city": "London"},
					},
				}},
			},
			FinishReasonPtr: &tool,
		}},
	}

	out, err := LLMResponseFromChatResponse(resp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.FinishReason != genai.FinishReasonStop {
		t.Fatalf("tool_calls should map to Stop, got %v", out.FinishReason)
	}
	if len(out.Content.Parts) != 1 || out.Content.Parts[0].FunctionCall == nil {
		t.Fatalf("expected one FunctionCall part, got %#v", out.Content.Parts)
	}
	fc := out.Content.Parts[0].FunctionCall
	if fc.Name != "get_weather" || fc.ID != "call_1" {
		t.Fatalf("unexpected function call: %#v", fc)
	}
	if fc.Args["city"] != "London" {
		t.Fatalf("expected args city=London, got %#v", fc.Args)
	}
}

func TestLLMResponseFromChatResponse_ErrorFinishReason(t *testing.T) {
	t.Parallel()

	errReason := krnkmodel.FinishReasonError
	resp := krnkmodel.ChatResponse{
		Choices: []krnkmodel.Choice{{
			Message:         &krnkmodel.ResponseMessage{Content: "boom"},
			FinishReasonPtr: &errReason,
		}},
	}
	out, err := LLMResponseFromChatResponse(resp)
	if err == nil {
		t.Fatal("expected error on FinishReasonError")
	}
	if out != nil {
		t.Fatalf("expected nil response, got %#v", out)
	}
	if !strings.Contains(err.Error(), "boom") {
		t.Fatalf("expected error to include model message, got %v", err)
	}
}

func TestLLMResponseFromChatResponse_NoChoices(t *testing.T) {
	t.Parallel()

	out, err := LLMResponseFromChatResponse(krnkmodel.ChatResponse{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out == nil || out.Content == nil {
		t.Fatal("expected a non-nil placeholder response")
	}
	if out.FinishReason != genai.FinishReasonOther {
		t.Fatalf("expected FinishReasonOther, got %v", out.FinishReason)
	}
}

func TestStreamChunkToLLMResponse_AccumulatesText(t *testing.T) {
	t.Parallel()

	state := NewStreamState()

	first, err := StreamChunkToLLMResponse(state, krnkmodel.ChatResponse{
		Model: "m",
		Choices: []krnkmodel.Choice{{
			Delta: &krnkmodel.ResponseMessage{Content: "hel"},
		}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if first == nil || !first.Partial || first.Content.Parts[0].Text != "hel" {
		t.Fatalf("expected partial 'hel', got %#v", first)
	}

	second, err := StreamChunkToLLMResponse(state, krnkmodel.ChatResponse{
		Choices: []krnkmodel.Choice{{
			Delta: &krnkmodel.ResponseMessage{Content: "lo"},
		}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if second == nil || second.Content.Parts[0].Text != "lo" {
		t.Fatalf("expected incremental delta 'lo', got %#v", second)
	}

	if state.text.String() != "hello" {
		t.Fatalf("expected accumulated text 'hello', got %q", state.text.String())
	}
	if state.modelID != "m" {
		t.Fatalf("expected modelID 'm', got %q", state.modelID)
	}
}

func TestStreamChunkToLLMResponse_AbsorbsEmpty(t *testing.T) {
	t.Parallel()

	state := NewStreamState()
	out, err := StreamChunkToLLMResponse(state, krnkmodel.ChatResponse{
		Choices: []krnkmodel.Choice{{Delta: &krnkmodel.ResponseMessage{}}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out != nil {
		t.Fatalf("expected nil response for empty delta, got %#v", out)
	}
}

func TestStreamChunkToLLMResponse_ErrorFinishReason(t *testing.T) {
	t.Parallel()

	state := NewStreamState()
	errReason := krnkmodel.FinishReasonError
	_, err := StreamChunkToLLMResponse(state, krnkmodel.ChatResponse{
		Choices: []krnkmodel.Choice{{
			Delta:           &krnkmodel.ResponseMessage{Content: "kaboom"},
			FinishReasonPtr: &errReason,
		}},
	})
	if err == nil {
		t.Fatal("expected error on FinishReasonError chunk")
	}
}

func TestStreamChunkToLLMResponse_NilState(t *testing.T) {
	t.Parallel()

	if _, err := StreamChunkToLLMResponse(nil, krnkmodel.ChatResponse{}); err == nil {
		t.Fatal("expected error on nil state")
	}
}

func TestFinalStreamResponse_AggregatesAll(t *testing.T) {
	t.Parallel()

	state := NewStreamState()
	state.modelID = "m"
	state.text.WriteString("hi")
	state.reasoning.WriteString("because")
	state.finishReason = krnkmodel.FinishReasonTool
	state.lastUsage = &krnkmodel.Usage{PromptTokens: 1, CompletionTokens: 2, TotalTokens: 3}

	// Simulate tool call accumulation through the public path.
	accumulateToolCall(state, krnkmodel.ResponseToolCall{
		ID: "call_1",
		Function: krnkmodel.ResponseToolCallFunction{
			Name:      "f",
			Arguments: krnkmodel.ToolCallArguments{"x": 1},
		},
	})

	out := FinalStreamResponse(state)
	if !out.TurnComplete {
		t.Fatal("expected TurnComplete=true")
	}
	if out.FinishReason != genai.FinishReasonStop {
		t.Fatalf("expected FinishReasonStop, got %v", out.FinishReason)
	}
	if out.UsageMetadata == nil || out.UsageMetadata.TotalTokenCount != 3 {
		t.Fatalf("unexpected usage: %#v", out.UsageMetadata)
	}
	if out.CustomMetadata[customMetadataKeyModel] != "m" {
		t.Fatalf("expected model ID in custom metadata, got %#v", out.CustomMetadata)
	}

	assertAggregatedParts(t, out.Content.Parts)
}

func assertAggregatedParts(t *testing.T, parts []*genai.Part) {
	t.Helper()

	var gotReasoning, gotText, gotTool bool
	for _, p := range parts {
		switch {
		case p.Thought:
			gotReasoning = true
			if p.Text != "because" {
				t.Fatalf("unexpected reasoning: %q", p.Text)
			}
		case p.FunctionCall != nil:
			gotTool = true
			assertToolCallPart(t, p.FunctionCall)
		case p.Text != "":
			gotText = true
			if p.Text != "hi" {
				t.Fatalf("unexpected text: %q", p.Text)
			}
		}
	}
	if !gotReasoning || !gotText || !gotTool {
		t.Fatalf("expected reasoning + text + tool parts, got %#v", parts)
	}
}

func assertToolCallPart(t *testing.T, fc *genai.FunctionCall) {
	t.Helper()

	if fc.Name != "f" || fc.ID != "call_1" {
		t.Fatalf("unexpected tool call: %#v", fc)
	}
	if x, _ := fc.Args["x"].(float64); x != 1 {
		t.Fatalf("expected args x=1, got %#v", fc.Args)
	}
}

func TestFinalStreamResponse_NilState(t *testing.T) {
	t.Parallel()

	out := FinalStreamResponse(nil)
	if out == nil || !out.TurnComplete {
		t.Fatal("expected non-nil TurnComplete response on nil state")
	}
	if out.FinishReason != genai.FinishReasonOther {
		t.Fatalf("expected FinishReasonOther, got %v", out.FinishReason)
	}
}

func TestFinalStreamResponse_Empty(t *testing.T) {
	t.Parallel()

	out := FinalStreamResponse(NewStreamState())
	if len(out.Content.Parts) != 1 || out.Content.Parts[0].Text != "" {
		t.Fatalf("expected single empty text part, got %#v", out.Content.Parts)
	}
}

func TestFinishReasonToGenai(t *testing.T) {
	t.Parallel()

	cases := map[string]genai.FinishReason{
		krnkmodel.FinishReasonStop:  genai.FinishReasonStop,
		krnkmodel.FinishReasonTool:  genai.FinishReasonStop,
		"length":                    genai.FinishReasonMaxTokens,
		krnkmodel.FinishReasonError: genai.FinishReasonOther,
		"":                          "",
		"weird_reason":              genai.FinishReasonOther,
	}
	for in, want := range cases {
		if got := finishReasonToGenai(in); got != want {
			t.Errorf("finishReasonToGenai(%q) = %v, want %v", in, got, want)
		}
	}
}

func TestFunctionArgsFromRawJSON(t *testing.T) {
	t.Parallel()

	if got := functionArgsFromRawJSON(""); len(got) != 0 {
		t.Fatalf("expected empty args for empty JSON, got %#v", got)
	}
	got := functionArgsFromRawJSON(`{"x":1}`)
	if got["x"].(float64) != 1 {
		t.Fatalf("expected parsed x=1, got %#v", got)
	}
	got = functionArgsFromRawJSON(`not json`)
	if got[rawFunctionArgsJSONKey] != "not json" {
		t.Fatalf("expected raw fallback, got %#v", got)
	}
}

func TestClampInt32(t *testing.T) {
	t.Parallel()

	cases := map[int]int32{
		0:                  0,
		42:                 42,
		math.MaxInt32:      math.MaxInt32,
		math.MaxInt32 + 1:  math.MaxInt32,
		-1:                 -1,
		math.MinInt32:      math.MinInt32,
		math.MinInt32 - 1:  math.MinInt32,
		math.MaxInt32 * 10: math.MaxInt32,
	}
	for in, want := range cases {
		if got := clampInt32(in); got != want {
			t.Errorf("clampInt32(%d) = %d, want %d", in, got, want)
		}
	}
}

func TestUsageToGenai_Nil(t *testing.T) {
	t.Parallel()

	if got := usageToGenai(nil); got != nil {
		t.Fatalf("expected nil, got %#v", got)
	}
}
