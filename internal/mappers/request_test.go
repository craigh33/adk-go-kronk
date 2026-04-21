package mappers

import (
	"encoding/json"
	"strings"
	"testing"

	krnkmodel "github.com/ardanlabs/kronk/sdk/kronk/model"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestMaybeAppendUserContent(t *testing.T) {
	t.Parallel()

	t.Run("empty slice gets a default user turn", func(t *testing.T) {
		t.Parallel()

		out := MaybeAppendUserContent(nil)
		if len(out) != 1 {
			t.Fatalf("expected 1 content, got %d", len(out))
		}
		if out[0].Role != genaiRoleUser {
			t.Fatalf("expected user role, got %q", out[0].Role)
		}
		if text := concatPartText(out[0].Parts); text == "" {
			t.Fatal("expected synthetic user text, got empty")
		}
	})

	t.Run("assistant-terminated history gets a continuation user turn", func(t *testing.T) {
		t.Parallel()

		in := []*genai.Content{
			genai.NewContentFromText("hello", genaiRoleUser),
			genai.NewContentFromText("hi there", genaiRoleModel),
		}
		out := MaybeAppendUserContent(in)
		if len(out) != 3 {
			t.Fatalf("expected 3 contents, got %d", len(out))
		}
		if out[2].Role != genaiRoleUser {
			t.Fatalf("expected final user turn, got role %q", out[2].Role)
		}
	})

	t.Run("user-terminated history is unchanged", func(t *testing.T) {
		t.Parallel()

		in := []*genai.Content{genai.NewContentFromText("go", genaiRoleUser)}
		out := MaybeAppendUserContent(in)
		if len(out) != 1 {
			t.Fatalf("expected 1 content, got %d", len(out))
		}
	})
}

func TestRequestFromLLMRequest_NilRequest(t *testing.T) {
	t.Parallel()

	if _, err := RequestFromLLMRequest(nil, false); err == nil {
		t.Fatal("expected error on nil LLMRequest")
	}
}

func TestRequestFromLLMRequest_BasicUserTurnAndSystemInstruction(t *testing.T) {
	t.Parallel()

	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hello", genaiRoleUser)},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: genai.NewContentFromText("be brief", genaiRoleSystem),
			Temperature:       new(float32(0.1)),
			TopP:              new(float32(0.9)),
			TopK:              new(float32(40)),
			MaxOutputTokens:   256,
			StopSequences:     []string{"STOP"},
			Seed:              new(int32(42)),
			FrequencyPenalty:  new(float32(0.5)),
			PresencePenalty:   new(float32(0.25)),
		},
	}

	d, err := RequestFromLLMRequest(req, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got := d["stream"]; got != true {
		t.Fatalf("expected stream true, got %v", got)
	}

	msgs, ok := d["messages"].([]krnkmodel.D)
	if !ok {
		t.Fatalf("expected []krnkmodel.D messages, got %T", d["messages"])
	}
	if len(msgs) != 2 {
		t.Fatalf("expected system + user, got %d messages: %#v", len(msgs), msgs)
	}
	if msgs[0]["role"] != krnkmodel.RoleSystem || msgs[0]["content"] != "be brief" {
		t.Fatalf("unexpected system message: %#v", msgs[0])
	}
	if msgs[1]["role"] != krnkmodel.RoleUser || msgs[1]["content"] != "hello" {
		t.Fatalf("unexpected user message: %#v", msgs[1])
	}

	for k, want := range map[string]any{
		"temperature":       float32(0.1),
		"top_p":             float32(0.9),
		"top_k":             40,
		"max_tokens":        256,
		"seed":              42,
		"frequency_penalty": float32(0.5),
		"presence_penalty":  float32(0.25),
	} {
		if got := d[k]; got != want {
			t.Errorf("param %q: got %v (%T), want %v (%T)", k, got, got, want, want)
		}
	}

	stops, ok := d["stop"].([]string)
	if !ok || len(stops) != 1 || stops[0] != "STOP" {
		t.Fatalf("unexpected stop sequences: %#v", d["stop"])
	}
}

func TestRequestFromLLMRequest_AssistantFunctionCall(t *testing.T) {
	t.Parallel()

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("what's the weather?", genaiRoleUser),
			{
				Role: genaiRoleModel,
				Parts: []*genai.Part{
					{Text: "let me check"},
					{FunctionCall: &genai.FunctionCall{
						ID:   "call_1",
						Name: "get_weather",
						Args: map[string]any{"city": "London"},
					}},
				},
			},
		},
	}

	d, err := RequestFromLLMRequest(req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, ok := d["stream"]; ok {
		t.Fatal("stream key should be absent when stream=false")
	}

	msgs, _ := d["messages"].([]krnkmodel.D)
	if len(msgs) != 3 {
		t.Fatalf("expected user + assistant + continuation user, got %d", len(msgs))
	}
	assistant := msgs[1]
	if assistant["role"] != krnkmodel.RoleAssistant {
		t.Fatalf("expected assistant role, got %v", assistant["role"])
	}
	if assistant["content"] != "let me check" {
		t.Fatalf("unexpected assistant text: %v", assistant["content"])
	}

	tcs, ok := assistant["tool_calls"].([]krnkmodel.D)
	if !ok || len(tcs) != 1 {
		t.Fatalf("expected 1 tool call, got %#v", assistant["tool_calls"])
	}
	tc := tcs[0]
	if tc["id"] != "call_1" || tc["type"] != "function" {
		t.Fatalf("unexpected tool call envelope: %#v", tc)
	}
	fn, _ := tc["function"].(krnkmodel.D)
	if fn["name"] != "get_weather" {
		t.Fatalf("unexpected function name: %v", fn["name"])
	}

	argsJSON, ok := fn["arguments"].(string)
	if !ok {
		t.Fatalf("expected arguments JSON string, got %T", fn["arguments"])
	}
	var parsed map[string]any
	if err := json.Unmarshal([]byte(argsJSON), &parsed); err != nil {
		t.Fatalf("arguments not valid JSON: %v", err)
	}
	if parsed["city"] != "London" {
		t.Fatalf("unexpected args: %v", parsed)
	}
}

func TestRequestFromLLMRequest_FunctionResponseBecomesToolMessage(t *testing.T) {
	t.Parallel()

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genaiRoleUser,
				Parts: []*genai.Part{
					{FunctionResponse: &genai.FunctionResponse{
						ID:       "call_1",
						Name:     "get_weather",
						Response: map[string]any{"temp": 22},
					}},
				},
			},
		},
	}

	d, err := RequestFromLLMRequest(req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	msgs, _ := d["messages"].([]krnkmodel.D)
	if len(msgs) == 0 {
		t.Fatalf("expected at least one message, got none")
	}
	tool := msgs[0]
	if tool["role"] != krnkmodel.RoleTool {
		t.Fatalf("expected tool role, got %v", tool["role"])
	}
	if tool["tool_call_id"] != "call_1" || tool["name"] != "get_weather" {
		t.Fatalf("unexpected tool envelope: %#v", tool)
	}
	if !strings.Contains(tool["content"].(string), `"temp":22`) {
		t.Fatalf("expected JSON response in content, got %v", tool["content"])
	}
}

func TestRequestFromLLMRequest_InlineImage(t *testing.T) {
	t.Parallel()

	req := &model.LLMRequest{
		Contents: []*genai.Content{{
			Role: genaiRoleUser,
			Parts: []*genai.Part{
				{Text: "describe this"},
				{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{0x89, 0x50}}},
			},
		}},
	}

	d, err := RequestFromLLMRequest(req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	msgs, _ := d["messages"].([]krnkmodel.D)
	if len(msgs) != 1 {
		t.Fatalf("expected 1 combined user message, got %d", len(msgs))
	}

	arr, ok := msgs[0]["content"].([]krnkmodel.D)
	if !ok {
		t.Fatalf("expected OpenAI content array, got %T", msgs[0]["content"])
	}
	if len(arr) != 2 {
		t.Fatalf("expected text + image block, got %d", len(arr))
	}
	if arr[0]["type"] != "text" || arr[0]["text"] != "describe this" {
		t.Fatalf("unexpected first block: %#v", arr[0])
	}
	if arr[1]["type"] != "image_url" {
		t.Fatalf("expected image_url block, got %#v", arr[1])
	}
	imgURL, _ := arr[1]["image_url"].(krnkmodel.D)
	if url, _ := imgURL["url"].(string); !strings.HasPrefix(url, "data:image/png;base64,") {
		t.Fatalf("expected data URL, got %q", url)
	}
}

func TestRequestFromLLMRequest_RejectsRemoteFileData(t *testing.T) {
	t.Parallel()

	req := &model.LLMRequest{
		Contents: []*genai.Content{{
			Role: genaiRoleUser,
			Parts: []*genai.Part{
				{FileData: &genai.FileData{FileURI: "gs://bucket/obj"}},
			},
		}},
	}
	if _, err := RequestFromLLMRequest(req, false); err == nil {
		t.Fatal("expected error for remote FileData")
	}
}

func TestRequestFromLLMRequest_RejectsUnsupportedInlineMime(t *testing.T) {
	t.Parallel()

	req := &model.LLMRequest{
		Contents: []*genai.Content{{
			Role: genaiRoleUser,
			Parts: []*genai.Part{
				{InlineData: &genai.Blob{MIMEType: "application/pdf", Data: []byte{1, 2}}},
			},
		}},
	}
	if _, err := RequestFromLLMRequest(req, false); err == nil {
		t.Fatal("expected error for application/pdf inline data")
	}
}

func TestRequestFromLLMRequest_RejectsModelRoleMedia(t *testing.T) {
	t.Parallel()

	req := &model.LLMRequest{
		Contents: []*genai.Content{{
			Role: genaiRoleModel,
			Parts: []*genai.Part{
				{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1}}},
			},
		}},
	}
	if _, err := RequestFromLLMRequest(req, false); err == nil {
		t.Fatal("expected error for model-role inline media")
	}
}

func TestRequestFromLLMRequest_ReasoningThoughtParts(t *testing.T) {
	t.Parallel()

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("hi", genaiRoleUser),
			{
				Role: genaiRoleModel,
				Parts: []*genai.Part{
					{Text: "let me think", Thought: true},
					{Text: "answer"},
				},
			},
		},
	}

	d, err := RequestFromLLMRequest(req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	msgs, _ := d["messages"].([]krnkmodel.D)
	assistant := findRoleMessage(msgs, krnkmodel.RoleAssistant)
	if assistant == nil {
		t.Fatal("assistant message missing")
	}
	if assistant["reasoning_content"] != "let me think" {
		t.Fatalf("expected reasoning content, got %v", assistant["reasoning_content"])
	}
	if assistant["content"] != "answer" {
		t.Fatalf("expected plain content, got %v", assistant["content"])
	}
}

func TestRequestFromLLMRequest_UnknownRoleErrors(t *testing.T) {
	t.Parallel()

	req := &model.LLMRequest{
		Contents: []*genai.Content{{Role: "alien", Parts: []*genai.Part{{Text: "x"}}}},
	}
	if _, err := RequestFromLLMRequest(req, false); err == nil {
		t.Fatal("expected error for unknown role")
	}
}

func TestRequestFromLLMRequest_ToolChoicePassthrough(t *testing.T) {
	t.Parallel()

	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("go", genaiRoleUser)},
		Config: &genai.GenerateContentConfig{
			Tools: []*genai.Tool{{
				FunctionDeclarations: []*genai.FunctionDeclaration{{Name: "f"}},
			}},
			ToolConfig: &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode: genai.FunctionCallingConfigModeAny,
			}},
		},
	}
	d, err := RequestFromLLMRequest(req, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if d["tool_selection"] != "required" {
		t.Fatalf("expected tool_selection=required, got %v", d["tool_selection"])
	}
	if _, ok := d["tools"]; !ok {
		t.Fatal("expected tools to be present")
	}
}

// =============================================================================

func TestNormalizeMIME(t *testing.T) {
	t.Parallel()

	cases := map[string]string{
		"":                         "",
		"image/png":                "image/png",
		"IMAGE/PNG":                "image/png",
		"image/png; charset=utf-8": "image/png",
		"image/png;charset=utf-8":  "image/png",
		"IMAGE/PNG; charset=UTF-8": "image/png",
	}
	for in, want := range cases {
		if got := normalizeMIME(in); got != want {
			t.Errorf("normalizeMIME(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestFunctionCallToToolCall_DefaultIDAndEmptyArgs(t *testing.T) {
	t.Parallel()

	tc, err := functionCallToToolCall(&genai.FunctionCall{Name: "do_thing"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if tc["id"] != "call_do_thing" {
		t.Fatalf("expected default id call_do_thing, got %v", tc["id"])
	}
	fn, _ := tc["function"].(krnkmodel.D)
	if fn["arguments"] != "{}" {
		t.Fatalf("expected empty args JSON, got %v", fn["arguments"])
	}
}

// =============================================================================
// Helpers

func findRoleMessage(msgs []krnkmodel.D, role string) krnkmodel.D {
	for _, m := range msgs {
		if m["role"] == role {
			return m
		}
	}
	return nil
}
