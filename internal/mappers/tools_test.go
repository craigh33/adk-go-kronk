package mappers

import (
	"slices"
	"strings"
	"testing"

	krnkmodel "github.com/ardanlabs/kronk/sdk/kronk/model"
	"google.golang.org/genai"
)

func TestToolsFromGenai_FunctionDeclarations(t *testing.T) {
	t.Parallel()

	cfg := &genai.GenerateContentConfig{
		Tools: []*genai.Tool{{
			FunctionDeclarations: []*genai.FunctionDeclaration{{
				Name:        "get_weather",
				Description: "Look up the weather",
				ParametersJsonSchema: map[string]any{
					"type": "OBJECT",
					"properties": map[string]any{
						"city": map[string]any{"type": "STRING"},
					},
					"required": []any{"city"},
				},
			}},
		}},
	}

	docs, err := toolsFromGenai(cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 tool doc, got %d", len(docs))
	}

	doc := docs[0]
	if doc["type"] != "function" {
		t.Fatalf("expected type=function, got %v", doc["type"])
	}
	fn, _ := doc["function"].(krnkmodel.D)
	if fn["name"] != "get_weather" || fn["description"] != "Look up the weather" {
		t.Fatalf("unexpected function envelope: %#v", fn)
	}
	params, _ := fn["parameters"].(map[string]any)
	if params["type"] != "object" {
		t.Fatalf("expected normalized lowercase type=object, got %v", params["type"])
	}
	props, _ := params["properties"].(map[string]any)
	city, _ := props["city"].(map[string]any)
	if city["type"] != "string" {
		t.Fatalf("expected normalized city.type=string, got %v", city["type"])
	}
}

func TestToolsFromGenai_SchemaFromParameters(t *testing.T) {
	t.Parallel()

	cfg := &genai.GenerateContentConfig{
		Tools: []*genai.Tool{{
			FunctionDeclarations: []*genai.FunctionDeclaration{{
				Name: "add",
				Parameters: &genai.Schema{
					Type: genai.TypeObject,
					Properties: map[string]*genai.Schema{
						"a": {Type: genai.TypeNumber},
						"b": {Type: genai.TypeNumber},
					},
				},
			}},
		}},
	}
	docs, err := toolsFromGenai(cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 doc, got %d", len(docs))
	}
	fn, _ := docs[0]["function"].(krnkmodel.D)
	params, _ := fn["parameters"].(map[string]any)
	if params["type"] != "object" {
		t.Fatalf("expected object type, got %v", params["type"])
	}
}

func TestToolsFromGenai_NoToolsReturnsNil(t *testing.T) {
	t.Parallel()

	if docs, err := toolsFromGenai(nil); err != nil || docs != nil {
		t.Fatalf("expected nil/nil, got %#v / %v", docs, err)
	}
	if docs, err := toolsFromGenai(&genai.GenerateContentConfig{}); err != nil || docs != nil {
		t.Fatalf("expected nil/nil, got %#v / %v", docs, err)
	}
}

func TestToolsFromGenai_FunctionWithoutNameIsSkipped(t *testing.T) {
	t.Parallel()

	cfg := &genai.GenerateContentConfig{
		Tools: []*genai.Tool{{
			FunctionDeclarations: []*genai.FunctionDeclaration{
				{Name: ""},
				{Name: "ok"},
			},
		}},
	}
	docs, err := toolsFromGenai(cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 doc, got %d", len(docs))
	}
	fn, _ := docs[0]["function"].(krnkmodel.D)
	if fn["name"] != "ok" {
		t.Fatalf("expected only 'ok' function, got %#v", fn)
	}
}

func TestToolsFromGenai_UnsupportedVariantRejected(t *testing.T) {
	t.Parallel()

	cfg := &genai.GenerateContentConfig{
		Tools: []*genai.Tool{
			{GoogleSearch: &genai.GoogleSearch{}},
			{CodeExecution: &genai.ToolCodeExecution{}},
		},
	}
	_, err := toolsFromGenai(cfg)
	if err == nil {
		t.Fatal("expected error for unsupported tool variants")
	}
	if !strings.Contains(err.Error(), "GoogleSearch") && !strings.Contains(err.Error(), "CodeExecution") {
		t.Fatalf("expected error to name unsupported variants, got %v", err)
	}
}

func TestUnsupportedToolVariantsFromGenai(t *testing.T) {
	t.Parallel()

	t.Run("nil is empty", func(t *testing.T) {
		t.Parallel()
		if out := unsupportedToolVariantsFromGenai(nil); len(out) != 0 {
			t.Fatalf("expected empty, got %v", out)
		}
	})

	t.Run("multiple variants", func(t *testing.T) {
		t.Parallel()
		tool := &genai.Tool{
			Retrieval:     &genai.Retrieval{},
			GoogleSearch:  &genai.GoogleSearch{},
			CodeExecution: &genai.ToolCodeExecution{},
			URLContext:    &genai.URLContext{},
			MCPServers:    []*genai.MCPServer{{}},
		}
		out := unsupportedToolVariantsFromGenai(tool)
		want := []string{"Retrieval", "GoogleSearch", "CodeExecution", "URLContext", "MCPServers"}
		for _, w := range want {
			if !contains(out, w) {
				t.Errorf("expected %q in unsupported list, got %v", w, out)
			}
		}
	})
}

func TestToolChoiceFromGenai(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		mode genai.FunctionCallingConfigMode
		want string
	}{
		{"any -> required", genai.FunctionCallingConfigModeAny, "required"},
		{"none -> none", genai.FunctionCallingConfigModeNone, "none"},
		{"auto -> auto", genai.FunctionCallingConfigModeAuto, "auto"},
		{"unspecified -> empty", genai.FunctionCallingConfigModeUnspecified, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			cfg := &genai.GenerateContentConfig{
				ToolConfig: &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: tc.mode}},
			}
			if got := toolChoiceFromGenai(cfg); got != tc.want {
				t.Fatalf("got %q, want %q", got, tc.want)
			}
		})
	}

	if got := toolChoiceFromGenai(nil); got != "" {
		t.Fatalf("nil cfg: got %q", got)
	}
	if got := toolChoiceFromGenai(&genai.GenerateContentConfig{}); got != "" {
		t.Fatalf("empty cfg: got %q", got)
	}
	if got := toolChoiceFromGenai(&genai.GenerateContentConfig{ToolConfig: &genai.ToolConfig{}}); got != "" {
		t.Fatalf("nil FunctionCallingConfig: got %q", got)
	}
}

func TestNormalizeSchemaTypes(t *testing.T) {
	t.Parallel()

	schema := map[string]any{
		"type": "OBJECT",
		"properties": map[string]any{
			"name": map[string]any{"type": "STRING"},
			"items": map[string]any{
				"type":  "ARRAY",
				"items": map[string]any{"type": "INTEGER"},
			},
			"any_of": []any{
				map[string]any{"type": "NULL"},
				map[string]any{"type": "NUMBER"},
			},
		},
	}

	normalizeSchemaTypes(schema)

	if schema["type"] != "object" {
		t.Errorf("root: got %v", schema["type"])
	}
	props := schema["properties"].(map[string]any)
	if props["name"].(map[string]any)["type"] != "string" {
		t.Errorf("name: got %v", props["name"])
	}
	items := props["items"].(map[string]any)
	if items["type"] != "array" {
		t.Errorf("items: got %v", items["type"])
	}
	if items["items"].(map[string]any)["type"] != "integer" {
		t.Errorf("items.items: got %v", items["items"])
	}
	anyOf := props["any_of"].([]any)
	if anyOf[0].(map[string]any)["type"] != "null" || anyOf[1].(map[string]any)["type"] != "number" {
		t.Errorf("any_of: got %v", anyOf)
	}
}

// =============================================================================
// Helpers

func contains(haystack []string, needle string) bool {
	return slices.Contains(haystack, needle)
}
