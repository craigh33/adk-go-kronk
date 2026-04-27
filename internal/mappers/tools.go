package mappers

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	krnkmodel "github.com/ardanlabs/kronk/sdk/kronk/model"
	"google.golang.org/genai"
)

// unsupportedToolVariantCount is a hint for the slice pre-allocation used by
// [unsupportedToolVariantsFromGenai]. It mirrors the bedrock provider so adding
// or removing a genai variant only requires updating this value.
const unsupportedToolVariantCount = 11

// toolsFromGenai converts the function declarations in cfg.Tools into
// Kronk-shaped tool documents. Non-function genai tool variants (for example
// GoogleSearch, CodeExecution, or MCPServers) are rejected with a clear error
// because Kronk's local models only consume OpenAI-style function
// declarations.
func toolsFromGenai(cfg *genai.GenerateContentConfig) ([]krnkmodel.D, error) {
	if cfg == nil || len(cfg.Tools) == 0 {
		return nil, nil
	}

	var docs []krnkmodel.D
	for _, t := range cfg.Tools {
		if t == nil {
			continue
		}
		var err error
		docs, err = appendFunctionDeclarationDocs(docs, t)
		if err != nil {
			return nil, err
		}
		if unsupported := unsupportedToolVariantsFromGenai(t); len(unsupported) > 0 {
			return nil, fmt.Errorf(
				"kronk provider does not support these genai tool variants: %s; use FunctionDeclarations instead",
				strings.Join(unsupported, ", "),
			)
		}
	}

	if len(docs) == 0 {
		return nil, nil
	}
	return docs, nil
}

func appendFunctionDeclarationDocs(docs []krnkmodel.D, t *genai.Tool) ([]krnkmodel.D, error) {
	for _, fd := range t.FunctionDeclarations {
		if fd == nil || fd.Name == "" {
			continue
		}
		params, err := functionParametersToJSONSchema(fd)
		if err != nil {
			return nil, fmt.Errorf("tool %q: %w", fd.Name, err)
		}
		docs = append(docs, krnkmodel.D{
			"type": "function",
			"function": krnkmodel.D{
				"name":        fd.Name,
				"description": fd.Description,
				"parameters":  params,
			},
		})
	}
	return docs, nil
}

// unsupportedToolVariantsFromGenai returns the set of non-function genai tool
// variants present on t. Callers treat a non-empty result as a request-time
// error so users see a clear, actionable message instead of silently losing
// the tool.
func unsupportedToolVariantsFromGenai(t *genai.Tool) []string {
	if t == nil {
		return nil
	}
	unsupported := make([]string, 0, unsupportedToolVariantCount)
	appendUnsupported := func(enabled bool, name string) {
		if !enabled || slices.Contains(unsupported, name) {
			return
		}
		unsupported = append(unsupported, name)
	}

	appendUnsupported(t.Retrieval != nil, "Retrieval")
	appendUnsupported(t.ComputerUse != nil, "ComputerUse")
	appendUnsupported(t.FileSearch != nil, "FileSearch")
	appendUnsupported(t.GoogleSearch != nil, "GoogleSearch")
	appendUnsupported(t.GoogleMaps != nil, "GoogleMaps")
	appendUnsupported(t.CodeExecution != nil, "CodeExecution")
	appendUnsupported(t.EnterpriseWebSearch != nil, "EnterpriseWebSearch")
	appendUnsupported(t.GoogleSearchRetrieval != nil, "GoogleSearchRetrieval")
	appendUnsupported(t.ParallelAISearch != nil, "ParallelAISearch")
	appendUnsupported(t.URLContext != nil, "URLContext")
	appendUnsupported(len(t.MCPServers) > 0, "MCPServers")

	return unsupported
}

func functionParametersToJSONSchema(fd *genai.FunctionDeclaration) (map[string]any, error) {
	if fd.ParametersJsonSchema != nil {
		if m, ok := fd.ParametersJsonSchema.(map[string]any); ok {
			normalizeSchemaTypes(m)
			return m, nil
		}
		b, err := json.Marshal(fd.ParametersJsonSchema)
		if err != nil {
			return nil, err
		}
		var m map[string]any
		if err := json.Unmarshal(b, &m); err != nil {
			return nil, err
		}
		normalizeSchemaTypes(m)
		return m, nil
	}
	if fd.Parameters == nil {
		return map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		}, nil
	}
	b, err := json.Marshal(fd.Parameters)
	if err != nil {
		return nil, err
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	normalizeSchemaTypes(m)
	return m, nil
}

// normalizeSchemaTypes recursively lowercases every "type" field value in a
// JSON Schema map. genai.Schema marshals Gemini-style uppercase type names
// (for example "STRING", "OBJECT", "ARRAY") but most OpenAI-compatible
// consumers — Kronk included — expect lowercase JSON Schema type names.

//nolint:gocognit // not concerned yet about complexity
func normalizeSchemaTypes(v any) {
	switch m := v.(type) {
	case map[string]any:
		for k, val := range m {
			if k == "type" {
				switch t := val.(type) {
				case string:
					m[k] = strings.ToLower(t)
				case []any:
					for i, item := range t {
						if s, ok := item.(string); ok {
							t[i] = strings.ToLower(s)
						}
					}
				case []string:
					for i, s := range t {
						t[i] = strings.ToLower(s)
					}
					m[k] = t
				}
			} else {
				normalizeSchemaTypes(val)
			}
		}
	case []any:
		for _, item := range m {
			normalizeSchemaTypes(item)
		}
	}
}

// toolChoiceFromGenai converts a genai tool-calling mode into the Kronk
// "tool_selection" value. Returns the empty string when no mapping applies so
// callers can omit the field entirely.
func toolChoiceFromGenai(cfg *genai.GenerateContentConfig) string {
	if cfg == nil || cfg.ToolConfig == nil || cfg.ToolConfig.FunctionCallingConfig == nil {
		return ""
	}
	switch cfg.ToolConfig.FunctionCallingConfig.Mode {
	case genai.FunctionCallingConfigModeAny:
		return "required"
	case genai.FunctionCallingConfigModeNone:
		return "none"
	case genai.FunctionCallingConfigModeAuto:
		return "auto"
	case genai.FunctionCallingConfigModeUnspecified, genai.FunctionCallingConfigModeValidated:
		// No meaningful mapping onto Kronk's tool_selection, let the model decide.
		return ""
	default:
		return ""
	}
}
