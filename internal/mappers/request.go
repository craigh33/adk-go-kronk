// Package mappers converts between ADK/genai types and the Kronk SDK request
// and response shapes used by github.com/ardanlabs/kronk.
package mappers

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	krnkmodel "github.com/ardanlabs/kronk/sdk/kronk/model"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

const (
	genaiRoleUser   = "user"
	genaiRoleModel  = "model"
	genaiRoleSystem = "system"
)

// MaybeAppendUserContent mirrors the Gemini / Bedrock provider behavior so
// empty histories or assistant-terminated turns still receive a valid final
// user message. Kronk's chat templates otherwise reject conversations that
// end in an assistant turn.
func MaybeAppendUserContent(contents []*genai.Content) []*genai.Content {
	if len(contents) == 0 {
		return append(contents, genai.NewContentFromText(
			"Handle the requests as specified in the System Instruction.", genaiRoleUser))
	}
	if last := contents[len(contents)-1]; last != nil && last.Role != genaiRoleUser {
		return append(contents, genai.NewContentFromText(
			"Continue processing previous requests as instructed. Exit or provide a summary if no more outputs are needed.",
			genaiRoleUser,
		))
	}
	return contents
}

// RequestFromLLMRequest converts an ADK LLMRequest into the Kronk chat
// "document" map expected by kronk.Kronk.Chat / ChatStreaming. When stream is
// true the document sets "stream": true so downstream helpers like
// ChatStreamingHTTP behave consistently, even though the streaming call site
// does not strictly require it.
func RequestFromLLMRequest(req *model.LLMRequest, stream bool) (krnkmodel.D, error) {
	if req == nil {
		return nil, errors.New("nil LLMRequest")
	}

	cfg := req.Config
	if cfg == nil {
		cfg = &genai.GenerateContentConfig{}
	}

	contents := MaybeAppendUserContent(append([]*genai.Content(nil), req.Contents...))

	messages, err := contentsToMessages(cfg, contents)
	if err != nil {
		return nil, err
	}
	if len(messages) == 0 {
		return nil, errors.New(
			"no messages to send to Kronk: every user/model part was empty or could not be mapped",
		)
	}

	d := krnkmodel.D{
		"messages": messages,
	}

	if stream {
		d["stream"] = true
	}

	applyInferenceParams(d, cfg)

	tools, err := toolsFromGenai(cfg)
	if err != nil {
		return nil, err
	}
	if len(tools) > 0 {
		d["tools"] = tools
		if choice := toolChoiceFromGenai(cfg); choice != "" {
			d["tool_selection"] = choice
		}
	}

	return d, nil
}

func contentsToMessages(cfg *genai.GenerateContentConfig, contents []*genai.Content) ([]krnkmodel.D, error) {
	messages := make([]krnkmodel.D, 0, len(contents)+1)

	if sys := systemInstructionText(cfg); sys != "" {
		messages = append(messages, krnkmodel.TextMessage(krnkmodel.RoleSystem, sys))
	}

	for _, c := range contents {
		if c == nil {
			continue
		}

		// genai uses "user" / "model" / "system"; Kronk expects the
		// OpenAI-compatible "user" / "assistant" / "system" / "tool".
		switch c.Role {
		case genaiRoleSystem:
			if text := concatPartText(c.Parts); text != "" {
				messages = append(messages, krnkmodel.TextMessage(krnkmodel.RoleSystem, text))
			}
			continue
		case genaiRoleUser:
			userMsgs, err := userContentToMessages(c.Parts)
			if err != nil {
				return nil, err
			}
			messages = append(messages, userMsgs...)
		case genaiRoleModel, "":
			assistant, err := modelContentToMessage(c.Parts)
			if err != nil {
				return nil, err
			}
			if assistant != nil {
				messages = append(messages, assistant)
			}
		default:
			return nil, fmt.Errorf("unsupported content role for Kronk: %q", c.Role)
		}
	}

	return messages, nil
}

func systemInstructionText(cfg *genai.GenerateContentConfig) string {
	if cfg == nil || cfg.SystemInstruction == nil {
		return ""
	}
	return concatPartText(cfg.SystemInstruction.Parts)
}

func concatPartText(parts []*genai.Part) string {
	var b strings.Builder
	for _, p := range parts {
		if p == nil || p.Text == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteString("\n")
		}
		b.WriteString(p.Text)
	}
	return b.String()
}

// userContentToMessages emits one or more Kronk messages for a user turn.
// FunctionResponse parts become standalone role:"tool" messages so Kronk can
// wire them back to the preceding tool_call id. Mixed media + text in a
// single user Content produce one message with an OpenAI-style content array.
func userContentToMessages(parts []*genai.Part) ([]krnkmodel.D, error) {
	acc := &userPartAccumulator{}

	for _, p := range parts {
		if p == nil {
			continue
		}
		if err := acc.addPart(p); err != nil {
			return nil, err
		}
	}

	return acc.finalize(), nil
}

type userPartAccumulator struct {
	out          []krnkmodel.D
	contentArray []krnkmodel.D
	plainText    strings.Builder
}

func (a *userPartAccumulator) flushTextOnly() {
	if a.plainText.Len() == 0 {
		return
	}
	a.out = append(a.out, krnkmodel.TextMessage(krnkmodel.RoleUser, a.plainText.String()))
	a.plainText.Reset()
}

func (a *userPartAccumulator) addPart(p *genai.Part) error {
	switch {
	case p.FunctionResponse != nil:
		a.flushTextOnly()
		toolMsg, err := functionResponseToToolMessage(p.FunctionResponse)
		if err != nil {
			return err
		}
		a.out = append(a.out, toolMsg)
	case p.InlineData != nil && len(p.InlineData.Data) > 0:
		block, err := inlineDataContentBlock(p)
		if err != nil {
			return err
		}
		if a.plainText.Len() > 0 {
			a.contentArray = append(a.contentArray, krnkmodel.D{
				"type": "text",
				"text": a.plainText.String(),
			})
			a.plainText.Reset()
		}
		a.contentArray = append(a.contentArray, block)
	case p.FileData != nil && p.FileData.FileURI != "":
		return fmt.Errorf(
			"kronk provider does not accept remote FileData URIs (%q); inline the bytes via InlineData instead",
			p.FileData.FileURI,
		)
	case p.Text != "":
		if a.plainText.Len() > 0 {
			a.plainText.WriteString("\n")
		}
		a.plainText.WriteString(p.Text)
	}
	return nil
}

func (a *userPartAccumulator) finalize() []krnkmodel.D {
	if len(a.contentArray) > 0 {
		if a.plainText.Len() > 0 {
			a.contentArray = append(a.contentArray, krnkmodel.D{
				"type": "text",
				"text": a.plainText.String(),
			})
			a.plainText.Reset()
		}
		a.out = append(a.out, krnkmodel.D{
			"role":    krnkmodel.RoleUser,
			"content": a.contentArray,
		})
	}
	a.flushTextOnly()
	return a.out
}

func modelContentToMessage(parts []*genai.Part) (krnkmodel.D, error) {
	var text strings.Builder
	var reasoning strings.Builder
	var toolCalls []krnkmodel.D

	for _, p := range parts {
		if p == nil {
			continue
		}
		var err error
		toolCalls, err = appendModelPart(p, &text, &reasoning, toolCalls)
		if err != nil {
			return nil, err
		}
	}

	if text.Len() == 0 && reasoning.Len() == 0 && len(toolCalls) == 0 {
		return nil, nil //nolint:nilnil // Empty assistant turn; caller skips nil result.
	}

	msg := krnkmodel.D{
		"role": krnkmodel.RoleAssistant,
	}
	if text.Len() > 0 {
		msg["content"] = text.String()
	} else {
		msg["content"] = ""
	}
	if reasoning.Len() > 0 {
		msg["reasoning_content"] = reasoning.String()
	}
	if len(toolCalls) > 0 {
		msg["tool_calls"] = toolCalls
	}
	return msg, nil
}

func appendModelPart(
	p *genai.Part,
	text, reasoning *strings.Builder,
	toolCalls []krnkmodel.D,
) ([]krnkmodel.D, error) {
	switch {
	case p.FunctionCall != nil:
		tc, err := functionCallToToolCall(p.FunctionCall)
		if err != nil {
			return toolCalls, err
		}
		return append(toolCalls, tc), nil
	case p.Thought:
		if p.Text != "" {
			appendWithNewline(reasoning, p.Text)
		}
		return toolCalls, nil
	case p.InlineData != nil || (p.FileData != nil && p.FileData.FileURI != ""):
		return toolCalls, errors.New(
			"kronk provider only supports text and tool_call parts on model-role content",
		)
	case p.Text != "":
		appendWithNewline(text, p.Text)
	}
	return toolCalls, nil
}

func appendWithNewline(b *strings.Builder, s string) {
	if b.Len() > 0 {
		b.WriteString("\n")
	}
	b.WriteString(s)
}

func functionCallToToolCall(fc *genai.FunctionCall) (krnkmodel.D, error) {
	if fc == nil {
		return nil, errors.New("nil FunctionCall")
	}
	id := fc.ID
	if id == "" {
		id = "call_" + fc.Name
	}
	argsJSON := "{}"
	if fc.Args != nil {
		b, err := json.Marshal(fc.Args)
		if err != nil {
			return nil, fmt.Errorf("marshal tool %q args: %w", fc.Name, err)
		}
		argsJSON = string(b)
	}
	return krnkmodel.D{
		"id":   id,
		"type": "function",
		"function": krnkmodel.D{
			"name":      fc.Name,
			"arguments": argsJSON,
		},
	}, nil
}

func functionResponseToToolMessage(fr *genai.FunctionResponse) (krnkmodel.D, error) {
	if fr == nil {
		return nil, errors.New("nil FunctionResponse")
	}
	id := fr.ID
	if id == "" {
		id = "call_" + fr.Name
	}

	var content string
	if len(fr.Response) > 0 {
		b, err := json.Marshal(fr.Response)
		if err != nil {
			return nil, fmt.Errorf("marshal tool %q response: %w", fr.Name, err)
		}
		content = string(b)
	}
	for _, part := range fr.Parts {
		if part == nil {
			continue
		}
		if part.InlineData != nil && len(part.InlineData.Data) > 0 {
			// Kronk tool results only carry textual content; inline binary
			// data is base64-encoded and appended so the model at least sees
			// it, but callers should prefer textual tool output.
			encoded := base64.StdEncoding.EncodeToString(part.InlineData.Data)
			if content != "" {
				content += "\n"
			}
			content += fmt.Sprintf("[inline %s base64]: %s", part.InlineData.MIMEType, encoded)
		}
	}

	return krnkmodel.D{
		"role":         krnkmodel.RoleTool,
		"name":         fr.Name,
		"tool_call_id": id,
		"content":      content,
	}, nil
}

func inlineDataContentBlock(p *genai.Part) (krnkmodel.D, error) {
	mime := normalizeMIME(p.InlineData.MIMEType)
	encoded := base64.StdEncoding.EncodeToString(p.InlineData.Data)

	switch {
	case strings.HasPrefix(mime, "image/"):
		return krnkmodel.D{
			"type": "image_url",
			"image_url": krnkmodel.D{
				"url": fmt.Sprintf("data:%s;base64,%s", mime, encoded),
			},
		}, nil
	case strings.HasPrefix(mime, "audio/"):
		return krnkmodel.D{
			"type": "input_audio",
			"input_audio": krnkmodel.D{
				"data": fmt.Sprintf("data:%s;base64,%s", mime, encoded),
			},
		}, nil
	case strings.HasPrefix(mime, "video/"):
		return krnkmodel.D{
			"type": "video_url",
			"video_url": krnkmodel.D{
				"url": fmt.Sprintf("data:%s;base64,%s", mime, encoded),
			},
		}, nil
	default:
		return nil, fmt.Errorf("kronk provider does not support inline mime type %q", p.InlineData.MIMEType)
	}
}

func applyInferenceParams(d krnkmodel.D, cfg *genai.GenerateContentConfig) {
	if cfg == nil {
		return
	}
	if cfg.Temperature != nil {
		d["temperature"] = *cfg.Temperature
	}
	if cfg.TopP != nil {
		d["top_p"] = *cfg.TopP
	}
	if cfg.TopK != nil {
		// genai TopK is a float but Kronk samplers treat it as an int count.
		d["top_k"] = int(*cfg.TopK)
	}
	if cfg.MaxOutputTokens > 0 {
		d["max_tokens"] = int(cfg.MaxOutputTokens)
	}
	if len(cfg.StopSequences) > 0 {
		d["stop"] = append([]string(nil), cfg.StopSequences...)
	}
	if cfg.Seed != nil {
		d["seed"] = int(*cfg.Seed)
	}
	if cfg.FrequencyPenalty != nil {
		d["frequency_penalty"] = *cfg.FrequencyPenalty
	}
	if cfg.PresencePenalty != nil {
		d["presence_penalty"] = *cfg.PresencePenalty
	}
}

// normalizeMIME returns the type/subtype in lowercase with parameters
// stripped. Clients sometimes send "image/png; charset=utf-8" which would
// otherwise fail prefix matches.
func normalizeMIME(mime string) string {
	mime = strings.TrimSpace(strings.ToLower(mime))
	if i := strings.IndexByte(mime, ';'); i >= 0 {
		mime = strings.TrimSpace(mime[:i])
	}
	return mime
}
