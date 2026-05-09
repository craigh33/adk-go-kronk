// Command kronk-files maps a local file to a Kronk chat request (ADK → genai
// → mappers) for debugging or dry runs without loading a GGUF.
//
//	go run . -dry-run -path ./sample.png
//	go run . -dry-run -combined -path ./notes.txt
//
// Set DOCUMENT_PATH if -path is omitted. Use -mime to override [http.DetectContentType].
package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	krnkmodel "github.com/ardanlabs/kronk/sdk/kronk/model"
	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-kronk/internal/mappers"
)

func main() {
	dryRun := flag.Bool("dry-run", false, "only run request mapping (no Kronk / model load)")
	combined := flag.Bool("combined", false, "put prompt and file bytes in the same Part (Web UI style)")
	prompt := flag.String(
		"prompt",
		"Briefly describe what this attachment is in one or two sentences.",
		"user text sent with the file",
	)
	pathFlag := flag.String("path", "", "path to a file, or set DOCUMENT_PATH")
	mimeFlag := flag.String("mime", "", "override MIME type (default: http.DetectContentType on file bytes)")
	flag.Parse()

	p := strings.TrimSpace(*pathFlag)
	if p == "" {
		p = strings.TrimSpace(os.Getenv("DOCUMENT_PATH"))
	}
	if p == "" {
		log.Fatal("usage: go run . -path /path/to/file (or set DOCUMENT_PATH)")
	}
	p = filepath.Clean(p)
	data, err := os.ReadFile(p) // #nosec G304 -- path from -path / env (operator-chosen)
	if err != nil {
		log.Fatalf("read file: %v", err)
	}

	mime := strings.TrimSpace(*mimeFlag)
	if mime == "" {
		mime = http.DetectContentType(data)
	}
	base := filepath.Base(p)

	req := buildLLMRequest(*prompt, data, mime, base, *combined)

	fmt.Printf(
		"File debug\n  path: %s\n  size: %d bytes\n  mime: %s\n  combined: %v\n\n",
		p,
		len(data),
		mime,
		*combined,
	)

	if !*dryRun {
		log.Fatal("only -dry-run is implemented here; use examples/kronk-web-ui for a full model + agent run")
	}

	if err := runDryRun(req); err != nil {
		log.Fatalf("mapping failed: %v", err)
	}
	fmt.Println("\nNext: run examples/kronk-web-ui with a vision/audio model and KRONK_MODEL_ID to invoke the model.")
}

func buildLLMRequest(prompt string, data []byte, mime, displayName string, combined bool) *model.LLMRequest {
	var parts []*genai.Part
	if combined {
		parts = []*genai.Part{{
			Text: prompt,
			InlineData: &genai.Blob{
				Data:        data,
				MIMEType:    mime,
				DisplayName: displayName,
			},
		}}
	} else {
		parts = []*genai.Part{
			genai.NewPartFromText(prompt),
			{
				InlineData: &genai.Blob{
					Data:        data,
					MIMEType:    mime,
					DisplayName: displayName,
				},
			},
		}
	}
	return &model.LLMRequest{
		Contents: []*genai.Content{{
			Role:  genai.RoleUser,
			Parts: parts,
		}},
		Config: &genai.GenerateContentConfig{
			MaxOutputTokens: 1024,
		},
	}
}

func runDryRun(req *model.LLMRequest) error {
	d, err := mappers.RequestFromLLMRequest(req, false)
	if err != nil {
		return err
	}
	fmt.Println("dry-run: mapping OK — Kronk chat document summary:")
	msgs, ok := d["messages"].([]krnkmodel.D)
	if !ok {
		return fmt.Errorf("unexpected messages type %T", d["messages"])
	}
	for i, msg := range msgs {
		role, _ := msg["role"].(string)
		fmt.Printf("  message[%d] role=%s ", i, role)
		switch c := msg["content"].(type) {
		case string:
			fmt.Printf("content=text (%d runes)\n", len([]rune(c)))
		case []krnkmodel.D:
			fmt.Printf("content parts=%d\n", len(c))
			for j, part := range c {
				typ, _ := part["type"].(string)
				fmt.Printf("    part[%d] type=%s\n", j, summarizePartType(typ, part))
			}
		default:
			fmt.Printf("content=%T\n", c)
		}
	}
	return nil
}

func summarizePartType(typ string, part krnkmodel.D) string {
	switch typ {
	case "text":
		t, _ := part["text"].(string)
		const previewRunes = 80
		runes := []rune(t)
		if len(runes) > previewRunes {
			preview := string(runes[:previewRunes])
			return fmt.Sprintf("text len=%d preview=%q…", len(t), preview)
		}
		return fmt.Sprintf("text %q", t)
	case "image_url":
		return "image_url (data URL)"
	case "input_audio":
		return "input_audio (data URL)"
	case "video_url":
		return "video_url (data URL)"
	default:
		return typ
	}
}
