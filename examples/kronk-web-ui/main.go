// Command kronk-web-ui demonstrates wiring the github.com/craigh33/adk-go-kronk
// Kronk provider into the ADK full launcher so the agent can be driven through
// the ADK web UI and REST API against a locally loaded GGUF model.
//
// The first time the program runs it downloads the llama.cpp libraries, the
// Kronk model catalog, and the selected GGUF model into the default Kronk
// install directories (see github.com/ardanlabs/kronk/sdk/tools for paths).
// Subsequent runs reuse the cached artifacts.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	krnk "github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/tools/catalog"
	"github.com/ardanlabs/kronk/sdk/tools/defaults"
	"github.com/ardanlabs/kronk/sdk/tools/libs"
	"github.com/ardanlabs/kronk/sdk/tools/models"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/genai"

	kronkllm "github.com/craigh33/adk-go-kronk/kronk"
)

const (
	defaultModelID      = "Qwen3-0.6B-Q8_0"
	installPhaseTimeout = 25 * time.Minute
)

func main() {
	if err := run(); err != nil {
		log.Fatalf("%v", err)
	}
}

func run() error {
	ctx := context.Background()

	modelID := strings.TrimSpace(os.Getenv("KRONK_MODEL_ID"))
	sourceURL := strings.TrimSpace(os.Getenv("KRONK_MODEL_URL"))
	if modelID == "" && sourceURL == "" {
		modelID = defaultModelID
		log.Printf("KRONK_MODEL_ID / KRONK_MODEL_URL unset, defaulting to catalog model %q", modelID)
	}

	mp, err := installSystem(ctx, modelID, sourceURL)
	if err != nil {
		return fmt.Errorf("install kronk runtime: %w", err)
	}

	llm, err := kronkllm.New(ctx, kronkllm.Config{
		ModelFiles: mp.ModelFiles,
	})
	if err != nil {
		return fmt.Errorf("build kronk llm provider: %w", err)
	}
	defer func() {
		closeCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if cerr := llm.Close(closeCtx); cerr != nil {
			log.Printf("close kronk llm: %v", cerr)
		}
	}()

	a, err := llmagent.New(llmagent.Config{
		Name:        "assistant",
		Description: "A helpful assistant running on a local Kronk model",
		Model:       llm,
		Instruction: "You reply briefly and clearly using only the information the user provides.",
		GenerateContentConfig: &genai.GenerateContentConfig{
			MaxOutputTokens: 1024,
		},
	})
	if err != nil {
		return fmt.Errorf("agent: %w", err)
	}

	launcherCfg := &launcher.Config{
		AgentLoader:     agent.NewSingleLoader(a),
		ArtifactService: artifact.InMemoryService(),
	}

	l := full.NewLauncher()
	if err := l.Execute(ctx, launcherCfg, os.Args[1:]); err != nil {
		return fmt.Errorf("run failed: %w\n\n%s", err, l.CommandLineSyntax())
	}
	return nil
}

// installSystem mirrors the kronk chat example: install llama.cpp libraries,
// pull down the catalog, then fetch the selected GGUF model. It returns the
// local models.Path the kronk provider will load from.
func installSystem(ctx context.Context, modelID, sourceURL string) (models.Path, error) {
	ctx, cancel := context.WithTimeout(ctx, installPhaseTimeout)
	defer cancel()

	lib, err := libs.New(libs.WithVersion(defaults.LibVersion("")))
	if err != nil {
		return models.Path{}, err
	}
	if _, err := lib.Download(ctx, krnk.FmtLogger); err != nil {
		return models.Path{}, err
	}

	ctlg, err := catalog.New()
	if err != nil {
		return models.Path{}, err
	}
	if err := ctlg.Download(ctx); err != nil {
		return models.Path{}, err
	}

	mdls, err := models.New()
	if err != nil {
		return models.Path{}, err
	}

	switch {
	case sourceURL != "":
		//nolint:gosec // G706: sourceURL comes from a developer-set env var; surfacing it in logs is intentional.
		log.Printf("downloading model from URL: %q", sourceURL)
		return mdls.Download(ctx, krnk.FmtLogger, sourceURL, "")
	default:
		//nolint:gosec // G706: modelID comes from a developer-set env var; surfacing it in logs is intentional.
		log.Printf("downloading model from catalog: %q", modelID)
		return ctlg.DownloadModel(ctx, krnk.FmtLogger, modelID)
	}
}
