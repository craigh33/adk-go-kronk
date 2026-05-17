// Command kronk-a2a demonstrates exposing a Kronk-backed ADK agent through
// A2A, then consuming that same in-process A2A server as a remote agent.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
	krnk "github.com/ardanlabs/kronk/sdk/kronk"
	krnkmodel "github.com/ardanlabs/kronk/sdk/kronk/model"
	"github.com/ardanlabs/kronk/sdk/tools/defaults"
	"github.com/ardanlabs/kronk/sdk/tools/libs"
	"github.com/ardanlabs/kronk/sdk/tools/models"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/remoteagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/server/adka2a"
	"google.golang.org/adk/session"
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

	a2aServerAddress, closeAgent, err := startKronkAgentServer(ctx)
	if err != nil {
		return err
	}
	defer closeAgent()

	remoteAgent, err := remoteagent.NewA2A(remoteagent.A2AConfig{
		Name:            "A2A Kronk assistant",
		Description:     "A remote ADK agent served over A2A and backed by a local Kronk model.",
		AgentCardSource: a2aServerAddress,
	})
	if err != nil {
		return fmt.Errorf("create remote agent: %w", err)
	}

	launcherCfg := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(remoteAgent),
	}

	l := full.NewLauncher()
	if err := l.Execute(ctx, launcherCfg, os.Args[1:]); err != nil {
		return fmt.Errorf("run failed: %w\n\n%s", err, l.CommandLineSyntax())
	}
	return nil
}

func startKronkAgentServer(ctx context.Context) (string, func(), error) {
	a, closeAgent, err := newKronkAgent(ctx)
	if err != nil {
		return "", nil, fmt.Errorf("create Kronk agent: %w", err)
	}

	listener, err := (&net.ListenConfig{}).Listen(ctx, "tcp", "127.0.0.1:0")
	if err != nil {
		closeAgent()
		return "", nil, fmt.Errorf("bind A2A server: %w", err)
	}

	baseURL := &url.URL{Scheme: "http", Host: listener.Addr().String()}
	log.Printf("Starting A2A server on %s", baseURL.String())

	httpServer := &http.Server{
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      30 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	go func() {
		agentPath := "/invoke"
		agentCard := &a2a.AgentCard{
			Name:               a.Name(),
			Description:        a.Description(),
			DefaultInputModes:  []string{"text/plain"},
			DefaultOutputModes: []string{"text/plain"},
			Skills:             adka2a.BuildAgentSkills(a),
			PreferredTransport: a2a.TransportProtocolJSONRPC,
			URL:                baseURL.JoinPath(agentPath).String(),
			Capabilities:       a2a.AgentCapabilities{Streaming: true},
		}

		mux := http.NewServeMux()
		mux.Handle(a2asrv.WellKnownAgentCardPath, a2asrv.NewStaticAgentCardHandler(agentCard))

		executor := adka2a.NewExecutor(adka2a.ExecutorConfig{
			RunnerConfig: runner.Config{
				AppName:        a.Name(),
				Agent:          a,
				SessionService: session.InMemoryService(),
			},
		})
		requestHandler := a2asrv.NewHandler(executor)
		mux.Handle(agentPath, a2asrv.NewJSONRPCHandler(requestHandler))

		httpServer.Handler = mux

		serveErr := httpServer.Serve(listener)
		if serveErr != nil && !errors.Is(serveErr, http.ErrServerClosed) {
			log.Printf("A2A server stopped with error: %v", serveErr)
			return
		}
		log.Printf("A2A server stopped")
	}()

	closeServerAndAgent := func() {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		if cerr := listener.Close(); cerr != nil && !errors.Is(cerr, net.ErrClosed) {
			log.Printf("close A2A listener: %v", cerr)
		}
		if serr := httpServer.Shutdown(shutdownCtx); serr != nil && !errors.Is(serr, http.ErrServerClosed) {
			log.Printf("shutdown A2A server: %v", serr)
		}
		closeAgent()
	}

	return baseURL.String(), closeServerAndAgent, nil
}

func newKronkAgent(ctx context.Context) (agent.Agent, func(), error) {
	modelID := strings.TrimSpace(os.Getenv("KRONK_MODEL_ID"))
	sourceURL := strings.TrimSpace(os.Getenv("KRONK_MODEL_URL"))
	if modelID == "" && sourceURL == "" {
		modelID = defaultModelID
		log.Printf("KRONK_MODEL_ID / KRONK_MODEL_URL unset, defaulting to catalog model %q", modelID)
	}

	mp, err := installSystem(ctx, modelID, sourceURL)
	if err != nil {
		return nil, nil, fmt.Errorf("install kronk runtime: %w", err)
	}

	cfg := kronkllm.Config{
		ModelFiles: mp.ModelFiles,
	}
	if strings.TrimSpace(mp.ProjFile) != "" {
		cfg.ModelOptions = append(cfg.ModelOptions, krnkmodel.WithProjFile(mp.ProjFile))
	}

	llm, err := kronkllm.New(ctx, cfg)
	if err != nil {
		return nil, nil, fmt.Errorf("build kronk llm provider: %w", err)
	}

	closeAgent := func() {
		closeCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if cerr := llm.Close(closeCtx); cerr != nil {
			log.Printf("close kronk llm: %v", cerr)
		}
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "kronk_a2a_assistant",
		Description: "A helpful assistant running on a local Kronk model and exposed over A2A.",
		Model:       llm,
		Instruction: "You reply briefly and clearly using only the information the user provides.",
		GenerateContentConfig: &genai.GenerateContentConfig{
			MaxOutputTokens: 512,
		},
	})
	if err != nil {
		closeAgent()
		return nil, nil, fmt.Errorf("agent: %w", err)
	}
	return a, closeAgent, nil
}

// installSystem installs llama.cpp libraries, then fetches the selected GGUF
// model. Catalog resolution is handled inside models.Download.
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

	mdls, err := models.New()
	if err != nil {
		return models.Path{}, err
	}

	switch {
	case sourceURL != "":
		//nolint:gosec // G706: sourceURL comes from a developer-set env var; surfacing it in logs is intentional.
		log.Printf("downloading model from URL: %q", sourceURL)
		return mdls.Download(ctx, krnk.FmtLogger, sourceURL)
	default:
		//nolint:gosec // G706: modelID comes from a developer-set env var; surfacing it in logs is intentional.
		log.Printf("downloading model from catalog: %q", modelID)
		return mdls.Download(ctx, krnk.FmtLogger, modelID)
	}
}
