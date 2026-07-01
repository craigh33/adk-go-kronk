<p align="center">
  <img
    src="docs/images/readme-header.jpg"
    alt="adk-go-kronk banner showing Agent Development Kit connected to Kronk"
    width="100%"
  />
</p>

# adk-go-kronk

[kronk](https://github.com/ardanlabs/kronk) implementation of the [`model.LLM`](https://pkg.go.dev/google.golang.org/adk/v2/model#LLM) interface for [adk-go](https://github.com/google/adk-go), so you can run agents on any Kronk-supported local GGUF model with the same ADK APIs you use for Gemini.

**Other providers:** [adk-go-bedrock](https://github.com/craigh33/adk-go-bedrock) · [adk-go-ollama](https://github.com/craigh33/adk-go-ollama)

## Requirements

- **Go** 1.26+ (aligned with `google.golang.org/adk/v2`)
- Enough disk space and (optionally) GPU for the selected GGUF model. The first run of any Kronk-backed program downloads the llama.cpp libraries, the Kronk model catalog, and the chosen model into Kronk's default install directories; subsequent runs reuse the cached artifacts. See the [Kronk README](https://github.com/ardanlabs/kronk#readme) for platform / GPU support.

## Install

```bash
go get github.com/craigh33/adk-go-kronk
```

Replace the module path with your fork or published path if you rename the module in `go.mod`.

## Usage

```go
ctx := context.Background()

cfg := kronk.Config{ModelFiles: []string{"/path/to/model.gguf"}}
// For vision / audio / video models, add krnkmodel.WithProjFile("/path/to/mmproj") via cfg.ModelOptions.

llm, err := kronk.New(ctx, cfg)
if err != nil {
    log.Fatal(err)
}
defer llm.Close(ctx)

a, err := llmagent.New(llmagent.Config{
    Name:        "assistant",
    Description: "A helpful assistant running on a local Kronk model",
    Model:       llm,
    Instruction: "You reply briefly and clearly.",
})
if err != nil {
    log.Fatal(err)
}

// Wire `a` into runner.New(...), a launcher, or call
// llm.GenerateContent(ctx, req, stream) directly.
```

The provider implements [`model.LLM`](https://pkg.go.dev/google.golang.org/adk/v2/model#LLM) on top of a loaded [`*kronk.Kronk`](https://pkg.go.dev/github.com/ardanlabs/kronk/sdk/kronk#Kronk) engine. The convenience constructor `kronk.New` owns the engine lifecycle; use `kronk.NewWithKronk` if you already have an engine you want to reuse.

The [`internal/mappers`](internal/mappers/) package holds genai ↔ Kronk conversions (requests, responses, tools, usage); the public provider surface is [`kronk`](kronk/).

### Model files

`Config.ModelFiles` is the slice of GGUF paths the Kronk engine will load. The easiest way to obtain them is via Kronk's own catalog / models helpers, for example:

```go
mdls, _ := models.New()
mp, _ := mdls.Download(ctx, krnk.FmtLogger, "Qwen3-0.6B-Q8_0")
cfg := kronk.Config{ModelFiles: mp.ModelFiles}
if mp.ProjFile != "" {
    cfg.ModelOptions = []krnkmodel.Option{krnkmodel.WithProjFile(mp.ProjFile)}
}
llm, _ := kronk.New(ctx, cfg)
```

Pass **`krnkmodel.WithProjFile(mp.ProjFile)`** in **`Config.ModelOptions`** when the catalog download includes an mmproj (required for Kronk’s image / audio / video pipeline).

See [`examples/kronk-web-ui`](examples/kronk-web-ui) for a complete runnable example including library and model installation (it passes **`ProjFile`** when present).

## Examples

Each example has its own `README.md` and `Makefile`:

- [`examples/kronk-web-ui`](examples/kronk-web-ui): ADK local web UI + REST API launcher backed by a Kronk-loaded GGUF model. Controlled via `KRONK_MODEL_ID` (catalog model ID, default `Qwen3-0.6B-Q8_0`) or `KRONK_MODEL_URL` (direct GGUF URL). First run downloads the llama.cpp libraries, the Kronk catalog, and the selected model; subsequent runs reuse the cache. Passes through **`ProjFile`** from the catalog download when present so multimodal models receive the mmproj.

- [`examples/kronk-a2a`](examples/kronk-a2a): in-process A2A server backed by a Kronk-loaded GGUF model, then consumed as a remote ADK agent. Uses the same `KRONK_MODEL_ID` / `KRONK_MODEL_URL` setup as the web UI example.

- [`examples/kronk-files`](examples/kronk-files): CLI that runs **`RequestFromLLMRequest`** only (`-dry-run`) to inspect how file bytes map to Kronk message blocks—no model load. Useful for debugging uploads and MIME sniffing (`http.DetectContentType` by default).

```bash
export KRONK_MODEL_ID=Qwen3-0.6B-Q8_0
make -C examples/kronk-web-ui run
```

```bash
make -C examples/kronk-a2a run ARGS='web api webui'
```

```bash
make -C examples/kronk-files run ARGS='-dry-run -path ./photo.jpg'
```

## How it maps to Kronk

- **Messages**: `genai` roles `user` and `model` map to Kronk `user` and `assistant`. `FunctionResponse` parts are emitted as standalone `role:"tool"` messages with `tool_call_id` so Kronk can thread them back to the originating tool call.
- **System instruction**: `GenerateContentConfig.SystemInstruction` is prepended as a `role:"system"` message.
- **Inference params**: `Temperature`, `TopP`, `TopK`, `MaxOutputTokens`, `StopSequences`, `Seed`, `FrequencyPenalty`, and `PresencePenalty` are passed through to Kronk.
- **Tools**: only `genai.Tool.FunctionDeclarations` are mapped (as OpenAI-shaped function tool entries with lowercased JSON Schema types). Non-function variants (Google Search, Code Execution, Retrieval, MCP servers, Computer Use, File Search, Google Maps, URL Context, etc.) are rejected early with a clear provider error. Use ADK's [`mcptoolset`](https://pkg.go.dev/google.golang.org/adk/v2/tool/mcptoolset) to bring MCP tools in as function declarations.
- **Multimodal input**: inline `image/*`, `audio/*`, and `video/*` bytes on user turns are sent as OpenAI-style `image_url` / `input_audio` / `video_url` data URLs (set `Blob.MIMEType` accordingly—callers often use `http.DetectContentType` or the browser-reported type). **Prompt text before attachment** in the same `genai.Part` maps to **text blocks before** media, matching Kronk’s `ImageMessage` / `AudioMessage` layout.
- **Other inline bytes**: if the MIME type is not `image/` / `audio/` / `video/`, valid **UTF-8** payloads become a single `text` part with a short header; otherwise bytes are embedded as **base64 inside `text`** (default max **4 MiB** raw bytes before encoding). No filename-based MIME guessing inside the provider—set `MIMEType` at the edge.
- **Multimodal engine setup**: Kronk requires a **projection (mmproj) file** for real vision/audio/video inference. Pass **`krnkmodel.WithProjFile`** via **`kronk.Config.ModelOptions`** when your model download provides `ProjFile` (see [`examples/kronk-web-ui`](examples/kronk-web-ui)).
- **Streaming**: when streaming is enabled the provider calls `ChatStreaming`, emits text deltas as `Partial:true` responses, and buffers tool calls, reasoning, usage, and finish reason into the final `TurnComplete:true` response.
- **Usage**: Kronk `Usage` maps to ADK `GenerateContentResponseUsageMetadata` (`PromptTokenCount`, `CandidatesTokenCount`, `TotalTokenCount`).

## Limitations

- **No native safety / guardrails**: ADK `SafetySettings` and `ModelArmorConfig` are not supported; Kronk runs local models with no built-in guardrail layer. Wrap the provider with your own policy layer if you need one.
- **Non-function tool variants unsupported**: Only function declarations are passed through to the model. All other `genai.Tool` variants return a request-time error with a clear message naming the unsupported variants.
- **Model-role media unsupported**: Only text, reasoning, and tool-call parts are permitted on assistant (`model`-role) turns. Inline or remote media on assistant turns will produce an error.
- **Remote `FileData` unsupported**: Kronk loads local models and does not fetch arbitrary remote URIs; callers must inline bytes via `InlineData`.
- **Default request timeout**: Kronk's `Chat` / `ChatStreaming` APIs require a context deadline. When the caller does not provide one the provider attaches a 2-minute default — override it with `context.WithTimeout` for long-running prompts.
- **No embeddings / rerank surface**: ADK `model.LLM` is chat-only; call the underlying `*kronk.Kronk` directly (via `Model.Engine()`) for embedding or rerank features if you need them.
- **Large binary attachments**: non-media inline payloads embedded as base64 text are rejected above **4 MiB** raw bytes by default.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for Makefile targets, required pre-commit setup, commit message conventions, and pull request guidelines. For new issues, use the [bug report](https://github.com/craigh33/adk-go-kronk/issues/new?template=bug_report.yml) or [feature request](https://github.com/craigh33/adk-go-kronk/issues/new?template=feature_request.yml) templates.

## License

Apache 2.0 — see [LICENSE](LICENSE).

[Contributing](CONTRIBUTING.md) · [Issues](https://github.com/craigh33/adk-go-kronk/issues) · [Security](https://github.com/craigh33/adk-go-kronk/security)
