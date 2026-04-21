# kronk-web-ui example

This example launches the ADK local web UI and REST API backed by the
[Kronk](https://github.com/ardanlabs/kronk) provider, serving a GGUF model
loaded locally on your machine.

## Prerequisites

- Go 1.25+
- Enough disk space for the llama.cpp libraries, Kronk catalog, and the
  selected model file (the first run downloads them into the Kronk default
  install directories; subsequent runs reuse the cached artifacts).

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KRONK_MODEL_ID` | `Qwen3-0.6B-Q8_0` | Model ID to pull from the Kronk catalog. Ignored when `KRONK_MODEL_URL` is set. |
| `KRONK_MODEL_URL` | _(unset)_ | Direct URL of a `.gguf` model to download. Overrides `KRONK_MODEL_ID`. |

See the [Kronk README](https://github.com/ardanlabs/kronk#readme) for the
current list of catalog model IDs and GPU / CPU platform support.

## Run

```bash
make -C examples/kronk-web-ui run
```

Or directly:

```bash
cd examples/kronk-web-ui
go run . web api webui
```

The first invocation may take several minutes while it downloads the
llama.cpp libraries and the selected model. Open the URL printed by the
launcher (typically `http://localhost:8000`) to chat with the agent through
the ADK web UI.
