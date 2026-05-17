# kronk-a2a example

This example is based on the ADK A2A example and the Bedrock A2A provider
example. It runs an in-process A2A server backed by a locally loaded Kronk
model, then connects to it as a remote agent.

## Prerequisites

- Go 1.26+
- Enough disk space for the llama.cpp libraries, Kronk catalog, and the
  selected model file. The first run downloads them into the Kronk default
  install directories; subsequent runs reuse the cached artifacts.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KRONK_MODEL_ID` | `Qwen3-0.6B-Q8_0` | Model ID to pull from the Kronk catalog. Ignored when `KRONK_MODEL_URL` is set. |
| `KRONK_MODEL_URL` | _(unset)_ | Direct URL of a `.gguf` model to download. Overrides `KRONK_MODEL_ID`. |

See the [Kronk README](https://github.com/ardanlabs/kronk#readme) for the
current list of catalog model IDs and GPU / CPU platform support.

## Run

```bash
make -C examples/kronk-a2a run
```

The launcher accepts ADK launcher commands and flags. For example:

```bash
make -C examples/kronk-a2a run ARGS='web api webui'
```
