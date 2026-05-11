# kronk-files — file → Kronk request mapping

Small CLI to exercise **inline file** mapping (`genai.Part` → Kronk chat document) **without** loading a GGUF. Use this when debugging ADK/Web UI uploads or verifying MIME handling.

MIME type for `-path` defaults to [`http.DetectContentType`](https://pkg.go.dev/net/http#DetectContentType) on the file bytes. Override with `-mime` when sniffing is wrong.

## Requirements

Same repo module as `adk-go-kronk`; **no** AWS or local model required for `-dry-run`.

## Commands

**Map-only (no GGUF):**

```bash
go run . -dry-run -path /path/to/image.png
```

**Combined prompt + file** (same `Part`, like some Web UIs):

```bash
go run . -dry-run -combined -path ./memo.txt
```

**Custom MIME:**

```bash
go run . -dry-run -path ./blob.bin -mime application/pdf
```

Environment variable **`DOCUMENT_PATH`** is used if `-path` is omitted.

## Full inference

This example only runs **`RequestFromLLMRequest`** and prints a summary. To call a real model, use [`examples/kronk-web-ui`](../kronk-web-ui) (or wire `kronk.New` yourself) with a **multimodal** catalog model and ensure **`WithProjFile`** is set when the download provides an mmproj — see the root [README](../../README.md).

## Makefile

```bash
make -C examples/kronk-files run ARGS='-dry-run -path ./file.png'
```
