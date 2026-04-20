<p align="center">
  <img
    src="docs/images/readme-header.jpg"
    alt="adk-go-kronk banner showing Agent Development Kit connected to Kronk"
    width="100%"
  />
</p>

# adk-go-kronk

[kronk](https://github.com/ardanlabs/kronk) implementation of the [`model.LLM`](https://pkg.go.dev/google.golang.org/adk/model#LLM) interface for [adk-go](https://github.com/google/adk-go), so you can run agents on any kronk-supported model with the same ADK APIs you use for Gemini.

## Requirements

- **Go** 1.25+ (aligned with `google.golang.org/adk`)
- **[golangci-lint](https://golangci-lint.run/welcome/install/)** if you run `make lint` (uses [.golangci.yaml](.golangci.yaml))

## Install

```bash
go get github.com/craigh33/adk-go-kronk
```

Replace the module path with your fork or published path if you rename the module in `go.mod`.

## Makefile

| Target | Description |
|--------|-------------|
| `make test` | Run unit tests |
| `make build` | Compile all packages |
| `make lint` | Run `golangci-lint run ./...` |
| `make pre-commit-install` | Install pre-commit hooks |

## Contributing / Development

### Pre-commit hooks

This project uses [pre-commit](https://pre-commit.com) to enforce code quality and commit hygiene. The following tools must be available on your `PATH` before installing the hooks:

| Tool | Purpose | Install |
|------|---------|---------|
| [pre-commit](https://pre-commit.com) | Hook framework | `brew install pre-commit` |
| [golangci-lint](https://golangci-lint.run/welcome/install/) | Go linter (runs `make lint`) | `brew install golangci-lint` |
| [gitleaks](https://github.com/gitleaks/gitleaks) | Secret / credential scanner | `brew install gitleaks` |

Once the tools are installed, wire the hooks into your local clone:

```bash
make pre-commit-install
```

This installs hooks for both the `pre-commit` stage and the `commit-msg` stage.

#### What the hooks do

| Hook | Stage | Description |
|------|-------|-------------|
| `trailing-whitespace` | pre-commit | Strips trailing whitespace |
| `end-of-file-fixer` | pre-commit | Ensures files end with a newline |
| `check-yaml` | pre-commit | Validates YAML syntax |
| `no-commit-to-branch` | pre-commit | Prevents direct commits to `main` |
| `conventional-pre-commit` | commit-msg | Enforces [Conventional Commits](https://www.conventionalcommits.org/) message format (`feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`) |
| `golangci-lint` | pre-commit | Runs `make lint` against all Go files |
| `gitleaks` | pre-commit | Scans staged diff for secrets/credentials |

## Usage

TODO: add usage example here, for now see the individual examples in the `examples/` directory.

## Examples

TODO: add example descriptions here, for now see the individual examples in the `examples/` directory.

## Limitations

TODO: add any known limitations here, for example around streaming, error handling, or supported Kronk features.

## License

Apache 2.0 — see [LICENSE](LICENSE).
