# online_decoder — Claude Guidelines

`online_decoder/` is the **standalone app root**. It must remain self-contained and portable — no imports from the parent monorepo.

## Architecture Reference

See [docs/backend_plan.md](docs/backend_plan.md) for the authoritative component architecture (Phase 1 offline training, Phase 2 live inference). Always consult it before adding new backend components.

## Directory Layout

```
online_decoder/
├── scripts/            — Root-local helper entrypoints for dev/integration tasks
├── src/backend/
│   ├── core/           — SettingsManager, Pydantic config models
│   ├── offline_phase/  — OfflinePreprocessor, ModelEvaluator, ModelTrainer
│   └── online_phase/   — LSLReceiver, RingBuffer, OnlinePreprocessor,
│                          LiveInferenceEngine, StreamWorker
├── tests/              — pytest suite, completely separate from src/
├── tools/lslproxy/     — LSLProxy.exe and Windows DLLs (hardware interface)
└── docs/               — architecture plans and documentation
```

## Dependency Management

- `requirements.txt` — runtime deps only (install for production)
- `requirements-dev.txt` — `-r requirements.txt` + test/dev tools
- `src/` code never imports test libraries (pytest, etc.)

## Running Tests

```bash
# From online_decoder/ root
pytest tests/
pytest tests/ -v --cov=src   # with coverage
python scripts/characterize_lsl.py --duration 10
```

## Config Schema

The experiment config lives in `experiment_config.yaml`. Schema is defined in `src/backend/core/config_models.py` (Pydantic v2). When the YAML schema changes, update the Pydantic models — that is the single source of truth for validation.

## When to Update This File

Update CLAUDE.md when:
- A new backend component directory is added under `src/backend/`
- A new top-level workflow directory is added (for example `scripts/`)
- A new project-wide convention is established (error handling, logging, naming, etc.)
- The config schema structure changes significantly
- New tooling is added that affects the development workflow

Do NOT update for every new file — only for structural or convention changes that affect how an AI agent reasons about the project.
