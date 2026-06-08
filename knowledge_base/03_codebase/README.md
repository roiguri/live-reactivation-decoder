# Codebase Structure

Back to the [Project Index](../START_HERE.md).

## Main Idea

This repository is the online stage of the reactivation decoder — a standalone PyQt6 app for real-time decoding.

The offline (semester-A) pipeline that informed this app's design lives in the parent [`reactivation-decoder`](https://github.com/roiguri/reactivation-decoder) repo. See [Offline Architecture](offline_architecture.md) for context.

## Stage Documentation

- [Offline Architecture](offline_architecture.md): Historical context for the parent repo's `src/` (semester-A evaluation pipeline)
- [Online Architecture](online_architecture.md): Current backend surface and planned live stack
- [Frontend Layout](../../docs/architecture/frontend_layout.md): PyQt6 UI layout reference — widget hierarchy, signal flow, shared widgets, styling
- [Implementation Docs](../../docs/README.md): Current backend contract and Phase 2 implementation plan

## Source Layout

- `src/backend/core/` — config models and `SettingsManager`
- `src/backend/offline_phase/` — Phase 1 training pipeline (preprocessing, model training, evaluation)
- `src/backend/online_phase/` — Phase 2 live stack (LSL receiver, online preprocessing, real-time inference)
- `src/frontend/` — PyQt6 UI
- `docs/` — implementation contracts and plans

The terms "offline" and "online" inside this repo refer to the **app's two phases** — Phase 1 trains on a recorded session, Phase 2 streams live data. Neither is the parent repo's semester-A `src/`.
