# Feature Plan — Per-Decoder Timepoint Selection (Phase 1 offline UI, Goal 19)

Branch: `feat/per-decoder-timepoints`

## Execution workflow (how we proceed)

- **Feature branch.** All work lands on `feat/per-decoder-timepoints` off `main`.
- **Plan committed to the repo.** This file is the versioned plan of record.
- **Approval gate after every stage.** Work stops after each stage below and waits for explicit
  approval before starting the next. No batching across stages.
- **Backend changes get extra care.** Before touching any `src/backend/` file (Stages 1 and 5),
  the exact change, rationale, blast radius, and verification are explained and approved first.

## Context

Each decoder is trained at a **single** timepoint, and the backend **already supports a
different timepoint per decoder**: `ModelTrainer.run_training()` accepts `dict[str, float]`,
the trained feature stays `n_channels` wide, and `DecoderPipelineMetadata.decoding_timepoints`
carries the per-task values. **There is no online impact** — `LiveInferenceEngine` runs every
decoder on the same sliding window regardless.

The gap is entirely in **Phase 1 (offline) UI**. Today the Evaluation screen (Node 4) already
renders per-decoder tabs (AUC curve + TGM heatmap + a SELECTED TIMEPOINT spinbox each), **but
every control mirrors one global `_selected_timepoint`** — picking 220 ms anywhere moves all
decoders. One global "Confirm Timepoint" button gates "Approve & Continue"; `evaluation_complete`
emits a single `float`, which flows to `TrainView.set_timepoint(float)` →
`orchestrator.run_training(float)`. The orchestrator then *auto-derives* per-task peaks via
`_derive_per_task_timepoints()` (argmax of each task's `diagonal_auc`, falling back to the shared
float) — so the operator can never actually choose a different timepoint per decoder.

This plan makes per-decoder timepoints **operator-selectable**, with the **Summary tab as the
control center** (per-decoder editable timepoint + per-decoder confirm), the individual decoder
tabs staying **editable and synced** to their own row, then threads a real per-decoder
`dict[str, float]` through training, the debug seed, and finally a cleanup stage that removes the
single-timepoint legacy paths.

### Confirmed decisions
- **Summary tab = control center**: each decoder has its own editable timepoint + its own Confirm.
- **Per-decoder tabs stay editable & synced**: a tab's spinbox / chart-click sets only that
  decoder; edits sync bidirectionally with the Summary row. Confirm lives only on Summary.
- **"Approve & Continue" enables only when every decoder is confirmed.**
- **Full cleanup of legacy single-timepoint paths happens as a separate final stage**, after the
  feature works end-to-end (so development never breaks the existing float path).

---

## Stage 1 — Backend: per-decoder suggestion source + dict training path

**Goal:** a single source of truth for each decoder's *suggested* peak, and let the orchestrator
accept an explicit per-decoder dict. The legacy float path stays intact this stage.

1. **`src/backend/offline_phase/evaluator.py`** — in `run_evaluation()`, add a per-task
   `peak_timepoint` (seconds) to each `tasks[name]` entry: `float(times[argmax(diagonal_auc)])`.
   This is already computed ad-hoc in three places (UI `_populate_decoder_peak_stats`,
   `_rebuild_summary_table`, and orchestrator `_derive_per_task_timepoints`); exposing it once
   makes it canonical. Keep the existing `suggested_timepoint` (cross-task avg) field for now.

2. **`src/backend/offline_phase/orchestrator.py`** — widen `run_training` to
   `timepoints: float | dict[str, float]`:
   - `dict` → use directly as the per-task map (no derivation); set the metadata's representative
     `decoding_timepoint = mean(values)`.
   - `float` → existing behavior (calls `_derive_per_task_timepoints`), unchanged.
   - Pass the resolved per-task dict to `ModelTrainer.run_training` (which already accepts dicts);
     store it in `metadata.decoding_timepoints`. `ModelTrainer` already raises `ValueError` for
     out-of-bounds timepoints (`_extract_features`) — surfaced to the UI via the worker error path.

3. **Tests** — `tests/offline_phase/test_evaluator.py`: assert `peak_timepoint` present and equals
   the argmax time. `tests/offline_phase/test_orchestrator.py`: add a dict-path case asserting
   `decoding_timepoints` reflects the passed dict and `decoding_timepoint == mean`.

---

## Stage 2 — EvaluationView: per-decoder selection + Summary control center

**The bulk of the work**, all in **`src/frontend/views/evaluation_view.py`**. Convert the single
global timepoint model to per-decoder.

**State (replace globals with per-task maps):**
- `_selected_timepoint: float` → `_selected_timepoints: dict[str, float]` (task → seconds).
- `_confirmed_timepoint: float|None` → `_confirmed: dict[str, bool]` (task → confirmed), cleared
  on every fresh eval and whenever that decoder's selection changes.
- Add `_suggested_timepoints: dict[str, float]` populated from `tasks[name]["peak_timepoint"]`
  (Stage 1); used to pre-fill each spinbox/marker and to drive the per-decoder amber deviation
  hint (reuse `_DEVIATION_WARN_MS`, now compared per-decoder against its own peak).

**Selection plumbing (de-globalize):**
- Replace `_set_selected_timepoint(t)` with `_set_decoder_timepoint(task_name, t)` — snaps to the
  nearest sample in `times`, writes only `_selected_timepoints[task_name]`, unconfirms that
  decoder, updates only that decoder's spinbox(es) (Summary row + decoder tab — both kept in
  sync), AUC value, chart marker, amber state, and confirm control.
- Per-decoder AUC/TGM chart `timepoint_clicked` → bind with the captured `task_name` so a click in
  the Red tab sets Red only (currently both connect to the global setter at
  `evaluation_view.py:717,738`).
- The **Summary overlay AUC chart** is inspection-only (a click on an all-decoder overlay is
  ambiguous): drop its `timepoint_clicked → set` wiring (`evaluation_view.py:453`). Optionally
  show each decoder's selected timepoint as a thin marker in its curve color (nice-to-have via a
  small `AUCChart` extension; not required).
- `_update_per_decoder_aucs` becomes per-decoder: each decoder's "AUC @ selected" reads at its own
  timepoint (not one shared idx). The "Avg" row = mean of each decoder's AUC@its-own-timepoint.

**Summary tab roster (control center):** replace the right-side stats panel's *global*
SELECTED TIMEPOINT spinbox + single Confirm + single Reset (`_build_stats_panel`,
`evaluation_view.py:457`) with a **per-decoder roster**. One row per decoder:
`[color dot + name] [timepoint spinbox (ms)] [AUC @ t] [Confirm ✓]`, plus a footer
`Reset all to suggested` and an overall `N / M confirmed` status. The bottom summary table
(`_build_summary_table`) stays as an at-a-glance read and keeps row-click → jump-to-tab; its AUC
column now reads each decoder's AUC at its own timepoint.

**Per-decoder tab (`_build_decoder_stats_card`, `evaluation_view.py:750`):** its spinbox now edits
only its own decoder (`_on_decoder_input_committed(name)` → `_set_decoder_timepoint(name, …)`),
synced to the Summary row. No confirm here (confirm lives on Summary).

**Confirm + gating:**
- Per-decoder Confirm toggles `_confirmed[name]`; selecting a new timepoint for a decoder clears
  its confirm (mirror existing single-confirm unconfirm-on-change logic).
- `_update_ready_state`: `page1_ready = self._done and all(self._confirmed.get(n) for n in tasks)`.
- `evaluation_complete = Signal(dict)` (was `Signal(float)`); `trigger_confirm` emits
  `dict(self._selected_timepoints)`.

---

## Stage 3 — Downstream wiring: Phase1Screen → TrainView → worker → debug walkthrough

1. **`src/frontend/screens/phase1_screen.py`** — `_on_evaluation_confirmed(timepoint: float)` →
   `(timepoints: dict)`; rename stash `self._selected_timepoint` → `self._selected_timepoints`;
   call `self._train_view.set_timepoints(timepoints)`.

2. **`src/frontend/views/train_view.py`** — `set_timepoint(float)` → `set_timepoints(dict)`;
   `_timepoint` → `_timepoints`; `trigger_run` builds `TrainingWorker(offline, self._timepoints)`;
   the "Trained at:" header (`_trained_at_lbl`, `train_view.py:165`) shows per-decoder values
   (e.g. `Red 220 · Green 180 · Yellow 240 ms`) read from `_timepoints`.

3. **`src/frontend/workers/training_worker.py`** — accept `timepoints: dict[str, float]`; `run()`
   calls `orchestrator.run_training(self._timepoints)`.

4. **`src/frontend/debug/phase1_screen_debug.py`** —
   - Eval-skip step (emits `evaluation_complete(timepoint)`): build a per-decoder dict from the
     loaded eval snapshot's `tasks[name]["peak_timepoint"]` and emit that dict.
   - Train-skip step (`set_timepoint(float(spec.metadata.decoding_timepoint))`,
     `phase1_screen_debug.py:351`): use `spec.metadata.decoding_timepoints` (the dict); fall back
     to `{name: decoding_timepoint}` for legacy snapshots with an empty dict.

5. **Tests** — `tests/test_phase1_*` / any EvaluationView headless tests: update for the
   `Signal(dict)` payload and per-decoder confirm gating (all-confirmed → ready). Add a focused
   test: confirming a subset leaves `ready=False`; confirming all → `evaluation_complete` carries
   the per-decoder dict.

---

## Stage 4 — Debug seed uses per-decoder suggested timepoints

**`scripts/demo_seed_debug_snapshots.py:128-130`** — replace the single
`suggested_t = float(eval_result["suggested_timepoint"])` + `orch.run_training(suggested_t)` with
a per-decoder dict built from `eval_result["tasks"][name]["peak_timepoint"]` (Stage 1), passed to
`orch.run_training(per_decoder)`. Re-seed `debug_snapshots/default/` so the `train_done.joblib`
snapshot carries genuinely per-decoder `decoding_timepoints`.

---

## Stage 5 — Full cleanup of legacy single-timepoint paths (done)

Done last so development never breaks the float path:
- **`orchestrator.py`**: `run_training` signature → `dict[str, float]` only; removed the float
  branch and `_derive_per_task_timepoints` (the UI/seed now supply explicit per-decoder dicts and
  the evaluator's `peak_timepoint` is the single suggestion source). Dropped the now-unused
  `numpy` import.
- **`artifact_models.py`**: **removed** the singular `decoding_timepoint` field entirely;
  `decoding_timepoints` is now the authoritative **required** field (`min_length=1`). (Originally
  we planned to keep `decoding_timepoint` as a representative mean, but it had no production
  consumer beyond the single-timepoint diagnostics, so it was removed.)
- **`phase1_screen_debug.py`**: dropped the legacy-snapshot fallback that read the singular field.
- **`ModelTrainer.run_training`**: kept the `float | dict` union — low-level API, directly
  unit-tested (`tests/offline_phase/test_trainer.py`); its missing-key `ValueError` is now the
  orchestrator's guard.
- **`evaluator.py`**: `suggested_timepoint` (cross-task avg) **kept** — still drives the
  EvaluationView summary-overlay reference line (a *separate* calculation from
  `decoding_timepoint`).
- **Tests**: `tests/offline_phase/test_orchestrator.py` float call-sites → per-decoder dicts;
  removed the `decoding_timepoint` assertions.

### Deferred follow-up — diagnostic scripts left stale
The four single-timepoint diagnostic scripts still read the now-removed `metadata["decoding_timepoint"]`:
`preproc_parity_check.py` (subscript → **KeyError** on new artifacts), `offline_inference_check.py`,
`inspect_decoder_internals.py`, `full_recording_live_inference_check.py` (`.get(...)` → `None`).
They were **left untouched on purpose** to avoid conflicts with in-flight Goal-18 work. They still
run against *existing* artifacts (which carry the old key) but break on any artifact trained after
Stage 5. **Fix when picked up:** replace the read with the local mean of `decoding_timepoints`,
e.g. `float(sum(d.values()) / len(d))`.

### Docs wrap-up (done)
- Goal 19 marked ✅ Done in `docs/Phase2_UI_Plan_M2.md`; the deferred follow-ups (stale
  diagnostic scripts; optional sklearn `penalty`→`l1_ratio` migration) are tracked there too so
  they survive after this feature plan is closed.
- `docs/backend_architecture.md` `run_training` signature + `decoding_timepoints` references updated
  (light targeted touch; the doc has broader pre-existing drift left untouched).

---

## Verification (end-to-end)

1. **Unit/integration**: `pytest tests/` — new evaluator `peak_timepoint`, orchestrator dict-path,
   and EvaluationView gating tests pass.
2. **Debug GUI walkthrough**: `python -m frontend.debug.main --profile default --phase2`
   (or the Phase 1 walkthrough) — at Node 4, each decoder pre-fills its own peak; set Red and
   Green to *different* timepoints, confirm each from the Summary tab; verify a per-decoder tab's
   spinbox/chart-click moves only that decoder and stays in sync with its Summary row; verify
   "Approve & Continue" stays disabled until all are confirmed.
3. **Train + inspect artifact**: complete training; load the saved `decoder_pipeline.joblib` and
   assert `metadata.decoding_timepoints` holds the **distinct** per-decoder values chosen (not all
   equal), and `decoding_timepoint == mean`.
4. **Seed**: re-run `scripts/demo_seed_debug_snapshots.py --profile default`; confirm
   `train_done.joblib`'s `decoding_timepoints` are per-decoder peaks.
5. **No online regression**: smoke `scripts/smoke_stream_worker.py` against the new pipeline —
   live inference behaves identically (feature stays `n_channels` wide; per-decoder timepoint is
   training-only).

## Risk notes
- Largest surface is `evaluation_view.py` (de-globalizing the selection + signal-blocking sync
  between Summary rows and decoder tabs). The existing `blockSignals` discipline around
  programmatic `setValue` must be preserved per-spinbox to avoid feedback loops.
- `evaluation_complete` payload type change ripples through Phase1Screen, TrainView, and the debug
  walkthrough together — land them in one stage (Stage 3) so the app is never half-wired.
