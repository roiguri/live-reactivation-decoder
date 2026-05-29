# Parallel-Port Trigger Decoding (offline path)

Back to [Reference Docs](README.md) or the [Project Index](../START_HERE.md).

---

## Status: REVERTED (2026-05-17)

**This approach has been reverted.** The recordings in use carry standard
`Stimulus,S11/S12/S13` markers in the `.vmrk`, so the offline pipeline now reads
events natively via `mne.io.read_raw_brainvision` → `mne.events_from_annotations`
(matching the reference pipeline `tomer_preprocessing_new.py`). The empty `.vmrk`
that motivated the analog decoder (open question #2 below) was a recording-side
artifact, not the intended encoding.

`OfflineOrchestrator._load_eeg_raw` no longer decodes the analog channel and no
longer calls `set_annotations`; `trigger_decoder.py` and `test_trigger_decoder.py`
were deleted. `scripts/create_test_eeg.py` now carries the source `.vmrk` stimulus
markers (clipped to the crop window) into generated fixtures. The online phase was
unaffected (it has its own separate LSL trigger path).

The rest of this document is retained as a historical record of the provisional
analog-decode approach and why it was tried.

---

### Original status note (historical): provisional

This document records the trigger-extraction approach that was used for
BindingDecoding recordings. It was **not final** — the choice was made under
uncertainty about whether the analog encoding was by design or a recording-side
misconfiguration. Investigation showed `.vmrk` markers were expected, so it was
reverted to the standard `mne.events_from_annotations` path (see banner above).

See [Open questions and review triggers](#open-questions-and-review-triggers) below for the specific things that prompted the revisit.

---

## The problem we found

The existing reference pipeline ([tomer_preprocessing.md](tomer_preprocessing.md)) extracts trial events with `mne.events_from_annotations(raw)`, which reads BrainVision markers from the `.vmrk` file. On the BindingDecoding recordings, that returns **0 events**:

```text
$ head -10 data/new_experiment/experiment/BindingDecoding103.vmrk
Brain Vision Data Exchange Marker File Version 1.0
[Common infos]
DataFile=BindingDecoding103.eeg
[Marker infos]
Mk1=New Segment,,1,1,0,00000000000000000000
```

Only the default `New Segment` is present — no `Stimulus,Sxx` markers. MNE doesn't surface `New Segment` as an annotation, so `events_from_annotations` yields an empty array and downstream `mne.Epochs(..., event_id={})` crashes with `max() iterable argument is empty`.

Inspection of all 65 channels showed that trial events are encoded as **voltage plateaus on the channel labelled "EMG"** (channel 65). The plateau height in volts × 1e4 equals the parallel-port byte value (code 11 → 1.1 mV, code 12 → 1.2 mV, etc.), matching the codes documented in [BMR Data Specification.md §Trigger Code Reference](BMR%20Data%20Specification.md#trigger-code-reference).

## The decision

Extract triggers from the analog channel during `OfflineOrchestrator._load_eeg_raw`, **before** the channel is dropped along with other non-EEG inputs. Implementation lives at [trigger_decoder.py](../../src/backend/offline_phase/trigger_decoder.py).

Rationale:

- It works on the only recordings we have. Without it, the pipeline crashes at epoch creation.
- It produces codes that match the canonical [trigger code table](BMR%20Data%20Specification.md#trigger-code-reference) (11→`show_red`, 12→`show_green`, 13→`show_yellow`, 16/17/18→scenes, etc.).
- It leaves downstream code (preprocessing, ICA, epoching, AutoReject) untouched. The decoder injects annotations onto `raw` in the BrainVision-convention `Stimulus/S<code>` format, so `mne.events_from_annotations` extracts the integer codes the same way it would for a healthy `.vmrk`.

## Signal shape

Each parallel-port event appears on the "EMG" channel as a brief voltage plateau with a hardware-smoothed leading edge:

```text
samples (voltage × 1e4):  0 0 0 0 0 1 1 2 3 4 5 6 8 9 10 11 12 13 14 14 15 15 15 16 16 16 16 16 17 17 17 17 17 17 17
                          └─ rest ─┘└─── ~2 ms ramp ───┘└────────────── plateau (settled at code 17) ─────────────┘
```

- **Plateau height** = parallel-port byte value (after multiplying by `VOLTAGE_TO_CODE_SCALE = 1e4`).
- **Pulse duration**: typically 5–40 ms with a ~2 ms ramp before the plateau settles. The ramp is hardware low-pass, not introduced by us.
- **Resting level**: ~0 V with small noise (well below the 0.5-code-unit threshold).
- **Sampling rate**: 5000 Hz in the current NeurOne setup, giving ~10–200 samples per plateau.

## Decoder algorithm

In one paragraph: the decoder scans the trigger channel for rising edges that cross the noise threshold, filters out blips shorter than `PULSE_MIN_DURATION_MS`, reads the peak value inside a `PLATEAU_WINDOW_MS` window after the edge, and emits one `mne.Annotations` entry per surviving pulse. The window-peak read (rather than reading the first above-noise sample) is what handles the leading ramp correctly.

Module-level constants (all in [trigger_decoder.py](../../src/backend/offline_phase/trigger_decoder.py)):

| Constant | Value | Purpose |
|---|---|---|
| `TRIGGER_CHANNEL_NAME` | `"EMG"` | Channel to read from |
| `VOLTAGE_TO_CODE_SCALE` | `1.0e4` | volts × scale → integer code |
| `NOISE_THRESHOLD_CODE_UNITS` | `0.5` | rising-edge threshold (in scaled units) |
| `PULSE_MIN_DURATION_MS` | `5.0` | drop pulses shorter than this |
| `PLATEAU_WINDOW_MS` | `10.0` | window after the rising edge for plateau peak |

Unit tests at [test_trigger_decoder.py](../../tests/offline_phase/test_trigger_decoder.py) cover clean pulses, multi-code recordings, sub-threshold noise, the ramp-handling case, the missing-channel error, and the `events_from_annotations` round-trip.

## Output format

`mne.Annotations` with descriptions like `"Stimulus/S 11"` (BrainVision convention, 3-character padded code). This matches what MNE would have generated from a healthy `.vmrk`, so `mne.events_from_annotations` extracts the integer code (`11`) directly — no special handling needed downstream.

The trigger channel itself is **dropped immediately after decoding** in [`OfflineOrchestrator._load_eeg_raw`](../../src/backend/offline_phase/orchestrator.py), along with any other non-EEG/EOG/ECG channels, so it does not pollute ICA, average referencing, or epoching.

## Open questions and review triggers

These are the specific things that would prompt revisiting the decision:

1. **Unknown codes in the decoded stream.** On `BindingDecoding103/experiment`, decoded data contains codes `7-10`, `14-15`, `19-20`, `24-25`, `33-40` that are not in the [BMR-published table](BMR%20Data%20Specification.md#trigger-code-reference). Epoching tolerates them — they appear in `raw.annotations` but are filtered out by `markers_mapping` in `experiment_config.yaml` — but they should be confirmed with Tomer. They may indicate either undocumented event types or a decoding artifact.

2. **Was the empty `.vmrk` by design or by accident?** Older recordings (`data/raw/sub_001/sub_001_task_TEP.vmrk`) contain proper `Stimulus,Sxx` markers in the `.vmrk`. If the new recordings' empty `.vmrk` was a recording-side misconfiguration, future recordings may ship with proper markers and the analog-channel decoder becomes redundant — or worse, *wrong* on a setup where "EMG" is genuinely an EMG channel.

3. **Hardware coupling of the voltage scale.** The `× 1e4` scale is whatever the BrainVision converter applies on this NeurOne setup. Different converter settings or different hardware would change the scale. The decoder has no way to auto-detect a wrong scale today — it would just produce wrong codes silently.

4. **Channel name assumption.** We hardcode `"EMG"`. If a recording uses a different label for the trigger channel, the decoder fails fast with `ValueError` — which is the desired safety behavior — but it means this doc has to be revisited if a different label appears in the wild.

## If we revert

If investigation determines that the analog-channel approach should be reverted to the standard `.vmrk` path:

- Remove the `decode_parallel_port_channel` call from `_load_eeg_raw`.
- Restore `events_from_annotations`-only flow (already the implicit behaviour from MNE — just delete the explicit `raw.set_annotations(...)` line).
- Update `markers_mapping` codes in `experiment_config.yaml` to whatever the new `.vmrk` provides.
- The unit tests at [test_trigger_decoder.py](../../tests/offline_phase/test_trigger_decoder.py) become dead code at that point.

Scope is roughly the same as the original commit, in reverse.

## Relationship to the online phase

The online path has its own (different) trigger decoder via [`LSLReceiver.decode_trigger_value`](../../docs/backend_architecture.md#lslreceiver-helper-functions), which uses `(int(raw_value) >> 8) & 0xFF` because the LSL stream from NeurOne carries the trigger as a packed 32-bit word. Our offline decoder uses `voltage × 1e4` because the BrainVision .eeg file stores the same underlying byte as a pre-scaled float32 voltage.

Two file formats, one logical signal. If the offline decision is revisited, this section should be revisited too — there may or may not be a corresponding online-phase change depending on whether the upstream hardware reconfiguration also affects the LSL stream.
