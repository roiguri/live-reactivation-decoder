# Online Causal Filtering for EEG Preprocessing

Back to [Reference Docs](README.md) or [Project Index](../START_HERE.md).

## The Core Decision: Causal vs. Zero-Phase

The offline pipeline applies its filters using MNE's default forward-backward processing — the IIR filter is applied **twice**: once forward, once backward. This cancels all phase shift (zero-phase output) and doubles the effective filter order.

Online streaming data cannot use forward-backward filtering. At time *t* only samples up to *t* are available, so backward passes are impossible. The online pipeline uses `scipy.signal.sosfilt` — a single forward pass with persistent filter state (`zi`) carried across every micro-batch.

The SOS coefficients come from `mne.filter.create_filter(..., method='iir')`, which returns the same Butterworth design used offline. The difference is only in how they are applied.

## Consequences

| Property | Offline (filtfilt) | Online (sosfilt) |
|---|---|---|
| Phase shift | None (zero-phase) | Yes — frequency-dependent |
| Effective stopband order | ~2× design order | Design order only |
| Waveform distortion | None | Temporal smearing, frequency-dependent |

## Group Delay Analysis

Group delay — how many milliseconds each frequency is shifted — was computed from the actual filter coefficients used in this project. After the migration there are **two cascaded IIR stages** that each contribute their own group delay:

- **High-pass at 0.1 Hz** — biggest contributor below ~2 Hz, fast decay above.
- **Low-pass at 40 Hz** — biggest contributor near 40 Hz, small contribution below.

Approximate combined delay (HP + notch + LP, 1000 Hz input rate, Butterworth design):

| Frequency | Group Delay | Band |
|---|---|---|
| 1 Hz | ~600 ms | Near DC (dominated by HP rolloff) |
| 2 Hz | ~130 ms | Delta |
| 5 Hz | ~30 ms | Theta |
| 10 Hz | ~15 ms | Alpha |
| 20 Hz | ~15 ms | Beta |
| 30 Hz | ~20 ms | Beta |
| 40 Hz | ~30 ms | Gamma low (at LP cutoff) |

Key observations:
- Delay is **not constant** across the passband — this is the defining characteristic of IIR filters.
- 5–30 Hz sees ~15–30 ms of delay; reasonable for spatial-pattern classification.
- Below 5 Hz delay grows rapidly and becomes very large (~600 ms at 1 Hz). This frequency range should not be relied on for latency-sensitive features.
- The 40 Hz LP cutoff adds modest delay (~10–15 ms) on top of the HP. The cascaded total is comparable to what a single Butterworth bandpass would have given.

## Practical Impact for This Project

The decoder classifies encoded image categories from spatial EEG patterns in a post-stimulus window. This is a pattern classification problem — the relevant information is *which spatial pattern is present*, not the exact peak latency of an ERP component.

The ~15–30 ms delay in the 5–40 Hz range:
- Shifts all features in a window slightly, but does not change which pattern is present.
- Does not affect ICA spatial filters, which are insensitive to uniform temporal shifts.
- Would matter for precise peak latency measurements, which is not what this decoder does.

The ICA was trained on zero-phase filtered data offline. The online distribution has the same frequency content but frequency-dependent phase. In practice this rarely harms classification accuracy for spatial pattern decoders.

**The one real concern:** the 1–5 Hz range has large, rapidly varying delay (30–600 ms). If features include delta or theta power and those bands carry signal, temporal smearing there is substantial. If this becomes a problem, raising the HP cutoff to 2–4 Hz online avoids the most dispersive region.

## Timestamp semantics — important caveat

`OnlinePreprocessor.process_batch()` returns `out_timestamps` that mark each kept sample with the LSL timestamp of the **input sample at the kept index** — not the effective time of the filtered value. Because the anti-alias FIR has a constant group delay of `(n_taps - 1) / 2 = 50 samples = 50 ms` (at 1000 Hz input), and the IIR HP/LP add their own variable delay (~10–30 ms in the 5–40 Hz band), the *true* effective time of `filtered[i]` is roughly 50–80 ms earlier than `out_timestamps[i]`.

For classifier inference triggered at "N ms after stimulus", this means the classifier operates on brain activity from `~(N - 60) ms` after stimulus rather than exactly N. The offset is **constant across all predictions**, so it doesn't degrade decoding — it effectively shifts the trained decoding window by ~60 ms. Worth knowing for any UI feature that wants to display "event happened at T", and for cross-comparing offline and online prediction timing.

Mitigations if it ever matters:
1. **Subtract the FIR group delay from `out_timestamps`** — one-line fix, accounts for the largest contribution.
2. **Train the classifier on data offset by the expected online delay** — corrects the issue end-to-end at training time.
3. **Use identical causal filters offline** — eliminates the offline-online phase mismatch entirely, at the cost of training on causal-filtered data.

## Notch Filter

The offline notch uses MNE's `notch_filter()`. Online, the equivalent is `scipy.signal.iirnotch(w0=50.0, Q=30, fs=1000)` converted to SOS via `scipy.signal.tf2sos`. Q=30 is a standard quality factor giving a narrow notch (about 3.3 Hz wide at −3 dB). The implementation matches the high-pass: single forward pass with persistent `zi`.

---

## Decimation: 1000 Hz → 100 Hz

After all spectral filtering, the data is decimated from 1000 Hz to the target rate (100 Hz, per the cited replay paper). Decimation has two sub-steps: anti-aliasing lowpass filter, then subsampling.

### Why not just subsample?

Taking every Nth sample without anti-aliasing first causes **aliasing** — frequencies above the Nyquist of the target rate (50 Hz at 100 Hz target) fold back into the signal as fake spurious content. The dedicated 40 Hz low-pass that runs immediately before decimation already kills most of what would alias. The FIR anti-alias inside `_decimate` is a defensive second line at 45 Hz (0.9 × Nyquist).

### Anti-aliasing FIR low-pass

A FIR low-pass is used (not IIR) because:
- FIR has linear phase — the decimated signal's temporal structure is not distorted.
- No feedback, so no instability risk with the `lfilter` state.
- Applied via `scipy.signal.lfilter(b, 1.0, data, axis=0, zi=zi)` with persistent `zi`.

**Parameters (for 1000 → 100 Hz):**
- Decimation factor: `decimation = input_sfreq / target_sfreq = 10` (integer-only — non-integer ratios raise at construction).
- Cutoff: `0.9 × (100 / 2) = 45 Hz` (10% margin below Nyquist).
- Taps: `10 × decimation + 1 = 101` taps.
- Design: `scipy.signal.firwin(101, 45, fs=1000)`.

**`zi` initialisation:** zero-init (not warm-started from `data[0]`). The FIR's step response is bounded and short (101 samples = 101 ms at 1000 Hz), much milder than the IIR transient. After decimation the transient spans ~10 output samples (~100 ms at 100 Hz). The first second of any EEG session is typically discarded (pre-stimulus baseline), absorbing the transient.

### Integer-ratio subsampling

For 1000 → 100 Hz the decimation ratio is a clean integer 10:1. The `_decimate` algorithm is:

```python
phase = self._decimate_phase
out_indices = np.arange(phase, n_in, self._decimation)
self._decimate_phase = (self._decimation - ((n_in - phase) % self._decimation)) % self._decimation
return filtered[out_indices], timestamps[out_indices]
```

`phase` tracks where the next kept sample lands inside the next chunk, carrying state across micro-batches so the output spacing is globally consistent.

**Why integer-only:** the previous implementation supported arbitrary ratios via polyphase resampling (e.g. 1000 → 256 reduces to 125/32). With the project's locked 100 Hz target the ratio is always integer, and the polyphase bookkeeping became dead weight. Construction asserts `input_sfreq % target_sfreq == 0`; non-integer ratios raise with a clear error rather than being silently approximated.

### Parameters stored in `__init__`

| Attribute | Value (1000→100 Hz) | Purpose |
|---|---|---|
| `_decimation` | 10 | Integer factor |
| `_decimate_fir` | ndarray (101,) | FIR low-pass coefficients |
| `_decimate_zi` | ndarray (100, n_ch) or None | FIR filter state across chunks |
| `_decimate_phase` | int ∈ [0, 10) | Subsampling phase counter |

---

## Alternatives Considered (for reference)

**A) Double the causal filter order to match filtfilt stopband.** Design at order 16 instead of 8 so the single-pass stopband matches filtfilt's magnitude response. Phase shift remains. Implemented by passing custom `iir_params` to `mne.filter.create_filter`.
- Pro: magnitude response matches offline.
- Con: more zi state, higher transient at session start, still has phase shift.

**B) Linear-phase FIR for HP/LP.** FIR filters can have linear phase (all frequencies delayed by the same constant amount), so waveform shape is preserved. Still causal — cannot be zero-phase without buffering.
- Pro: uniform delay, no waveform distortion.
- Con: 0.1 Hz highpass at 1000 Hz requires thousands of taps (hundreds of ms latency). Impractical for real-time micro-batch processing.

**C) Accept the IIR causal single-pass (current approach).** Standard for online EEG/BCI systems. The frequency content is correct; only the within-frequency phase is shifted. Accepted by the field for classification tasks.
- Pro: minimal latency, matches MNE's filter design, simple implementation.
- Con: not bit-exact with training distribution; low-frequency range is heavily dispersive.

**Chosen approach: C.** If offline–online accuracy gap appears in practice, try A first — it is a one-line change to the filter design call.
