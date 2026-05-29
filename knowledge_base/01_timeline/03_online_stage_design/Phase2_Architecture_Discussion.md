  # Phase 2 Architecture Discussion

> **Document Type:** Design Discussion / Decision Document
> **Status:** Under Consideration
> **Date:** 2025-01-XX
>
> This document captures the architectural discussion and design options for Phase 2 (Online Live Inference) implementation. It is NOT a source of truth or final specification, but rather a record of design decisions and trade-offs being considered.

---

# Phase 2 (Online Live Inference) Implementation Plan

## Architecture Analysis

### Current Design (from `backend_architecture.md`)

**Data Flow (Every ~10ms):**
1. LSLReceiver pulls new EEG chunks from hardware stream
2. New samples → RingBuffer (maintains rolling 2-second window at 1000 Hz)
3. OnlinePreprocessor receives **entire 2000-sample buffer**
4. Preprocessor processes **entire buffer**: filter → decimate → ICA → return newest timepoint
5. LiveInferenceEngine predicts on single timepoint (64 channels at 250Hz)
6. StreamWorker emits probabilities to UI

### Critical Design Issue: Stateless Processing is Scientifically Risky

**The Problem:**
The current design has a **fundamental correctness issue**, not just inefficiency. It reprocesses the entire 2-second window every 10ms, which means:

1. **Signal Distortion**: IIR filters have memory - they depend on previous inputs and outputs. Restarting the filter creates **artificial transients** and **edge effects**.

2. **Newest Sample is Most Affected**: The newest filtered sample (exactly what we predict on!) may depend on a fake filter startup condition.

3. **Insufficient Warmup**: With a 1 Hz lower cutoff, filter memory can be relatively long. A 2-second buffer is not obviously safe for transients to fully decay.

**Example of Redundancy:**
- At t=0ms: Process samples 0-2000 (filter restart)
- At t=10ms: Process samples 10-2010 (filter restart - discontinuity!)
- At t=20ms: Process samples 20-2020 (filter restart - discontinuity!)

> **Expert insight:** "The bad part is not 'it wastes CPU.' The bad part is 'the newest filtered sample may depend on a fake filter startup condition.'"

**This is a correctness problem, not just a performance problem.**

### Understanding Phase 1 (Offline Training)

**IMPORTANT:** Phase 1 is NOT the semester A analysis in the parent `reactivation-decoder` repo. It's the offline training phase of this app.

**Phase 1 Preprocessing Pipeline** (from `tomer_preprocessing.md` + `experiment_config.yaml`):
1. **Load Raw**: .vhdr file at 1000Hz
2. **Bandpass Filter**: 1-40Hz IIR + 50Hz notch (ZERO-PHASE)
3. **Resample**: 1000Hz → 250Hz
4. **Bad Channels**: Detect and interpolate
5. **Average Reference**: Re-reference all channels
6. **ICA**: Fit 20 components (fastica), remove artifacts
7. **Epoch**: Extract -0.2s to +0.8s around triggers
8. **Baseline**: Baseline correction (start to 0s)
9. **AutoReject**: Drop/repair bad epochs
10. **Export online_state**: ICA unmixing matrix, bad channel indices, avg ref projection

**Phase 1 Training** (ModelEvaluator + ModelTrainer):
1. Cross-validation (k=5) to find optimal timepoint
2. Extract spatial features at that timepoint: (n_epochs, n_channels)
3. Train LDA classifiers (solver='lsqr', shrinkage='auto')
4. Save models + online_state to decoder_pipeline.joblib

### Phase 2 (Online/Live Inference) Requirements

**What We Actually Need:**
1. **Continuous stream** of EEG at 1000 Hz from LSL
2. **Causal Filtering**: Apply 1-40Hz bandpass + 50Hz notch (IIR with state)
3. **Spatial transforms**: Bad channel removal, average reference, ICA unmixing
4. **Decimation**: 1000 Hz → 250 Hz
5. **Feature extraction**: Most recent spatial pattern (n_channels array)
6. **Inference**: Features → LDA decoder → probability
7. **NO temporal smoothing** - not part of Phase 1 training!

### Design Options 

Since temporal smoothing is NOT part of Phase 1 training, the architecture is simpler:

#### Option A: Stateless - Full Buffer Processing ⚠️ NOT RECOMMENDED

```
Every ~10-20ms:
- Pull entire ring buffer (e.g., 2000 samples @ 1000Hz)
- Filter entire buffer (IIR, but reprocessing old samples)
- Bad channel removal (spatial, instant)
- Average reference (spatial, instant)
- ICA unmixing (spatial, instant)
- Decimate entire buffer to 250Hz → 500 samples
- Extract NEWEST sample (64 channels at 250Hz)
- Predict
```

**Pros:** Simple, no state management, easy to debug

**Cons (CRITICAL):**
- **Signal distortion**: Filter restarts create artificial transients and edge effects
- **Newest sample corrupted**: The prediction target depends on fake filter startup
- **Insufficient warmup**: 2s buffer may not be enough for 1 Hz cutoff to stabilize
- **Redundancy**: Reprocessing 99% same data, high CPU, wasteful
- **Correctness risk**: Creates artificial boundary conditions that don't exist in continuous EEG

> **Expert warning:** "Since IIR filters have memory, restarting the filter on every rolling buffer creates artificial boundary conditions. That may distort the newest sample, especially with a low cutoff around 1 Hz."

**Use only as emergency fallback with validation against stateful output.**

#### Option B: Stateful/Incremental - Process New Samples Only
```
Initialization:
- Initialize IIR filter states (zi) for bandpass + notch
- Load online_state (ICA matrix, bad channels, avg ref)

Each time new LSL data arrives (timing depends on LSL buffering):
- Pull NEW samples only (chunk size varies: 1-10+ samples @ 1000Hz)
- Filter new samples using filter state (update zi) → filtered_samples
- Drop bad channels from filtered_samples (spatial)
- Apply average reference to filtered_samples (spatial)
- Apply ICA unmixing matrix to filtered_samples (spatial)
- Decimate: accumulate until we have enough for 250Hz output
- When 250Hz sample ready: predict immediately
```
**Pros:** Minimal redundancy, efficient, lowest latency, scalable
**Cons:** Complex state management (filter zi), decimation accumulation logic

**Note:** LSL chunk arrival rate depends on hardware buffering configuration. Need to characterize actual behavior (see Test 0 below).

#### Option C: Stateful Micro-batch - Process Small Chunks ✅ RECOMMENDED

```
Initialization:
- Initialize IIR filter states (zi) using sosfilt_zi
- Load online_state

Every 40ms (batch):
- Pull batch of new samples (40 samples @ 1000Hz)
- Filter batch with persistent state (update zi) → continuous filtering!
- Apply spatial transforms (bad channels, avg ref, ICA)
- Decimate batch to 250Hz (10 samples) with persistent phase counter
- Emit all 10 predictions, display latest
- Save filter state for next batch
```

**Pros:**
- **Correctness**: Continuous filtering without artificial boundaries
- **Balance**: Efficient without excessive complexity
- **Debuggable**: NumPy/SciPy/sklearn work well on arrays
- **Acceptable latency**: 40ms is fine for events lasting 100-500ms
- **No redundancy**: Each sample processed exactly once

**Cons:** Slight batching latency (40ms), need to maintain filter state

> **Expert recommendation:** "Use micro-batching for simplicity, but keep state for correctness. Batching does not mean we forget the past."

**Key Implementation Details:**
- Use `scipy.signal.sosfilt` (not lfilter) for numerical stability
- Maintain persistent `zi` state across batches
- Maintain persistent decimation phase counter (global sample counter)
- Make batch size configurable (can reduce to 20ms, 10ms later)

### Critical Questions to Resolve

1. **Processing Architecture:**
   - Stateless (simple, wasteful) vs Stateful (efficient, complex) vs Micro-batch (balance)?
   - What's the priority: simplicity, efficiency, or low latency?

2. **IIR Filter State Management:**
   - scipy.signal supports stateful filtering with `zi` parameter
   - Need to maintain filter state between calls
   - How to initialize: `lfilter_zi` for steady-state startup

3. **Processing Order (must match Phase 1):**
   - Phase 1: Raw → Filter → Resample → Bad channels → Avg ref → ICA → Epochs
   - Phase 2: Raw → **Causal Filter** → Bad channels → Avg ref → ICA → Decimate → Features
   - ICA is applied AFTER filtering and spatial preprocessing

4. **Processing Frequency:**
   - Stateless: Every 10-20ms (limited by CPU cost of reprocessing)
   - Stateful: Every 4ms (as data arrives)
   - Micro-batch: Every 20-40ms (batch size)
   - Trade-off: latency vs CPU vs implementation complexity

5. **Decimation Strategy:**
   - 1000Hz → 250Hz = 4:1 decimation
   - Stateful approach: accumulate 4 samples before outputting 1
   - Need proper anti-aliasing filter before decimation

6. **Buffer Requirements:**
   - Stateless: Need ~200-500ms for filter warmup (avoid edge artifacts)
   - Stateful: No large buffer needed, just filter state
   - Micro-batch: Small buffer for batch size (20-40ms)

## Final Data Flow Architecture

**Main Loop (Stateful Micro-Batch):**
```python
while experiment_running:
    # 1. Collect batch (40ms @ 1000Hz = 40 samples)
    batch, timestamps = lsl_receiver.get_batch(batch_ms=40)

    # 2. Preprocess with persistent state
    features, feature_timestamps = preprocessor.process_batch(batch, timestamps)
    # Returns 10 samples @ 250Hz

    # 3. Predict all 10 samples
    probabilities = inference_engine.predict(features)

    # 4. Log all predictions, display latest
    logger.write(feature_timestamps, probabilities)
    ui.update(probabilities[-1])
```

**StreamWorker Implementation:**
```
StreamWorker.run() loop:
  1. LSLReceiver.pull_new_data() → (timestamps, eeg_chunk, markers)
  2. Accumulate into 40ms batch
  3. When batch ready:
     a. OnlinePreprocessor.process_batch(batch):
        - Apply sosfilt bandpass with persistent zi_bandpass
        - Apply sosfilt notch with persistent zi_notch
        - Drop bad channels (fixed mapping from Phase 1)
        - Apply average reference
        - Multiply by ICA unmixing matrix (from Phase 1)
        - Decimate with persistent sample_counter (40 → 10 samples)
        - Return (features, timestamps) for all 10 samples @ 250Hz
     b. LiveInferenceEngine.predict(features) → 10 probabilities
     c. StreamWorker.emit(probabilities, timestamps, markers)
  4. Clear batch accumulator, keep filter states
```

**Key Points:**
- **NO RingBuffer** - just small batch accumulator
- **Persistent states**: zi_bandpass, zi_notch, sample_counter
- **Log ALL predictions**, not just latest
- **Total latency**: ~50ms (40ms batch + 10ms processing)

## Architectural Decisions (Expert-Validated)

### 1. Processing Strategy: Stateful Micro-Batch ✅

**DECISION:** Implement **Stateful Micro-batch (Option C)** with 40ms batches.

**Rationale:**
- **Correctness first**: Avoids artificial filter restarts and signal distortion
- **Balanced complexity**: Simpler than per-sample, more correct than stateless
- **Acceptable latency**: 40ms is fine for events lasting 100-500ms
- **Engineering benefits**: NumPy/SciPy/sklearn work well on arrays
- **Already accepting mismatch**: Since we're accepting training/inference mismatch, we should NOT add extra distortions

> **Expert reasoning:** "Because we are not doing online-emulation, we should not try to 'perfectly match' the offline pipeline. Instead, we should build the cleanest practical online approximation. The architecture decision becomes even clearer: we should avoid adding extra avoidable distortions in the online pipeline."

### 2. Filter State Management: Use sosfilt with Persistent State ✅

**DECISION:** Use `scipy.signal.sosfilt` (NOT lfilter) with persistent `zi` state.

**Implementation:**
```python
from scipy.signal import butter, sosfilt, sosfilt_zi

# Initialize - use Second-Order Sections for numerical stability
sos_bandpass = butter(
    N=4,
    Wn=[1, 40],
    btype="bandpass",
    fs=1000,
    output="sos",  # Important!
)
sos_notch = butter(
    N=4,
    Wn=50,
    btype="bandstop",
    fs=1000,
    output="sos",
)

# Initialize persistent filter states (one per channel)
zi_bandpass = sosfilt_zi(sos_bandpass)  # Shape: (n_sections, 2)
zi_bandpass = np.tile(zi_bandpass[:, None, :], (1, 64, 1))  # (n_sections, 64, 2)

zi_notch = sosfilt_zi(sos_notch)
zi_notch = np.tile(zi_notch[:, None, :], (1, 64, 1))

# Process batch - filter continues from previous state
filtered, zi_bandpass = sosfilt(sos_bandpass, chunk, axis=1, zi=zi_bandpass)
filtered, zi_notch = sosfilt(sos_notch, filtered, axis=1, zi=zi_notch)
```

> **Expert guidance:** "SciPy documentation explicitly recommends sosfilt over lfilter for most filtering tasks. Second-order sections reduce numerical problems compared to direct-form polynomial coefficients."

### 3. Decimation Implementation: Persistent Phase Counter ✅

**DECISION:** Manual decimation with persistent global phase counter.

**Critical Requirement:** Decimation phase must persist across batches!

**Implementation:**
```python
class OnlinePreprocessor:
    def __init__(self, ...):
        self.sample_counter = 0  # Global persistent counter

    def process_batch(self, batch, timestamps):
        # ... filtering ...

        # Decimate 1000Hz → 250Hz with phase tracking
        n_samples = filtered.shape[1]
        start_idx = self.sample_counter
        keep_indices = []

        for i in range(n_samples):
            if (start_idx + i) % 4 == 0:  # Keep every 4th sample
                keep_indices.append(i)

        decimated = filtered[:, keep_indices]
        timestamps_250hz = timestamps[keep_indices]

        self.sample_counter += n_samples  # Update for next batch

        return decimated.T, timestamps_250hz
```

> **Expert warning:** "Don't reset decimation phase every batch! The decimation phase must persist. Use a global sample counter."

**Why Not scipy.signal.decimate()?**
- Built for batch processing, not streaming with persistent state
- Manual approach gives full control over phase alignment across batches

### 4. Spatial Transform Order ✅

**DECISION:** Follow Phase 1 order with fixed transforms from offline preprocessing.

**Processing Pipeline:**
```
Raw → Causal Filter → Bad channels → Avg reference → ICA → Decimate → Features → LDA
```

**Detailed Spatial Preprocessing:**
1. **Bad-channel handling**: Use offline-detected bad channels (fixed mapping)
   - Don't try online bad-channel detection in first implementation
2. **Average reference**: Subtract mean across channels
3. **ICA artifact projection**: Apply fixed ICA unmixing matrix from offline
   - Don't try online ICA fitting in first implementation
4. **Feature extraction**: Extract spatial pattern (n_channels array)
5. **LDA prediction**: Apply trained LDA classifier

> **Expert guidance:** "Don't try online bad-channel detection or online ICA fitting in first implementation. Use fixed spatial transforms from offline."

### 5. Component Breakdown ✅

**Final 4 Components:**
1. **LSLReceiver** - Hardware interface (LSL proxy management + data pull)
2. **OnlinePreprocessor** - Stateful filtering + spatial transforms + decimation
3. **LiveInferenceEngine** - Model loading + prediction
4. **StreamWorker** - QThread orchestration loop with batch accumulator

**Note:** No RingBuffer needed! Just small batch accumulator in StreamWorker.

## Summary: What Phase 2 Really Needs to Do

1. **Receive LSL stream** at 1000 Hz (LSLReceiver)
2. **Causal IIR filtering** with state (1-40Hz + 50Hz notch)
3. **Spatial preprocessing**: bad channels → avg ref → ICA unmixing
4. **Decimate** to 250 Hz
5. **Extract** most recent spatial features (64 channels)
6. **Predict** with LDA models
7. **Emit** probabilities to UI

**NO temporal smoothing, NO large buffers, NO complex windowing!**

---

## Detailed Concept Explanations

### 1. Why Stateless is Scientifically Risky (Not Just Inefficient)

**Question:** In stateless full buffer processing, what are the real problems?

**Answer:** The issue is **NOT primarily CPU waste**. The deeper problem is **signal correctness**.

**The Real Problem: Filter Artifacts**

IIR filters are recursive - they depend on previous inputs and outputs:
```python
# IIR filter equation (simplified):
# y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
#
# To compute y[n], you need:
# - Current input: x[n]
# - Past inputs: x[n-1], x[n-2]
# - Past outputs: y[n-1], y[n-2]
```

**What happens in stateless processing:**
- Every 10ms: restart filter from scratch on 2s buffer
- Filter assumes this is the START of the EEG recording
- Creates artificial transients and boundary effects
- **The newest sample (what we predict on!) depends on fake startup**

**Why 2s buffer may not be enough:**
- With 1 Hz lower cutoff, filter memory can be relatively long
- Transients may not fully decay in 2 seconds
- The newest filtered sample is most affected

> **Expert insight:** "The bad part is not 'it wastes CPU.' The bad part is 'the newest filtered sample may depend on a fake filter startup condition.' Since IIR filters have memory, restarting the filter on every rolling buffer creates artificial boundary conditions. That may distort the newest sample, especially with a low cutoff around 1 Hz."

**Secondary Problem: CPU Waste**
- We filter 2000 samples to get 1 new prediction
- 99.95% of the work is redundant
- High CPU load, battery drain

**Latency is actually acceptable (~16ms), but correctness and efficiency are not.**

### 2. Buffer vs Batches - What's the Difference?

**Buffer (RingBuffer):**
- **Purpose:** Store historical data for **later use**
- **Size:** Large (e.g., 2 seconds = 2000 samples)
- **Access pattern:** Read the **full buffer** contents
- **Example use:** Need 500ms of past data for edge-artifact-free filtering

```python
# RingBuffer usage
ring_buffer = RingBuffer(size=2000)  # 2 seconds @ 1000Hz

# Every 10ms:
new_chunk = lsl_receive()  # e.g., 10 samples
ring_buffer.append(new_chunk)
full_window = ring_buffer.get_all()  # Returns 2000 samples
process(full_window)  # Process all 2000 samples
```

**Batch (BatchBuffer/Accumulator):**
- **Purpose:** Accumulate **new samples** until you have enough to process
- **Size:** Small (e.g., 40ms = 40 samples)
- **Access pattern:** Accumulate until full, then process and **empty**
- **Example use:** Wait for 40 new samples before processing

```python
# BatchBuffer usage
batch_buffer = []

# Continuous loop:
new_chunk = lsl_receive()  # e.g., 4 samples
batch_buffer.extend(new_chunk)

if len(batch_buffer) >= 40:  # Got enough for a batch
    process(batch_buffer)  # Process only these 40 NEW samples
    batch_buffer = []  # Empty and start fresh
```

**Key Differences:**

| Feature | RingBuffer | BatchBuffer |
|---------|-----------|-------------|
| Size | Large (seconds of data) | Small (milliseconds of new data) |
| Retains old data? | Yes, circular | No, empties after processing |
| Purpose | Keep history | Accumulate new arrivals |
| Processing | Reprocess same data repeatedly | Process each sample once |

**In our case:**
- **Stateless** uses a **RingBuffer** because it reprocesses the full 2s history
- **Stateful Micro-batch** uses a **BatchBuffer** because it accumulates 40ms of NEW samples
- **Fully stateful** uses **neither** - processes immediately as samples arrive!

**CRITICAL: Batching ≠ Forgetting**

Batching is NOT about forgetting the past - it's about implementation convenience.

**Bad Batching (Stateless):**
```python
take 40 ms
filter from scratch  # WRONG! Creates artifacts
forget everything
take next 40 ms
filter from scratch  # WRONG! Discontinuity
```

**Good Batching (Stateful Micro-Batch):**
```python
take 40 ms
filter using previous filter state  # Continuous!
save new filter state
take next 40 ms
continue from saved filter state  # Continuous!
```

> **Expert clarification:** "Batching does not mean we forget the past. This is the important part. We process 40ms chunks for simplicity, but we keep the filter state across chunks so the filter behaves like it is seeing one continuous EEG stream."

**Benefits of Batching:**
- NumPy/SciPy/sklearn work well on arrays
- Easier to debug than per-sample processing
- More robust implementation
- 40ms latency acceptable for events lasting 100-500ms

### 3. Stateful Filtering with `zi` - Deep Dive

**The Problem with Naive Filtering:**

When you filter in chunks, you get discontinuities at chunk boundaries:

```python
# WRONG way - stateless
chunk1 = [1.0, 2.0, 3.0, 4.0]
chunk2 = [5.0, 6.0, 7.0, 8.0]

filtered1 = lfilter(b, a, chunk1)  # Filter assumes chunk1 is entire signal
filtered2 = lfilter(b, a, chunk2)  # Filter assumes chunk2 is entire signal

# Problem: There's a discontinuity between filtered1[-1] and filtered2[0]
# because lfilter doesn't "remember" the previous chunk's state!
```

**What is Filter State (`zi`)?**

IIR filters are recursive - they depend on **previous outputs**. The `zi` (initial conditions) parameter captures this "memory":

```python
# IIR filter equation (simplified):
# y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
#
# To compute y[n], you need:
# - Current input: x[n]
# - Past inputs: x[n-1], x[n-2]
# - Past outputs: y[n-1], y[n-2]
#
# zi stores these past values!
```

**Stateful Filtering - Maintaining Continuity:**

```python
# CORRECT way - stateful
from scipy.signal import lfilter, lfilter_zi, butter

# Design filter once
b, a = butter(4, [1, 40], btype='bandpass', fs=1000)

# Initialize filter state (steady-state for first sample value)
zi = lfilter_zi(b, a) * chunk1[0]

# Process chunk 1
filtered1, zi = lfilter(b, a, chunk1, zi=zi)
# zi now contains the filter's "memory" at end of chunk1

# Process chunk 2 - continues from where chunk1 left off
filtered2, zi = lfilter(b, a, chunk2, zi=zi)
# zi updated again

# Result: filtered1 and filtered2 connect smoothly, as if
# we filtered the entire [chunk1 + chunk2] in one go!
```

**Visual Analogy:**

Think of filtering like a running average:
```
Stateless (no zi):
Chunk 1: avg([1,2,3,4]) = 2.5
Chunk 2: avg([5,6,7,8]) = 6.5  ← Forgot about chunk 1!

Stateful (with zi):
Chunk 1: running_avg([1,2,3,4]) = 2.5, remember state
Chunk 2: running_avg([5,6,7,8], prev_state) = 4.5  ← Smooth transition!
```

**What `zi` Actually Contains:**

For a typical IIR filter:
```python
zi.shape  # e.g., (max(len(a), len(b)) - 1, n_channels)

# For a 4th-order Butterworth filter:
# zi contains the last 4 inputs and last 4 outputs
# needed to compute the next filtered sample
```

**Comparison:**

| Aspect | Stateless (no zi) | Stateful (with zi) |
|--------|------------------|-------------------|
| Filter each chunk | Independently | Continues from previous |
| Boundary artifacts | YES - discontinuities | NO - smooth |
| Memory | None between chunks | Maintains zi |
| Correctness | Wrong for streaming | Correct for streaming |
| Complexity | Simple | Need to track zi |

**Real-world impact:**

```python
# Stateless - edge artifacts every chunk
# Each chunk treated as separate signal → filter assumes zeros before/after
# Results in "ringing" artifacts at chunk boundaries

# Stateful - continuous filtering
# Filter "remembers" previous samples → no artifacts
# Results are identical to filtering entire stream at once
```

**Why This Matters for Phase 2:**

Phase 1 used **zero-phase filtering** on complete epochs:
```python
raw.filter(1, 40, method='iir')  # Filters entire recording at once
```

Phase 2 must use **causal filtering** on streaming data:
```python
# Cannot filter future data!
# Must maintain filter state between chunks
filtered, zi = lfilter(b, a, new_chunk, zi=zi)
```

**The zi approach ensures Phase 2 filtering is mathematically equivalent to Phase 1, just done incrementally.**

---

## Summary Table: Architecture Comparison

| Feature | Stateless | Stateful Micro-batch | Fully Stateful |
|---------|-----------|---------------------|----------------|
| **Latency** | ~16ms | ~50ms | Variable (depends on LSL)* |
| **CPU Efficiency** | Very poor | Good | Excellent |
| **Memory** | 2s buffer | 40ms batch + zi | Just zi |
| **Complexity** | Simple | Medium | Complex |
| **Redundancy** | 99.95% | 0% | 0% |
| **Filter artifacts** | ⚠️ YES - artificial transients | ✅ None | ✅ None |
| **Signal correctness** | ⚠️ Risky | ✅ Correct | ✅ Correct |
| **Debuggability** | Easy | Good (arrays) | Harder |
| **Best for** | ❌ Emergency fallback only | ✅ **Recommended** | Production optimization |

*Fully stateful latency depends on LSL chunk arrival rate. If LSL sends samples frequently (every 1-10ms), latency is minimal. Need to characterize with Test 0.

**Key Insight:** Stateless doesn't just waste CPU - it creates **artificial filter startup conditions** that distort the newest sample (what we predict on!). With a 1 Hz cutoff, a 2s buffer may not be enough for transients to decay.

> **Expert recommendation:** "The clean first implementation should be stateful micro-batch with configurable batch size."

**Important Note on Latency:**
- **Sample generation rate**: 1000 Hz = 1 sample every **1 millisecond**
- **LSL chunk arrival**: Variable, depends on hardware buffering (unknown until Test 0)
- **Our batching**: Accumulate until 40ms worth of samples, then process
- Total pipeline latency = LSL arrival variability + batch accumulation (40ms) + processing (~10ms) + UI (~10ms)

---

## Accepted Limitations and Critical Considerations

### Training/Inference Mismatch: Accepted

**What is the mismatch?**

**Offline Training (Phase 1):**
- Zero-phase filtering (bidirectional, uses future data)
- Full recording available
- Epoching around triggers
- Baseline correction (start to 0s)
- AutoReject for epoch cleaning
- Clean, artifact-free preprocessing

**Online Inference (Phase 2):**
- Causal IIR filtering (forward-only, no future data)
- Streaming data
- No epoching
- No baseline correction
- No AutoReject
- Practical, real-time approximation

**Expected Consequence:**
Online classifier performance may be somewhat worse than offline validation metrics.

**Decision: Accept the mismatch**

We are NOT implementing an online-emulation training pipeline because:
- Time constraints
- Adds significant complexity
- Would require reprocessing localizer data
- Would require retraining/revalidating models at new optimal timepoint

> **Expert reasoning:** "Since we are not doing online-emulation, we should not try to 'perfectly match' the offline pipeline. Instead, we should build the cleanest practical online approximation."

**Mitigation Strategy:**
Keep the online pipeline as clean and deterministic as possible:
- Use stateful filtering (no artificial artifacts)
- Use persistent decimation phase
- Use fixed spatial transforms from Phase 1
- Document and log this limitation clearly

> **Critical insight:** "Because we're already accepting offline/online preprocessing mismatch, we should NOT add ANOTHER source of signal distortion (stateless filter restarts). The architecture decision becomes even clearer: we should avoid adding extra avoidable distortions in the online pipeline."

### Phase Delay: Understanding the Impact

**What is Phase Delay?**

Causal filters shift signal components in time because they can't use future samples.

**Impact on Temporal Patterns:**
- **Offline zero-phase filtering**: Pattern at 350ms appears at 350ms
- **Online causal filtering**: Same pattern may appear at 370-430ms
- The exact delay depends on filter design and frequency content

**Why It Matters:**
- The offline optimal timepoint (e.g., 350ms) may no longer be optimal online
- Category information is still present, just shifted later in time
- The LDA classifier is trained on zero-phase patterns at 350ms
- Online, it receives causal patterns that are temporally shifted

**Does It Break Real-Time Detection?**

Probably not, if total latency stays below acceptable limit (~100ms).

**Total delay includes:**
- Filter group delay (~20-60ms depending on design)
- Batch accumulation delay (40ms)
- Processing time (~10ms)
- UI/communication delay (~10ms)
- **Total: ~80-120ms**

If events last 100-500ms, we're within acceptable range.

**Critical Requirement: Timestamp Logging**

Log timestamps carefully for every prediction:
- EEG sample LSL timestamp
- Wall-clock time when prediction emitted
- Classifier probability
- Batch number
- Optional: Latency estimate

This allows post-hoc analysis of actual latency and temporal alignment.

### Validation Tests: Must Do Before Real Experiment

#### Test 0: Characterize LSL Chunk Delivery ✅ FIRST TEST

**Purpose:** Understand actual LSL buffering behavior before designing the pipeline.

**Method:**
```python
# Simple LSL characterization script
import pylsl
import time
import numpy as np

streams = pylsl.resolve_stream('type', 'EEG')
inlet = pylsl.StreamInlet(streams[0])

chunk_sizes = []
inter_arrival_times = []
last_time = time.perf_counter()

for i in range(1000):  # Sample for ~10 seconds
    chunk, timestamps = inlet.pull_chunk()

    if len(chunk) > 0:
        current_time = time.perf_counter()
        inter_arrival_times.append(current_time - last_time)
        chunk_sizes.append(len(chunk))
        last_time = current_time
    else:
        time.sleep(0.0001)  # Small sleep if no data

print(f"Chunk sizes: mean={np.mean(chunk_sizes):.1f}, "
      f"std={np.std(chunk_sizes):.1f}, "
      f"min={min(chunk_sizes)}, max={max(chunk_sizes)}")
print(f"Inter-arrival time: mean={np.mean(inter_arrival_times)*1000:.1f}ms, "
      f"std={np.std(inter_arrival_times)*1000:.1f}ms")
```

**What This Tells Us:**
- **Chunk size**: How many samples LSL sends per chunk (e.g., 1, 4, 10, 40?)
- **Arrival frequency**: How often chunks arrive (e.g., every 1ms? 4ms? 10ms?)
- **Variability**: Is it consistent or variable?

**Decision Impact:**
- If LSL sends 1-2 samples frequently → Option B (fully stateful) makes sense
- If LSL sends ~10-40 samples per chunk → Our Option C (40ms batches) aligns well
- If LSL buffers heavily (>100 samples) → May need to adjust batch size

**This test should be run FIRST before making final implementation decisions.**

#### Test 1: Chunked Filtering = Continuous Filtering ✅ CRITICAL

**Purpose:** Verify that stateful filtering with persistent state works correctly.

**Method:**
1. Take saved EEG recording
2. **A:** Filter entire recording in one call (causal)
3. **B:** Filter same recording in 40ms batches with persistent state
4. Compare A and B sample-by-sample

**Expected Result:**
- Outputs should match almost exactly (except very beginning warmup)
- If they don't match, state handling is WRONG

**This test must pass before using the pipeline in real experiments.**

#### Test 2: Stateless vs Stateful Comparison (If Considering Stateless Fallback)

**Purpose:** Quantify the distortion caused by stateless processing.

**Method:**
1. **A:** True continuous stateful causal filtering
2. **B:** Rolling-buffer stateless filtering
3. Compare newest sample from B against corresponding sample from A
4. Compare classifier outputs

**Decision Criteria:**
- If close enough for classifier outputs, stateless may be acceptable as emergency fallback
- If not close, don't use stateless

#### Test 3: End-to-end Latency Logging

**Purpose:** Measure actual latency, don't just estimate.

**Method:**
During dry run with saved data or live stream, log:
- LSL sample timestamp
- Time batch received
- Time features computed
- Time prediction computed
- Time UI updated

**Analyze:** Actual latency distribution and identify bottlenecks.

### Pragmatic Fallback Hierarchy

**If time is extremely short:**

**Best Version (Recommended):**
✅ Stateful micro-batch
✅ 40ms batches
✅ Causal filtering with sosfilt
✅ Persistent filter states (zi)
✅ Persistent decimation phase
✅ Validation: chunked vs continuous filter

**Acceptable Simpler Version:**
✅ Stateful micro-batch
✅ 40ms batches
✅ Causal filtering with sosfilt
✅ Persistent states
⚠️ Minimal validation only

**Emergency Fallback (If Desperate):**
⚠️ Stateless rolling buffer
⚠️ Longer buffer (>2 seconds, preferably 5s)
⚠️ Causal filtering from scratch
❌ **Must compare against stateful output on saved data**
❌ Document that this is a compromise

**Avoid if At All Possible:**
❌ 2-second stateless buffer
❌ IIR filtering from scratch every iteration
❌ No comparison to stateful output
❌ Use directly in real experiment without validation
❌ **This is the risky version!**

> **Expert warning:** "We should not implement the fully stateless approach as the main real-time pipeline. It is simple, but the issue is not just CPU. If time gets tight, stateless can be used as a temporary baseline or emergency fallback, but only if we compare it against stateful filtering on saved data and show that the outputs are close enough."

---

## Final Decision Summary

### ✅ Decisions Made (Expert-Validated)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Architecture** | Stateful Micro-Batch | Balance of correctness, simplicity, and efficiency |
| **Batch Size** | 40ms (configurable) | Acceptable latency for 100-500ms events |
| **Filter Implementation** | `scipy.signal.sosfilt` | Numerical stability, SciPy recommended |
| **Filter State** | Persistent `zi` across batches | Continuous filtering, no artificial boundaries |
| **Decimation** | Manual with persistent phase counter | Full control over phase alignment |
| **Spatial Transforms** | Fixed from Phase 1 | No online bad-channel detection or ICA fitting |
| **Training/Inference Mismatch** | Accepted | No online-emulation pipeline due to time constraints |
| **Validation Required** | Chunked vs continuous filter | Must pass before real experiment |

### ❌ Decisions Rejected

| Option | Reason for Rejection |
|--------|---------------------|
| **Stateless as main approach** | Creates artificial filter restarts, signal distortion |
| **lfilter instead of sosfilt** | Numerical instability with high-order filters |
| **scipy.signal.decimate()** | Doesn't support persistent state for streaming |
| **Online-emulation training** | Time constraints, complexity, requires retraining |
| **Online bad-channel detection** | Defer to future, use fixed transforms first |
| **Online ICA fitting** | Defer to future, use fixed transforms first |

### 📝 Documentation Requirements

**Must Document in Experiment Notes:**
1. Training/inference mismatch (zero-phase vs causal)
2. Phase delay from causal filtering
3. No baseline correction in online
4. No AutoReject in online
5. Fixed spatial transforms from Phase 1
6. Expected: slightly degraded online performance vs offline validation

### 🔧 Implementation Files to Create

1. `src/backend/online_phase/lsl_receiver.py`
2. `src/backend/online_phase/preprocessor.py`
3. `src/backend/online_phase/inference.py`
4. `src/backend/online_phase/stream_worker.py`
5. `tests/online_phase/test_preprocessor_stateful.py`

### 🎯 Next Steps

1. **FIRST:** Run Test 0 to characterize LSL chunk delivery behavior
2. Adjust batch size if needed based on Test 0 results
3. Implement LSLReceiver
4. Implement OnlinePreprocessor with persistent state
5. Implement LiveInferenceEngine
6. Implement StreamWorker with batch accumulation
7. **CRITICAL:** Run Test 1 validation (chunked vs continuous filter must match)
8. Dry run with saved data
9. Live experiment

---

## Document Status

**Status:** ✅ Architecture Decisions Finalized

**Implementation:** Ready to proceed with stateful micro-batch approach

**Expert Review:** Completed - all insights integrated

**Last Updated:** January 2025
