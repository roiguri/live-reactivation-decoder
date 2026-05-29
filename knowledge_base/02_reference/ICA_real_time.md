# Real-Time ICA for EEG Processing

Back to [Reference Docs](README.md) or [Project Index](../START_HERE.md).

## Overview

Independent Component Analysis (ICA) is a blind source separation technique used to decompose multi-channel EEG data into independent components representing distinct sources of neural and non-neural activity. In EEG processing, ICA is primarily used for artifact rejection by identifying and removing components associated with eye blinks, muscle activity, cardiac signals, and other non-brain sources.

The key insight is that ICA learns a mixing matrix that describes how independent sources are combined in the observed electrode signals. By inverting this matrix, we obtain an unmixing matrix that can separate the original sources, allowing us to filter out artifacts.

## Approach for Online Decoder

For real-time EEG decoding, two main approaches exist for applying ICA to streaming data:

### Static ICA Approach

**Core Concept**: Train the ICA model on functional localizer data (offline), extract the unmixing matrix, then apply it to the live data stream during online decoding.

**Implementation**:
1. During offline preprocessing, fit ICA on functional localizer EEG data
2. Extract and save the unmixing matrix (weights and sphere)
3. In real-time, apply the unmixing matrix via simple matrix multiplication to incoming data
4. Remove artifact components (based on offline identification)
5. Project cleaned data back to electrode space

**Advantages**:
- **Simple implementation**: Just matrix multiplication on incoming data
- **Low computational cost**: No iterative optimization during runtime
- **Predictable behavior**: Components are fixed and pre-identified
- **Consistency**: Same artifact rejection criteria as offline analysis

**Trade-offs (from instructor guidance)**:
- **Sensitivity to electrode shifts**: If sensors move slightly or electrode impedance changes after the localizer, the fixed matrix may lose accuracy
- **Non-stationarity**: Assumes the sources stay identical over time; if new artifacts appear that weren't in the localizer (e.g., different muscle tension, new environmental noise), a static matrix won't know how to handle them
- **Session-specific**: May require recalibration if experimental setup changes between sessions
- **Limited adaptability**: Cannot adjust to gradual changes in signal properties over time

### Dynamic ICA Approach (ORICA)

**Core Concept**: Continuously update the ICA unmixing matrix during real-time processing, allowing the model to adapt to changing signal conditions.

**Implementation**:
- Use Online Recursive ICA (ORICA) algorithm
- Incrementally update unmixing matrix as new data arrives
- Adaptively identify and track artifact components
- Continuously apply updated separation to streaming data

**Advantages**:
- **Adapts to changing conditions**: Handles electrode shifts and impedance changes
- **Manages non-stationarity**: Can identify new artifacts that emerge during recording
- **Robust to long sessions**: Maintains performance despite gradual signal drift
- **No calibration required**: Learns directly from streaming data

**Trade-offs**:
- **Higher computational cost**: Requires iterative updates with each data block
- **Complexity**: More sophisticated implementation than static approach
- **Initial convergence**: May require warm-up period before components stabilize
- **Component tracking**: Artifact components may need to be re-identified as matrix updates

## Technical Implementations

### ORICA (MATLAB)

**Repository**: https://github.com/goodshawn12/orica

**Description**:
ORICA is a MATLAB-based implementation of online recursive ICA specifically designed for real-time adaptive blind source separation of high-density EEG data.

**Key Features**:
- Real-time processing capability that operates incrementally as new data arrives
- High-density EEG support for complex multi-channel neurological signals
- Computational efficiency optimized for practical deployment
- Integration with EEGLAB, BCILAB, and REST (Real-time EEG Source-mapping Toolbox)

**Technical Approach**:
- Uses online recursive least squares (RLS) whitening
- Recursive independent component analysis
- Instantaneous incremental convergence upon presentation of new data
- Eliminates traditional ICA stationarity assumptions
- Reduces data requirements for convergence

**Applications**:
- Artifact rejection in real-time systems
- Feature extraction for clinical monitoring
- Brain-computer interface signal classification

**Requirements**:
- MATLAB
- Compatible with EEGLAB data formats (.set/.fdt files)
- Includes example test scripts and sample 16-channel EEG data

### SpyICA (Python)

**Repository**: https://github.com/alejoe91/spyica

**Description**:
SpyICA is a Python package originally designed for spike sorting but includes robust implementations of both offline and online ICA algorithms suitable for EEG artifact rejection.

**Key Features**:
- Three operational modes:
  1. **Offline FastICA**: Classical ICA for batch processing
  2. **Offline ORICA**: ORICA applied to complete datasets
  3. **Online ORICA**: True streaming processing mode

**Technical Approach**:
- Implements ORICA algorithm in Python
- Processes data "as if coming as an online stream"
- Adaptive continuous learning from streaming data
- Integration with SpikeInterface ecosystem

**Advantages for This Project**:
- **Python-based**: Easier integration with existing Python/MNE stack
- **Flexible modes**: Can test ORICA offline before deploying online
- **Modern tooling**: Integrates with contemporary Python neuroscience tools
- **Active development**: Part of broader spike sorting ecosystem

**Requirements**:
- Python
- Compatible with SpikeInterface RecordingExtractor objects
- Returns standard SortingExtractor results

## Educational Resources

### Practical Guide for EEG ICA
**Video**: [Practical Guide for EEG ICA](https://www.youtube.com/watch?v=AKCK7DXa0gY&t=1305s)

A hands-on tutorial covering:
- ICA fundamentals for EEG
- Artifact identification
- Component interpretation
- Practical implementation tips

### In-Depth ICA Series
**Playlist**: [ICA Deep Dive](https://www.youtube.com/watch?v=kWAjhXr7pT4&list=PLXc9qfVbMMN2uDadxZ_OEsHjzcRtlLNxc)

Comprehensive series covering:
- Mathematical foundations of ICA
- Different ICA algorithms (FastICA, Infomax, etc.)
- Applications to EEG
- Advanced topics and edge cases

## Recommendations for This Project

### For Offline Stage (Current Implementation)

**Approach**: Standard ICA as implemented in MNE-Python

**Workflow**:
1. Load preprocessed epochs from functional localizer
2. Fit ICA on concatenated data (recommended: at least 1 minute of data)
3. Manually inspect components using MNE visualization tools
4. Identify artifact components (eye blinks, muscle, cardiac)
5. Apply ICA to remove artifacts from all datasets
6. Save cleaned data and ICA solution

**Current Status**: Implemented in the parent [`reactivation-decoder`](https://github.com/roiguri/reactivation-decoder) repo (offline pipeline).

### For Online Stage (Planned Implementation)

**Decision Needed**: Static vs Dynamic ICA

#### Option A: Static ICA Approach (Recommended for Initial Implementation)

**Rationale**:
- Simpler to implement and debug
- Consistent with offline preprocessing
- Minimal computational overhead
- Predictable performance

**Implementation**:
1. Use ICA solution from functional localizer preprocessing
2. Load unmixing matrix at start of online session
3. Apply to incoming LSL data stream via matrix multiplication
4. Remove pre-identified artifact components
5. Project back to electrode space

**When to reconsider**:
- If electrode impedances change significantly during session
- If new artifact sources emerge not present in localizer
- If sessions are very long (>2 hours) with substantial drift

---

## Bad Channel Interpolation for Online Preprocessing

This section covers how bad channel interpolation — a step that happens before ICA in the offline pipeline — is handled at inference time. It is closely related to the static ICA approach above because both involve freezing a transform at Phase 1 and reapplying it online as a matrix multiply.

### Why this problem exists

In the offline pipeline, MNE detects flat or noisy EEG channels and replaces them using spherical spline interpolation. The ICA model is then trained on data where those bad channels already carry interpolated signals. Online, the live EEG stream will include those same bad channels with real (degraded) signals. If we feed those directly into the ICA transform, the distribution no longer matches what the model was trained on.

The fix is to replicate the same interpolation online. The problem is that MNE's `interpolate_bads()` is not available at inference time, and running it on every micro-batch would be slow and require the full MNE stack.

### Why a precomputed weight matrix works

Spherical spline interpolation is a **linear operation**: the interpolated value of a bad channel at any time point is a fixed weighted sum of the good channels at the same time point. The weights depend only on the electrode positions on the scalp, not on the signal values. This means:

- The same weight vector applies at every time point
- The weights never change during a session
- The entire interpolation for all bad channels reduces to a single matrix multiply:

```
X[:, bad_indices] = X[:, good_indices] @ W
```

where `W` is a precomputed matrix of shape `(n_good_channels, n_bad_channels)`.

### How the weight matrix is extracted: the identity-basis trick

MNE does not expose the interpolation weights directly, so they are extracted using **basis vector probing** (a standard numerical linear algebra technique). The idea: if you feed MNE a signal where exactly one good channel has value 1 and all others are 0, the bad channel's output is the weight for that good channel.

To extract all weights at once, feed an identity matrix as the signal (one column per channel, one row per time point):

```python
identity_data = np.eye(n_eeg)   # shape: (n_channels, n_channels)
                                 # at time t, only channel t = 1
```

After running `interpolate_bads()` on this data:
- `interp_data[bad_k, t]` = weight of channel `t` on bad channel `k`

Reading the bad-channel rows at the good-channel time points gives `W`.

**Implementation** in `OfflinePreprocessor._compute_interp_weights()` (`src/backend/offline_phase/preprocessor.py`):

```python
eeg_info = mne.pick_info(self.raw.info, sel=eeg_picks)   # preserves positions
test_raw = mne.io.RawArray(np.eye(n_eeg), eeg_info)
test_raw.info["bads"] = list(self._bad_channels)
test_raw.interpolate_bads(reset_bads=False)

interp_data = test_raw.get_data()                         # (n_eeg, n_eeg)
W = interp_data[np.ix_(bad_local_indices, good_local_indices)].T  # (n_good, n_bad)
```

`W` is stored in the artifact under `online_state["interp_weights"]` and applied by `OnlinePreprocessor` on every micro-batch.

### Is this wasteful?

No. The identity trick runs once, offline, during artifact export. The input is 64×64 floats (~32 KB). At inference time the cost is a single `(n_samples, n_good) @ (n_good, n_bad)` multiply — the cheapest possible operation.

### Alternative approaches and their tradeoffs

| Approach | How | Tradeoff |
|---|---|---|
| **Identity-basis trick** (chosen) | Feed identity matrix to `interpolate_bads()`, read weights | Public API only; guaranteed correct; trivially cheap at runtime |
| **MNE private API** | Call `mne.channels.interpolation._make_interpolation_matrix(pos_good, pos_bad)` directly | More explicit, but private API — may break on MNE version changes |
| **Spherical splines from scratch** | Implement Perrin et al. (1989) weight formula using electrode positions | Transparent and zero MNE dependency, but ~30 lines of non-trivial math requiring careful validation against MNE output |
| **Zero out bad channels** | Set bad channel values to 0 each batch | Simplest, but creates distributional mismatch: ICA was trained on interpolated data, not zeros |

The identity trick was chosen because it uses only the public `interpolate_bads()` call, meaning it is guaranteed to produce exactly the same result as the offline pipeline regardless of how MNE implements the interpolation internally.

---

#### Option B: Dynamic ICA Approach (For Robust Long-Term Use)

**Rationale**:
- More robust to changing conditions
- Handles non-stationarity inherently
- Better for long experimental sessions
- Adapts to individual subject variability

**Implementation Options**:
1. **SpyICA** (Python):
   - Easier integration with Python/MNE stack
   - Can test offline ORICA mode first
   - Modern, maintained codebase

2. **ORICA** (MATLAB):
   - More mature, extensively validated
   - Integrated with EEGLAB ecosystem
   - Would require MATLAB in runtime stack

**Considerations**:
- Requires more development and testing time
- Higher computational cost during runtime
- Need strategy for initial component identification
- May require warm-up period before stable performance

## What This Document Doesn't Cover

- **Detailed ICA mathematics and theory**: For theoretical foundations, see the educational resources above
- **MNE-Python ICA implementation details**: See [MNE documentation](https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html) for API details
- **Specific ICA component interpretation**: See [tomer_preprocessing.md](tomer_preprocessing.md) for project-specific preprocessing notes
- **Hardware-specific artifact patterns**: See [Lab Equipment & LSL.md](../01_timeline/03_online_stage_design/Lab%20Equipment%20%26%20LSL.md) for equipment-related artifacts
- **Offline decoder implementation**: See the parent [`reactivation-decoder`](https://github.com/roiguri/reactivation-decoder) repo's `src/` for the offline preprocessing pipeline
- **Online system architecture**: See [Historical Online System Architecture.md](../01_timeline/03_online_stage_design/Historical%20Online%20System%20Architecture.md) for full real-time processing pipeline context
- **`OnlinePreprocessor` implementation details**: See `docs/OnlinePreprocessor_Implementation_Plan.md` for the full commit-by-commit plan
