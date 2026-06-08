# Historical Online System Architecture

> **Historical Design Note**: This document records an earlier target architecture. It is **not** the current implementation contract for this app. Some parts of this document, including older buffering assumptions, are now obsolete. For the current committed backend surface and active Phase 2 direction, see [../../../docs/backend_architecture.md](../../../docs/architecture/backend_architecture.md).

Documentation of the real-time inference system architecture for the Reactivation Decoder project.

## Overview

The online system extends the offline decoding pipeline into a real-time application. After training category-specific decoders on functional localizer data (Phase 1), the system streams live EEG, preprocesses it, and runs continuous inference to detect memory reactivation events (Phase 2). When reactivation is detected, the system emits a digital trigger that can be consumed by other lab software for closed-loop interventions.

---

## 1. Physical Hardware Chain

The signal chain from brain to digital data:

1. **EEG Cap** (Easycap): Electrodes on the participant's scalp pick up microvolt-level voltage fluctuations from neural activity.
2. **Amplifier** (Bittium NeurOne Tesla): The cap connects via wires to the amplifier, which digitizes the analog electrode voltages at a fixed sampling rate. The amplifier connects to the NeurOne Main Unit via fiber optic cables.
3. **NeurOne Main Unit**: The central hub. Communicates with amplifiers and with the acquisition computer. Has **two separate Ethernet ports**:
   - **Control Ethernet** (IP: `192.168.200.200`): Connects to the acquisition/lab computer for recording, display, and control.
   - **DigiOut Ethernet** (IP: `192.168.200.220`): A dedicated real-time output port that streams a copy of the EEG data as UDP packets to a second computer. This is the path used for real-time applications.
4. **Acquisition Computer** (IP: `192.168.200.201`): Runs NeurOne recording software. Connected to the Control Ethernet port. Handles signal visualization, impedance checking, recording to disk, and experiment protocol management.
5. **Processing Computer / Laptop** (IP: `192.168.200.240`): Runs the decoder application (and the LSL proxy if used). Connected to the DigiOut Ethernet port via a dedicated cable.

**Note**: A built-in Ethernet port is recommended on the processing computer. USB-to-Ethernet adapters can be problematic for real-time streaming (per Bittium's guidance).

---

## 2. NeurOne DigiOut — Real-Time Data Output

### How It Works

NeurOne does **not** natively output LSL streams. Instead, the NeurOne Main Unit has a dedicated hardware module called **DigiOut** that streams EEG data as raw **UDP packets** in a proprietary binary format. This is a hardware-level feature — data is pushed directly from the Main Unit's pipeline, ensuring deterministic latency of <3ms with no jitter.

The DigiOut sends UDP packets containing channel-interleaved sample bundles to a configured target IP address and port. The protocol is fully documented in the NeurOne User Manual (Appendix 10).

### DigiOut Configuration (in NeurOne Software)

Configured under Protocol → Real-time Out tab → Digital Out:

| Setting | Value |
|---|---|
| Target IP Address | `192.168.200.240` (the processing computer) |
| Target UDP Port | `50000` |
| Packet Frequency | 1000 Hz |
| Send MeasurementStart/End packets | Enabled |
| Inputs | Select desired channels for streaming |

### UDP Packet Format (Summary)

All fields are big-endian. Key packet types:

- **MeasurementStart Packet** (type 1): Sent when measurement begins. Contains sampling rate, channel count, channel-to-input mapping, scaling info, and trigger definitions.
- **Samples Packet** (type 2): Contains channel-interleaved EEG samples as 24-bit integers (int24). Header includes packet sequence number, channel count, sample count, first sample index, and timestamp.
- **Trigger Packet**: Contains trigger event information.

Sample values require scaling based on amplifier type and channel mode (Tesla AC: divide by 20, Tesla DC: divide by 100).

### Existing Open-Source Implementations

Several open-source projects have implemented NeurOne DigiOut UDP receivers:

- **EStiMo** (Python): Open-source TMS monitoring toolbox with a complete NeurOne UDP receiver, ring buffer, and dropped packet detection. Repository: https://nugit.drcmr.dk/Tools/EStiMo — key file: `connection/NeurOne.py`. Uses EasyCap montages (same cap system as our lab).
- **NeuroSimo** (C++ with Python): Open-source closed-loop TMS platform. Receives NeurOne DigiOut directly via UDP.
- **BEST Toolbox** (MATLAB): Open-source brain stimulation experiment toolbox with NeurOne support.

All published real-time NeurOne research uses DigiOut UDP directly — no published work was found using LSL with NeurOne.

---

## 3. EEG Data Path — Two Options

### Option A: LSL Proxy (Confirmed Available)

Bittium provides an official **LSL proxy application** (`LSLproxy.exe`) that bridges DigiOut UDP to LSL. It runs on the processing computer, receives the UDP packets from DigiOut, and republishes them as a standard LSL stream.

**Data path**: NeurOne DigiOut → UDP → `LSLproxy.exe` → LSL stream → Decoder application (via `pylsl`)

**Setup** (instructions from Bittium support):
1. Unzip the LSL proxy into a folder on the processing computer
2. Connect an Ethernet cable from NeurOne DigiOut port to the processing computer (use built-in network card, not USB adapter)
3. Set the processing computer's IP to `192.168.200.240`
4. In NeurOne software: go to Protocol → Real-time Out tab, enable Digital, add desired inputs, configure settings (target IP `192.168.200.240`, UDP port `50000`, packet frequency `1000`, enable MeasurementStart/End packets)
5. Disable firewall on the processing computer
6. Run `LSLproxy.exe`
7. Start NeurOne measurement
8. The LSL proxy should display: "MEASUREMENTSTART", block size, and trigger info

**Advantages**: Standard LSL interface. The decoder app uses `pylsl` for all data access. Clean separation of concerns.

**Disadvantages**: Extra conversion step adds latency. Dependency on a proprietary executable. Limited debugging if issues arise.

**Status**: Confirmed available from Bittium. Download link pending from Simo-Pekka Simonaho (Bittium support).

### Option B: Direct UDP (Fallback)

The decoder application receives NeurOne's UDP packets directly, parsing the binary protocol itself. LSL is still used for PsychoPy markers and trigger output — only the EEG input bypasses LSL.

**Data path**: NeurOne DigiOut → UDP → Decoder application (custom receiver)

**Advantages**: No proxy dependency. One fewer hop on the EEG path. Full control over packet handling. Proven approach used by all published NeurOne real-time research. Reference implementation available in EStiMo's `NeurOne.py` (~200 lines of Python).

**Disadvantages**: Must implement and maintain the UDP receiver and binary parser.

**Decision**: Try Option A first (LSL proxy). Fall back to Option B if the proxy is unreliable or adds unacceptable latency. The EStiMo `NeurOne.py` code provides a ready-made starting point for Option B.

---

## 4. Lab Streaming Layer (LSL)

### What It Is

LSL (Lab Streaming Layer) is an open-source library for real-time streaming of time-series data between applications. It is not a standalone application — it is a code library that programs link against. Any program can use LSL to publish data (create an **outlet**) or receive data (open an **inlet**).

### How It Works

LSL operates over network sockets. When a program creates an LSL outlet, it opens a network socket and continuously pushes data samples into it. The stream exists only in memory and network buffers — nothing is written to disk.

LSL provides a built-in **discovery mechanism**: any program on the same machine or local network can query "what LSL streams are available?" and get back a list of active streams with their metadata (name, type, channel count, sampling rate).

LSL also handles **clock synchronization** between machines and maintains internal buffers on both sides, so brief processing delays don't cause data loss.

### Role in Our System

Regardless of whether Option A or B is used for EEG input, LSL is used for:

- **PsychoPy marker stream**: PsychoPy has native LSL support and sends event markers via LSL.
- **Decoder trigger output**: The decoder publishes trigger events via LSL for PsychoPy or other lab software to consume.
- **EEG stream** (Option A only): If the LSL proxy is used, the EEG data is also available as an LSL stream.

---

## 5. Machine Configuration (Confirmed)

The NeurOne hardware design dictates a **two-machine setup**:

- **Machine 1 — Acquisition Computer** (lab computer): Runs NeurOne software and PsychoPy. Connected to the NeurOne Control Ethernet port. Handles recording, display, and stimulus presentation.
- **Machine 2 — Processing Computer** (laptop): Runs the decoder application (and LSL proxy if using Option A). Connected to the NeurOne DigiOut Ethernet port via a dedicated cable.

These two machines are **not directly connected to each other**. They each have their own dedicated cable to a different port on the NeurOne Main Unit. Communication between the decoder app and PsychoPy (markers, triggers) occurs via LSL over a separate network connection (e.g., WiFi, or a second Ethernet link through a switch).

### Network Topology

```
                         ┌──────────────────────┐
                         │  NeurOne Main Unit    │
                         │                       │
                         │  [Control Ethernet]───┼──── Cable 1 ──── Acquisition Computer
                         │   192.168.200.200     │                  192.168.200.201
                         │                       │                  • NeurOne software
                         │                       │                  • PsychoPy
                         │                       │                  • Recording to disk
                         │                       │
                         │  [DigiOut Ethernet]────┼──── Cable 2 ──── Processing Computer
                         │   192.168.200.220     │                  192.168.200.240
                         │                       │                  • LSLproxy.exe (Option A)
                         │                       │                  • Decoder application
                         └──────────────────────┘

PsychoPy ◄──── LSL (markers/triggers) ────► Decoder Application
         (over WiFi or separate network link)
```

### Where PsychoPy Runs

PsychoPy runs on the **acquisition computer** (Machine 1). This is because it needs to send hardware triggers to NeurOne (for embedding event markers in the EEG recording) and because it controls stimulus presentation to the participant.

### PsychoPy ↔ Decoder Communication

Since PsychoPy and the decoder run on different machines, they need a network path for LSL marker and trigger streams. Options include WiFi (both on the same WiFi network), a USB Ethernet adapter on the processing computer creating a second network interface, or a network switch connecting both machines. This is separate from the DigiOut Ethernet connection which is dedicated to EEG data.

**Decision Pending**: How to establish the network link between PsychoPy and the decoder application.

---

## 6. Software Components and Data Flow

### 6.1 NeurOne Software (Acquisition Computer)

Runs on the acquisition computer. Controls the amplifier, displays live signals, records raw data to disk. Configures the DigiOut settings for real-time streaming.

### 6.2 PsychoPy (Acquisition Computer)

Presents the experimental paradigm (e.g., retrieval cues) to the participant. Sends event markers via an LSL outlet. Can listen to the decoder's trigger stream to implement closed-loop responses.

### 6.3 LSL Proxy (Processing Computer, Option A only)

`LSLproxy.exe` provided by Bittium. Receives DigiOut UDP packets and republishes as an LSL stream. Runs as a standalone background process.

### 6.4 Decoder Application (Processing Computer)

The central component. Internally consists of:

- **Data Input Layer**: Receives EEG data — either via LSL inlet (Option A) or direct UDP socket (Option B). Also receives PsychoPy markers via LSL.
- **Ring Buffer**: A circular buffer that accumulates incoming EEG samples. Must hold at least enough data for the preprocessing window (smoothing duration), typically a few seconds.
- **Real-time Preprocessor**: Replicates the offline preprocessing pipeline in real time — channel selection (EEG only, drop EOG), temporal smoothing over the buffer, and feature scaling. Must produce features identical to what the decoder was trained on.
- **Decoder Engine**: Holds the trained per-category models (loaded from Phase 1 output). Takes preprocessed features and runs each category decoder in parallel to produce probability estimates.
- **Decision Logic**: Applies the configurable decision rules — probability threshold, sustained activation duration, and conflict resolution when multiple decoders fire simultaneously.
- **Trigger Output**: Publishes decoder decisions via an LSL outlet for PsychoPy or other lab software.
- **User Interface**: Live visualization showing probability traces, connection/latency status, decoder controls, decision history, and a trigger event log.

### Data Flow (Option A — LSL Proxy)

```
EEG Cap → Amplifier → NeurOne Main Unit
                           │
              ┌────────────┼────────────────┐
              │ Control ETH│   DigiOut ETH  │
              ▼            │                ▼
     Acquisition PC        │       Processing PC
     ┌─────────────┐       │       ┌──────────────────┐
     │ NeurOne SW   │       │       │ LSLproxy.exe     │
     │ (recording)  │       │       │   ▼              │
     │              │       │       │ LSL EEG stream   │
     │ PsychoPy     │───LSL markers──►│   ▼              │
     │              │◄──LSL triggers──│ Decoder App     │
     └─────────────┘       │       │  • Preprocessor  │
                           │       │  • Decoder Engine│
                           │       │  • Decision Logic│
                           │       │  • UI            │
                           │       └──────────────────┘
```

### Data Flow (Option B — Direct UDP)

```
EEG Cap → Amplifier → NeurOne Main Unit
                           │
              ┌────────────┼────────────────┐
              │ Control ETH│   DigiOut ETH  │
              ▼            │                ▼
     Acquisition PC        │       Processing PC
     ┌─────────────┐       │       ┌──────────────────┐
     │ NeurOne SW   │       │       │ UDP Receiver     │
     │ (recording)  │       │       │   ▼              │
     │              │       │       │ Ring Buffer      │
     │ PsychoPy     │───LSL markers──►│   ▼              │
     │              │◄──LSL triggers──│ Decoder App     │
     └─────────────┘       │       │  • Preprocessor  │
                           │       │  • Decoder Engine│
                           │       │  • Decision Logic│
                           │       │  • UI            │
                           │       └──────────────────┘
```

---

## 7. Continuous Sliding Window Decoding

### Approach

Unlike the offline pipeline where epochs are locked to stimulus events, Phase 2 uses a **continuous sliding window**. The system decodes constantly on a rolling basis, regardless of external events, continuously asking whether the current spatial pattern across EEG channels resembles a reactivation signature.

### Mechanism

1. **Buffer Accumulation**: Incoming EEG samples are continuously appended to a ring buffer.
2. **Decode Stride**: At regular intervals (e.g., every 20–50ms), the system extracts the most recent preprocessed data from the buffer.
3. **Feature Extraction**: Temporal smoothing is applied over the buffer contents, then scaling, producing a feature vector of shape `(n_channels,)` — matching the spatial pattern format the decoder was trained on.
4. **Parallel Decoding**: Each per-category decoder processes the feature vector and outputs a class probability.
5. **Decision Evaluation**: Probabilities are checked against the configured threshold and sustained activation rules.

### Preprocessing Consistency

The real-time preprocessor must replicate the offline pipeline exactly (same channel selection, same smoothing method and window, same scaler parameters). Any divergence means the decoder receives features from a different distribution than it was trained on, degrading performance. The scaler parameters (e.g., median/IQR values for RobustScaler) are saved during Phase 1 training and loaded for live use.

### Design Parameters

| Parameter | Description | Typical Range |
|---|---|---|
| Decode stride | How often a new probability estimate is produced | 20–50 ms |
| Buffer duration | How much EEG history is maintained | 2–5 seconds |
| Smoothing window | Temporal smoothing applied before feature extraction | Matches offline (e.g., 40ms) |
| Probability threshold | Minimum decoder confidence to trigger | Configurable (e.g., 0.85) |
| Sustained duration | How long threshold must be exceeded continuously | Configurable |

**Decision Pending**: Decode stride value. Shorter stride gives smoother probability traces and faster reaction but increases CPU load.

---

## 8. Latency Budget

**Requirement**: Total latency from EEG acquisition to decoder decision must remain ≤ 100ms.

| Stage | Estimated Latency |
|---|---|
| Amplifier digitization + NeurOne processing | ~2–3 ms (DigiOut hardware guarantee) |
| DigiOut UDP transmission | <1 ms |
| LSL proxy conversion (Option A only) | ~1–2 ms |
| Buffer + windowing | Depends on stride |
| Preprocessing (smoothing, scaling) | ~1–5 ms |
| Decoder inference (all categories) | ~1–5 ms |
| Decision logic + trigger output | <1 ms |

The dominant variable is the decode stride — if you decode every 50ms, the worst-case "waiting" latency before your newest data gets processed is 50ms. Stride selection should account for this within the overall 100ms budget.

---

## 9. LSL Streams Specification

### EEG Stream (Option A only — from LSL Proxy)

- **Source**: `LSLproxy.exe` on processing computer
- **Type**: TBD (will be determined when proxy is tested)
- **Channels**: Depends on DigiOut configuration in NeurOne
- **Sampling Rate**: Matches DigiOut packet frequency (1000 Hz as configured)

### Marker Stream (from PsychoPy)

> **Decision Pending**: Stream format and content need to be verified based on the PsychoPy experiment design.

- **Type**: `Markers` (tentative)
- **Channels**: 1 (string markers) (tentative)
- **Sampling Rate**: Irregular (event-driven)
- **Content**: Experimental event codes — specific marker names TBD based on the paradigm

### Trigger Stream (from Decoder Application)

> **Decision Pending**: Trigger format, naming convention, and content need to be decided.

- **Type**: `Markers` (tentative)
- **Channels**: 1 (string markers) (tentative)
- **Sampling Rate**: Irregular (event-driven)
- **Content**: Decoder trigger events — specific trigger codes and format TBD

---

## 10. Reference Resources

| Resource | Type | Relevance |
|---|---|---|
| NeurOne User Manual (Appendix 10) | Documentation | DigiOut UDP protocol specification |
| EStiMo toolbox (`connection/NeurOne.py`) | Python source | Complete NeurOne UDP receiver implementation (~200 lines) |
| NeuroSimo | C++/Python source | Alternative NeurOne receiver + closed-loop framework |
| BEST Toolbox | MATLAB source | Another NeurOne real-time integration reference |
| Bittium support (Simo-Pekka Simonaho) | Contact | LSL proxy provider, open support request |
| `lsl_test.py` | Our script | LSL stream discovery and data verification |

---

## 11. Open Decisions

| Decision | Options | Impact |
|---|---|---|
| EEG data path | Option A (LSL proxy) vs Option B (direct UDP) | Architecture complexity, latency, dependencies |
| Sampling rate | 500 Hz vs 1000 Hz | Processing load, temporal resolution |
| Decode stride | 20ms / 30ms / 50ms | Probability trace smoothness, CPU load, latency |
| PsychoPy ↔ Decoder network link | WiFi / second Ethernet / switch | Marker and trigger latency, reliability |
| Marker stream usage | Logging only vs gating (decode only during retrieval periods) | Computational savings, experimental flexibility |
| PsychoPy marker format | Stream type, channel format, marker names | Integration with decoder app, event logging |
| Trigger output format | Stream type, channel format, trigger codes | Integration with PsychoPy and recording systems |
| Processing computer | Which laptop/PC to use, built-in Ethernet required | Hardware procurement |

---

## 12. Lab Equipment Notes

Confirmed equipment in the lab (from lab visit):

- Bittium NeurOne Tesla EEG System with DigiOut hardware module
- Easycap EEG Recording Caps
- sync2brain bossdevice (already connected to DigiOut — confirms DigiOut is functional)
- NeurOne software on acquisition computer (IP: `192.168.200.201`)
- PsychoPy installed (version TBD)

## What This Document Doesn't Cover

- **Implementation details and component interfaces**: Detailed Python class interfaces, method signatures, and data flow specifics (see [../../../docs/backend_architecture.md](../../../docs/architecture/backend_architecture.md))
- **UI/UX design**: User interface screens, workflows, and visual prototypes (see [Reactivation Decoder PRD.md](Reactivation%20Decoder%20PRD.md))
- **Behavioral experiment design**: BMR experiment stages, feature space, and output file formats (see [../../02_reference/BMR Data Specification.md](../../02_reference/BMR%20Data%20Specification.md))
- **Offline decoder implementation**: See the parent [`reactivation-decoder`](https://github.com/roiguri/reactivation-decoder) repo's `src/` for the existing semester-A analysis pipeline.
- **Preprocessing parameter details**: Specific filtering, epoching, and artifact rejection parameters used in offline analysis (see [../../02_reference/tomer_preprocessing.md](../../02_reference/tomer_preprocessing.md))
