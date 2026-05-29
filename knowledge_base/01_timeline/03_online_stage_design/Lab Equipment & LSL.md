What We Set Out To Do
Understand and build the infrastructure for the online (real-time) stage of the Reactivation Decoder — specifically how live EEG data gets from the amplifier to your decoder application.

What We Learned and Built
Architecture

The online system has two machines: acquisition PC (runs NeurOne software + PsychoPy) and decoding laptop (runs decoder application)
These connect to two separate Ethernet ports on the NeurOne Main Unit — not to each other directly
NeurOne does not natively support LSL. It has its own proprietary real-time output called DigiOut which sends raw UDP packets in a binary format

LSL Proxy

Bittium provides an official LSLproxy.exe that receives NeurOne's DigiOut UDP packets and republishes them as an LSL stream
Configuration is minimal — Settings.ini only has port=50000 and stream_name=NeuroneStream
No channel metadata is embedded in the LSL stream (no channel names)
The proxy applies the Tesla AC scaling (divides by 20) before publishing — so LSL values are already in microvolts

Confirmed Stream Properties (from real hardware)

Stream name: NeuroneStream
Type: EEG
Channels: 65 (64 EEG + 1 trigger channel)
Sampling rate: 1000 Hz (configured in NeurOne — note: appeared as 5000 Hz in one session, needs confirmation)
No channel metadata in the stream

Trigger Channel

Channel 65 (index 64) is the trigger channel, embedded in every EEG sample
Trigger values arrive as large numbers because NeurOne packs multiple trigger sources into one integer using a defined bit layout (from the manual):

Bits 1-2: Isolated A
Bits 3-4: Isolated B
Bits 8-15: Parallel port code ← the PsychoPy trigger


Decoding formula: trigger_code = (int(raw_value) >> 8) & 0xFF
This is confirmed by both the manual's bit layout spec and by observation

PsychoPy Experiment

The experiment (BindingMemory repo) already has parallel port triggers implemented
Trigger codes are defined in ParallelPortEnums in Enums.py
No LSL marker stream needed from PsychoPy — triggers arrive embedded in the EEG stream's trigger channel
This simplifies the architecture: your decoder only needs one LSL inlet (the EEG stream), not two

Tools Built

neurone_simulator.py — sends fake NeurOne DigiOut UDP packets for testing without hardware. Verified against EStiMo's parser and against LSLproxy.exe
lsl_test.py — discovers LSL streams, pulls data, runs sanity checks, runs diagnostics on failure
monitor_triggers.py — monitors channel 65 in real time, prints raw and decoded trigger values
record_lsl.py — records LSL stream to .npz file for offline development
view_lsl.py — live scrolling EEG plot
build_portable.py — builds a self-contained portable Python environment (no install needed on lab laptop)
Various .bat launcher files for all scripts

Lab Infrastructure Confirmed

NeurOne DigiOut hardware module is present (confirmed by bossdevice being connected to it)
LabRecorder is installed on the acquisition PC — can record .xdf files
Decoding laptop has Python 3.12 and pylsl already installed
Full chain tested and working: NeurOne → DigiOut → LSLproxy → LSL → lsl_test


Consequences for the Project
Architecture is simpler than expected:

One LSL inlet for EEG (from LSLproxy)
No separate marker stream — triggers are inside the EEG stream
One LSL outlet for decoder triggers to PsychoPy (still needed for closed-loop)

Preprocessing must handle:

65 channels, with channel 64 being the trigger — must be separated from EEG before preprocessing
No channel labels from the stream — channel-to-electrode mapping must be hardcoded based on NeurOne's Protocol configuration
Sampling rate needs to be confirmed as either 1000 Hz or 5000 Hz — this significantly affects the preprocessing pipeline

Trigger decoding is solved:

(int(raw_value) >> 8) & 0xFF gives the PsychoPy trigger code
Need to get Enums.py from the BindingMemory repo to map codes to events

Open items remaining:

Confirm actual sampling rate being used
Get Enums.py to map trigger codes to experiment events
Get channel mapping from NeurOne Protocol settings
Confirm network path between decoding laptop and acquisition PC for the trigger output LSL stream (currently they're on separate networks)
Record real EEG data with a cap on for offline development
Build the LSL stream playback script for home development
## What This Document Doesn't Cover

- **UI implementation details**: User interface screens and application design (see [Reactivation Decoder PRD.md](Reactivation%20Decoder%20PRD.md))
- **Full system architecture**: Complete data flow and processing pipeline (see [Historical Online System Architecture.md](Historical%20Online%20System%20Architecture.md))
- **Behavioral data formats**: Experiment output files and data structure (see [../../02_reference/BMR Data Specification.md](../../02_reference/BMR%20Data%20Specification.md))
- **Preprocessing algorithms**: Specific signal processing methods beyond basic LSL data acquisition (see [../../02_reference/tomer_preprocessing.md](../../02_reference/tomer_preprocessing.md))
