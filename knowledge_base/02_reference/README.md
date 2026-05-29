# Reference Docs

Back to the [Project Index](../START_HERE.md).

These documents are not the project timeline. They are support material.

## Purpose

Use this section for domain and implementation reference:

- experiment structure
- behavioral data layout
- trigger meanings
- preprocessing notes
- artifact rejection and ICA approaches
- parameter presets from related work

## Files

- [BMR Data Specification.md](BMR%20Data%20Specification.md): the Binding Memory Retrieval experiment data model and output files
- [tomer_params.md](tomer_params.md): parameter and trigger definitions from Tomer's pipeline notes
- [tomer_preprocessing.md](tomer_preprocessing.md): example preprocessing pipeline from Tomer's code
- [ICA_real_time.md](ICA_real_time.md): real-time ICA approaches (static vs dynamic), technical implementations (ORICA, SpyICA), online artifact rejection recommendations, and bad channel interpolation via precomputed weight matrices
- [parallel_port_trigger_decoding.md](parallel_port_trigger_decoding.md): provisional decision record for the offline trigger-extraction approach on BindingDecoding recordings — runs upstream of the preprocessing steps in `tomer_preprocessing.md`
- [online_filtering.md](online_filtering.md): causal vs. zero-phase filtering decision, group delay analysis across EEG bands, alternatives considered, and notch filter approach for online preprocessing
- [ui_demo/](ui_demo/): React screen mockups used as visual design reference for the PyQt6 frontend — Phase1Screen.jsx (training pipeline), Phase2Screen.jsx (live inference), WelcomeScreen.jsx; source repo `https://github.com/roiguri/decoder_gui`

## Relevance

These files are useful inputs for both stages:

- offline: understanding triggers, labels, and preprocessing expectations
- online: aligning live trigger decoding and event semantics with the experiment

But these are reference files, not the main architecture source of truth.
