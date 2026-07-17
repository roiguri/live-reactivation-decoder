# App Walkthrough

## Launch

*The launcher offers two entry points: start a new training run from scratch, or open live inference directly from a folder a previous run already trained into.*

![Launch](../assets/walkthrough/01-launch.png)

## Offline Phase - Training

### Pipeline Settings

*Loads the experiment configuration and output directory, then displays the fixed preprocessing and model-evaluation settings for review before the run begins.*

![Pipeline Settings](../assets/walkthrough/02-settings.png)

### Data Loading

*Selects the recording folder and loads the BrainVision `.vhdr` file into the session.*

![Load Data](../assets/walkthrough/03-load-data.png)

### Preprocessing

*The preprocessing step starts from a single control, then runs filtering, bad-channel marking, ICA, and epoching automatically.*

![Preprocessing - ready](../assets/walkthrough/04a-preprocessing-ready.png)

*MNE's interactive browser, where the operator inspects the raw traces and marks noisy channels to exclude.*

![Preprocessing - bad channels](../assets/walkthrough/04b-preprocessing-badchannels.png)

*The full set of ICA components shown as topomaps, each labelled with its ICLabel category and confidence; the operator toggles which components to remove.*

![Preprocessing - ICA components](../assets/walkthrough/04c-preprocessing-ica.png)

*A summary of the cleaned data: epochs retained per class and the number of ICA components removed.*

![Preprocessing - complete](../assets/walkthrough/04d-preprocessing-complete.png)

### Model Evaluation

*Cross-validation runs across the decoders one at a time - here the animate decoder is complete and the inanimate decoder is running.*

![Evaluation - progress](../assets/walkthrough/05a-eval-progress.png)

*The summary tab: AUC across epoch time for all decoders, where clicking the curve selects the timepoint used for inference.*

![Evaluation - AUC](../assets/walkthrough/05b-eval-auc.png)

*Each trained decoder also gets its own tab, presenting that decoder's AUC-over-time curve alongside its temporal-generalization matrix (train × test).*

![Evaluation - per-decoder AUC and TGM](../assets/walkthrough/05c-eval-tgm.png)

### Train & Save

*The train step ready to run, before the final decoders are fit at the confirmed timepoints.*

![Train & Save - ready](../assets/walkthrough/06a-train-ready.png)

*Trains the final decoders at the selected timepoint and saves the pipeline artifact, with a spatial topomap for each decoder.*

![Train & Save - topomaps](../assets/walkthrough/06b-train.png)

## Online Phase - Live Inference

*The full live screen before inference starts: status header, decoder and decision-settings sidebar, and the empty decision, probability, and event-locked regions awaiting a stream.*

![Live - idle](../assets/walkthrough/07-live-idle.png)

*Available LSL streams on the network are discovered automatically and presented to choose from (here the replayed `NeuroneStream`).*

![Live - target dialog](../assets/walkthrough/08-live-target-dialog.png)

*The full live screen during inference - everything together: status and latency header, decision tiles, streaming probability chart, and event-locked view.*

![Live - running](../assets/walkthrough/09-live-running.png)

*The live header shows the inference status, the selected decode target, and a latency readout (rolling ~1 s average): Pipeline is the compute time to process one micro-batch (preprocessing plus inference), while E2E is the end-to-end latency from a sample arriving to its prediction.*

![Live - header](../assets/walkthrough/10-live-topbar.png)

*The decision tiles above the live probability chart: each decoder's class probability streams in real time, and a tile lights up in the decoder's colour when it latches over threshold.*

![Live - decisions and probabilities](../assets/walkthrough/11-live-decision-probability.png)

*An event-locked view that freezes the decoder outputs around each trigger event, with controls to browse the captured history.*

![Live - event-locked view](../assets/walkthrough/12-live-frozen-event.png)

*The live control sidebar: toggle each decoder's visibility and set the decision threshold and sustain length. The Start/Halt button sits at its foot.*

![Live - settings](../assets/walkthrough/13-live-settings.png)
