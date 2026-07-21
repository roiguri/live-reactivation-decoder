# Replacing the decoder

The app is built so the **hull** (acquisition, streaming, decisions, logging, and
the UI) is separate from the **brain** (the decoder that turns EEG into
predictions). You can replace the brain, anywhere from a different model family up
to a completely different decoding paradigm, without touching the hull, as long as
you honor one small contract. This page describes that contract and the levels of
change it supports.

The current brain is a **per-timepoint spatial classifier**: one decoder per task,
each trained at a single post-stimulus timepoint, applied independently to every
incoming sample. That is a choice, not a constraint of the architecture. See
[backend.md](backend.md) for the full backend picture.

## The hull and the brain

The hull is fixed and does not care what the brain is:

```
LSLReceiver -> StreamWorker micro-batch loop -> prediction_ready
                                                     |
                              DecisionEngine, LiveSessionLogger, live charts
```

Everything downstream of `prediction_ready` (the decision layer, the logger, the
charts) consumes only a **per-decoder probability stream**. Whatever produces that
stream is the brain, and it is the only thing you replace.

## The contract

There are three touch points, and a new brain only has to satisfy them.

1. **Into the brain.** `OnlinePreprocessor.process_batch(eeg, timestamps)` returns
   `(features, timestamps)`, where `features` has shape `(n_samples, n_channels)`,
   one spatial vector per decimated timepoint.
2. **The brain.** `LiveInferenceEngine.predict(features)` returns
   `{task_name: probability_vector}`, one probability per input row. The models it
   holds are **duck-typed**: any object exposing `predict_proba(2D) -> 2D` and a
   `classes_` attribute qualifies. Nothing here is tied to scikit-learn.
3. **Out to the hull.** The worker emits
   `prediction_ready(dict[name -> np.ndarray], timestamps, markers)`. That per-decoder
   probability stream is the entire boundary the rest of the app sees.

## Three levels of change

**Level 1: a new model, same shape.** Keep the per-timepoint spatial approach and
swap the estimator. Change `decoders.model` (and `params`) in the config, or drop
any object with `predict_proba` and `classes_` (for example a wrapped PyTorch or
Keras model) into the artifact's `models` dict. The hull is untouched.

**Level 2: a new feature space, still per-sample.** Change how features are built
in `OnlinePreprocessor` and mirror it in the offline path, but keep `predict`
returning one probability per sample. The hull is still untouched.

**Level 3: a different paradigm.** Move away from per-timepoint decoding entirely
(temporal windows, sequence models, end-to-end networks). Replace
`LiveInferenceEngine`, or have your model keep an internal rolling buffer, and
replace the offline training so it produces your kind of artifact. A windowed model
can still emit one probability per incoming sample, so even here the
`prediction_ready` contract can stay exactly the same.

## Two ways to plug in a new brain

### Route A: keep the artifact, swap the model

The artifact `decoder_pipeline.joblib` is a dict with three keys: `models`,
`online_state`, and `metadata` (see [backend.md](backend.md)). Produce one whose
`models` are your objects, and `AppSession.build_live_stream_session` loads it
unchanged. This is the right route whenever your brain is still per-sample.

A minimal custom model is just an object with the two required members:

```python
import numpy as np

class MyDecoder:
    """Any object with predict_proba + classes_ is a valid live decoder."""

    classes_ = [0, 1]  # positive class defaults to label 1

    def __init__(self, net):
        self._net = net

    def predict_proba(self, features):
        # features: (n_samples, n_channels) at the online target rate.
        pos = np.asarray(self._net(features)).reshape(-1, 1)  # (n_samples, 1)
        return np.hstack([1.0 - pos, pos])                    # (n_samples, 2)
```

Bundle it into an artifact and save it where Phase 2 expects it:

```python
import joblib

artifact = {
    "models": {"my decoder": MyDecoder(net)},
    "online_state": online_state,  # frozen preprocessing state (below)
    "metadata": {
        "feature_width": n_channels,               # must match features.shape[1]
        "decoding_timepoints": {"my decoder": 0.3},
    },
}
joblib.dump(artifact, "models/decoder_pipeline.joblib")
```

`online_state` is the frozen preprocessing state the default `OnlinePreprocessor`
replays (channel indices, bad-channel interpolation, the fitted ICA matrices, the
pre-whitener). Reuse the one Phase 1 wrote, or, if you also change preprocessing,
supply your own (Route B). To point the app at an existing output folder, use
**Open Live from Existing Output** on the Settings screen.

### Route B: replace the inference engine

For a genuinely different paradigm, provide your own engine and wire it into the
live run the same way `AppSession.build_live_stream_session` does:

```python
worker = StreamWorker(
    receiver=receiver,
    preprocessor=preprocessor,       # or your own preprocessor
    inference_engine=MyEngine(...),  # exposes predict(features) -> {name: probs}
    batch_size_samples=40,
)
```

`MyEngine.predict(features)` must return `{task_name: np.ndarray}` with one
probability per input row. For a windowed or stateful model, keep the rolling
buffer inside the engine and still emit one probability per input sample, so the
`prediction_ready` contract and every hull consumer keep working unchanged.

## What is still coupled to the timepoint approach

The online path is cleanly decoupled, but the **offline path is not**. These parts
assume per-timepoint decoding and would need rework for a different paradigm:

- `metadata.decoding_timepoints` and `feature_width` describe a single-timepoint
  spatial vector.
- The Phase 1 **Model Evaluation** screen (the temporal-generalization curve where
  the operator picks a timepoint) and `ModelTrainer` are built around choosing and
  training at one timepoint.
- The preprocessing recipe is shared by both phases through
  `preprocessing_constants.py`. A new paradigm may want a different recipe.

For a non-timepoint brain, either add a matching Phase 1 training flow, or skip
Phase 1 and hand-produce the artifact as in Route A.
