# Binding Memory Retrieval (BMR) — Data Specification

## Overview

This document describes the data output of the Binding Memory Retrieval (BMR) experiment, a PsychoPy-based neuroscience paradigm that studies how subjects encode and retrieve bindings between objects and contextual features (color and scene). The experiment is designed for EEG/fMRI recording.

The experiment code lives at `github.com/tomermuller/BindingMemory`. It generates all behavioral data automatically — no manual steps are required beyond entering the subject ID at launch.

---

## Experiment Summary

The experiment has 3 stages, run sequentially in a single session:

**Stage 1 — Functional Localizer:** Subjects view repeated images of individual colors and scenes. After each image, a Hebrew word appears and the subject judges whether the word matches the image (right arrow \= match, left arrow \= no match). This establishes baseline neural signatures for each feature in isolation. 6 features (3 colors × 70 repeats \+ 3 scenes × 70 repeats), shuffled with max 2 consecutive same features.

**Stage 2 — Binding Learning \+ Test (5 blocks):** The core task. Each block has 3 phases:

1. **Learning:** 9 trials per block. Subject sees an object (e.g. a shoe) tinted a specific color (e.g. red) composited onto a scene background (e.g. kitchen) for 3 seconds, then rates memorization difficulty (1–5).  
2. **Break game:** \~100 seconds. Subject counts how many times a rectangle becomes brighter. This is a distractor to prevent rehearsal.  
3. **Test:** Same 9 objects shown plain (no color, no scene). Subject has 3 seconds to press a key indicating they remember the binding. If they press: they report what they remember (color / scene / both), then select the specific feature from the 3 options per category.

45 total binding trials (9 per block × 5 blocks), 45 unique objects.

**Stage 3 — Partial Retrieval Test:** Only objects where the subject got BOTH color and scene correct in Stage 2 are used. A cue image (either a generic "color" probe or "scene" probe) is shown first, then the object, then the subject must pick the specific feature for the cued category. This tests whether a partial cue triggers retrieval of the associated feature.

---

## Feature Space

| Category | Features | RGBA (for coloring objects) |
| :---- | :---- | :---- |
| Colors | `red`, `green`, `yellow` | (255,0,0), (0,255,0), (255,255,0) |
| Scenes | `living_room`, `bathroom`, `kitchen` | N/A (background images) |

Note: `blue` exists in the enum definitions (RGBA and Hebrew translation) but is NOT used in the experiment — `COLOR_TO_IMAGE` and `CATEGORY_TO_FEATURES` only include red, green, yellow.

Objects are 45 everyday items (e.g. `shoe`, `closet`, `laptop`, `hat`, `bulb`). Each object appears exactly once across the entire experiment, assigned to one block.

---

## File Structure

All output is saved under `subject_answer/final_data/subject_<id>/`. Every file is generated automatically when the experiment runs.

subject\_answer/final\_data/subject\_\<id\>/

├── functional\_localizer/

│   ├── subject\_\<id\>\_\<time\>\_function\_localizer\_stage.csv

│   └── subject\_\<id\>\_\<time\>\_function\_localizer\_stage.json

├── true\_answers/

│   ├── subject\_\<id\>\_\<time\>\_true\_answers.csv

│   ├── subject\_\<id\>\_\<time\>\_true\_answers.json

│   └── subject\_\<id\>\_\<time\>\_difficulty.json

├── subject\_answer/

│   ├── subject\_\<id\>\_\<time\>\_subject\_answer.csv

│   └── subject\_\<id\>\_\<time\>\_subject\_answer.json

├── combined\_data/

│   └── subject\_\<id\>\_\<time\>\_combined.csv

└── partial\_retrival/

    ├── subject\_\<id\>\_\<time\>\_partial\_retrival.csv

    └── subject\_\<id\>\_\<time\>\_partial\_retrival.json

The `<time>` component is formatted as `MM-DD-YYYY_HH-MM` (e.g. `04-05-2026_11-55`).

Crash recovery backups are written per-trial to `subject_answer/temp/subject_<id>/` but are not part of the final output.

---

## File Descriptions

### 1\. Functional Localizer CSV

**Path:** `functional_localizer/subject_<id>_<time>_function_localizer_stage.csv` **Rows:** \~420–540 (depends on feature count × repeats) **One row per trial.**

| Column | Type | Description |
| :---- | :---- | :---- |
| (index) | int | Pandas row index (unnamed first column) |
| `subject` | str | Subject ID |
| `trial_index` | int | 0-indexed trial number within the localizer |
| `feature` | str | Feature shown: `red`, `green`, `yellow`, `living_room`, `bathroom`, `kitchen` |
| `word_question` | str | The word shown for judgment (same string space as `feature`) |
| `user_answer` | str | `right` (subject said "match") or `left` (subject said "no match") |
| `is_right` | bool | Whether the subject's answer was correct |
| `feature_appear` | timestamp | Image onset |
| `feature_disappear` | timestamp | Image offset (1.5s after onset) |
| `question_appear` | timestamp | Word shown on screen |
| `answer_time` | timestamp | Subject keypress |

**Usage notes:**

- The match/mismatch is randomized 50/50 per trial.  
- Non-matching words are always from the SAME category (e.g. if a red swatch was shown, the false word might be "green" or "yellow", never "kitchen").  
- If the subject answered wrong, a 3-second error message was shown (not reflected in the CSV — only visible in timing gaps).

### 2\. True Answers CSV (Ground Truth)

**Path:** `true_answers/subject_<id>_<time>_true_answers.csv` **Rows:** 45 (one per binding trial) **Records what was actually shown to the subject during learning.**

| Column | Type | Description |
| :---- | :---- | :---- |
| (index) | int | Pandas row index |
| `subject` | str | Subject ID |
| `trial` | int | Global trial number (1–45) |
| `object` | str | Object name (e.g. `closet`, `shoe`) — matches PNG filename stem |
| `colors` | str | Color assigned: `red`, `green`, or `yellow` |
| `scenes` | str | Scene assigned: `living_room`, `bathroom`, or `kitchen` |
| `object_appear` | timestamp | Binding stimulus onset |
| `feature_disappear` | timestamp | Binding stimulus offset (3s after onset) |
| `difficulty_question_appear` | timestamp | Difficulty rating prompt shown |
| `difficulty_answer_time` | timestamp | Subject pressed 1–5 |

### 3\. Difficulty Ratings JSON

**Path:** `true_answers/subject_<id>_<time>_difficulty.json` **Format:** Flat dict mapping trial-in-block index (as string) to rating (int 1–5).

{"0": 3, "1": 4, "2": 5, "3": 4, "4": 4, "5": 2, "6": 5, "7": 5, "8": 4}

Note: Keys are trial-in-block indices (0–8), NOT global trial numbers. This resets each block. The combined CSV resolves this to global trial context.

### 4\. Subject Answer CSV (Test Phase Responses)

**Path:** `subject_answer/subject_<id>_<time>_subject_answer.csv` **Rows:** 45 (one per test trial) **Records the subject's responses during the test phase of Stage 2\.**

| Column | Type | Description |
| :---- | :---- | :---- |
| (index) | int | Pandas row index |
| `subject` | str | Subject ID |
| `trial` | str | Trial label (e.g. `trial_1`, `trial_2`, ...) — note: this is test-phase ordering, NOT learning-phase ordering. Objects are re-shuffled for test. |
| `object` | str | Object name |
| `colors` | str or empty | Color the subject selected, or empty if they didn't report remembering color |
| `scenes` | str or empty | Scene the subject selected, or empty if they didn't report remembering scene |
| `retrival_success` | bool | `TRUE` if subject pressed a key within 3s (indicating they remembered something), `FALSE` if they timed out |
| `retrival_report_color` | bool | Subject claimed to remember the color |
| `retrival_report_scene` | bool | Subject claimed to remember the scene |
| `object_appear` | timestamp | Plain object shown |
| `object_disappear` | timestamp | Object removed, retrieval window starts |
| `start_retrival_time` | timestamp | Same as object\_disappear (retrieval countdown begins) |
| `retrival_time` | timestamp | Subject keypress OR 3s timeout |
| `retrival_question_appear` | timestamp | "What do you remember?" screen shown |
| `retrival_report_time` | timestamp | Subject selected color/scene/both/nothing |
| `colors_question_appear` | timestamp | Color selection screen shown (empty if not asked) |
| `colors_answer_time` | timestamp | Subject picked a color (empty if not asked) |
| `scenes_question_appear` | timestamp | Scene selection screen shown (empty if not asked) |
| `scenes_answer_time` | timestamp | Subject picked a scene (empty if not asked) |

**Important behavioral logic:**

- If `retrival_success` is FALSE → the trial ends immediately, all subsequent columns are empty.  
- If the subject pressed a key but then reported "nothing" (up arrow) → `retrival_report_color` and `retrival_report_scene` are both FALSE, feature columns are empty.  
- If they reported "color only" → only `colors` is filled, `scenes` is empty.  
- If they reported "both" → both `colors` and `scenes` are filled.  
- The order of color vs. scene questions is randomized per trial.

### 5\. Combined Data CSV (Primary Analysis File)

**Path:** `combined_data/subject_<id>_<time>_combined.csv` **Rows:** 45 (one per binding trial) **Merges the ground truth (what was shown) with the test responses (what the subject answered). This is the main file for analysis.**

| Column | Type | Description |
| :---- | :---- | :---- |
| `subject` | str | Subject ID |
| `block` | int | Block index (0–4) |
| `object` | str | Object name |
| `binding_trial` | int | Global trial number (1–45) |
| `binding_trial_in_block` | int | Trial index within block (0–8) |
| `colors` | str | Ground truth color shown |
| `scenes` | str | Ground truth scene shown |
| `difficulty` | int | Difficulty rating (1=easy, 5=hard) |
| `test_trial` | str | Test-phase trial label (e.g. `trial_7`) |
| `subject_color` | str or empty | Color the subject answered |
| `subject_scene` | str or empty | Scene the subject answered |
| `color_correct` | bool | `subject_color` \== `colors` |
| `scene_correct` | bool | `subject_scene` \== `scenes` |
| `both_correct` | bool | Both color and scene correct |
| `color_rt_ms` | int or empty | Response time for color selection (ms) |
| `scene_rt_ms` | int or empty | Response time for scene selection (ms) |
| `first_question` | str or empty | Which feature was asked first: `colors` or `scenes` |
| `color_question_order` | int or empty | 1 if color was asked first, 2 if second |
| `scene_question_order` | int or empty | 1 if scene was asked first, 2 if second |
| `binding_object_appear` | timestamp | Learning phase: binding stimulus onset |
| `binding_feature_disappear` | timestamp | Learning phase: binding stimulus offset |
| `binding_difficulty_question_appear` | timestamp | Learning phase: difficulty prompt |
| `binding_difficulty_answer_time` | timestamp | Learning phase: difficulty keypress |
| `test_object_appear` | timestamp | Test phase: plain object shown |
| `test_object_disappear` | timestamp | Test phase: object removed |
| `test_start_retrival_time` | timestamp | Test phase: retrieval window start |
| `test_retrival_time` | timestamp | Test phase: keypress or timeout |
| `test_retrival_question_appear` | timestamp | Test phase: "what do you remember?" |
| `test_retrival_report_time` | timestamp | Test phase: color/scene/both selected |
| `test_colors_question_appear` | timestamp | Test phase: color options shown |
| `test_colors_answer_time` | timestamp | Test phase: color selected |
| `test_scenes_question_appear` | timestamp | Test phase: scene options shown |
| `test_scenes_answer_time` | timestamp | Test phase: scene selected |

**Key relationships:**

- `binding_trial` is the global ordering (1–45), sequential across blocks.  
- `test_trial` is the test-phase ordering — objects are re-shuffled within each block for the test, so `binding_trial` 1 might be `test_trial` 7\.  
- RT columns are empty when the subject didn't report remembering that feature or timed out.  
- `both_correct` is the filter used to select objects for Stage 3\.

### 6\. Partial Retrieval CSV

**Path:** `partial_retrival/subject_<id>_<time>_partial_retrival.csv` **Rows:** Variable (equals the number of `both_correct == TRUE` trials from Stage 2). Subject 101 had 37; Subject 102 had 31\. **One row per trial.**

| Column | Type | Description |
| :---- | :---- | :---- |
| `subject` | str | Subject ID |
| `trial` | str | Trial label (e.g. `trial_1`) |
| `object` | str | Object name |
| `probe` | str | Which category was cued: `colors` or `scenes` |
| `retrival_success` | bool | Subject pressed within 3s |
| `is_remember` | bool | Subject confirmed they remember (after pressing) |
| `subject_answer` | str or empty | The feature the subject selected for the probed category |
| `probe_appear` | timestamp | Cue image onset |
| `probe_disappear` | timestamp | Cue image offset (1s after onset) |
| `object_appear` | timestamp | Object shown |
| `object_disappear` | timestamp | Object removed |
| `start_retrival_time` | timestamp | Retrieval window start |
| `retrival_time` | timestamp | Keypress or timeout |
| `colors_question_appear` | timestamp | Color options shown (only if probe was `colors`) |
| `colors_answer_time` | timestamp | Color selected |
| `scenes_question_appear` | timestamp | Scene options shown (only if probe was `scenes`) |
| `scenes_answer_time` | timestamp | Scene selected |

**Important:**

- The `probe` column tells you which category was cued. Only that category's question timestamps will be filled — the other pair will be empty.  
- The probe is randomly assigned per trial (`random.choice([colors, scenes])`), so the split is roughly 50/50 but not exact.  
- `subject_answer` contains the feature name the subject picked (e.g. `red`, `kitchen`), NOT whether it was correct. To determine correctness, you must cross-reference with the ground truth in `true_answers` or `combined_data`.  
- Only objects with `both_correct == TRUE` in the combined CSV are included. The `_load_correct_objects` method reads the most recent combined CSV to determine this list.

---

## Timestamp Format

All timestamps follow the pattern: `YYYY-MM-DD_HH-MM-SS.mmm`

Example: `2026-04-05_13-19-10.788`

This is millisecond precision (the code uses Python's `%f` microsecond format but truncates to 3 decimal places).

To parse in Python:

from datetime import datetime

fmt \= "%Y-%m-%d\_%H-%M-%S.%f"

dt \= datetime.strptime("2026-04-05\_13-19-10.788", fmt)

To parse in pandas:

df\['object\_appear'\] \= pd.to\_datetime(df\['object\_appear'\], format="%Y-%m-%d\_%H-%M-%S.%f")

---

## JSON Files

Each CSV has a companion JSON file containing the same data in nested dict format. The JSON preserves the original hierarchical structure:

**Functional localizer JSON** — list of trial dicts:

\[

  {

    "feature": "red",

    "word\_question": "yellow",

    "user\_answer": "left",

    "is\_right": true,

    "trial\_index": 0,

    "trial\_times": {

      "feature\_appear": "2026-04-05\_11-56-16.479",

      "feature\_disappear": "2026-04-05\_11-56-18.000",

      "question\_appear": "2026-04-05\_11-56-19.562",

      "answer\_time": "2026-04-05\_11-56-20.473"

    }

  }

\]

**True answers JSON** — dict keyed by global trial number:

{

  "1": {

    "closet": {"colors": "red", "scenes": "living\_room"},

    "trial\_times": {"object\_appear": "...", "feature\_disappear": "...", ...}

  }

}

**Subject answer JSON** — dict keyed by trial label:

{

  "trial\_1": {

    "shoe": {"colors": "yellow", "scenes": "bathroom"},

    "retrival\_success": true,

    "retrival\_report\_color": true,

    "retrival\_report\_scene": true,

    "trial\_times": {"object\_appear": "...", ...}

  }

}

The CSVs are flattened versions of these structures and are generally easier to work with for analysis.

---

## EEG/fMRI Integration

The experiment sends parallel port trigger codes at every key event. These are NOT stored in the behavioral data files — they are recorded by the external EEG/fMRI acquisition system as event markers in the continuous neural signal.

To align EEG data with behavioral data, match the trigger timestamps (from the behavioral CSVs) to the event markers in the EEG recording. The parallel port address is `0x5EFC`.

### Trigger Code Reference

| Code | Event |
| :---- | :---- |
| 1 | Start baseline recording |
| 2 | Start functional localizer |
| 3 | Start binding learning block |
| 4 | Start test phase block |
| 5 | Start break game |
| 6 | Start partial retrieval |
| 11 | Show red |
| 12 | Show green |
| 13 | Show yellow |
| 16 | Show living room |
| 17 | Show bathroom |
| 18 | Show kitchen |
| 21 | Stop red |
| 22 | Stop green |
| 23 | Stop yellow |
| 26 | Stop living room |
| 27 | Stop bathroom |
| 28 | Stop kitchen |
| 31 | Show attention question |
| 32 | Answer attention question |
| 41 | Show binding trial (learning) |
| 42 | Stop binding trial |
| 43 | Show difficulty question |
| 44 | Answer difficulty question |
| 51 | Show object in test trial |
| 53 | Start retrieval window |
| 54 | Answer during retrieval window |
| 55 | Show retrieval question (what do you remember?) |
| 56 | Answer retrieval question |
| 61 | Show color answer options |
| 62 | Answer color question |
| 66 | Show scene answer options |
| 67 | Answer scene question |
| 71 | Show probe (partial retrieval) |
| 72 | Stop probe |
| 73 | Show partial retrieval remember question |
| 74 | Answer partial retrieval remember question |

---

## Existing Data

The repository includes completed data for 2 subjects:

| Subject | Date | Both Correct (Stage 2\) | Partial Retrieval Trials |
| :---- | :---- | :---- | :---- |
| 101 | 2026-04-05 | 37/45 (82%) | 37 |
| 102 | 2026-04-12 | 31/45 (69%) | 31 |

---

## Object List

45 objects are used per experiment, drawn from a pool of 46 PNGs:

`air conditioner`, `apple`, `baby`, `ball`, `brush`, `bucket`, `build a bear`, `bulb`, `candle`, `cat`, `chair`, `clock`, `closet`, `coat`, `dog`, `dustpan`, `fan`, `flower pot`, `game`, `glasses`, `gloves`, `hat`, `headphones`, `iphone`, `key`, `keyboard`, `ladder`, `laptop`, `lego`, `man`, `mog`, `painting`, `paper`, `phone`, `shirt`, `shoe`, `sock`, `sponge`, `sprey`, `sweeper`, `tie`, `towel`, `trash can`, `wallet`, `wipes`, `woman`

Objects are randomly shuffled per subject and divided into 5 blocks of 9\. Object assignment to blocks varies across subjects.

---

## Quick Start for Analysis

import pandas as pd

\# Load primary analysis file

combined \= pd.read\_csv("subject\_answer/final\_data/subject\_101/combined\_data/subject\_101\_04-05-2026\_11-55\_combined.csv")

\# Overall accuracy

print(f"Color accuracy: {combined\['color\_correct'\].mean():.1%}")

print(f"Scene accuracy: {combined\['scene\_correct'\].mean():.1%}")

print(f"Both accuracy:  {combined\['both\_correct'\].mean():.1%}")

\# Accuracy by block

print(combined.groupby('block')\['both\_correct'\].mean())

\# Accuracy by difficulty

print(combined.groupby('difficulty')\['both\_correct'\].mean())

\# Load partial retrieval

partial \= pd.read\_csv("subject\_answer/final\_data/subject\_101/partial\_retrival/subject\_101\_04-05-2026\_11-55\_partial\_retrival.csv")

\# Cross-reference partial retrieval with ground truth

true\_answers \= pd.read\_csv("subject\_answer/final\_data/subject\_101/true\_answers/subject\_101\_04-05-2026\_11-55\_true\_answers.csv")

partial\_merged \= partial.merge(true\_answers\[\['object', 'colors', 'scenes'\]\], on='object')

\# Check partial retrieval accuracy

color\_probes \= partial\_merged\[partial\_merged\['probe'\] \== 'colors'\]

color\_probes\['correct'\] \= color\_probes\['subject\_answer'\] \== color\_probes\['colors'\]

scene\_probes \= partial\_merged\[partial\_merged\['probe'\] \== 'scenes'\]

scene\_probes\['correct'\] \= scene\_probes\['subject\_answer'\] \== scene\_probes\['scenes'\]  

## What This Document Doesn't Cover

- **EEG preprocessing**: Signal filtering, artifact rejection, epoching, and feature extraction methods (see [tomer_preprocessing.md](tomer_preprocessing.md))
- **Real-time system architecture**: Online decoder implementation and LSL integration (see [../01_timeline/03_online_stage_design/Historical Online System Architecture.md](../01_timeline/03_online_stage_design/Historical%20Online%20System%20Architecture.md))
- **UI/UX design**: Application screens and user workflows (see [../01_timeline/03_online_stage_design/Reactivation Decoder PRD.md](../01_timeline/03_online_stage_design/Reactivation%20Decoder%20PRD.md))
- **Offline decoder implementation**: See the parent [`reactivation-decoder`](https://github.com/roiguri/reactivation-decoder) repo's `src/` for the existing semester-A analysis code.
- **Hardware specifications**: EEG equipment and network configuration (see [../01_timeline/03_online_stage_design/Lab Equipment & LSL.md](../01_timeline/03_online_stage_design/Lab%20Equipment%20%26%20LSL.md))
