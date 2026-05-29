# Project Index

This knowledge base supports the online stage of the reactivation decoder. It is organized around chronology and authority, not just topic.

## Project Context

This repo is the online stage of the project — the standalone PyQt6 app for real-time decoding. Initial work to evaluate the decoding pipeline, done before this app was built, lives in the parent `reactivation-decoder` repo. See [Related Work](#related-work).

## Timeline Order

The project story is:

1. work plan
2. mid-term presentation
3. online PRD and system architecture

That order matters. Older documents stay in the knowledge base, but they do not carry the same authority as the newer online-stage design documents.

## Authority Rules

When documents disagree:

1. newer phase beats older phase
2. implemented code in `src/` beats docs
3. hardware-confirmed notes beat earlier architectural assumptions

## Start Here

- Timeline and document authority: [01_timeline/README.md](01_timeline/README.md)
- Code structure: [03_codebase/README.md](03_codebase/README.md)
- Experiment and preprocessing reference docs: [02_reference/README.md](02_reference/README.md)
- How to add new material: [ADDING_TO_THE_KNOWLEDGE_BASE.md](ADDING_TO_THE_KNOWLEDGE_BASE.md)

## Current Source Of Truth

- Implementation docs: [../docs/](../docs/)
- Current online product/design direction: [01_timeline/03_online_stage_design/README.md](01_timeline/03_online_stage_design/README.md)
- Hardware and LSL integration facts: [01_timeline/03_online_stage_design/Lab%20Equipment%20%26%20LSL.md](01_timeline/03_online_stage_design/Lab%20Equipment%20%26%20LSL.md)

## Folder Layout

```text
repository/
├── src/                    — online app source (frontend + backend)
├── docs/                   — frequently updated implementation docs
├── tests/                  — automated tests
├── scripts/                — utility and replay scripts
├── tools/                  — bundled binaries (LSLProxy)
└── knowledge_base/
    ├── START_HERE.md
    ├── 01_timeline/
    │   ├── 01_work_plan/
    │   ├── 02_mid_term/
    │   └── 03_online_stage_design/
    ├── 02_reference/
    └── 03_codebase/
        ├── README.md
        ├── offline_architecture.md
        └── online_architecture.md
```

## What Each Section Means

- `01_timeline/`: project evolution, with clear separation between historical and current docs
- `02_reference/`: data, trigger, and preprocessing notes that support implementation
- `03_codebase/`: explanation of this repo's structure, plus historical context for the parent repo's offline pipeline
- `../docs/`: frequently updated implementation contracts and plans for the online app

## Related Work

- Parent repo: [`reactivation-decoder`](https://github.com/roiguri/reactivation-decoder) — initial work to evaluate the decoder before this app was built. The semester-A offline evaluation pipeline lives under its `src/`.

## Maintenance

If you are adding a new document to the knowledge base, follow [ADDING_TO_THE_KNOWLEDGE_BASE.md](ADDING_TO_THE_KNOWLEDGE_BASE.md).
