# Adding To The Knowledge Base

Back to the [Project Index](START_HERE.md).

This file is written so an agent can be told:

`Add this document to the knowledge base`

and know where it belongs and what to update.

## Main Rule

Do not add documents by topic alone. Add them by role in the project.

Classify every new item into one of these buckets:

1. timeline
2. reference
3. codebase explanation

## Step 1: Decide The Bucket

Use `01_timeline/` if the document is part of the project story or project decisions over time.

Examples:

- project plans
- milestone decks
- design decisions
- PRDs
- architecture documents
- meeting artifacts that define the direction of the project

Use `02_reference/` if the document supports implementation but is not itself the project story.

Examples:

- data specifications
- trigger maps
- preprocessing notes
- parameter files
- experiment documentation

Use `03_codebase/` if the document explains the structure of the repo or how code is organized.

Examples:

- explanations of what `src/` is
- explanations of future online-stage placement
- notes about how to navigate the code

## Step 2: If It Is Timeline Material, Choose The Phase

Inside `01_timeline/`, place the document in the phase it belongs to:

- `01_work_plan/` = original scope and planning
- `02_mid_term/` = checkpoint / transition material
- `03_online_stage_design/` = current online-stage planning, PRD, architecture, and hardware/integration docs

If a new document clearly belongs to one of those phases, put it there.

If it introduces a genuinely new phase that does not fit the current three-phase story, do this:

1. create a new numbered phase folder under `01_timeline/`
2. add a short `README.md` for that phase
3. update [01_timeline/README.md](01_timeline/README.md)
4. update [START_HERE.md](START_HERE.md)

Do not create a new phase folder unless the document really represents a new stage of the project.

## Step 3: Place The File

Rules:

- keep the original document if it is already a useful artifact
- use a clear filename
- do not duplicate the same document in multiple places
- prefer putting the file directly in the correct folder rather than creating unnecessary subfolders

If the document arrives with a messy filename, rename it only if the new name is clearly better and still recognizable.

## Step 4: Update The Local README

After adding a document, update the nearest README in that section so the file is discoverable.

Examples:

- if you add a PRD to `01_timeline/03_online_stage_design/`, update [01_timeline/03_online_stage_design/README.md](01_timeline/03_online_stage_design/README.md)
- if you add a new reference note, update [02_reference/README.md](02_reference/README.md)

Add only a short line:

- file name
- what it is
- why it matters

## Step 5: Update The Top-Level Index Only If Needed

Update [PROJECT INDEX - start here](PROJECT%20INDEX%20-%20start%20here) only when the new item changes navigation or project structure.

Examples that should update the top-level index:

- a new phase folder
- a new major “current source of truth” document
- a new top-level knowledge-base guide

Examples that usually should not update the top-level index:

- one more reference note
- one more document inside an existing phase folder

## Special Rule For Codebase Changes

If the repo itself changes in a way that affects the project story, update the codebase docs too.

Important current structure:

- This repo is the online stage of the project (standalone PyQt6 app). Its source lives in `src/`, implementation docs in `docs/`.
- The parent `reactivation-decoder` repo holds the semester-A offline pipeline; this knowledge base keeps historical context for it in [03_codebase/offline_architecture.md](03_codebase/offline_architecture.md).

When new top-level directories are created or existing ones change significantly, update:

- [03_codebase/README.md](03_codebase/README.md)
- [03_codebase/offline_architecture.md](03_codebase/offline_architecture.md) (for parent-repo offline context)
- [03_codebase/online_architecture.md](03_codebase/online_architecture.md) (for online stage changes)
- [START_HERE.md](START_HERE.md)

## When To Update Existing Docs

Update existing knowledge base documents when:

1. **New phase starts**: Add supersession notes to previous phase documents
2. **Code structure changes**: Update 03_codebase/ when new top-level directories are added
3. **Hardware/integration changes**: Update Lab Equipment & LSL doc and add notes to superseded sections
4. **Dependencies change**: Update references to libraries, tools, or external systems
5. **Architectural decisions change**: Add supersession notes and decision history
6. **New document supersedes old**: Mark the old document with a supersession note

When adding supersession notes, use this template:

```markdown
> **Historical Document**: This reflects [context]. For current direction, see [link](path)
```

## Minimal Agent Checklist

When an agent adds a new document, it should do all of these:

1. classify the document
2. place it in the correct folder
3. update the nearest README
4. update the top-level index only if navigation changed
5. avoid inventing a new category unless the existing structure truly does not fit
6. check if the new document makes older documents outdated and add supersession notes if needed

## Default Behavior

If unsure:

- prefer an existing folder over creating a new one
- prefer `02_reference/` over mislabeling something as a project-phase document
- prefer updating a local README instead of expanding the top-level index
