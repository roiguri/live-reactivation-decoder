# Operator Manual (Quarto book)

The operator manual is a [Quarto](https://quarto.org) book. Sources are plain
Markdown (`.qmd`) — you never write or edit LaTeX. The **PDF is the current
deliverable**; HTML is configured too so the same sources can render an in-app
Help book later with no content changes.

## Structure

```
docs/manual/
├── _quarto.yml          # book config + chapter order (edit this to add chapters)
├── index.qmd            # preface
└── chapters/
    ├── 01-overview.qmd
    ├── 02-installation.qmd
    ├── 03-hardware-lsl.qmd
    ├── 04-phase1-offline.qmd
    ├── 05-phase2-live.qmd
    ├── 06-configuration.qmd
    ├── 07-troubleshooting.qmd
    └── 90-debug-walkthrough.qmd   # appendix
```

To add a chapter: create the `.qmd` file and add its path to the `chapters:`
list in `_quarto.yml`. Numbering, the table of contents, and cross-references
update automatically.

## One-time setup (per machine)

Quarto is a standalone binary — **not** a pip package, so there is nothing to
add to `requirements*.txt` for it.

1. Install the Quarto CLI: <https://quarto.org/docs/get-started/>
   (or `winget install quarto`, `brew install quarto`).
2. Install a TeX distribution for PDF output — Quarto manages a lightweight
   one for you:

   ```bash
   quarto install tinytex
   ```

## Build

From `docs/manual/`:

```bash
quarto render            # builds every configured format (PDF + HTML)
quarto render --to pdf   # PDF only — the milestone-1 deliverable
quarto preview           # live-reloading preview while writing
```

Output is written to `docs/manual/_book/` (git-ignored). The PDF lands at
`_book/Live-Reactivation-Decoder---Operator-Manual.pdf`.
