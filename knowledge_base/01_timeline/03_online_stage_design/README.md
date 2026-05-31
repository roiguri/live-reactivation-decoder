# Online Stage Design

Back to [Timeline And Authority](../README.md).

## Role In The Project

This is the current planning layer for the online stage.

This folder should drive the next implementation decisions more than the earlier folders do.

Frequently updated implementation contracts and checklists live in [`docs/`](../../../docs/). This folder keeps the higher-level product, hardware, and design context.

## Relevance Today

Status: `current planning`

Use this folder to answer:

- what the online system should do
- how the online UI is intended to work
- how the lab hardware and LSL path actually behave
- what constraints the real-time implementation must respect

## Files And Roles

- [Reactivation Decoder PRD.md](Reactivation%20Decoder%20PRD.md): product flow and UX intent for the online system
- [Historical Online System Architecture.md](Historical%20Online%20System%20Architecture.md): older target architecture and system context, kept for reference
- [Lab Equipment & LSL.md](Lab%20Equipment%20%26%20LSL.md): confirmed lab-side operational facts and integration notes
- [Phase2_Implementation_Plan.md](../../../docs/Phase2_Implementation_Plan.md): current implementation plan for the missing Phase 2 components
- [Decoder Pipeline Investigation.md](Decoder%20Pipeline%20Investigation.md): end-to-end record of the offline/online parity investigation — root causes, fixes applied, before/after comparison, open questions for live deployment

## How To Use These Together

- Use the PRD for screen flow, operator workflow, and expected behaviors.
- Use the implementation plan in `docs/` for the current build sequence and planned component behavior.
- Use the historical architecture doc for older system-level context and hardware/data-flow background.
- Use the lab-equipment note when you need the most grounded hardware/LSL facts.

If these documents differ in detail, prefer:
- committed code first
- hardware-confirmed notes for integration facts
- the implementation plan for current Phase 2 work
- the historical architecture doc only for legacy context
