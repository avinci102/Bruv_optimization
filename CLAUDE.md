# CLAUDE.md

## What this is
A Windows program that helps people identify fish in BRUV (baited remote underwater video)
footage from a single GoPro. A model finds and tracks the fish. A human confirms each
species. Results are exported into the owner's Excel viewing template.

**Read `docs/SPEC.md` before doing anything.** It holds every decision made so far, the
proposed architecture, the build phases, and the open questions.

## Working rules
- **CPU-first.** Assume no NVIDIA GPU. Use a GPU automatically when one is present.
- **Packaging.** Keep it installer-friendly (a PyInstaller + Inno Setup installer comes
  later):
  - avoid dependencies that are hard to bundle;
  - keep model inference behind interfaces so torch can later be swapped for ONNX/OpenVINO.
- **The human decides species.** Show the model's top-3 guesses, but never pre-select one.
- **Abundance is MaxN, computed from per-frame detections.** The ~10 s continuity rule is
  only for deduplicating review cards, never for counts.
- **Excel template rules:**
  - write by header name, never by column letter;
  - always write to a copy;
  - read dropdown lists from the sheet's data-validation rules.
- **No true length yet.** It's a mono camera; leave `length_ind` blank and log relative
  size in the `app_details` sheet.
- **Large files.** Don't commit videos or model weights; keep them out of git via
  `.gitignore`.
- **Tests.** Every pipeline stage gets pytest coverage using synthetic data (fake
  detector, synthetic video, mock template).

## Owner preferences
- When evaluating tools or models, weigh independent results and community feedback over
  vendor claims.
- Keep the MVP lean, and take the clearest path to a working result.
