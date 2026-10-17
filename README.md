# Bruv_optimization
Program focused on simplifying fish identification for monocamera rigs.

A model finds and tracks every fish in BRUV footage from a single GoPro, a human confirms
each species on one card per fish, and the results go into a copy of the Excel viewing
template: one `record` row per sighting, plus a `MaxN` tab and an `app_details` tab.

Status: **MVP, phases 1–4 built** (scaffold, pipeline, review UI, export), tested on
synthetic video. Not yet run on real footage or with the real template. See
[`docs/SPEC.md`](docs/SPEC.md) for the decisions, the architecture and what's next.

## Try it in two minutes (no footage or models needed)

```bat
scripts\setup_windows.bat
scripts\run_windows.bat demo
```

This makes a synthetic two-chapter "GoPro" video and a mock template, processes them, and
opens the review page in your browser. Label the cards, then click **Export to Excel**.

On macOS/Linux: `pip install -e . && python -m bruv_app demo`.

## Real footage

1. Install the models (CPU is fine): `scripts\setup_windows.bat --ml`
2. Download a Community Fish Detector RF-DETR checkpoint from its
   [releases page](https://github.com/filippovarini/community-fish-detector/releases)
   into `models\` (git ignores that folder). **Nano (640)** is the CPU default; Small or
   Medium (1024) suit a GPU. BioCLIP 2 downloads itself from Hugging Face on first run.
3. Process. Point it at the GoPro files or their folder, in any order. Chapters are
   ordered by GoPro naming and joined into one timeline. This step is resumable: run the
   same command again after a stop.

   ```bat
   scripts\run_windows.bat process --project projects\site1 ^
       --template "templates\viewing template v.2024.xlsx" ^
       --weights models\cfd-rf-detr-nano-640-2026.02.02.cp-011.20260706-release.pth ^
       --start 120 --duration 3600 D:\GoPro\site1
   ```

4. Review: `scripts\run_windows.bat review --project projects\site1`
5. Export from the page, or with `scripts\run_windows.bat export --project projects\site1`.
   The original template is never modified.

Useful options: `--fps` (default 1 on CPU, 2 with a GPU), `--start`/`--duration` (an
analysis window, e.g. to skip the descent), `--clip-names index` (use `clip_1, clip_2…`
instead of the GoPro file name). Everything else is in `<project>\settings.json`.

## Review keys

`1` `2` `3` pick a model guess · `/` search species, `Enter` records it · `U` unknown fish ·
`N` not a fish · `S` skip · `←` `→` move between cards. Nothing is pre-selected.

## Development

```bash
pip install -r requirements-dev.txt -e .
pytest
```

The tests use synthetic data only (fake detector, synthetic video, mock template).

Layout: `bruv_app/` holds the package (video → detect → track → continuity → classify →
store → metrics → template/export, plus server and static UI). `tests/` holds the tests,
and `scripts/` the Windows helpers.
