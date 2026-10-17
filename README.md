# Bruv_optimization
Program focused on simplifying fish identification for monocamera rigs.

A model finds and tracks every fish in BRUV footage from a single GoPro, a human confirms
each species on one card per fish, and the results go into a copy of the Excel viewing
template: one `record` row per sighting, plus a `MaxN` tab and an `app_details` tab.

Status: **MVP, phases 1–4 built** (scaffold, pipeline, review UI, export). First real run
done on Windows: a 2-minute GoPro clip with the owner's template, real model weights, and
the export checked in real Excel. Validation against a manual review (phase 5) is next.
See [`docs/SPEC.md`](docs/SPEC.md) for the decisions, the architecture and what's next, and
[`docs/SOURCES.md`](docs/SOURCES.md) for every outside source, model and license used.

## Run it on your own footage (Windows, one step)

1. Get the code: on GitHub, click **Code → Download ZIP**, then unzip it anywhere.
2. Put your GoPro `.MP4` file(s) and your Excel template (`.xlsx`) together in one
   folder.
3. Drag that folder onto **`quickstart.bat`** in the unzipped code (or double-click it
   and paste the folder path).

The first run installs everything. It can install Python 3.13 for you if it's missing
(through Python's install manager, `py install 3.13`, when you have it; otherwise
winget), and it downloads the fish detector and species model, a few GB in all. After
that it processes the video and opens the review page in your browser. Keep the black
window open while you review. Exports land in a `bruv_project\exports` folder inside your
footage folder, and the original template is never changed. Run it again at any time to
pick up where you left off.

**Python:** use 3.12, 3.13 or 3.14. 3.15 is too new: PyTorch has no 3.15 build yet.

**Time and memory on a laptop CPU.** On an Intel i7-10810U (6 cores, no NVIDIA GPU), a
2-minute clip took about 18 minutes: 2 for detection, 13 for the species model on 486
fish crops, 3 for the species list. The species model's time grows with the number of
fish tracks, so expect hours for a 1-hour video until the model runs on a faster
runtime. The species model needs about 1.5 GB of free memory. If the window stops with
no error message, close other programs and run `quickstart.bat` again: it resumes.

## Try the demo (no footage or models needed)

```bat
scripts\setup_windows.bat
scripts\run_windows.bat demo
```

This makes a synthetic two-chapter "GoPro" video and a mock template, processes them, and
opens the review page in your browser. Label the cards, then click **Export to Excel**.

On macOS/Linux: `pip install -e . && python -m bruv_app demo`.

## Real footage, step by step (advanced)

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
analysis window, e.g. to skip the descent), `--clip-names` (`auto`, the default, uses
`clip_1, clip_2…` when the template's `clip_in` dropdown lists them and the GoPro file
name otherwise; `file` or `index` force one). Everything else is in
`<project>\settings.json`.

## Sharing a test clip

GoPro files are large (a 2-minute 4K clip is around 750 MB). To send one over a size-limited
upload, make a small copy with the bundled ffmpeg:

```bat
scripts\run_windows.bat shrink D:\GoPro\GX010123.MP4 --max-mb 38
```

This writes `D:\GoPro\small\GX010123.MP4` under 38 MB: 1280 px wide, 10 fps, H.264, no
audio. The pipeline itself decodes at 1280 px and samples 1–2 fps, so nothing it uses is
lost. Keep the GoPro file name so chapter ordering still works. Use `--start`/`--duration`
to send just part of a clip.

No setup on that PC? HandBrake works too. Pick the "Fast 720p30" preset, set the
framerate to 10, set the average bitrate to about 2,000 kbps (for 2 minutes), and remove
the audio track.

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
