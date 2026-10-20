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
winget), and it downloads the fish detector and species model, a few GB in all. On a PC
with an NVIDIA graphics card it installs the GPU version of PyTorch (about 3 GB more),
which makes processing many times faster; if the card's driver is too old, it says so
and runs on the CPU. After
that it processes the video and opens the review page in your browser. Keep the black
window open while you review. Exports land in a `bruv_project\exports` folder inside your
footage folder, and the original template is never changed.

**Long runs and interruptions.** Every slow step saves its work as it goes, so a run that
stops (a crash, a restart, the window closed) picks up where it left off when you run
`quickstart.bat` again: detection every 4 frames, the species model every 64 fish crops,
the species names every 20. While it processes, it keeps Windows from going to sleep
(closing a laptop's lid still does). If processing crashes, `quickstart.bat` waits 30
seconds and carries on by itself, up to twice. Your reviews are saved as you make them.

**Python:** use 3.12, 3.13 or 3.14. 3.15 is too new: PyTorch has no 3.15 build yet.

**Time and memory.** Videos are sampled at 4 frames per second: fish move little between
frames, which keeps each fish on its own track. On a laptop CPU (Intel i7-10810U, no
NVIDIA GPU), detection takes about 1.2 s per frame and the species model 1.1 s per fish
crop. Tiny fish skip the species model (on the first clip that was over half its work),
so a 2-minute clip takes about 16 minutes, and a 45-minute video about 5–6 hours. A PC
with a recent NVIDIA card should take well under an hour. The species model needs about
1.5 GB of free memory. If the window stops with no error message, close other programs
and run `quickstart.bat` again: it resumes.

## Fish that aren't on your template's list

The review page and the species model also offer the names in
[`bruv_app/extra_species.txt`](bruv_app/extra_species.txt): 33 reef-fish families
(triggerfish, butterflyfish, batfish / spadefish, fusilier and so on), marked "not in
template" on the page. Your template stays as it is. A card recorded with one of these names is exported as
the template's "other …" species, with the name in the comment, and the `MaxN` tab counts
each name on its own.

To change the list, edit that file (one name per line). To use a different list for a
project, for example in another region, put a file named `extra_species.txt` next to its
Excel template. Changing the list only recomputes the model's guesses, about 2 seconds
per new name; your cards and reviews stay.

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

Useful options: `--fps` (default 4), `--start`/`--duration` (an
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
audio. The pipeline itself decodes at 1280 px and samples 4 fps, so nothing it uses is
lost. Keep the GoPro file name so chapter ordering still works. Use `--start`/`--duration`
to send just part of a clip.

No setup on that PC? HandBrake works too. Pick the "Fast 720p30" preset, set the
framerate to 10, set the average bitrate to about 2,000 kbps (for 2 minutes), and remove
the audio track.

## Review keys

`1` `2` `3` pick a model guess · `/` search species, `Enter` records it · `U` unknown fish ·
`N` not a fish · `E` equipment · `S` skip · `←` `→` move between cards. Nothing is
pre-selected.

- **Cards that may hold more than one fish** show a warning box: two of the card's pieces
  look less alike than the joining rule asks, the pieces clearly favour different
  species, or the card holds two boxes in one frame. A **Pieces** row under the photo
  shows each tracked stretch with its own best view and guess (click one to go to it).
  Answering such a card asks you to confirm that all its pieces are one fish; otherwise
  use `M` or tick and move. The card's line also names the tracker's closest call, a
  place to look if the card may have jumped to another fish.
- **Equipment** (`E`): a card showing the bait arm, canister or rope is recorded as not a
  fish, and cards that sit in the same place and look the same are queued under
  **Show → ⚙ likely equipment**. There, "Mark the … that match throughout as equipment"
  answers in one go the cards whose every sighting is in the rig's place with the rig's
  colours; a card with sightings elsewhere (a fish the tracker joined onto the rig) is
  left for you, with a note. A batch can be undone. Nothing is removed without you.

- **A fish on neither list:** type its name in Comment, then pick the template's
  "other …" species in the search box. The page won't record that entry without a name.
  All such fish share one row in the `MaxN` tab, so for a fish you see often, add its
  name to the extra list instead.
- **When a card is wrong:**
  - `M` (multiple fish): the card mixes different fish. It breaks into its tracked
    pieces, each one a new card to review.
  - `X` (split): one tracked stretch swaps fish. Pick the sighting where the other fish
    starts (click it, or use `↑` `↓`) and press `X`. That sighting and the later ones
    become a new card.
  - **Tick and move:** for the other fish's sightings, e.g. after two fish crossed. Tick
    them (the box on each thumbnail, `Space` on the selected one, `Ctrl`+click, or
    `Shift`+click for a range), then press `X`: they all move to one new card. `Esc`
    clears the ticks.
  - `P` (same fish as the card before it): folds this card into the card just before it
    in the video, whatever the sort (its number is on the button). That card then needs
    reviewing again.

  Each fix is logged, and the export lists them on a `corrections` sheet. They show where
  tracking goes wrong, for tuning it and as training data.
- **Choosing which cards to see.** Three controls sit at the top of the page, and each
  choice shows how many cards it would give:
  - **Status:** pending (not reviewed yet), skipped, reviewed, or all. The page opens on
    pending, or on skipped once nothing is pending.
  - **Show:** all cards, or only those where one of the template's own species is the
    model's 1st guess, or in its top 2 or top 3. Such a card is marked ★ with that
    species. The model can be wrong either way, so review the other cards too.
  - **Show tiny fish:** tiny fish never reach 40 px across (in the 960 px wide frames the
    page shows; `tiny_fish_px` in the project's settings), so nobody can identify them.
    They get cards of their own, never joined to other fish, and stay hidden until you
    tick this. Left unreviewed, they stay out of the counts, and the progress line shows
    them apart ("tiny fish: 0 of 177"): hidden is not the same as reviewed. Each card
    shows its fish's size at its largest.
  - **Sort:** biggest fish first (the default), or in time order. Big fish are the
    clearest and the ones the counts rest on, so if a review stops early, what's left is
    the small, doubtful tail. Time order helps to follow a fish that comes and goes.

## Teaching it your fish

The species model never changes, but your confirmed fish can sharpen its guesses. Add
examples to a library on your PC: reviewed projects you trust, or photos sorted into one
folder per species (folder names as in your species list).

```bat
scripts\run_windows.bat learn --add projects\site1
scripts\run_windows.bat learn --add-photos D:\fish_photos
scripts\run_windows.bat learn
```

The last command lists what the library holds. Guesses blend in the nearest confirmed
examples only once that has beaten the species model alone on recordings it didn't learn
from: each recording is guessed from the others, with the blend's weight chosen without
it (at least 20 examples from 3 or more recordings, by 5 points). So a thin or patchy
library can't make guesses worse. Examples are grouped by the footage itself (the videos'
fingerprints), so a copy of the same video in another folder counts once, and adding it
again replaces it. Cards with warnings you never confirmed stay out. Processing a project
tells you whether its guesses use the library. Adding to the library refreshes a
project's guesses the next time it's processed; reviews stay. `--remove <folder>` takes
a project or photo folder back out.

## Checking new rules against your answers

To see whether a change (new settings, a new version) builds better cards, re-run a copy
of a project you've reviewed and score it against the original:

```bat
scripts\run_windows.bat score --project projects\site1_rerun --answers projects\site1
```

It counts the cards that now mix sightings you answered differently, and the groups of
sightings you pulled apart (splits, moves, "Multiple fish") that are on one card again.
It matches sightings by source frame and box, one to one, so it works across frame
rates too, and it refuses two projects of different footage.

## Measuring accuracy

These tools follow the independent accuracy review (docs/SPEC.md §13). None of them
changes a default: new settings wait until they've been tested on annotated footage.

- **Every result is traceable.** Each processing step records its settings, the code
  version and the model files (by fingerprint); each video is fingerprinted, and a
  read-only snapshot of the project's results is saved in `snapshots\` after processing.
  `snapshot --project … --label before-review` saves one by hand. The export's
  `provenance` sheet lists all of it.
- **The export says how complete the review is.** `review_status` and
  `review_by_minute` count cards and sightings by status, tiny fish apart. Until every
  card is answered, the `MaxN` rows say "provisional". `checks` lists what could count
  one fish twice (two boxes on one fish, or one card with two boxes at once).
- **Answer keys.** `annotation-sheet --project … --window 600-620 --out key.csv` starts
  a sheet of boxes for a stretch of video (every row "unchecked" until someone checks
  it). Fill in each fish's kind, individual and species, and add the fish the detector
  missed. `evaluate --project … --annotations key.csv` then scores detection, tracking,
  joining and MaxN separately; `--thresholds 0.1 0.2 0.3 0.4` adds detection at other
  score thresholds, from the boxes the run kept (down to 0.1).
- **Crop sources (E1).** `crop-test --project … --limit 60` crops the same sightings from
  the review JPEG, the same frame without JPEG, the 1280 px decode and the original video
  frame, checks each against the saved frame, and compares the species guesses
  (`experiments\` in the project). Answered cards turn agreement into accuracy.
- **How the footage was shot.** `capture --project … --lens wide --housing "flat port"`
  records what the video file can't tell (lens mode, stabilisation, housing).

## Development

```bash
pip install -r requirements-dev.txt -e .
pytest
```

The tests use synthetic data only (fake detector, synthetic video, mock template).

Layout: `bruv_app/` holds the package (video → detect → track → continuity → classify →
store → metrics → template/export, plus server and static UI; provenance, evaluate,
diagnostics, equipment and crops for measuring accuracy). `tests/` holds the tests, and
`scripts/` the Windows helpers.
