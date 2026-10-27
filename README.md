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
and runs on the CPU. To check what a PC will use, run
`.venv\Scripts\python -m bruv_app check-setup` in the code folder: it names the card,
the PyTorch build, and whether processing will run on the card or the CPU. After
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

`1` `2` `3` pick a model guess (a family, then its species) · `F` the family only ·
`/` search species, `Enter` records it · `U` unknown fish · `N` not a fish · `E` equipment ·
`S` skip · `Y` yes, same as the card beside it · `R` same answer as a look-alike · `L` answer
look-alikes together · `←` `→` move between cards. Nothing is pre-selected.

- **The family is required, the species optional.** If you can tell the family but not
  the species, `F` records the family alone (the template's "grouper family" entry). A
  family the template has no such entry for (moray eels) is recorded by its name and
  exported as the template's "other" entry, with the name in the comment. `U` stays
  "unknown fish", for a fish whose family you can't tell.
- **The picture is the fish's best side-on view:** of each card's sightings, the page shows
  the one that is sharp, big and long rather than tall (a fish lying across the frame, not
  facing the camera), and not cut off by the frame's edge.
- **The same fish over and over.** Short tracks keep each card to one fish, so a fish that
  circles the bait comes back as many cards. Two cards look alike when their pictures'
  embeddings are 0.85 alike or more (the bar for joining two pieces of one fish):
  - A card that looks like one you've answered says so ("Looks like #4202, which you
    answered unicornfish"), and `R` gives it the same answer. With no such card, `R`
    repeats your last answer.
  - **Show → ⚡ quick confirm** lists the cards at least 0.90 alike to one you've
    answered, most alike first (on the NP339 answers, 99.8% of answered cards that alike
    had the same answer). Each shows the two pictures side by side: `Y` gives it that
    answer, or answer as usual if it's a different fish. Every card is still your click,
    and each answer records what was suggested and whether you took it.
  - After you answer a card, a button offers its unanswered look-alikes. `L` opens them
    as a list you scroll down, in three bands: very alike (93% or more), alike (90–93%),
    and maybe (85–90%, which start unticked). Your card's picture stays at the top; point
    at a card to see it large beside it. A card made of several tracked pieces shows one
    picture per piece. Untick any that differ, and the rest get your answer together.
    Cards with warnings, or with a piece that isn't 0.85 alike to yours, are left to answer
    one by one, and the batch can be undone. What you left ticked or unticked is logged.
  - After a rebuild, a card holding sightings you answered before says so ("Before the
    cards were rebuilt you answered unicornfish for 117 of this card's 126 sightings"),
    and `R` gives the whole card that answer.
  - **Sort → look-alikes together** puts each group of look-alike cards back to back, so
    a run of the same kind of fish goes quickly with `R`.

  On the first 185 answers of the 44-minute NP339 drop, replayed in order, the closest
  card answered earlier named the fish right on 40 of 45 cards and was never wrong; the
  model alone got 24. It learns the fish of one video as you go; it doesn't make the
  model better at the next video (that's the library, below).
- **Still objects are hidden.** The bait arm, canister, float line and rocks sit still for
  the whole drop. A box that hardly differs from the scene's background (the median of
  the frames in the minute either side) makes no card and isn't counted, while a fish
  swimming in front of the arm still does. The header says how many boxes were hidden;
  click it to check a sample. On the NP339 answers this hid 91% of the sightings answered
  "not a fish" and none of the 1,689 answered as fish. A fish that stays motionless for
  over a minute would be hidden too, so look at the sample now and then.

- **Guesses by family first** (the menu next to "Model guesses"; "by species" shows the
  model's top three names instead). `1` `2` `3` pick a family: one with a single name
  (unicornfish) is recorded at once; otherwise its names follow, best first. `1`–`9` pick
  one, `F` records just the family ("grouper family", "unknown shark"), `Esc` goes back.
  Families come from your template's groups (its type list); a mixed list such as
  "species of interest" isn't one, so its names stand alone. They're ranked by their best
  name's score, as species are. The export's `MaxN_by_family` sheet counts each family's
  fish frame by frame, whether they were answered by species or by family.

- **Cards that may hold more than one fish** show a warning box: two of the card's pieces
  look less alike than the joining rule asks, the pieces clearly favour different
  species, or the card holds two boxes in one frame. A **Pieces** row under the photo
  shows each tracked stretch with its own best view and guess (click one to go to it).
  Answering such a card asks you to confirm that all its pieces are one fish; otherwise
  use `M` or tick and move. The card's line also names the tracker's closest call, a
  place to look if the card may have jumped to another fish.
- **Equipment** (`E`), for equipment that moves (a swaying rope, a diver's hand): a card
  showing it is recorded as not a fish, and cards that sit in the same place and look the same are queued under
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
    tick this. They're outside the review's scope: left unanswered, they stay out of the
    counts without making them provisional, and the export's `review_status` sheet says
    how many there were. Each card shows its fish's size at its largest.
  - **Sort:** biggest fish first (the default), in time order, or look-alikes together.
    Big fish are the clearest and the ones the counts rest on, so if a review stops early,
    what's left is the small, doubtful tail. Time order helps to follow a fish that comes
    and goes.
- **Only the time on the seabed.** If the video starts on deck or during the descent, set
  the review window. Give each end as a clip and the time in it, as the review page shows
  them, or as time across the whole video:

  ```bat
  .venv\Scripts\python -m bruv_app review-window --project my_project --start "GX010042 1:30" --end "GX030042 2:10"
  ```

  It prints where each clip starts, and the window both ways. Cards wholly outside it are
  hidden and not counted, and neither are sightings outside it; the export's
  `review_status` sheet records the window. `--clear` removes it.
- **Time per card.** Each answer records how long the card was open (up to 10 minutes), and
  a look-alike batch shares its time between its cards. It's for planning reviews and for
  measuring which steps save time.

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

To remake a reviewed project's cards with new rules and keep your answers:

```bat
scripts\run_windows.bat rebuild --project projects\site1 --latest-rules
```

`--latest-rules` adopts the current defaults (still objects hidden, tracks that bridge at
most one missed frame, contested links refused); `--track-gap 0` links consecutive frames
only. A new card gets your earlier answer only when every one of its sightings had that
answer; the rest wait for you, and the command says how many. The cards before and after
are frozen in `snapshots\`. A rebuild that is stopped part way loses no answers: run it
again. Projects made before these rules keep their old behaviour until rebuilt this way.

### Trying rules side by side

To see what a rule does to your own project before adopting it, try it on a copy:

```bat
scripts\run_windows.bat try-rules --project projects\site1 --name faint --set track_faint=true
scripts\run_windows.bat try-rules --project projects\site1 --name smooth --set track_velocity_smoothing=0.5
scripts\run_windows.bat compare-rules --project projects\site1
```

Each trial is a copy of the project's database in `trials\`, sharing its frames, rebuilt
with the changed settings; the project itself isn't touched. The first one also builds a
`baseline`: the project rebuilt with its own rules, so every trial is compared with cards
the machine built, not ones you've since fixed. `compare-rules` scores each against what
you did in the project:

| Column | What it counts | Better |
|---|---|---|
| cards, to review | all cards, and those not tiny and inside the review window | fewer |
| mix answers | cards holding sightings you answered differently (two fish on one card) | fewer |
| fixes undone | groups of sightings you pulled apart that are on one card again | fewer |
| followed boxed | the share of your followed fish's visits that have a box | higher |
| cards/fish | cards each followed fish ends up on | closer to 1 |
| card changes | times a followed fish's card changes | fewer |
| cards w/ another | cards holding a followed fish that also hold a box where it wasn't | fewer |

The follow columns need fish followed in the project ("Follow this fish"); a few followed
through crossings and busy moments say the most. `review --project … --trial faint` opens a
trial's cards on the review page to look at them; answers given there stay in the trial.

Options to try, all off until measured (docs/SPEC.md §15):
- `track_faint=true`: boxes scoring 0.10–0.30 may carry on a track that found no box in a
  frame (they never start one), so a fish that fades for a moment keeps its card. Only
  projects processed since faint boxes were kept have any; `try-rules` says when a
  project has none.
- `track_velocity_smoothing=0.5`: each link keeps half the track's earlier course, so one
  wobble doesn't turn the direction the tracker expects (0 is today's).
- Any other setting works the same way, e.g. `track_max_missed_s=0.5` or `gap_seconds=5`.

## Measuring accuracy

These tools follow the independent accuracy review (docs/SPEC.md §13). Defaults change
only once measured on answers: the still objects, the shorter tracks and the refused
contested links were measured on the first 185 NP339 answers.

- **Every result is traceable.** Each processing step records its settings, the code
  version and the model files (by fingerprint); each video is fingerprinted, and a
  read-only snapshot of the project's results is saved in `snapshots\` after processing.
  `snapshot --project … --label before-review` saves one by hand. The export's
  `provenance` sheet lists all of it.
- **One record row per answered card.** A card is one fish while it stays on screen, so
  each is a row with a count of 1; cards you joined (`P`) are one row, cards answered
  together as look-alikes stay separate. The most at once is MaxN's, on the `MaxN` sheet,
  never a sum of rows. Each MaxN row (by species and by family) has an `evidence_frame`:
  the frame it comes from, with the fish it counted boxed, saved beside the workbook in
  a `…_maxn` folder, to check the number. `record_rows: "grouped"` in a project's
  settings gives the rows from before (same species within `sighting_gap_s` share one).
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
- **Following a fish** (ground truth for the cards). On a card, **Follow this fish**
  shows its best view, then steps through the frames from there: click the fish's box in
  each frame (a dashed, faint box too), click the fish itself if it has no box (or press
  `B` if you can't point to it), and `O` when it isn't there. Each click moves on. It goes
  forward until the fish has been gone for 5 seconds (or 10), in case it comes back, then
  back from the start the same way. `Backspace` undoes a frame, and `Z` zooms in where you
  last saw it. At the end it lists the cards the fish's boxes are on, where its card
  changed, and any of those cards with a box where the fish wasn't (a swap onto another
  fish, to split with `X`). One follow is one fish's visit. When unsure which fish it is, press `O` and let
  it go: a short, sure follow is worth more than a long guess.

  `evaluate --project …` scores the follows with no answer key needed: how much of each
  visit the run boxed, how many cards each fish ended on, and why its card changed (near
  another fish, a gap in its boxes, the frame's edge). Only the followed fish were
  annotated in those frames, so they score identity, not the detector's false boxes.
  `follow-sheet --project … --out follows.csv` saves them as an answer key, to score a
  re-run of the same footage.
- **Crop sources (E1).** `crop-test --project … --limit 60` crops the same sightings from
  the review JPEG, the same frame without JPEG, the 1280 px decode and the original video
  frame, checks each against the saved frame, and compares the species guesses
  (`experiments\` in the project). Answered cards turn agreement into accuracy.
- **How the footage was shot.** `capture --project … --lens wide --housing "flat port"`
  records what the video file can't tell (lens mode, stabilisation, housing).
- **Other species models.** `model-test --project … --model bioclip-2.5 fishial --key
  answer_key.xlsx` scores today's BioCLIP 2, BioCLIP 2.5, the two together and Fishial on
  the same crops of your answered cards, plus how often your earlier answers would have
  named them. Fishial answers from its own list of species, so it's scored by family, and
  as a second opinion (when it and BioCLIP 2 agree or disagree, who was right); its
  `model.pt` goes in the app's model folder under `fishial-v0.10.2\extracted`. With a
  key, it also checks the key's clip_1 sightings, against chance. BioCLIP 2.5 is a 3.9 GB
  download, read from disk as needed, and about half as fast; the embeddings are saved in
  `model_test\` for a second run. The project is only read, so its review page can stay
  open. On NP339 neither BioCLIP 2.5 nor Fishial beat BioCLIP 2 overall (SPEC §14).

## Volunteers reviewing in a browser (optional)

A school can let volunteers review in a web browser, with nothing to install, on its own
free Cloudflare site. The processing PC publishes each video as files and can then be
switched off. Volunteers sign in with an access pass or their email. Their answers and
fixes (split, move ticked sightings, multiple fish) are saved on the site. The
coordinator pulls each volunteer's answers back into the project and exports as usual.
Setup and costs: [docs/HOSTED.md](docs/HOSTED.md).

```
python -m bruv_app publish --project D:\bruv\A1 --video A1 --site D:\bruv\site   (add --follow to follow fish online)
python -m bruv_app site volunteers --site D:\bruv\site --csv volunteers.csv
python -m bruv_app site serve --site D:\bruv\site          (try it on this PC)
python -m bruv_app pull --project D:\bruv\A1 --answers answers-A1.json --volunteer Ana
```

## Development

```bash
pip install -r requirements-dev.txt -e .
pytest
```

The tests use synthetic data only (fake detector, synthetic video, mock template). With
[Node.js](https://nodejs.org) installed, they also check the hosted site's browser code
against the review server (`tests/test_hosted_js.py`); without it, that test is skipped.

Layout: `bruv_app/` holds the package (video → detect → still → track → continuity →
classify → store → metrics → template/export, plus server and static UI with lookalike;
provenance, evaluate, diagnostics, equipment, crops and modeltest for measuring accuracy;
publish, site, pull and `hosted/` for the volunteers' review site). `tests/` holds the tests, and
`scripts/` the Windows helpers.
