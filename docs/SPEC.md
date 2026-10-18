# BRUV Fish ID Assistant — Spec & Handoff

Status: **MVP built (phases 1–4). First real run done on Windows (§9).** This document
captures every decision made so far. Items marked **[Decided]** came from the project owner (Vinci).
Items marked **[Proposed]** are Claude's recommended defaults — change them freely. Section
8 lists what changed after the first code review of this spec.

---

## 1. Goal

A Windows desktop program that turns a BRUV (baited remote underwater video) recording
into a filled-in copy of the owner's Excel viewing template, with a human confirming every
species ID.

The program:

1. Splits the video into frames.
2. Finds every fish (unidentified at this stage).
3. Links detections of the same fish over time (continuity), so each fish is reviewed once.
4. Shows the user one card per unique fish, with the model's top guesses **and** a
   species dropdown. The user's pick is what gets recorded.
5. Exports per-sighting rows into the template's `record` sheet plus a MaxN summary tab.

**[Decided]** Accuracy target: ≥ 90%, reached mainly through existing pre-trained models
plus human review, not by training from scratch.

---

## 2. Decided requirements

- **[Decided] Rig:** a single GoPro (mono camera). No stereo. True length can't be measured
  yet. Visual scale markers placed next to the BRUV come in a later iteration.
- **[Decided] Platform:** Windows.
- **[Decided] Hardware:** assume **no NVIDIA GPU** (CPU-first). Use a GPU automatically if
  one is present. Vinci can use his PC for heavier trials.
- **[Decided] Users:** Vinci first; the end goal is non-technical users. Build so a
  one-click installer can be added later without rework. No installer is needed in v1.
- **[Decided] Export:** one row per sighting (unique fish, or a school) in the template's
  `record` sheet, **plus** a separate MaxN-per-species tab.
- **[Decided] Review UX:** show the model's top-3 guesses as buttons next to the species
  dropdown. **Nothing is pre-selected**; the user must click. While model confidence is
  low, the user's answer is always the recorded answer.
- **[Decided] Continuity rule:** if a fish leaves the frame and comes back within ~10 s,
  treat it as the same fish. The window should be configurable.
- **[Decided] Required outputs per sighting:** the timestamp when the fish appears, its
  size, and its species.
- **[Decided] Extra species list (Oct 2026).** The app keeps its own list of 33 reef-fish
  families (`bruv_app/extra_species.txt`) on top of the template's species list, and the
  template stays unchanged.
  - A card recorded with an extra name is exported as the template's "other" entry, with
    the name in the comment.
  - MaxN counts each extra name on its own.
  - Cold-water families (salmon, cod) are left out; fusiliers are their own name rather
    than part of "snapper family".

### Important nuance on counting

The 10-second continuity rule exists to **cut review work**, so the same snapper isn't
labelled 20 times. It must **not** drive abundance. BRUV abundance is reported as
**MaxN**: the maximum number of individuals of a species visible in a single frame. MaxN
is used precisely because you can't know whether a returning fish is the same individual.
Compute counts from per-frame detections, not from merged tracks.

---

## 3. Excel template (observed from a photo; real file still needed)

The workbook is named roughly `…V viewing template v.2024`. It has three sheets:

- `dropdownlist (dont touch this)`: source lists for the dropdowns.
- `info (fill this first)`: deployment metadata.
- `record (fill this next)`: one row per observation.

The visible `record` headers, in order, are: `clip_in`, `time_in`, `clip_out`, `time_out`,
`type`, `species`, `count`, `length_ind`, `school_min`, `school_max`, `sex`, `behavior`,
`comment`.

- Columns A–F were off-screen in the photo, and columns K, L, P, Q, U are hidden. Their
  names are unknown.
- Row 2 is an example: `clip_1 | 00:00:00 | clip_1 | 00:00:00`. Times are time values
  within a clip, and clips correspond to GoPro chapter files.

**[Proposed] Rules for working with the template**

- **Write by header name, never by column letter.** Leave unknown or hidden columns blank.
- **Read dropdown lists from the data-validation rules** on the `record` sheet (formula1
  → range on the dropdown sheet, or a named range). Fall back to a column on the dropdown
  sheet whose header matches.
- **Always write to a copy** of the template; never modify the original.
- **Until the real file arrives**, build a mock template with the same sheet names,
  headers, and validations, for tests only.

---

## 4. Proposed architecture [Proposed]

**Stack:**

- **Language:** Python 3.11+.
- **UI:** local web app — a FastAPI backend plus plain HTML/JS, no frontend build step.
  It opens in the default browser, with pywebview as an option later.
- **Storage:** SQLite per project, so work can be resumed.
- **Why this stack:** it packages cleanly with PyInstaller + Inno Setup later.

### Pipeline

1. **Ingest.** Accept one or more files, since GoPro splits recordings into chapters.
   Order chapters by GoPro naming (`GX01xxxx`, `GX02xxxx`, …) and build one continuous
   timeline, while keeping each file name for `clip_in` / `clip_out`. Support an
   analysis window (start offset and duration), for example to skip the descent.
2. **Frame sampling.** Decode with an ffmpeg pipe via `imageio-ffmpeg` (bundled binary,
   hardware decode where available). No OpenCV: the bundled ffmpeg is the only decoder. Default is **1 fps on
   CPU**, 2–5 fps with a GPU. Store a downscaled JPEG of each sampled frame for the review
   UI, at roughly 150 MB per 30 minutes at 960 px.
3. **Detection.** Use the **Community Fish Detector**
   (github.com/filippovarini/community-fish-detector), **RF-DETR release** (see §8).
   - Single `fish` class, trained on a dataset of over 1.9M images built from 17 open
     datasets.
   - Models: RF-DETR Nano (640 px), Small and Medium (1024 px). The inference code
     (`rfdetr`) is **Apache-2.0**. The CFD repo states no license for the weights
     themselves (see `docs/SOURCES.md`). The older YOLOv12x model is deprecated by its
     authors, and its inference code (Ultralytics) is AGPL.
   - Default: **Nano on CPU**, Small/Medium with a GPU. It sits behind a `Detector`
     interface so it can be swapped for an ONNX/OpenVINO export later (rfdetr 1.11+ ships
     both exporters). That should help on Intel CPUs (a vendor claim, so benchmark it).
     The installer can drop torch only once the classifier is also off torch (BioCLIP 2
     runs through open_clip, which needs it).
4. **Tracking.** Use a lightweight in-house tracker: IoU plus centre distance, matched as a
   linear assignment (SciPy's `linear_sum_assignment`, the same optimum as the Hungarian
   algorithm). It avoids an ultralytics dependency so torch can be dropped
   later. At 1 fps tracks will fragment; step 5 fixes that.
5. **Continuity merge.** Merge tracklet B into A when all of these hold:
   - B starts within N seconds (default 10) after A ends;
   - they never overlap in time (two fish on screen at once can't be one fish);
   - their appearance embeddings are similar (cosine similarity ≥ threshold);
   - their box-size ratio is plausible.

   Use a greedy merge on best similarity. Keep the original tracklets so the UI can
   **split** a wrong merge.
6. **Classification.** Use **BioCLIP 2** zero-shot through open_clip:
   - Load it with `open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')`.
   - **Restrict the labels to the template's species dropdown.** *Plus the app's extra
     list since Oct 2026 (§2, §10).*
   - Per merged fish, take the top 3–5 crops (scored on detection confidence, size, and
     sharpness), average their image embeddings, and compute softmax over the cached text
     embeddings.
   - Output the top-3 guesses with their probabilities.
   - Reuse the same embeddings for the continuity merge in step 5.
   - Fallback when the model can't load: a colour-histogram embedding with no species
     guesses.
7. **Review UI** — one card per unique fish, in time order. Each card shows:
   - the best crop, plus the full frame with the box drawn, and a filmstrip of the other
     sightings;
   - top-3 guess buttons (keyboard shortcuts 1/2/3) and a searchable species dropdown;
   - buttons for **Unknown fish**, **Not a fish**, **Split**, and **Skip**;
   - optional fields for type, sex, and behavior, using the template's dropdowns.

   Show progress as "N of M reviewed."
8. **Export.**
   - **Sightings → `record` sheet.**
     - [Proposed] Group same-species fish whose time spans overlap or sit within the gap
       window into one sighting.
     - `time_in` / `time_out` are the first and last seen, with the matching clip names.
     - `count` is the peak simultaneous count within the sighting.
     - Leave `length_ind` blank until scale markers exist.
     - Leave `school_min` / `school_max` blank until their meaning is confirmed.
   - **New `MaxN` sheet:** one row per species, with MaxN, the clip and time of MaxN, and
     the time of first arrival. Only human-confirmed labels count.
   - **New `app_details` sheet** (keeps the user's template clean): one row per fish with
     track id, model top-3 and confidences, the user's pick, whether they agree, relative
     size (box length as % of frame width), and the crop file path. This sheet becomes
     the training and accuracy dataset.

### Package layout (suggested)

```
bruv_app/
  config.py      settings + defaults (fps, imgsz, gap_seconds, thresholds)
  video.py       chapter ordering, timeline, frame sampler
  detect.py      Detector interface; CFD RF-DETR impl; fakes for tests
  track.py       tracker
  continuity.py  merge/split logic
  classify.py    BioCLIP 2 zero-shot; fallback embedder
  template.py    read dropdowns, write export
  metrics.py     sightings + MaxN + first arrival
  store.py       SQLite project store
  pipeline.py    orchestration with progress callbacks
  server.py      FastAPI routes
  static/        index.html, app.js, styles.css
tests/
scripts/  setup_windows.bat, run_windows.bat
```

---

## 5. Build phases

1. **Scaffold.**
   - Package skeleton, config, a mock template, and a pytest setup.
   - A synthetic test video, made by moving fish crops over a background, so the
     pipeline can be tested without real footage.
2. **Pipeline.** Ingest, sampling, detection, tracking, continuity, and classification,
   all persisted to SQLite and resumable.
3. **Review UI.** The card workflow, keyboard shortcuts, split/merge, and progress.
4. **Export.** Record rows, the MaxN tab, and the app_details tab, all written into a copy
   of the template.
5. **Validation on real footage.**
   - Compare against a manual annotation of the same video: species list, per-species
     MaxN, and time of first arrival.
   - Track model-vs-user agreement per species. Auto-accepting a species is only allowed
     once its agreement rate proves high (for example ≥95% over a meaningful sample).
6. **Windows packaging.** PyInstaller plus an Inno Setup installer. Prefer an ONNX/OpenVINO
   runtime here to drop torch.
7. **Scale markers.** Convert pixels to cm from markers next to the BRUV, then fill
   `length_ind`.

---

## 6. Open questions (need Vinci)

1. **The real template `.xlsx`.** *Received (Oct 2026) and used for the first real run;
   kept out of the repo.*
2. **A sample clip**, 1–3 minutes of real footage. *Received: a 2-minute clip, used for
   the first real run (§9).*
3. **Species list format.** Scientific names, common names, or codes? Which region?
4. **What do `type`, `school_min`, and `school_max` mean** in your workflow?
5. **Analysis window.** Fixed soak time (for example 30 or 60 minutes from settle), or
   the full video?
6. **Clip naming.** Use the GoPro file name or `clip_1`, `clip_2`, …?

---

## 7. Background research (from planning conversation)

- **Published BRUV results:** fully automatic species ID typically lands well below 90%.
  Semi-automated (model plus human) workflows reach roughly 90%+ MaxN agreement.
  - **SharkTrack** (arXiv 2407.20623): 89% species-specific MaxN accuracy on unseen
    locations, with about 2 minutes of human time per hour of video.
  - **Reef shark study** (IRD): fully automatic F-measures of 0.43–0.85 rose to 0.82–0.97
    once a human was in the loop. *(Citation missing: no authors, title or DOI were
    recorded, so this can't be checked yet.)*
- **Fishial.ai** self-reports 70–90% accuracy; this is a vendor claim. *(Their public
  GitHub repo reports 93.22% classification accuracy instead. Where 70–90% came from is
  unknown.)*
- **VIAME** (NOAA/Kitware) is the established open-source alternative for detection,
  tracking, and annotation, and is worth comparing against.
- **General vision-language chatbots** score poorly on open-ended fish species ID (the
  VLM4Bio benchmark); don't rely on them.
- **The biggest accuracy levers after human review:**
  - restricting labels to the regional species list;
  - fine-tuning on a few hundred labelled crops from the owner's own footage.

---

## 8. Changes after the first code review (Oct 2026)

The approach and the review-card layout in §4.7 stand. These points changed while
building the MVP:

1. **Detector: RF-DETR instead of YOLOv12x.** The Community Fish Detector repo now
   marks YOLOv12x as deprecated and ships RF-DETR Nano/Small/Medium, whose inference
   code is Apache-2.0. That removes the AGPL concern. (The CFD repo states no license for
   the weights; ask the authors before any commercial distribution.) Nano should also be
   much lighter on CPU, though that hasn't been benchmarked here. Its
   reported AP (0.596 vs 0.609 for Medium) is on the authors' own unpublished validation
   split. They say the absolute values mean little, but the column is fine for comparing
   these models. Phase 5 should still compare Nano and Medium on our own footage.
2. **Excel dropdowns survive export.** Excel stores dropdowns that point at another
   sheet in an `x14` extension block, which openpyxl drops on load (with only a
   warning). The app reads validations from the raw sheet XML (standard, named-range,
   literal, and x14), and re-attaches any dropped ones on every sheet of the exported
   copy, with their error and prompt settings, so typed values outside the list are
   still rejected. Two other kinds of loss are now reported at export: features openpyxl
   warns about (e.g. extended conditional formatting) are logged, and shapes and text
   boxes, which openpyxl drops with no warning, are detected and flagged. *Checked in
   real Excel during the first real run (§9): the re-attached dropdowns work, but two
   other openpyxl problems made Excel refuse the file until they were fixed.*
3. **Frame timestamps are exact.** ffmpeg's `fps` filter by default returns the frame
   about half a step *after* the timestamp. Sampling now uses `fps=N:round=up`, which a
   test checks.
4. **The tracker has a cheap colour check.** At 1 fps, a fish passing close to another
   could be handed the other's track (seen on the synthetic video). A colour-histogram
   gate (`track_min_appearance`) now blocks obvious swaps. It is cheap and adds no
   dependencies. A mixed card would put the wrong species into MaxN, so this matters
   for accuracy.
5. **Split works at any sighting**, not only between tracklets, so the reviewer can fix
   a swap inside one track. Merge folds a card into the previous one and sends it back
   for review.
6. **Where record rows start.** The export appends after the last filled row of the
   `record` sheet, so the example row 2 is kept. (See open question 7.)
7. **Unknown fish** become `record` rows with a blank species and the comment
   "Unidentified fish". They are left out of the MaxN tab.
8. **Guesses can be recomputed without the model.** Label text embeddings are cached in
   the project, so a split or merge in the UI recomputes guesses without loading
   torch.
9. **BioCLIP 2 is called the way its authors call it.** Each label's text embedding is
   the average over CLIP's 80 prompt templates, as in Imageomics' `pybioclip`, rather
   than a single "a photo of …" prompt. (BioCLIP 2's own evaluation uses the same 80
   plus "a photo of {c}.".) Crops are squashed to 224×224, as in `pybioclip`, rather
   than centre-cropped, which would cut the head and tail off a long fish. For
   camera-trap and rare-species tests, their evaluation labels classes with full
   taxonomic names plus common names ("Animalia Chordata … *Lutjanus campechanus* with
   common name red snapper"). Doing the same needs a species-to-taxonomy lookup, so it
   waits on open question 3.
10. **Detector loading is guarded.** If a checkpoint's resolution doesn't match the one
    in its CFD file name, it is reloaded at the right resolution. Older checkpoints would
    otherwise run at the RF-DETR default (Nano 384 instead of 640) without any error.
    `rfdetr` and `open_clip` are held to the minor versions whose APIs were checked
    (1.11.x and 3.3.x). The load error now says when a file may be incomplete, so a
    re-download is the obvious next step.
11. **Licensing to sort out before the installer (phase 6).**
    - The ffmpeg that imageio-ffmpeg bundles is a GPLv3 build, so the installer must
      ship the GPL text and offer its source.
    - `rfdetr` pulls in PyAV through `supervision`. PyAV's FFmpeg libraries link x264
      and x265, which are GPL, so exclude PyAV from the bundle (the app never uses it).
    - tqdm and certifi are MPL-2.0, so ship their notices.
    - BioCLIP 2 is MIT according to its authors. Still check the Hugging Face model card
      and the TreeOfLife-200M terms (one component is CC BY-NC 4.0) before distributing.
    - The CFD weights have no stated license.

    Details are in `docs/SOURCES.md`.

### Status by phase

1. Scaffold — **done.** Package, config, mock template, synthetic video, pytest.
2. Pipeline — **done.** Resumable detection at 4 fps, a tracker that follows each fish's
   course, continuity with tiny fish kept apart, BioCLIP 2 with a histogram fallback, and
   a library of confirmed fish that can sharpen the guesses (§12). RF-DETR and BioCLIP 2
   run with real weights on Windows (§9).
3. Review UI — **done.** Cards, keys for the guesses, unknown, not a fish, multiple
   fish, skip, split, tick-and-move and merge (with a log of corrections), searchable
   species, optional fields, status and template-species filters, tiny fish hidden until
   asked for, progress (§11, §12).
4. Export — **done.** Record rows by header name, MaxN tab, app_details tab with model vs
   user agreement, relative size and the saved best crop.
5. Validation on real footage — **started.** First real run done (§9); a `score` command
   checks re-runs against reviewed copies (§12). The comparison with Vinci's manual
   annotation of the same video is still to do.
6. Windows installer — not started (`scripts/*.bat` cover setup for now).
7. Scale markers — not started.

### Known risks to watch on real footage

- **Schools** create many simultaneous cards (one per fish), since cards on screen
  together can't merge. If that is slow to review, add "apply this label to the similar
  cards in this time window".
- **Decode speed.** 4K GoPro HEVC decoded on the CPU can take as long as detection.
  `-hwaccel auto` is on by default. *On Vinci's PC it engages (DXVA2 on Intel UHD
  graphics): a 2.7K H.264 clip decodes in the same time as on the CPU (~48 s for 2
  minutes), but with ~6× less CPU time, which leaves the CPU to the detector.*
- **MaxN at 1 fps** can slightly undercount compared with frame-by-frame manual review.
  Compare it in phase 5, and raise `--fps` if needed.

### New open question

7. Should the export **replace the example row 2** in `record`, or append below it as
   it does now?

---

## 9. First real run on Windows (Oct 2026)

Run through `quickstart.bat` on Vinci's PC (Windows 11, Intel i7-10810U with 6 cores,
16 GB RAM, Intel UHD graphics, no NVIDIA GPU), with a 2-minute 2.7K clip (H.264, 30 fps)
and the real template.

**Results**

- **73 fish cards** from 759 detections in 120 sampled frames (250 tracklets).
- **About 18 minutes of processing** for 2 minutes of video, plus a one-off install and
  model download of about 25 minutes:
  - detection (RF-DETR Nano at 1 fps, decode included): 1 min 55 s;
  - tracking: 34 s;
  - BioCLIP 2 on 486 crops: 12 min 37 s, including ~1 min to load the model;
  - species-list embeddings and guesses: 2 min 56 s.

  BioCLIP 2 (ViT-L) costs ~1.6 s per crop on this CPU and grows with the number of
  tracks, so a 1-hour video would take hours. A faster runtime (ONNX/OpenVINO, phase 6),
  fewer crops per track, or skipping tiny fish are the levers.
- **Guesses were implausible for most cards.** Most fish in this clip are species the
  template's list doesn't include. Zero-shot classification restricted to the list has no
  "none of these" answer, so it names the closest listed species. Given the right
  names, BioCLIP 2 was confident and correct (two test crops scored 1.00). Cards of
  listed families got plausible guesses.
- **Review burden.** About half the detections are small, distant fish (under 40 px
  across a 1280 px frame) that neither a person nor the model can identify. 3 cards are
  the rig's bait arm.

**Fixed during the run**

1. `quickstart.bat` installs Python 3.13 through Python's install manager
   (`py install 3.13`) when it's there. It also treats a crash (a negative exit code) as
   a failure instead of ending silently.
2. **Dependent dropdowns are read.** A list like `=INDIRECT(<cell>)`, where each option of
   a parent list names a range, is now read. The export fills the parent column (e.g.
   `type`) from the chosen species, and the review page hides that field.
3. **Clip names default to `auto`.** They become `clip_1, clip_2…` when the template's
   `clip_in` dropdown lists them.
4. **A cross-sheet dropdown that covers two ranges no longer crashes the export.** Its
   `sqref` contains a space.
5. **BioCLIP 2 no longer exceeds Windows' memory limit.** Loading it crashed a 16 GB PC
   that had other programs open. open_clip's loader peaks at ~3.5 GB of commit, and
   PyTorch keeps freed memory reserved. The model is now built on the meta device, with
   its weights mapped read-only from the checkpoint file. That needs ~1.3 GB and gives
   identical weights (all 446 tensors checked).
6. **Species-list embeddings are ~4× faster.** Each prompt is encoded at its own length
   instead of all 77 tokens. Results are identical, and every load checks this against
   open_clip's own encoder.
7. **Fallback runs are rebuilt.** A run that fell back to colour histograms (because
   BioCLIP 2 couldn't load) is rebuilt with the model on the next run, until reviewing
   starts.
8. **Real Excel now opens the export.** Before this fix, Excel refused every export,
   and even a plain openpyxl re-save of this template. Two openpyxl bugs caused it:
   - an external link saved with alternate paths loses the relationship it points at;
   - a rich-text run that holds only spaces loses `xml:space="preserve"`.

   Both are repaired at export. The export also now keeps rich text (bold and italic
   words inside a cell), and gives new cells their column's format.
9. **A Windows-only download bug.** A failed model download couldn't delete its `.part`
   file.

**Checked in real Excel.** The checks used late-bound automation, because the Office
type library that PowerShell needs is broken on that PC.

- The export opens.
- These all match the template:
  - sheets, named ranges and external links;
  - every value and formula (one example time differs by 1e-16 of a day);
  - hidden columns, widths and heights;
  - the dropdown rules at 58 sampled cells;
  - comments and rich text.
- All 20 values the export wrote under dropdowns pass Excel's own validation. That
  includes the dependent species list and the cross-sheet clip list.

openpyxl still loses a few things, all cosmetic:

- one sheet's printer settings;
- Google Sheets' comment metadata;
- the dynamic-array flag on two example formulas (they become legacy array formulas with
  the same result);
- the external link's alternate paths.

**Open points from this run** (for Vinci, with §6):

- How to export catch-all list entries (e.g. an "other" species). Today they're grouped
  into one record row and one MaxN row, like a real species. *Mostly answered by §10:
  extra names are exported as "other" but kept apart.*
- Whether to hide or skip tiny, distant fish.
- Whether the model should be allowed to name off-list fish as a hint. *Answered: yes,
  through the extra list (§10).*
- Where record rows should start when a template holds example rows.

---

## 10. Extra species and resuming long runs (Oct 2026)

**Extra species.** Vinci proposed 47 more families. 35 of them weren't in the template's
list; the others are already covered at species level (e.g. reef sharks, stingrays,
remora, sea chub). Salmon and cod were left out as cold-water families, which leaves 33,
in `bruv_app/extra_species.txt`.

- **The model needs no training for them.** BioCLIP 2 works from the names alone. On the
  first real run's 73 cards, adding the names changed the share of clearly identifiable
  cards with the right family as top guess:
  - before: 2 of 23;
  - after: 13 of 23 (17 of 23 in the top two).
- **Look-alike families are the remaining errors:** batfish vs angelfish, butterflyfish vs
  angelfish or wrasse, and some triggerfish guessed as surgeonfish.
- **Family names in the prompts made things worse,** so the labels are common names.
- **Next lever, if needed:** a small classifier trained on BioCLIP embeddings of the
  owner's own confirmed cards (seconds on a CPU once there are ~20–50 cards per family).
  Retraining BioCLIP itself isn't needed.

**Resuming.** Every slow stage now saves as it goes:

- detection every 4 frames (as before);
- crop embeddings every 64 crops, kept across reruns (they don't depend on tracking);
- species-name embeddings every 20 names, also reused when the list grows.

A changed name list recomputes only the guesses, so cards and reviews stay. Processing
keeps Windows from idle-sleeping, and `quickstart.bat` retries a crash twice.

*Checked on a copy of the real project:*

- adding the 33 names took 2 minutes and gave the same top guess as the experiment on all
  73 cards;
- a run killed hard at crop 128 of 486 restarted at 128, not 0. It finished in 10 minutes
  with the same 73 cards and top guesses as an uninterrupted run.

**Next topic:** keeping one fish on one card (continuity), and whether sampling more often
than 1 frame per second helps.

---

## 11. Frame rate, review keys and a template-species filter (Oct 2026)

**Frame-rate test.** The first real clip, processed at 1, 2 and 4 frames per second:

| | 1 fps | 2 fps | 4 fps |
|---|---|---|---|
| Cards | 73 | 75 | 89 |
| Cards stitched from 4+ tracks | 27 | 35 | 45 |
| Tracks | 250 | 288 | 431 |
| Single-sighting tracks | 47% | 38% | 42% |
| Processing time | ~15 min | 23 min | 27 min |

The times leave out the species-name embeddings, which the runs shared.

- **More frames don't fix the main problem: cards that mix different fish,** often of
  different species. Most cards stitched from several tracks do. One label covers the
  whole card, so mixing also distorts MaxN.
- **The cause is weak identity cues, not the frame rate:**
  - the colour histogram covers the whole box, which is mostly water or reef;
  - BioCLIP 2 sees most small, blurry fish as alike;
  - the tracker's missed-frame tolerance (`track_max_missed`) counts frames, so tracks
    break sooner at higher rates. Scaling it to about one second brought 4 fps back to
    ~290 tracks, but those tracks still swapped fish.
- **Decision: stay at 1 fps.** 2 fps may still be worth it on a GPU, for MaxN.
  *Superseded (§12): trials now run at 3–4 fps, and tracking adapts to the rate.*
- **Next lever:** compare colour on the middle of the box (the fish itself), in both the
  tracker and the stitching, with a stricter stitching threshold. Vinci wants to see how
  reviewing goes with the new keys first. *Tested in §12: colour from the middle of the
  box didn't help; the whole box's colour does, when joining pieces.*

**Review keys** (see the README):

- `M`, multiple fish: the card goes back to its tracked pieces, each a new pending card.
- `↑` `↓` select a sighting, then `X` splits there. `P` folds the card into the previous
  one. (`[` `]` came first; the arrows are easier to find.)
- An `O` key for the template's "other …" entry was tried and dropped. Fish on neither
  list are rare, and the search box covers them. The page refuses that entry without a
  name in the comment.
- "Pending only" became a **Status** menu: pending, skipped, reviewed or all.
- Every split, merge and "multiple fish" is logged in a `corrections` table (time,
  action, card, details as JSON) and exported as a `corrections` sheet. It shows where
  tracking and stitching fail, for tuning them and as training data.

**Template-species filter.** The page can show only the cards that have one of the
template's own species 1st among the guesses, in the top 2, or in the top 3. It marks
those cards ★. Of the first real run's 72 cards still to review, that's 27, 43 and 50
cards.

The filter needs one more rule to be useful. 16 of the 27 "1st" cards are distant fish
that all get the same guess. A rule on the fish's largest sighting separates them; the
model's probability doesn't:

| Rule | 1st | top 2 | top 3 |
|---|---|---|---|
| None | 27 | 43 | 50 |
| Largest sighting ≥ 40 px | 10 | 21 | 24 |
| Template guess p ≥ 0.3 | 9 | 10 | 10 |

- Sizes are the box's longer side in the 960 px wide frames the review page saves. 40 px
  there is about 53 px in the detector's 1280 px frames. That's a little stricter than the
  size below which nobody can identify a fish (40 px at 1280 px, §9).
- At 40 px all 16 distant fish drop out, plus one other small card.
- The distant fish score 0.12–0.33, in the same range as the other 11 cards (0.14–0.70).
  A cut at 0.3 still keeps 4 of them, and drops 6 of the others.
- **Added:** a "Fish ≥ 40 px" toggle next to the Show menu, and each card shows its
  fish's size at its largest. (A second group of menu choices came first; the toggle is
  cleaner.) *Then (§12): tiny fish are kept apart and hidden by default, behind a "Show
  tiny fish" toggle.*

**More names.** Vinci suggested 26 more names:

- 19 were already covered, either on the app's list (e.g. moorish idol, filefish,
  lizardfish, scorpionfish, lionfish, flounder) or on the template's.
- Spadefish is the batfish family, so that entry is now "batfish / spadefish". Two names
  for one fish would split the model's guesses between them.
- Coral bream and five small sand-dwelling families (sand diver, worm fish, dragonet,
  pipefish, stargazer) were left out as not critical.

---

## 12. Keeping one fish on one card (Oct 2026)

**What the first corrections showed.** Vinci's first 6 corrections (4 "Multiple fish", 2
splits) broke 21 joins between tracked pieces:

- BioCLIP 2 rated the broken joins 0.82–0.97 similar, so a stricter similarity bar alone
  can't separate them;
- 13 of the 21 had a fish under 40 px on one side;
- a shorter time window (5 s instead of 10 s) only reshuffled the joins: the same 28 cards
  still had 4 or more pieces.

Those answers are few (9 reviewed cards), and Vinci asked not to lean on them. So the rules
below were set from comparisons that need no answers: two fish in the same frame are always
different fish, and two halves of one tracked stretch are the same fish. Vinci's answers
were a cross-check.

**Built**

1. **Tiny fish** (under 40 px at their largest, `tiny_fish_px`) join no other piece, and
   the review page hides their cards until "Show tiny fish" is ticked. Left unreviewed,
   they stay out of the export and out of the progress count.
2. **Tick and move.** Ticked sightings move to a new card with X (`move_out` in the
   corrections), e.g. the other fish after two fish crossed.
3. **4 fps by default.** Trials now run at 3–4 fps. How long a fish may go unseen and keep
   its track is set in seconds (`track_max_missed_s`, 1 s), not frames.
4. **Direction of travel in the tracker** (`track_direction`). A match is scored against
   where the fish was or where it was heading, whichever fits better, so two crossing fish
   keep their own tracks. A first version scored against the heading only; at 1 fps fish
   turn so much that it joined a butterflyfish to a triggerfish.
5. **Colour check when joining pieces** (`merge_min_colour` 0.7, the whole box's colour),
   and a higher BioCLIP similarity bar (`merge_min_similarity` 0.85, was 0.80).
6. **`score` command.** It scores a re-run against a reviewed copy: cards mixing
   different answers, and groups of sightings the reviewer pulled apart that are joined
   again. Sightings are matched by frame time and box, so it works across frame rates.
7. **Training loop** (`learn` command, `learn.py`). A library on the PC holds confirmed
   examples: reviewed projects, and photos sorted by species. Guesses blend in a vote of
   the nearest examples only once that beats the model alone on sources it didn't learn
   from (at least 20 examples, by 5 points). BioCLIP 2 itself is never retrained.

**Tested and not adopted**

- **Colour from the middle of the box** (Vinci's colour rule) told same-fish pairs from
  different-fish pairs slightly worse than the whole box: AUC 0.79 against 0.82, for
  nearby fish of 40 px or more (0.77 for the middle 40%). A fish's own pixels change more
  between frames than its surroundings as it turns.
- **A speed limit across gaps.** Fish moved up to about a tenth of the frame width per
  second, so after a 5–10 s gap they could be almost anywhere. The limit cut 4 of the 18
  bad joins, for 10 more cards, and exceptions for fish that left by an edge or swam off
  let most of them back in. The same idea works better inside the tracker (item 4).
- **Comparing pieces by species guess** (the overlap of their guess probabilities): AUC
  0.82, against 0.86 for the raw BioCLIP similarity, and almost no gain on top of
  similarity and colour.

**Joining rules**, for fish of 40 px or more on the 4 fps copy of the first clip (32
same-fish pairs, 622 different-fish pairs):

| Rule | Same-fish joins kept | Different-fish joins blocked |
|---|---|---|
| Similarity ≥ 0.80 (before) | 100% | 40% |
| Similarity ≥ 0.80, colour ≥ 0.7 | 100% | 53% |
| Similarity ≥ 0.85, colour ≥ 0.7 (now) | 91% | 65% |
| Similarity ≥ 0.90, colour ≥ 0.7 | 75% | 83% |

The same-fish pairs are parts of one tracked stretch, so a fish that returns after a gap
is probably kept less often than shown.

**The first clip at 4 fps, scored against Vinci's 1 fps answers** (all new rows: tiny fish
apart, colour ≥ 0.7):

| Rules | Cards to review | Tiny fish (hidden) | Cards mixing answers | Pulled-apart groups joined again |
|---|---|---|---|---|
| Old | 89 | – | 2 | 8 of 23 |
| New, similarity 0.80, no direction of travel | 48 | 182 | 2 | 9 of 23 |
| New, similarity 0.80 | 48 | 177 | 1 | 6 of 23 |
| New, similarity 0.85 (the defaults) | 59 | 177 | 1 | 5 of 23 |

The remaining mixed card joins a triggerfish with something answered "not a fish" inside
one tracked stretch; tick and move fixes it in one step. Vinci's project now runs at 4 fps
with these rules; the 1 fps project with the earlier reviews is kept next to it as
`bruv_project_1fps`.

**Next**

- Compare with Vinci's manual answer key of the same clip (phase 5).
- More examples. Fully reviewed videos make the scores trustworthy; confirmed fish or
  photos per species feed the library.

**Field habits that help the model** (suggested to BRUV users):

- a marker at a set distance (a rough size for fish near it, a real counting zone, and a
  visibility check);
- the same camera settings on every drop: linear lens, stabilisation off, white balance
  locked;
- a level, still rig, good light and a clean lens;
- ropes and the bait arm away from the middle of the view (the detector took the bait arm
  for a fish 3 times on the first clip);
- a slate or hand signal when the divers leave, which marks where the analysis starts.
