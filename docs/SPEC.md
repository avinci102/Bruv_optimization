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

**Speed-ups for long videos.** Ahead of a full 45-minute run:

- **Tiny fish skip the species model** (`guess_tiny_fish` off). On the first clip at 4 fps
  they were 363 of the 632 crops it would embed, and their guesses were blobs named after
  whatever species looked closest. Cards and joins are unchanged (same 236 cards and
  scores on a re-run); tiny-fish cards just show no guesses.
- **The setup installs the GPU version of PyTorch** when an NVIDIA card can use it.
  `bruv_app/gpu.py` asks `nvidia-smi` for the driver and compute capability and picks
  CUDA 13.0 (RTX 20 series on, driver 580+), CUDA 12.6 (GTX 900 to RTX 40, driver 528+),
  or the CPU build, with a message when a driver update would help. PyTorch 2.14.1 has
  Windows builds for Python 3.13 in both, with a matching torchvision. Untested on a GPU
  so far: the first full run is that test.

Estimate for 45 minutes at 4 fps, scaled from the 2-minute clip (fish activity varies
along a drop, so ±50%):

| Step | Laptop CPU (measured rates) | NVIDIA GPU PC (typical, not measured) |
|---|---|---|
| Detection, 10,800 frames | ~3.5 h | ~10–20 min, mostly decoding |
| Species model, ~6,000 crops (tiny fish skipped) | ~1.8 h | ~1–2 min |
| Tracking, joining, guesses | minutes | minutes |
| Total | ~5–6 h | well under an hour |

**Next**

- Compare with Vinci's manual answer key of the same clip (phase 5).
- More examples. Fully reviewed videos make the scores trustworthy; confirmed fish or
  photos per species feed the library.

**Field habits that help the model** (suggested to BRUV users):

- a marker at a set distance (a rough size for fish near it, a real counting zone, and a
  visibility check);
- the same camera settings on every drop (stabilisation off, white balance locked), with
  the lens or field-of-view mode recorded (`capture`). Whether a linear or dewarped mode
  helps the model is untested, and it changes the area the counts cover (§13, E11);
- a level, still rig, good light and a clean lens;
- ropes and the bait arm away from the middle of the view (the detector took the bait arm
  for a fish 3 times on the first clip);
- a slate or hand signal when the divers leave, which marks where the analysis starts.

## 13. Accuracy foundations: the independent review (Oct 2026)

An independent accuracy and architecture review (V2, 20 Oct 2026, of revision `859d1f2`)
recommended keeping the architecture and strengthening parts of it, in five tasks, under
mandatory scientific safeguards. It judged the earlier joining experiments (§12) weak
evidence: halves of tracked stretches stood in for "the same fish", same-frame pairs for
"different fish", all from one clip. So **no threshold, model or tracker changed here.**
This section builds the tools to measure, and the safeguards. New defaults wait for
annotated footage.

**B0, the baseline:** the settings at `859d1f2`. That means 4 fps, frames decoded at 1280 px
and saved at 960 px (JPEG 85), detector 0.30, the tracker and joining rules of §12, tiny
fish under 40 px with no guesses, and 3 crops per piece, 5 per card, padding 0.1. Vinci's
44-minute run used 55 px for tiny fish. Older versions, B0 among them, still open
projects made by this one: a test pins the frames table to B0's columns, and B0's own
code was run on such a project.

**Found while building**

- **Frame times after a resume were one step late** (0.25 s at 4 fps, 1 s at 1 fps). After
  a seek, ffmpeg's fps filter numbered frames from the first one it decoded. Counts were
  right; times, and any crop taken from the video, were not. Fixed (the filter is anchored
  with `start_time`), and tested with a 29.97 fps video whose frames show their own
  number. Vinci's three projects each ran in one go from the start, so theirs were right.
- **The camera is a DJI Osmo Action 4**, not a GoPro: 2688 × 1512, 10-bit HEVC, 29.97 fps.
  The 2-minute clip is a re-encode of it.
- Decoding chosen frames from the long video stalled until its end (ffmpeg held the last
  frame back), and ran a 16 GB PC out of memory alongside the species model. Both fixed.

**Built, by the review's tasks**

1. **A reproducible benchmark.**
   - Every stage run is recorded (`runs`): settings, code version, the models by
     fingerprint, and whether BioCLIP 2 or the colour-histogram stand-in ran.
   - The footage is fingerprinted, giving a deployment id.
   - Every frame keeps its source frame number (`frame_sources`).
   - The cards as first built are frozen (`original_groups`).
   - Processing ends with a read-only snapshot.
   - An evaluation sidecar (`evaluate.py`) holds annotated boxes keyed by deployment and
     source frame. It scores detection, tracking (A: changes of fish inside a piece) and
     joining (B: wrong joins of pure pieces) separately, plus MaxN. Boxes are matched one
     to one, and `score` now matches the same way.
2. **Native-resolution crops.**
   - `grab_frames` decodes exact source frames.
   - `crop-test` (E1) compares A (the 960 px JPEG), A′ (the same frame without JPEG),
     B (1280 px) and C (the original frame). Every arm uses the same sightings, boxes
     and padding. Each crop is checked against the saved frame first.
   - Results are split by native size, image position and tiny or not.
   - Capture details (`capture`) record the camera, size, lens mode, stabilisation and
     housing.
   - The embedding cache key now names the weights, the model's input and the crop
     source.
   - E11 (correcting the lens) needs a calibration of this camera and housing, and isn't
     done.
3. **Continuity.**
   - The tracker records each link's cost, its margin over the next-best option, and how
     alike the box is to the track so far.
   - The merge records every join it weighed (`merge_edges`, about 15,000 on the
     44-minute video), with its evidence and decision.
   - Cards get warnings (`diagnostics.py`): pieces less alike than a join needs (the A~B,
     B~C, A unlike C chain), pieces that clearly disagree on species, or two boxes at once.
     A flagged card is answered for all its pieces only after the reviewer confirms they
     are one fish.
   - `evaluate` reports changes of fish by link margin.
   - The tracker's missed-time allowance is pinned by tests. At 4 fps, 1.0 s bridges
     1.25 s; at 2.5 fps, 1.0 s rounds to 2 frames (1.2 s). (§14 lets 0 mean consecutive
     frames only, and makes 0.25 s the default.)
4. **Equipment and detector threshold.**
   - Equipment (`E`, `equipment.py`) becomes a reference, and look-alike cards are queued
     for review. Only cards whose every sighting matches can be answered in one go;
     batches can be undone.
   - The detector keeps boxes down to 0.1 (`det_floor`), but only 0.30 and up are tracked.
   - `evaluate --thresholds` draws detection curves without re-running the detector.
5. **Labels.**
   - Each answer keeps its scope, the detections it was given for, and counts only
     those.
   - The export reports how complete the review is, tiny fish apart, and marks MaxN
     provisional until done. It also lists possible double counts.
   - Record rows use their own `sighting_gap_s`, not the merge window.
   - The library groups examples by recording and checks itself nested, with at least 3
     recordings. Cards with unconfirmed warnings stay out of it.

**The review's safeguards, and where they live**

| Safeguard | Where |
|---|---|
| Detection, identity, grouping, label and abundance kept apart | detections and runs; `original_groups`; cards; `reviews` with scope; `metrics.py` |
| Confirmation tied to a reviewed scope; a flagged card inspected first | `reviews.scope`; warnings and confirmation (`server.py`) |
| One fish, one detection; no deduplication by card | `double_count_checks`; MaxN counts detections per frame |
| Review completeness at export; hidden tiny fish not counted as reviewed | `review_status`, `review_by_minute`, provisional MaxN; the page's progress line |
| Record-row grouping apart from merge tuning | `sighting_gap_s` |

**First measurements on the 44-minute video** (exploratory, few answers)

- **Warnings:** 83 of the 437 cards with several pieces (959 cards not tiny).
  - 75 have pieces that look different.
  - 29 have pieces that clearly disagree on species. Flagging any difference in top
    guess would have marked 79; ordinary doubt between look-alikes (butterflyfish or
    angelfish) was most of it.
- **Equipment:** marking one bait-canister card queued 94 cards.
  - The 70 that match throughout were all the canister.
  - The 24 left out, each with sightings elsewhere or unlike the rig, included three
    whose best view is a real fish joined onto a canister track, and a diver's hand.
- **Alignment:** at 11 spots across the video, the saved review frame differed from its
  own source frame by about 3 grey levels, and from the neighbouring frames by 4.0–4.6.
  Its frame numbers are right.
- **E1, 20 pieces (47 crops, all aligned), no answers yet:**

| Crops from | Top guess differs from today's (crops) | (pieces) | Mean top score |
|---|---|---|---|
| A, review JPEG (today) | – | – | 0.58 |
| A′, same 960 px frame, no JPEG | 15 of 47 | 4 of 20 | 0.59 |
| B, 1280 px decode | 20 of 47 | 8 of 20 | 0.60 |
| C, original 2688 px frame | 20 of 47 | 9 of 20 | 0.62 |

Many guesses sit close to a tie, so even JPEG loss tips a third of them. Which source is
right more often needs answered cards.

**Vinci's projects, frozen after the first answers (24 Oct).** Both review projects were
processed before cards were frozen at build. Their cards as first built were rebuilt by
re-running tracking and joining on a copy (same detections, embeddings and settings). Every
card nobody had touched came out identical: 4,520 of 4,520 on the 44-minute video, and 230
of 230 on the 2-minute clip. Those cards are now frozen in each project, with the tracker's
link evidence and the merge's join evidence, plus a snapshot.

The first fixes, scored against the cards as built:

- 44-minute video: 8 fixes on 7 cards. Five cut inside one tracked piece, where the tracker
  followed the wrong fish. Three separated pieces of different fish that had been joined:
  unicornfish-looking pieces that the species model can't tell apart.
- No card warning had flagged those 7 cards. But at 4 of the 6 places where a tracked piece
  was cut, the tracker's record had marked that link as one of the card's two closest calls.
- Close calls are too common to flag (about 1 link in 9, on 519 of 706 cards with links).
  Showing each card's closest call is the useful form.
- 2-minute clip: the warnings caught 2 of its 3 joining errors.

**Families first (24 Oct).** Vinci asked whether identifying the family first, then the
species, would help. Measured with the model's own scores:

| | Exact answer | Family of the top guess | Family by adding up its names |
|---|---|---|---|
| 26 answered cards | 21 | 22 | 22 |
| Answer key, grouper sightings (29) | 0 | 13 | 17 |
| Answer key, snapper sightings (20) | 0 | 2 | 2 |
| Answer key, parrotfish sightings (6) | 0 | 0 | 0 |

- Most answers so far are already family-level names (unicornfish, butterflyfish), so for
  them the species and the family are the same.
- Adding up a family's names helps groupers, but on cards the model is unsure of, long lists
  win by their length: sharks or rays became the top family for 64 of the 962 cards not tiny.
  So families are ranked by their best name.
- Snappers and parrotfish go unrecognised either way: the model names other families for
  them, which only a better model, better prompts or better crops can change.
- The review page offers the family-first step (`families.py`). The export counts MaxN per
  family, frame by frame.
- Next to try (E8): ask the model about family names directly, rather than adding up
  species.

**Next, and what each needs**

- **Annotated footage.** The review's design is 12 deployments, a 20 s random window and
  a 20 s challenge window each, split by deployment before any tuning. A page to draw
  missed fish's boxes would make the sidecar practical; `annotation-sheet` starts one
  from a run's boxes.
- **Experiments E2–E4** (tracker and joining sweeps), card-level E6 (re-tracking at each
  threshold) and E10 (detector sizes), on those annotations.
- **E8:** taxonomy (stable taxon ids and ranks), abstention and calibration. Needs the
  template's species list and labelled examples.
- **E9:** review tiers, tested with reviewers.
- **E11:** a lens calibration for the camera and housing.
- Later: confusion and error-factor tables, once there are enough verified examples.

## 14. After the first 185 answers (24 Oct 2026)

Vinci answered 185 cards of the 44-minute NP339 drop, in about an hour of active review,
and found it slow going: the same fish and the same mistakes again and again, and the bait
arm, which never moves, answered card after card. Measured on those answers:

- **Where the time went.** 131 of the 185 answers were "not a fish": 121 the rig (115 the
  canister), at about 2 s each through the equipment queue. The 49 fish answers took about
  44 s each, because nearly every fish card needed fixing: 109 fixes (80 sightings moved
  out, 16 splits, 13 "multiple fish"). 43 of them involved the rig: 25 moved rig sightings
  out of a card, 18 moved a fish out of a rig card. The other 66 separated two fish. The
  tracker made 95 of the 109 errors, the joining step 14.
- **Why the equipment queue fell short.** It matches a card by the box's place and
  colour, and the detector boxes the arm differently each time (the canister, the canister
  and pole, larger boxes): 5 references and still 121 answers. It also acts once cards are
  built, when the arm has already been joined to fish.
- **The model repeats its mistakes.** Its first guess was right on 28 of 53 fish cards. It
  called unicornfish "napoleon wrasse" 7 times; the canister "tasselled wobbegong" 54
  times and the pole "cornetfish" 26.
- **The same fish comes back.** 29 unicornfish cards over 20 minutes; 19 of them began
  less than a minute after an earlier unicornfish card ended.

**Changed.** New defaults; a project made before keeps its settings until rebuilt
(`Settings.load` reads a missing new setting as off).

1. **Still objects are hidden before tracking** (`still.py`). Each box is compared with
   the scene's background: the median of frames 2 s apart over the minute either side, at
   a quarter of the saved size, where a pixel differs when a colour channel moves by more
   than 25 levels. On the answered sightings, half of the equipment boxes had 1% or less
   of their pixels changed, half of the fish boxes 47% or more. Hiding boxes at least 6%
   of the frame wide with under 10% changed hides 91% of the sightings answered "not a
   fish" and none of the 1,689 answered as fish. Without the width limit it also caught
   33 sightings of tiny, distant fish. A sample of what it hides on unanswered cards showed the
   canister, the pole, the float line and rocks. The cut-offs were chosen on these same
   answers; the gap is wide, but the next drop is the real test. Hidden boxes stay in the
   project (`det_change`), make no cards and aren't counted. The review page shows a sample,
   since a fish motionless for over a minute would be hidden too.
2. **Shorter tracks, and contested links refused** (`track_max_missed_s` 0.25: at 4 fps a
   fish may miss one frame; `track_refuse_contested`: no link where another fish or box
   was the better match). The tracker re-run on the same detections, scored on the 49
   fish-to-fish links Vinci's fixes cut (the rig's links aside):

| Tracking (still objects hidden) | Pieces | Wrong links avoided |
|---|---|---|
| 1.0 s, as before (rig tracked) | 5,483 | 0 of 49 |
| 1.0 s | 5,202 | 1 |
| 0.25 s | 7,964 | 14 |
| Consecutive frames only (0) | 10,772 | 20 |
| **0.25 s, contested links refused** | **9,794** | **33** |
| Consecutive frames only, contested refused | 12,165 | 32 |

   Three quarters of the wrong links (37 of 49) joined consecutive frames, which a shorter
   gap can't reach; 4 bridged one missed frame and 8 more. Refusing contested links reaches
   the consecutive ones.
   More pieces mean more cards of one fish; the look-alikes (4) answer those together.
   `missed_frames` now allows 0 (consecutive frames only); it used to raise 0 to one frame.
3. **Rebuilding keeps the answers** (`rebuild`). The answers are set aside per detection
   first, so a stopped rebuild loses none. A new card gets an earlier answer only when
   every one of its detections had that answer; cards with some answered and some new
   detections, or with different answers, wait for the reviewer and are counted. New
   cards never take the number of a card that's gone. Before, a card merged away could
   lend its number, and with it its answer's scope, to the next new card.
4. **Look-alikes** (`lookalike.py`): a card's look is its best crops' mean embedding, and two
   cards look alike at 0.85, the merge's bar for two pieces of one fish. A card that looks
   like an answered one shows that answer for one key (`R`); after an answer, its
   unanswered look-alikes can be answered together from a grid of their pictures (`L`),
   unticking any that differ, with undo; cards with warnings are left out. A sort puts
   look-alike groups together. Replaying the 185 answers in order, the closest card
   answered earlier named 40 of 45 fish cards right and was never wrong (5 had nothing
   alike enough); the model alone got 24 of 45. It named 106 of 110 "not a fish" cards
   right. This learns the fish of one video as you go; the library is still what carries
   them to the next video.
5. **Species models side by side** (`modeltest.py`, `model-test`): the same crops of the
   answered cards scored by today's BioCLIP 2, another model, and both together, plus how
   often earlier answers would have named them. BioCLIP 2.5 Huge (Imageomics, Feb 2026)
   has a ViT-H/14 image encoder, about twice BioCLIP 2's ViT-L/14, trained on
   TreeOfLife-200M plus 19M images. Imageomics report 61.3 against 55.6 on their zero-shot
   species benchmarks and 48.5 against 39.8 on FishNet; none is underwater video, and on
   plankton both stay near 5. Its 3.9 GB file is mapped like BioCLIP 2's. On Vinci's 49
   cards answered with a name, the same crops for both:

| | First guess right | Right family first | In the top 3 |
|---|---|---|---|
| BioCLIP 2 | 28 | 29 | 36 |
| BioCLIP 2.5 | 23 | 26 | 34 |
| Both, scores averaged | 25 | 28 | 32 |
| BioCLIP 2, plus earlier answers | 48 | 49 | – |
| BioCLIP 2.5, plus earlier answers | 46 | 47 | – |

   By family (right family first, BioCLIP 2 against 2.5): unicornfish 14 of 29 with
   either, butterflyfish 9 against 7 of 9, triggerfish 5 against 2 of 8, grouper 1 against
   3 of 3. BioCLIP 2.5 calls the canister a crown-of-thorns starfish (53 cards) where
   BioCLIP 2 says tasselled wobbegong. It took 3.0 s a crop on this PC, against about 2 s
   for BioCLIP 2. Few cards, three families, and many cards of the same few fish: no
   reason here to switch models or to average them, while earlier answers lift either to
   nearly every card.

   On the answer key's clip_1 sightings, a sighting counts as found when a card on screen
   then has the right family first. A median of 19 cards share the screen with a sighting,
   so each model is set against what calling families at random, at its own rates, would
   find:

| Key sightings | BioCLIP 2 | by chance | BioCLIP 2.5 | by chance |
|---|---|---|---|---|
| Grouper (29) | 13 | 12 | 27 | 20 |
| Snapper (21) | 3 | 4 | 9 | 7 |
| Parrotfish (6) | 0 | 0 | 0 | 0 |

   BioCLIP 2 does no better than chance on these families; BioCLIP 2.5 calls far more fish
   groupers (53 of the 822 cards tested), and finds somewhat more grouper sightings than
   chance would. Judged only by the card overlapping each sighting longest, BioCLIP 2.5
   finds 5 of the 29 grouper sightings and BioCLIP 2 none. The key has no boxes, so it
   can't say which card is the grouper: answered grouper and snapper cards are what would
   settle it. BioCLIP 2.5 calls unicornfish "bumphead parrotfish" (11 cards) as BioCLIP 2
   calls them "napoleon wrasse" (7). Embedding the 2,300 crops took about 10 s each here.

**Rebuilt on a copy of the NP339 project** (the rules above, the answers carried):

| | Before | After |
|---|---|---|
| Cards, not tiny | 1,089 (184 answered) | 1,145 (55 answered) |
| Tiny cards (hidden until asked for) | 3,579 | 6,790 |
| Boxes hidden as still objects | – | 1,622 |

- 78 cards got their earlier answer (55 not tiny). Most of the 121 rig answers aren't
  needed any more: the rig is hidden.
- 95 cards hold sightings answered before and new ones, mostly unicornfish (37),
  butterflyfish (25) and triggerfish (19): shorter pieces joined with pieces nobody had
  answered. They wait for the reviewer, and say what most of their sightings were
  answered, for one key (`R`). 5 cards mix different answers, 4 of them a fish with "not
  a fish" (rig remnants).
- 4 sightings answered as fish are now hidden as still objects.
- Vinci's 141 separations: 35 separated the rig, now hidden; 55 stay apart; 51 are on one
  card again. In 48 of those 51 the separated part was never answered; 1 has the same
  answer on both sides, and 2 put different answers together again (one by the tracker,
  one by the merge).
- The 1,090 pending cards, not tiny, fall into 99 look-alike groups. The biggest card has
  103 unanswered look-alikes: 28 at 0.90 or more, 40 between 0.85 and 0.90, and 35 with
  warnings, left out of answering together.
- On the original answers, pairs of answered cards alike enough to suggest each other had
  the same answer 98.7% of the time at 0.85, 99.8% at 0.90, and all but 2 of 4,928 pairs
  at 0.93. The look-alike grid starts with only the cards at 0.90 or more ticked.

**Considered and not adopted (an outside list of suggestions)**

- Training the detector on rig images: needs a GPU and a training set; the still test does
  the job and keeps fish in front of the arm.
- A 0.5 s gap: 8 of the 49 wrong links spanned more than 0.5 s.
- A stricter 10-second join: the joining step made 14 of the 109 errors, and a stricter
  join means more cards of one fish.
- 5–10 varied crops per card: guesses already pool up to 5 crops; pooling every saved crop
  scored 28 of 49, the same as today, and voting across crops 24. Crops not yet embedded
  are still untested.
- Fishial as a second opinion (`fishial.py`, `model-test --model fishial`). Its 352 MB
  classifier (v0.10.2: DinoV2 with sub-center ArcFace) knows 866 species by scientific
  name, many of them American and European fishing species; its genera are mapped to the
  template's families. It is fast here (0.39 s a crop) but did poorly on this footage:

| | Right family first, your 49 named cards | Key grouper (chance) | Key snapper (chance) |
|---|---|---|---|
| BioCLIP 2 | 29 | 13 (12) of 29 | 3 (4) of 21 |
| BioCLIP 2.5 | 26 | 27 (20) | 9 (7) |
| Fishial | 8 | 16 (12) | 9 (8) |

  By family on the named cards it got triggerfish 7 of 8 (BioCLIP 2: 5), but unicornfish 0
  of 29, butterflyfish 1 of 9 (it calls them batfish) and grouper 0 of 3. With BioCLIP 2 it
  agreed on the family for 10 cards, 6 of them right; of the 39 where they disagreed,
  BioCLIP 2 was right on 23 and Fishial on 2, so agreement isn't a confidence signal
  either. Having no "other" answer, it calls the canister a doctorfish (45 cards) or a
  whitetip reef shark (39). Not adopted.

## 15. Pilot workflow decisions (26 Oct 2026)

Agreed with Vinci in the workflow session, for the pilot (alpha on 30 Oct, two volunteer
waves, handoff on 10 Nov) and the target state: an open-source program that anyone runs
offline on their own device, GPU optional.

**What the outputs rest on** (this replaces the earlier rule that continuity only
deduplicates review cards):

| Output | How it's counted | What it depends on |
|---|---|---|
| Individual sightings (record rows of type "individual") | One tracked chain is one sighting, while the fish stays on screen; time in and out and behaviour come from the card | Reliable cards: one fish per card, with no swaps even within a species |
| Schools (type "school", school min/max) | Counted frame by frame across the school's frames | Correct boxes and labels in each frame |
| MaxN per species | The most fish of that species in any one frame | Correct labels |

- MaxN goes on the summary sheet only, with the frame it comes from so it can be
  checked. It isn't used for other classifications, isn't summed across time, and isn't
  a total of individuals.
- Continuity is therefore a primary accuracy goal. Cards are measured against clicked
  ground truth (below), not assumed.
- Not built yet: the export still groups same-species fish within `sighting_gap_s` and
  counts each row's busiest frame (`metrics.compute_sightings`). That changes in the
  wave-1 window.

**Answers have two levels.** The family is required and the species is an optional drill-down.
A family-only answer is the "species unknown" case. `U` stays "Unknown fish". Tiny fish
are out of the volunteers' scope.

**Quick confirm now, auto-accept later.**
- The pilot's quick-confirm queue offers cards at least 0.90 alike to a card a person
  has answered (99.8% of answered pairs that alike had the same answer, §14).
- A person accepts or rejects every card. Each decision is logged with the model's
  confidence and the similarity.
- Auto-accept comes only after the logs give at least directional evidence. The target
  is 85% family-level accuracy for the suggestions shown, on deployments not used for
  tuning. The model's own first-guess accuracy is tracked apart.

**Continuity: how it will be improved**
- A click-the-same-fish tool. It follows one fish frame by frame: a click moves to the
  next frame, there are keys for "here but no box" and "not in this frame", and it keeps
  stepping 5–10 s after the fish leaves in case it returns. It writes the evaluation
  file's `individual` column. It also gives a pattern report of where the tracker
  disagrees: crossings, turns, lighting, frame edge, missed detections. It is built in
  wave 1, for Vinci first, and for volunteers in wave 2 if the flow holds.
- Candidate changes, each measured on clicked tracks, volunteers' fixes and the answer
  keys before it becomes a default:
  - continue tracks through faint boxes (0.10–0.30), never starting a card from one;
  - a smoother motion prediction;
  - colour taken from the fish's own pixels, using the still-object background;
  - BioCLIP appearance at every link on a GPU, with today's rules on CPU.
- Not adopted: joining or splitting pieces on species guesses (they flip when a fish
  turns).
- 4 against 8 frames a second is tested offline on dataset A. `score` matches sightings
  across frame rates. Datasets A and B are processed at 4 fps, to stay comparable with
  NP339.

**Built for the alpha** (branch `claude/pilot-workflow`, before the 30 Oct freeze):
- Two-level answers. `families.resolve_answer` checks the answer and returns the recorded
  name and family; `reviews.family` stores the family. `F` is always offered in the
  family step. A family with no "family" entry in the template (moray eels) is recorded by
  its key and exported as the "other" entry with the name in the comment.
- Quick-confirm queue (Show → ⚡ quick confirm), for cards at least 0.90 alike to an
  answered card: both pictures side by side, `Y` takes the answer. Each answer's
  `reviews.details` holds `suggestion` (the look-alike card and the model's first guess),
  `matched` ("same", "same family" or "different"), `queue` ("quick" when answered from
  the queue) and `seconds` (time on the card, up to 600).
- The look-alike list (`L`) scrolls, in bands (≥0.93, 0.90–0.93, 0.85–0.90 unticked),
  shows one picture per tracked piece, and keeps the answered card pinned for comparison.
  A card is left to answer one by one if any piece is under 0.85 alike to the answered
  card (`lookalike.every_piece_alike`), or it has warnings. The `suggestion_log` table
  records each card shown, with the decision: answered, left or unticked.
- Best views prefer side-on, whole fish (`views.best_view`: quality × length-to-height,
  less at the frame's edge).
- Review window (`review-window` command, meta `review_window`): cards wholly outside it
  are hidden and not counted, and neither are detections outside it.
- Tiny fish are out of scope: unanswered, they don't make the counts provisional; the
  `review_status` sheet counts them.
- `check-setup` says what processing will run on (graphics card, PyTorch build, device).
- For static hosting, `lookalike.neighbours` precomputes each card's look-alikes at publish.

## 16. Hosted review for the pilot (26 Oct 2026)

Agreed with Vinci in the architecture session. The program stays open-source and
offline-first: each user processes and reviews on their own device, GPU optional, and
Vinci is never a choke point. For the pilot, volunteers review on a website instead, as a
way to collect data. The website is an optional kit that a school deploys on its own free
Cloudflare account. Setup, limits and measurements: [HOSTED.md](HOSTED.md).

**Decisions**
- No tunnel and no server on the processing PC. The GPU PC only processes and publishes;
  it can be off while volunteers review. (This replaces the tunnel plan in the shared
  rules.)
- Static Cloudflare Pages site on a new free account for the project: the pictures and
  card data as files. A Pages Function with a D1 database handles sign-in and stores
  answers, fixes and feedback.
- Sign-in with an access pass or the invited email, with no password or verification:
  the data isn't sensitive. The pictures are public files; sign-in protects the answers.
- Each volunteer gets 2 videos per wave (about 10 videos a wave). Their copy of their own
  video is the review of record (the answer key). Overlap is only targeted: calibration
  assignments of chosen cards, each answered in the volunteer's own copy.
- Wave 2's pictures replace wave 1's. Answers stay in the database.
- Live fixes online: split, move ticked sightings, and "multiple fish". Joining two
  cards waits. Tiny fish and cards outside the review window aren't published.
- One page, one API. The review page is unchanged. The site's copy loads two scripts
  that answer the same `/api` requests and picture addresses in the browser.
- Corrections can be consolidated into the master project with traceability: `pull`
  records the volunteer (`reviews.reviewer`, a new column, added by the add-only
  migration), the time, the site's card number, the corrections log, and snapshots before
  and after.
- No fallback to a local server. Volunteers running the program locally is a later
  option.
- Each volunteer's answers to the look-alike suggestions are kept for training only if
  they opt in (target state; the pilot logs them, `suggestion_log`).

**How it works**
- `publish` writes each video as static files, using the review server's own endpoint
  code:
  - one WebP sheet per card, with the close-ups the page shows and up to 8 whole scenes;
  - the card data;
  - the float16 BioCLIP descriptions, so the browser can redo the light work after a fix.
- `hosted/core.js` mirrors the server's answers, including after a fix: guesses,
  families, warnings, pieces, look-alikes and look-alike groups.
- `hosted/runtime.js` connects it to the page. It isn't a service worker, because those
  are blocked in some browsers and managed laptops.
- What a volunteer does is an event log (D1 `events`). It is saved before it shows as
  done, and replayed on top of the published cards each time the page opens.
- `pull` replays one volunteer's log onto the project with the store's own methods. It
  refuses a log for a different video than the project was published as
  (`meta.published`).

**Checked**
- `tests/test_hosted_js.py` (needs Node.js) drives `core.js` with the page's own
  requests: answers, warnings, a look-alike batch and undo, and every online fix. It
  checks that a reload replays to the same state. It then pulls the events into the
  project and compares the Python server's answers with the browser's, field by field.
  They match. The same was checked by hand in a browser on the demo.
- `tests/test_hosted.py` covers publish, the volunteer list, the /sync API (through the
  local copy of the site), pull and the commands.
- The 44-minute test video publishes to 1,153 files and about 263 MB. Its largest file,
  17.8 MB, is under the 25 MiB limit.

**Open**
- Descriptions over 25 MiB (a video about 60% longer than the test video) would need
  splitting.
- A report of agreement between volunteers on the calibration cards.
- Joining two cards online.
- Following a fish online (wave 2, once Vinci has tried it on the PC; not built yet). It
  steps through every sampled frame, not just the card's frames: from the card's start
  minus `gone_s` to its end plus `gone_s`, with every box in those frames (faint and
  still-hidden ones too, from `det_floor` up) and their card numbers. That's much more to
  publish than a card's sheet. Its state (`follows`, `follow_marks`) and next step
  (`follow.next_step`) could run in the browser as the review does.
