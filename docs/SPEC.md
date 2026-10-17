# BRUV Fish ID Assistant — Spec & Handoff

Status: **MVP built (phases 1–4), tested on synthetic data only.** This document captures
every decision made so far. Items marked **[Decided]** came from the project owner (Vinci).
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
   - **Restrict the labels to the template's species dropdown.**
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

1. **The real template `.xlsx`.** Add it to `templates/`. It's needed for the hidden
   columns, the `info` fields, and the dropdown contents.
2. **A sample clip**, 1–3 minutes of real footage. Don't commit large videos; share via
   Drive or attach.
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
   boxes, which openpyxl drops with no warning, are detected and flagged. *Still to
   check: open an export in real Excel. LibreOffice reads the re-attached dropdowns
   correctly.*
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
2. Pipeline — **done.** Resumable detection, tracker, continuity, BioCLIP 2 with a
   histogram fallback. RF-DETR and BioCLIP code paths are written against their current
   APIs but **not yet run with real weights** (the build sandbox couldn't download them).
3. Review UI — **done.** Cards, keys 1/2/3/U/N/S, searchable species, optional fields,
   split/merge, progress.
4. Export — **done.** Record rows by header name, MaxN tab, app_details tab with model vs
   user agreement, relative size and the saved best crop.
5. Validation on real footage — **next.** Needs the real template and a sample clip.
6. Windows installer — not started (`scripts/*.bat` cover setup for now).
7. Scale markers — not started.

### Known risks to watch on real footage

- **Schools** create many simultaneous cards (one per fish), since cards on screen
  together can't merge. If that is slow to review, add "apply this label to the similar
  cards in this time window".
- **Decode speed.** 4K GoPro HEVC decoded on the CPU can take as long as detection.
  `-hwaccel auto` is on by default; check that it engages on Vinci's PC.
- **MaxN at 1 fps** can slightly undercount compared with frame-by-frame manual review.
  Compare it in phase 5, and raise `--fps` if needed.

### New open question

7. Should the export **replace the example row 2** in `record`, or append below it as
   it does now?
