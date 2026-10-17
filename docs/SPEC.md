# BRUV Fish ID Assistant — Spec & Handoff

Status: **spec only, no code yet.** This document captures every decision made so far so
work can start in Claude Code without re-asking. Items marked **[Decided]** came from the
project owner (Vinci). Items marked **[Proposed]** are Claude's recommended defaults —
change them freely.

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
   hardware decode where available), with OpenCV as the fallback. Default is **1 fps on
   CPU**, 2–5 fps with a GPU. Store a downscaled JPEG of each sampled frame for the review
   UI, at roughly 150 MB per 30 minutes at 960 px.
3. **Detection.** Use the **Community Fish Detector**
   (github.com/WildHackers/community-fish-detector).
   - It is a YOLOv12x model with a single `fish` class.
   - It was trained on a dataset of over 1.9M images built from 17 open datasets.
   - The README says it expects `imgsz=1024`. It is **AGPL**, which matters for any
     commercial distribution.
   - On CPU, use `imgsz=640` as the default for speed. Put it behind a `Detector`
     interface so it can be swapped for an ONNX/OpenVINO export later, which is a big win
     on Intel CPUs and also shrinks the installer by dropping torch.
4. **Tracking.** Use a lightweight in-house tracker: IoU plus centre distance, matched with
   the Hungarian algorithm. It avoids an ultralytics dependency so torch can be dropped
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
  detect.py      Detector interface; CFD YOLO impl; fake for tests
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
    once a human was in the loop.
- **Fishial.ai** self-reports 70–90% accuracy; this is a vendor claim.
- **VIAME** (NOAA/Kitware) is the established open-source alternative for detection,
  tracking, and annotation, and is worth comparing against.
- **General vision-language chatbots** score poorly on open-ended fish species ID (the
  VLM4Bio benchmark); don't rely on them.
- **The biggest accuracy levers after human review:**
  - restricting labels to the regional species list;
  - fine-tuning on a few hundred labelled crops from the owner's own footage.
