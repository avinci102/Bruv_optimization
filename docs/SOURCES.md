# Sources and references

Everything outside this repository's own code that informed it: where it came from, what
it was used for, and whether it was checked. All sources were collected on **2026-10-17**
while building the MVP and during a follow-up audit.

## Summary

- **Other apps: none.** No connected apps or services (Notion, Gmail, Google Drive,
  Google Calendar, Slack, or any other) were read from or written to.
- **Production sources: none.** No production systems, databases, private datasets or
  personal data were used.
- **Other repositories: public, open-source ones only, read-only.** None of them is yours
  or private. They are listed in [section 3](#3-external-repositories-public-read-only).
- **Copied code: none.** Two things follow published examples:
  - the detector's load and predict calls match the Community Fish Detector's own
    inference script;
  - BioCLIP 2's 80 prompt templates are imported from the `open_clip` package at run time,
    not copied into this repo.
- **Unverified items are marked.** Some facts come from general knowledge rather than a
  source checked here. They are listed in [section 7](#7-from-general-knowledge-not-checked-against-a-primary-source)
  so they can be confirmed later.

Status labels used below:

- **verified**: checked against a primary source or by experiment.
- **partly**: checked against secondary sources only.
- **unverified**: not checked.
- **blocked**: the source exists but couldn't be reached from the build sandbox.

---

## 1. Inputs from inside this project

| Source | Used for | Notes |
|---|---|---|
| `docs/SPEC.md` (first version) | All requirements, decisions, architecture, phases | Written in an earlier Claude planning chat with the owner and committed as `e233720`. Its **[Decided]** items come from the owner. |
| `CLAUDE.md` | Working rules (CPU-first, template rules, MaxN, tests) | Same planning chat. |
| The owner's Excel viewing template (`…V viewing template v.2024`) | Sheet names, `record` headers, hidden columns K/L/P/Q/U, example row 2 | Known **only from a photo** described in the spec. The real file isn't in the repo. The mock template (`bruv_app/demo.py`) **invents** the A–F header names, the `info` fields and the type/sex/behavior lists. |
| The owner's message: a ~750 MB 2-minute clip and a ~40 MB upload limit | The `shrink` command's default `--max-mb 38`, and the README example | A request from the owner, not an external source. |

## 2. Other apps, services and production systems

None were used. Several connectors were available in the session (Notion, Gmail, Google
Drive, Google Calendar, and others), and none of them was called. No production
database, API or private data source was read.

## 3. External repositories (public, read-only)

| Repository | What was read | Used for | Status |
|---|---|---|---|
| [filippovarini/community-fish-detector](https://github.com/filippovarini/community-fish-detector) (the spec's `WildHackers/community-fish-detector` URL resolves to the same README) | `README.md`, `requirements.txt`, `rf_detr_batch_inference.py` | Detector choice: YOLOv12x is deprecated, RF-DETR Nano/Small/Medium replace it. Also checkpoint names, resolutions (640/1024), AP figures, the `rfdetr>=1.8.3` floor, the `from_checkpoint`/`predict` calls, and the 0.3 threshold from the README example. | verified |
| [Imageomics/bioclip-2](https://github.com/Imageomics/bioclip-2) | `README.md`, `LICENSE`, `src/evaluation/classification.py`, `slurm/eval.sh` | Model id, ViT-L/14 (768-d), MIT license, and how zero-shot class embeddings are built (80-template ensemble; taxonomic plus common names) | verified |
| [Imageomics/pybioclip](https://github.com/Imageomics/pybioclip) | `src/bioclip/_constants.py`, `src/bioclip/predict.py` | Confirms `hf-hub:imageomics/bioclip-2`, the 224×224 "squash" preprocessing and the template ensemble | verified |
| [roboflow/rf-detr](https://github.com/roboflow/rf-detr) | `LICENSE`, README license section | Apache-2.0 for the package and the Nano/Small/Medium/Large classes. "Plus" models are PML 1.0 and aren't used. | verified |
| [LibreOffice/core](https://github.com/LibreOffice/core) (`sc/source/filter/inc/worksheetfragment.hxx`), [EPPlus](https://github.com/EPPlusSoftware/EPPlus) (`ExcelDataValidation.cs`) | Source comments and code | Independent evidence for the Excel x14 data-validation extension and its URI | partly (Microsoft's spec was blocked) |
| [nkrabben/digphoto](https://github.com/nkrabben/digphoto) (`…/gopro-action/file-naming.md`) | GoPro file-naming summary, which cites GoPro's own article | GX/GH chapter naming, and GL low-res proxies being excluded | partly (GoPro's site was blocked) |
| [HandBrake/HandBrake](https://github.com/HandBrake/HandBrake) (`preset/preset_builtin.json`) | Preset list | The "Fast 720p30" preset named in the README | verified |
| [filippovarini/sharktrack](https://github.com/filippovarini/sharktrack), [Wye-Foundation/Fishial-Fish-Identification](https://github.com/Wye-Foundation/Fishial-Fish-Identification), [VIAME/VIAME](https://github.com/VIAME/VIAME), [Imageomics/VLM4Bio](https://github.com/Imageomics/VLM4Bio) | READMEs and licenses | Checking the background research in SPEC §7 (see [section 8](#8-research-cited-in-the-spec-inherited-from-the-planning-chat)) | partly |

## 4. Package registries and package source code

| Source | What was checked | Used for | Status |
|---|---|---|---|
| PyPI JSON API (`pypi.org/pypi/<name>/json`) | Versions, licenses and dependencies of every package in [section 9](#9-licenses-of-dependencies-and-models) | The license table; the `rfdetr`/`open_clip` pins | verified |
| `rfdetr` 1.11.2 and 1.8.3 wheels (PyPI) | `from_checkpoint` (the `device`, `resolution` and `trust_checkpoint` arguments; safe loading), `predict` (input types; boxes returned in input-image pixels; list vs single return), the `model_config.resolution` defaults | `bruv_app/detect.py` | verified (read, not run with weights) |
| `open_clip_torch` 3.3.0 wheel (PyPI) | `create_model_and_transforms` (return order, `image_resize_mode`), `get_tokenizer('hf-hub:…')`, `logit_scale` (log space), `encode_image`/`encode_text`, `OPENAI_IMAGENET_TEMPLATES` | `bruv_app/classify.py` | verified (read, not run with weights) |
| `imageio-ffmpeg` 0.6.0 (installed, plus the Windows wheel) | `read_frames`/`write_frames`/`get_ffmpeg_exe` behaviour, the frame-size warning in `_parsing.py`, and the bundled ffmpeg builds and their licenses | `bruv_app/video.py`, licensing notes | verified |
| `openpyxl` 3.1.5 (installed) | `worksheet/_reader.py` (drops extensions and warns), `xml/constants.py` (`{CCE6A557-…}` = Data Validation) | `bruv_app/template.py` | verified |

## 5. Models

| Model | Source | Used for | Status |
|---|---|---|---|
| Community Fish Detector RF-DETR Nano 640 / Small 1024 / Medium 1024 (release `2026.07.06-release`) | CFD README and releases | Fish detection | **Weights not downloaded** (blocked), so never run here |
| BioCLIP 2 (`hf-hub:imageomics/bioclip-2`; Gu, Stevens et al., NeurIPS 2025, arXiv 2505.23883; trained on TreeOfLife-200M) | Imageomics GitHub repos | Zero-shot species guesses and crop embeddings | **Weights not downloaded** (Hugging Face blocked), so never run here |

## 6. Experiments run while building (no external source)

- **ffmpeg `fps` filter timing.** On a 10 fps video whose frame *i* has brightness *i*,
  the default `round=near` returned frames 0.4 s late, and `round=up` returned them
  exactly, including after `-ss` seeking. A test checks this
  (`tests/test_video.py::test_sampled_frame_times_are_exact`).
- **openpyxl and x14 dropdowns.** Loading the mock template keeps 2 of its 4 dropdowns
  and warns. The app's export keeps all 4, with their settings.
- **LibreOffice 24.2 (headless)** opened an exported workbook and read all 4 dropdowns as
  proper lists. **Real Excel has not been tested yet.**
- **`shrink` size accuracy.** A deliberately noisy 1080p 30 fps clip came out at
  5.9 MB against a 6.3 MB target.
- **End to end on a synthetic video.** MaxN matches ground truth: red 2, yellow 3,
  green 1 (`tests/test_pipeline.py`).

## 7. From general knowledge, not checked against a primary source

These shaped the code but weren't confirmed against an authoritative source here:

- That Excel 2010+ writes cross-sheet dropdowns in the x14 extension (supported by
  EPPlus and LibreOffice, but not confirmed from Microsoft's spec).
- That Excel accepts a cross-sheet reference in a *standard* dropdown, which the export
  relies on. LibreOffice does accept it. **Check in real Excel.**
- The GoPro naming cut-offs: GX/GH from HERO6, and GOPR then GP01… on HERO5 and older.
- MaxN as the BRUV abundance metric, and time of first arrival (standard practice; no
  citation in the repo).
- Variance of the Laplacian as a sharpness score for picking crops.
- The common names of the six demo species. They are real western Atlantic / Gulf of
  Mexico fishes used only as fake labels, and aren't taken from the owner's list.
- SQLite UPSERT support (believed to need SQLite 3.24+, which current Python builds should exceed; not checked).

## 8. Research cited in the spec (inherited from the planning chat)

SPEC §7 came from the earlier planning conversation, not from sources read here. The
audit checked what it could:

| Claim in the spec | Check | Status |
|---|---|---|
| SharkTrack (arXiv 2407.20623): 89% species-specific MaxN accuracy, ~2 min human time per hour | The paper id and authors match the SharkTrack repo (Varini et al., 2024). The figures couldn't be checked because arXiv was blocked. | partly |
| Reef shark study (IRD): F-measure 0.43–0.85 rising to 0.82–0.97 with a human | No authors, title or DOI were recorded | unverified: **citation needed** |
| Fishial.ai self-reports 70–90% | Their public repo reports **93.22%** classification accuracy (866 classes). The 70–90% figure's source is unknown. | conflicts |
| VIAME is the established open-source alternative | Repo confirmed (NOAA-funded, Kitware; BSD-3 core) | verified |
| VLM4Bio: general vision-language models do poorly at fish ID | Benchmark confirmed (arXiv 2408.16176; Fish-10K, 495 species). The finding itself wasn't re-read. | partly |
| "Fully automatic BRUV ID is well below 90%; semi-automated reaches ~90%+" | No citation | unverified |

## 9. Licenses of dependencies and models

Versions are the ones checked on PyPI on 2026-10-17.

| Component | Role | License | Status |
|---|---|---|---|
| numpy 2.5.4 | core | BSD-3-Clause (+ bundled 0BSD/MIT/Zlib/CC0) | verified |
| scipy 1.18.1 | core | BSD-3-Clause (bundled OpenBLAS/LAPACK BSD, GCC runtime GPL-3.0 with runtime exception) | verified |
| pillow 13.0.0 | core | MIT-CMU | verified |
| imageio-ffmpeg 0.6.0 | core | BSD-2-Clause (the wrapper) | verified |
| ffmpeg bundled by imageio-ffmpeg (Windows: 7.1 gyan.dev "essentials"; Linux: 7.0.2 static) | core, binary | **GPL-3.0-or-later** (`--enable-gpl --enable-version3`, includes x264/x265; not "nonfree") | verified |
| openpyxl 3.1.5 | core | MIT | verified |
| fastapi 0.143.3, pydantic 2.x, starlette | core | MIT / MIT / BSD-3-Clause | verified |
| uvicorn 0.54.0 | core | BSD-3-Clause | verified |
| torch 2.14.1 / torchvision 0.29.1 | ml | Apache-2.0 + BSD family / BSD-3-Clause | verified |
| rfdetr 1.11.x | ml | Apache-2.0 | verified |
| open_clip_torch 3.3.x | ml | MIT | verified |
| supervision 0.31.0 (via rfdetr) | ml, transitive | MIT | verified |
| PyAV 19 (via supervision) | ml, transitive | BSD-3-Clause, but bundles **LGPL-3.0 FFmpeg libs plus x264/x265** | verified |
| transformers, timm, huggingface-hub | ml, transitive | Apache-2.0 | verified |
| pytest 9.1.1, httpx 0.28.1 | dev only | MIT, BSD-3-Clause | verified |
| CFD RF-DETR weights | model | **Not stated.** The README's "Apache" column describes the inference code only. The repo has no LICENSE file, and the training data is a mix of datasets with different licenses. | unverified |
| BioCLIP 2 weights | model | MIT, as stated by the authors on GitHub (the Hugging Face card was blocked) | partly |
| TreeOfLife-200M (BioCLIP 2's training data) | model data | Not stated in the sources reached. One component, a GBIF download, is CC BY-NC 4.0. | unverified |

**Before the Windows installer (phase 6):**

- **ffmpeg (GPLv3).** Ship the GPLv3 text and provide the source for that exact ffmpeg
  build. The app runs ffmpeg as a separate program, which is generally treated as
  separate from the app's own code.
- **PyAV.** Exclude it from the PyInstaller bundle (the app never uses it), or comply
  with the licenses of the libraries it bundles.
- **CFD weights.** Ask the CFD authors about a weights license before any commercial
  distribution.
- **BioCLIP 2.** Check the Hugging Face model card and the TreeOfLife-200M terms, and
  confirm BioCLIP 2's license against them.

## 10. Blocked during the work

These hosts were refused by the sandbox's network policy, so anything that depends on
them is unverified above:

- `huggingface.co` (BioCLIP 2 weights and model card)
- `github.com` release downloads (CFD weights)
- `learn.microsoft.com` (the [MS-XLSX] spec)
- `gopro.com` / `community.gopro.com`
- `arxiv.org`
- `lila.science`
- `ffmpeg.org`
- `python.org`
- `download.pytorch.org`

## 11. Tools used for checking only (not part of the app)

- Playwright with the sandbox's pre-installed Chromium, for a click-through of the
  review page.
- pyflakes, for lint.
- LibreOffice (headless), to open an exported workbook.

None of these is a dependency or is referenced by the app.
