# Sources and references

Everything outside this repository's own code that informed it: where it came from, what
it was used for, and whether it was checked. All sources were collected on **2026-10-17**,
while building the MVP and during two follow-up checks (a sources audit and a review of
this file).

## Summary

- **Other apps: one, read-only.** During the sources audit, helper agents used the
  **GitHub connector** 17 times to read public repository data. Nothing was written.
  Some of those searches covered all of public GitHub, beyond this session's one-repo
  scope. No other connected app or service (Notion, Gmail, Google Drive, Google
  Calendar, Slack, or any other) was used. Details are in [section 2](#2-other-apps-services-and-production-systems).
- **Production sources: none.** No production systems, databases, private datasets or
  personal data were used.
- **Other repositories: public ones only, read-only.** None of them is yours or private.
  Most are open source, with three exceptions: EPPlus is source-available (Polyform
  Noncommercial), and the Community Fish Detector and digphoto repos state no license.
  Nothing was copied from any of them. They are listed in [section 3](#3-external-repositories-public-read-only).
- **Copied code: none.** Several pieces follow published examples:
  - The detector's load and predict calls are *equivalent* to the Community Fish
    Detector's own script, but not identical (this app passes `device`, can reload at the
    right resolution, and passes numpy arrays rather than PIL images).
  - The resolution check is modelled on the same script.
  - The prompt-ensemble averaging and the 224×224 "squash" preprocessing follow
    Imageomics' `pybioclip`.
  - CLIP's 80 prompt templates are imported from the `open_clip` package at run time,
    not copied into this repo.
- **First real run (Oct 2026, on the owner's PC):**
  - no connectors were used;
  - inputs were the owner's template and clip, read locally;
  - Excel and the Windows event log were used for checking;
  - packages, Python 3.13 and the two models were downloaded from their usual hosts.

  Details are in sections 1, 4–6 and 11.
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
| The owner's real viewing template and a 2-minute clip, on the owner's PC (Oct 2026) | The first real run (SPEC §9): template reading (dependent dropdowns, clip list), the export, and the Excel checks | Read on the owner's PC only. Neither is in the repo; the tests use invented names. |
| The owner's list of fish families, with common and family names (chat, Oct 2026) | `bruv_app/extra_species.txt` (SPEC §10): the 33 names that weren't in the template, minus salmon and cod as the owner agreed | The owner's own list. Which names the template already covers was checked by hand from general knowledge of the families (e.g. that whiprays are stingrays, Dasyatidae). |

## 2. Other apps, services and production systems

- **GitHub connector: read-only, 17 calls, all during the sources audit.**
  - 8 code searches: for the Excel x14 extension URI, for EPPlus and XlsxWriter
    data-validation code, and for GoPro naming notes.
  - 7 repository searches: Community Fish Detector, BioCLIP 2, SharkTrack, VLM4Bio,
    Fishial and VIAME.
  - 1 release listing and 1 root-folder listing, both for the Community Fish Detector.

  All of these returned public data only. The unscoped code searches went beyond the
  single repository this session is set up for. Nothing was created, changed or posted.
- **Claude Code's environment documentation** was read once, to explain the sandbox's
  network limits. This is product documentation, not your data.
- **Nothing else.** Notion, Gmail, Google Drive, Google Calendar, Slack and the other
  connectors available in the session were not called. No production database, API or
  private data source was read.

## 3. External repositories (public, read-only)

| Repository | What was read | Used for | Status |
|---|---|---|---|
| [filippovarini/community-fish-detector](https://github.com/filippovarini/community-fish-detector) (the spec's `WildHackers/community-fish-detector` URL resolves to the same README) | `README.md`, `README-yolo.md`, `requirements.txt`, `rf_detr_batch_inference.py`, the release list | Detector choice: YOLOv12x is deprecated, RF-DETR Nano/Small/Medium replace it. Also checkpoint names, resolutions (640/1024), AP figures, CFD's own `rfdetr>=1.8.3` requirement (this app requires ≥1.11.2), the `from_checkpoint`/`predict` calls and resolution check, and the 0.3 threshold from the README example. | verified |
| [Imageomics/bioclip-2](https://github.com/Imageomics/bioclip-2) | `README.md`, `LICENSE`, `HISTORY.md`, `src/evaluation/classification.py`, `src/training/imagenet_zeroshot_data.py`, `slurm/eval.sh`, `gh-pages/index.html` (citations: weights labelled MIT; one GBIF component is CC BY-NC 4.0) | Model id, ViT-L/14 (768-d), MIT license, and how its evaluation builds class embeddings: the 80 CLIP templates plus "a photo of {c}." (81 in all), with taxonomic plus common names ("taxon_com") for camera-trap and rare-species tests | verified |
| [Imageomics/pybioclip](https://github.com/Imageomics/pybioclip) | `README.md`, `src/bioclip/_constants.py`, `src/bioclip/predict.py` | Confirms `hf-hub:imageomics/bioclip-2`, the 224×224 "squash" preprocessing and the 80-template ensemble this app follows | verified |
| [Imageomics/TreeOfLife-toolbox](https://github.com/Imageomics/TreeOfLife-toolbox) | `README.md`, `docs/README` | BioCLIP 2's training-data provenance (images come from GBIF, EOL, BIOSCAN-5M, FathomNet) | partly |
| [roboflow/rf-detr](https://github.com/roboflow/rf-detr) | `LICENSE`, README license section | Apache-2.0 for the package and the Nano/Small/Medium/Large classes. "Plus" models are PML 1.0 and aren't used. | verified |
| [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip), [pytorch/vision](https://github.com/pytorch/vision), [imageio/imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) | `LICENSE` files; open_clip `factory.py`; imageio-ffmpeg README and release script | Licenses (MIT, BSD-3-Clause, BSD-2-Clause), the `hf-hub:` prefix, and where the bundled ffmpeg binaries come from | verified |
| [LibreOffice/core](https://github.com/LibreOffice/core) (`sc/source/filter/inc/worksheetfragment.hxx`), [EPPlus](https://github.com/EPPlusSoftware/EPPlus) (`ExcelDataValidation.cs`, `license.md`), [XlsxWriter](https://github.com/jmcnamara/XlsxWriter) 3.2.9 source, plus GitHub code-search hits in openxlsx, openxlsx2, umya-spreadsheet and eec | Source comments and code | Independent evidence for the Excel x14 data-validation extension, its URI and namespaces | partly (Microsoft's spec was blocked) |
| [nkrabben/digphoto](https://github.com/nkrabben/digphoto) (`…/gopro-action/file-naming.md`) | GoPro file-naming summary, which cites GoPro's own article | The X/H codec letters (HEVC/H.264) and the chapter digits in characters 3–4 of GoPro names | partly (GoPro's site was blocked) |
| [HandBrake/HandBrake](https://github.com/HandBrake/HandBrake) (`preset/preset_builtin.json`) | Preset list | The "Fast 720p30" preset named in the README | verified |
| x264 (`mirror/x264`) and x265 (`videolan/x265`) `COPYING`; FFmpeg's `configure`; Pillow's Windows build files | License texts and build lists | The licenses of codecs bundled inside PyAV and the ffmpeg binary | verified |
| [filippovarini/sharktrack](https://github.com/filippovarini/sharktrack), [Wye-Foundation/Fishial-Fish-Identification](https://github.com/Wye-Foundation/Fishial-Fish-Identification), [VIAME/VIAME](https://github.com/VIAME/VIAME), [Imageomics/VLM4Bio](https://github.com/Imageomics/VLM4Bio) | READMEs and licenses | Checking the background research in SPEC §7 (see [section 8](#8-research-cited-in-the-spec-inherited-from-the-planning-chat)) | partly |

## 4. Package registries and package source code

| Source | What was checked | Used for | Status |
|---|---|---|---|
| PyPI JSON API (`pypi.org/pypi/<name>/json`) | Versions, licenses and dependencies of every package in [section 9](#9-licenses-of-dependencies-and-models) | The license table; the `rfdetr`/`open_clip` version limits | verified |
| PyTorch CPU wheel index (`download.pytorch.org/whl/cpu`) | Installed `torch 2.14.1+cpu` and `torchvision 0.29.1+cpu` from it on Windows (Python 3.13) | `scripts/setup_windows.bat` and `quickstart.bat` install torch from it | install verified; the `+cpu` wheels' bundled libraries and licenses are still uninspected (the torch row in §9 is from PyPI) |
| `open_clip_torch` 3.3.0 `factory.py`, `pretrained.py`, `model.py`, `transformer.py` (installed) | How the hub config and weights are loaded (`load_weights`, `download_pretrained_from_hf`), the random-init-then-copy load, `encode_text`, the causal mask | `bruv_app/classify.py` (meta-device build, mapped weights, short-context text path) | verified by experiment |
| `openpyxl` 3.1.5 `workbook/external_link/external.py`, `writer/excel.py`, `reader/excel.py` (installed) | External links keep only their first relationship; `rich_text=True` keeps rich text | `bruv_app/template.py` export repairs | verified by experiment |
| The `.safetensors` format (8-byte header length, JSON header, raw tensor data) | Reading BioCLIP 2's checkpoint without the `safetensors` reader, whose memory map counts against Windows' commit limit | `bruv_app/classify.py` (`map_safetensors`) | verified: every tensor equals a plain read |
| `rfdetr` 1.11.2 and 1.8.3 wheels (PyPI) | `from_checkpoint` (the `device`, `resolution` and `trust_checkpoint` arguments; safe loading), `predict` (input types; boxes returned in input-image pixels; list vs single return), the `model_config.resolution` defaults | `bruv_app/detect.py` | verified (read, not run with weights) |
| `open_clip_torch` 3.3.0 wheel (PyPI) | `create_model_and_transforms` (return order, `image_resize_mode`), `get_tokenizer('hf-hub:…')`, `logit_scale` (log space), `encode_image`/`encode_text`, `OPENAI_IMAGENET_TEMPLATES` | `bruv_app/classify.py` | verified (read, not run with weights) |
| `imageio-ffmpeg` 0.6.0 (installed, plus the Windows wheel) | `read_frames`/`write_frames`/`get_ffmpeg_exe` behaviour, the frame-size warning in `_parsing.py`, and the bundled ffmpeg builds and their licenses | `bruv_app/video.py`, licensing notes | verified |
| `openpyxl` 3.1.5 (installed) | `worksheet/_reader.py` (drops extensions and warns), `xml/constants.py` (`{CCE6A557-…}` = Data Validation); shapes and text boxes are dropped with no warning | `bruv_app/template.py` | verified |

## 5. Models

| Model | Source | Used for | Status |
|---|---|---|---|
| Community Fish Detector RF-DETR Nano 640 / Small 1024 / Medium 1024 (release `2026.07.06-release`) | CFD README and releases | Fish detection | Nano downloaded (116 MB) and run on Windows in the first real run; Small and Medium not tried |
| BioCLIP 2 (`hf-hub:imageomics/bioclip-2`; Gu, Stevens et al., NeurIPS 2025, arXiv 2505.23883; trained on TreeOfLife-200M) | Imageomics GitHub repos; weights from Hugging Face | Zero-shot species guesses and crop embeddings | Downloaded (1.7 GB) and run on Windows in the first real run; the model card is still unread |

## 6. Experiments run while building (no external source)

- **ffmpeg `fps` filter timing.** On a 10 fps video whose frame *i* has brightness *i*,
  the default `round=near` returned frames 0.4 s late, and `round=up` returned them
  exactly, including after `-ss` seeking. A test checks this
  (`tests/test_video.py::test_sampled_frame_times_are_exact`).
- **openpyxl and x14 dropdowns.** Loading the mock template keeps 2 of its 4 dropdowns
  and warns. The app's export keeps all 4, with their settings, on every sheet. Shapes
  and text boxes are dropped by openpyxl with no warning, so the export now looks for
  them and warns.
- **LibreOffice 24.2 (headless)** opened an exported workbook and read all 4 dropdowns as
  proper lists. **Real Excel has not been tested yet.**
- **`shrink` size accuracy.** A deliberately noisy 1080p 30 fps clip came out at
  5.9 MB against a 6.3 MB target.
- **End to end on a synthetic video.** MaxN matches ground truth: red 2, yellow 3,
  green 1 (`tests/test_pipeline.py`).
- **First real run on the owner's PC (SPEC §9).** Timings per stage from the project's
  stage log, and the cards' guesses judged by eye on contact sheets of every card.
- **Hardware decode.** ffmpeg 7.1 `-benchmark` on the 2-minute clip at 1 fps: 47.9 s with
  `-hwaccel auto` (it chose DXVA2) and 47.7 s without, with 54 s and 354 s of CPU time.
- **BioCLIP 2 memory.** Measured with Windows' per-process commit counter.
  - open_clip's own load reached 3.18 GB after its random init and peaked at 3.6 GB.
  - The meta-device build with mapped weights peaked at 1.26 GB, with the same speed
    (~1.6 s per crop).
  - The `safetensors` reader's own memory map failed with "paging file is too small"
    when free commit was low. That is why the checkpoint is read directly.
- **Extra species (SPEC §10).** The 73 real cards were re-scored from their stored image
  embeddings with the new names added. Accuracy was judged by eye on contact sheets of
  all 73 cards, on the 23 cards whose family was clear from the image (Claude's own IDs,
  not checked by an expert).
- **Short-context text encoding.** At every load, `encode_text` and the short path are
  compared on two prompts (they agree within 1e-4). The short path took 2.6 s per label
  (80 prompts), against 9.8 s for the full context.
- **Excel opening the export (real Excel, hidden, late-bound through `cscript`).**
  - Even a plain openpyxl re-save of the template was refused (error 0x800A03EC).
  - Bisecting found two causes: the external link's relationship id, and one rich-text
    cell holding a space-only run.
  - Each fix alone was then confirmed in Excel.
  - The final export matches the template cell by cell, and Excel's own
    `Validation.Value` accepts every value written under a dropdown.

## 7. From general knowledge, not checked against a primary source

These shaped the code but weren't confirmed against an authoritative source here:

- That Excel 2010+ writes cross-sheet dropdowns in the x14 extension (supported by
  EPPlus and LibreOffice, but not confirmed from Microsoft's spec).
- That Excel accepts a cross-sheet reference in a *standard* dropdown, which the export
  relies on. *Now verified in real Excel (Office 16, Oct 2026): the re-attached list
  validates values.*
- The GoPro naming cut-offs: GX/GH from HERO6, and GOPR then GP01… on HERO5 and older.
- That GoPro's `GL…` files are low-resolution `.LRV` proxies, which the app skips.
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
| torch 2.14.1 / torchvision 0.29.1 | ml | Apache-2.0 + BSD family / BSD-3-Clause | verified on PyPI; the `+cpu` wheels the setup script installs were not inspected |
| rfdetr 1.11.x (required: `>=1.11.2,<1.12`) | ml | Apache-2.0 | verified |
| open_clip_torch 3.3.x (required: `>=3.3,<3.4`) | ml | MIT | verified |
| supervision 0.31.0 (via rfdetr) | ml, transitive | MIT | verified |
| PyAV 19 (via supervision) | ml, transitive | BSD-3-Clause binding. Its bundled FFmpeg libraries say LGPL-3.0, but they link **libx264/libx265 (GPL-2.0-or-later)**, so shipping it brings GPL obligations. | verified |
| tqdm, certifi (via rfdetr, open_clip, requests) | ml, transitive | **MPL-2.0** (tqdm: MPL-2.0 AND MIT) | verified |
| transformers, timm, huggingface-hub | ml, transitive | Apache-2.0 | verified |
| pytest 9.1.1, httpx 0.28.1 | dev only | MIT, BSD-3-Clause | verified |
| CFD RF-DETR weights | model | **Not stated.** The README's "Apache" column describes the inference code only. The repo has no LICENSE file, and the training data is a mix of datasets with different licenses. | unverified |
| BioCLIP 2 weights | model | MIT, as stated by the authors on GitHub (the Hugging Face card was blocked) | partly |
| TreeOfLife-200M (BioCLIP 2's training data) | model data | Not stated in the sources reached. One component, a GBIF download, is CC BY-NC 4.0. | unverified |

**Before the Windows installer (phase 6):**

- **ffmpeg (GPLv3).** Ship the GPLv3 text and provide the source for that exact ffmpeg
  build. The app runs ffmpeg as a separate program, which is generally treated as
  separate from the app's own code.
- **PyAV.** Exclude it from the PyInstaller bundle (the app never uses it). Otherwise its
  x264/x265 libraries bring GPL obligations.
- **MPL-2.0 files (tqdm, certifi).** Ship their license notices, and say where their
  source is available.
- **CFD weights.** Ask the CFD authors about a weights license before any commercial
  distribution.
- **BioCLIP 2.** Check the Hugging Face model card and the TreeOfLife-200M terms, and
  confirm BioCLIP 2's license against them.

## 10. Blocked during the work

These hosts were refused by the build sandbox's network policy, so anything that depends
on them is unverified above. On the owner's PC (first real run), `huggingface.co`,
GitHub release downloads, `download.pytorch.org` and `python.org` were reachable and were
used only to install the packages, Python 3.13 and the two models:

- `huggingface.co`: BioCLIP 2's weights, model card and TreeOfLife-200M card
- `github.com` release downloads and `api.github.com`: the CFD weights (repository data
  came through the GitHub connector instead)
- `learn.microsoft.com`: the [MS-XLSX] spec
- `gopro.com` / `community.gopro.com`: GoPro's naming article
- `arxiv.org`, `api.crossref.org`, `api.semanticscholar.org`: the SharkTrack and IRD
  figures and citations
- `www.fishial.ai`: the source of the spec's 70–90% figure
- `lila.science`: the CFD training-data licenses
- `www.gbif.org`, `www.marinespecies.org`, `www.fishbase.se`: checking the demo species
  names
- `pyinstaller.org`, `jrsoftware.org`: the installer tools' licenses
- `imageomics.github.io`, `viame.kitware.com`, `handbrake.fr`: project sites (their
  GitHub sources were read instead)
- `ffmpeg.org`, `python.org`, `download.pytorch.org`

## 11. Tools used for checking only (not part of the app)

- Playwright with the sandbox's pre-installed Chromium, for a click-through of the
  review page.
- XlsxWriter, used once to build a test workbook with a text box and a data bar.
- pyflakes, for lint.
- LibreOffice (headless), to open an exported workbook.
- On the owner's PC:
  - Microsoft Excel (Office 16), driven by JScript through `cscript`, to open and
    compare exports;
  - the Windows event log, to diagnose the out-of-memory crash;
  - the built-in browser of the Claude desktop app, for the review page.

None of these is a dependency or is referenced by the app.
