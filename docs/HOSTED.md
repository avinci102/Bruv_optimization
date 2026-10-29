# The hosted review site (volunteer kit)

Volunteers review cards in a web browser, with nothing to install. The processing PC
finds and tracks the fish as usual, then **publishes** each video as plain files. A free
Cloudflare site serves those files and keeps each volunteer's answers. The coordinator
**pulls** the answers back into the project and exports to Excel as usual.

The processing PC can be switched off while volunteers review. Nothing on the site depends
on it, and nobody needs a tunnel or a server of their own.

This is optional. The program itself stays offline-first: anyone can process and review
on their own computer without any of this. A school that wants volunteers to help can
set the site up on its own free Cloudflare account by following this page. The design
decisions behind it are in [SPEC.md §16](SPEC.md).

## How it fits together

| Where | What runs there | What it holds |
|---|---|---|
| Processing PC | `process`, then `publish` | The videos, the project, the master answers |
| Cloudflare Pages (static files) | Nothing: files only | One picture file ("sheet") per card, and the card data |
| Cloudflare Pages Function + D1 database | Sign-in and saving | Volunteers, who reviews what, every answer and fix, feedback |
| Volunteer's browser | The normal review page, plus `hosted/runtime.js` and `hosted/core.js` | Nothing kept: rebuilt from the files and their saved answers each time the page opens |

The review page is the same page as on the PC. Two scripts added to the site's copy of
it answer the page's requests in the browser, from the published files. Answers and fixes
are saved to the database before they show as done.

## What it costs

Nothing, on Cloudflare's free plan, for a pilot of this size:

| Free-plan limit | What uses it | Pilot estimate |
|---|---|---|
| 20,000 files per site | One sheet per card, plus 4 files per video; with `--follow`, one more file per 20 frames | About 1,150 per 44-minute video (about 1,680 with `--follow`), so 11 to 15 videos per site |
| 25 MiB per file | The largest file is a video's descriptions (`emb.bin`) | 17.8 MB for 44 minutes |
| Static requests | Pictures and card data | Unlimited and free |
| 100,000 function requests a day | Sign-in, each answer or fix, 2 per page load | 5 volunteers × ~1,000 a day = 5,000 |
| 100,000 database rows written a day | About 3 per answer (the row and its 2 indexes) | About 15,000 |
| 5 GB database | Answers are small text | Well under 1% |

The 44-minute test video came to 1,153 files and about 263 MB (details at the end of
this page).

When one site is full, make a second site (a second Pages project, same steps) for the
other videos. Both can share the database.

## One-time setup (about 30 minutes)

You need the processing PC, a terminal in the program's folder, and an email address for
the new Cloudflare account.

1. **Make a free Cloudflare account** for the project at <https://dash.cloudflare.com/sign-up>.
   Use a project or school address rather than a personal one, so it can be handed over.
2. **Install Node.js** (the LTS version) from <https://nodejs.org>. It brings `npx`, which
   runs Cloudflare's `wrangler` tool. Nothing else needs installing.
   - Open a **new** terminal afterwards: one that was already open doesn't know about Node.
   - In Windows PowerShell, type `npx.cmd` wherever this page says `npx`. Plain `npx` is
     blocked there with "running scripts is disabled on this system".
3. **Sign wrangler in** to that account. A browser page opens: click **Allow** within about
   two minutes, or wrangler gives up and you run it again.
   ```
   npx wrangler login
   ```
4. **Make the site folder** by publishing a first video (see "Each wave" below), then go
   into it: `cd <site folder>`.
5. **Make the database** and note the `database_id` it prints:
   ```
   npx wrangler d1 create bruv-review
   ```
   If it offers to add the database to a configuration file, answer **No**: it would make a
   `wrangler.jsonc`, which wrangler reads instead of the `wrangler.toml` below. Copy
   `wrangler.toml.example` to `wrangler.toml` in the site folder and paste the id in.
6. **Make the database's tables:**
   ```
   npx wrangler d1 execute bruv-review --remote --file schema.sql
   ```
7. **Make the site**, from the site folder:
   ```
   npx wrangler pages project create bruv-review --production-branch main
   ```
   When an AI assistant runs this for you, recent wrangler versions can create the newer
   kind of Pages site, which runs on Workers. This kit is built for the classic kind.
   Wrangler keeps to the classic kind when the folder holds a `functions` folder, which
   publishing (step 4) puts there.
8. **Give it a secret** for signing volunteers' sessions. Make a random one with
   `python -c "import secrets; print(secrets.token_urlsafe(32))"`, then paste it when asked:
   ```
   npx wrangler pages secret put SESSION_SECRET --project-name bruv-review
   ```

The site's address is `https://bruv-review.pages.dev` (or the name you chose). If that
name is taken, Cloudflare adds a few letters; the create command prints the address.

## Each wave

1. **Process** the videos on the processing PC as usual.
2. **Publish** them, each with a short name volunteers will see (letters, digits, `-`, `_`):
   ```
   python -m bruv_app publish --project D:\bruv\A1 D:\bruv\A2 --video A1 A2 --site D:\bruv\site
   ```
   It reports the files against the 20,000 limit. Publishing a video again replaces that
   video's files; other videos in the site folder are kept.

   To put a rule trial online (`try-rules`), add `--trial <name>` with a video name of its
   own, such as `--video A1-faint`. Pull its answers with the same `--trial <name>`: they go
   into the trial, not the project.

   To let people **follow a fish** on the site (frame by frame, as on the PC), add
   `--follow`. It publishes every sampled frame and every box. It adds about 390 MB and
   11 minutes for a 44-minute video, so use it for the videos where you want follows. The
   review page of such a video also shows the real scene at every moment.
3. **List the volunteers** in a CSV file (Excel can save one), with these columns:

   | name | email | pass | role | videos | calibration |
   |---|---|---|---|---|---|
   | Vinci | vinci@example.org | | admin | | |
   | Ana | ana@example.org | | volunteer | A1 A2 | A3=12;15;40 |
   | Ben | | | volunteer | A3 A4 | A1 |

   - **pass**: leave it empty and one is made (like `BRUV-H3Y7-F47A`). Keep the passes the
     command writes back: a volunteer is known by their pass, so their answers stay theirs
     when you change the list.
   - **role**: `admin` sees everyone's progress and downloads the answers.
   - **videos**: the volunteer's own videos. Their review of those is the review of record.
   - **calibration**: shared checks for comparing volunteers. A video alone means all its
     cards; `A3=12;15;40` means just those cards. Use these only where you want to measure
     agreement. Each volunteer answers in their own copy.

   Then make the database's list and load it:
   ```
   python -m bruv_app site volunteers --site D:\bruv\site --csv volunteers.csv
   npx wrangler d1 execute bruv-review --remote --file volunteers.sql
   ```
4. **Try it on this PC first** (the same site, with a local database):
   ```
   python -m bruv_app site serve --site D:\bruv\site
   ```
5. **Put it online**, from the site folder:
   ```
   npx wrangler pages deploy --branch main
   ```
   `--branch main` makes it the live site. Without it, a site folder inside a git checkout
   would go to a preview address instead.
6. **Send the invitations** (see the template below).
7. **Watch progress**: sign in with an `admin` pass. The start page shows each volunteer's
   answers, fixes and follows per video, and when they were last active.
8. **Collect the answers.** On the start page, click **Download answers: A1**, then pull
   each volunteer into the project:
   ```
   python -m bruv_app pull --project D:\bruv\A1 --answers answers-A1.json --volunteer Ana
   ```
   - Pull the volunteer whose own video it is into that video's project. Their answers
     land on the same cards, with their name and the time they gave them
     (`reviews.reviewer`). Their splits, moves and "multiple fish" fixes become that
     project's cards; new cards get the project's next numbers, and the command prints the
     mapping.
   - Pull a calibration volunteer into a **copy** of the project, to compare with.
   - A snapshot is taken before and after, and the pull is recorded in the run log.
   - It refuses answers for a different video than the project was published as.
   - Their follows come too, with the name they had on the site (`F3-v2`: follow 3 of
     volunteer 2), so `evaluate` and `compare-rules` score them on this PC.
9. **Export** to Excel as usual.

### Replacing wave 1's pictures with wave 2's

- Delete the old videos' folders (`site\public\videos\<name>`) or start a new site folder,
  publish wave 2, and deploy. A deploy replaces the whole live site with the folder.
- Cloudflare keeps each earlier deploy reachable at its own preview address until you
  delete it: in the dashboard, Workers & Pages → bruv-review → Deployments → ⋯ → Delete.
- Saved answers stay in the database whatever you deploy.
- **If you re-process a video** (new tracking rules, say), its cards change. Publish it
  under a new name (`A1-w2`). Answers saved against the old name belong to the old cards.

## The invitation

> Hi Ana,
>
> Thank you for helping identify fish in our baited underwater videos.
>
> 1. On a laptop or desktop (the page is built for the keyboard), open
>    https://bruv-review.pages.dev/start
> 2. Sign in with your access pass: **BRUV-H3Y7-F47A** (or with this email address).
> 3. Click your video. Your answers save as you go, so you can stop and come back
>    any time, on any computer.
> 4. The keys are shown next to each button. When you finish a session, use the
>    "videos · feedback" link at the bottom right to tell us how it went.

## What works differently online

| On the PC | Online |
|---|---|
| Join two cards (merge, `P`) | Not yet: the page says to note the card numbers in a comment. Splitting, moving ticked sightings and "multiple fish" all work. |
| Export to Excel | The coordinator exports, after pulling. |
| Bait-arm batch marking | Not on the site; each card's Equipment answer works. |
| Whole scenes for every moment | With `--follow`, the same. Otherwise up to 8 whole scenes per card; other moments show the nearest one, dimmed, with the fish as seen at that moment, and say so. |
| Follow this fish, and following as its own task | With `--follow`, the same, with a "Follow fish in <video>" button per video on the start page. Each volunteer's follows and box fixes are their own: "taken" and "shared" look at their own follows; after pulling, the PC sees everyone's. |
| Tiny fish and cards outside the review window | Not published. |

## Privacy and access

- Signing in needs no password, by decision: the data isn't sensitive. Anyone with a
  volunteer's pass or invited email can sign in as them.
- The pictures and card data are ordinary public files. Anyone who knows the site's
  address and a video's name can download them. Signing in protects the answers, not the
  pictures. Don't publish footage that must stay private.
- Volunteers see and change only their own answers. Admins see progress and download
  everyone's.
- Nothing on the site contains this PC's folder names or user name: publish keeps file
  names only.

## How it works (for developers)

- **`bruv_app/publish.py`** calls the review server's own endpoint functions, so the
  published answers are what the local server would give. It writes, per video, under
  `public/videos/<name>/`:
  - `manifest.json`: the summary, settings, species names and families, chapter
    timeline, and the still-object sample;
  - `cards.json`: every card's list entry and detail, its sightings (box, time, quality,
    piece, description row, size in pixels), its pieces, look, published look-alike
    neighbours (≥0.85), and where its pictures are on its sheet;
  - `emb.bin`: float16 descriptions (BioCLIP embeddings) in this order: species names,
    crops, pieces, card looks;
  - `sheets/<card>.webp`: the card's close-ups and up to 8 whole scenes (640 px wide);
  - with `--follow`, `follow/`: every sampled frame, 20 to a WebP strip (`<n>.webp`, 640
    px a frame, 4 to a row), `frames.json` (every frame's id and exact time, in order) and
    `boxes.json` (every box in each frame, as `follow.frame_boxes` lists them).
  It also records `meta.published` in the project, which `pull` checks.
- **`hosted/core.js`** answers every `/api/...` request the page makes, mirroring the
  server: the summary and card list, card details, look-alikes and their answers, labels
  (with the same warnings check), undo, split, move and "multiple fish". After a fix it
  works out the new cards' guesses, families, warnings, pieces, look-alikes and look-alike
  groups from the published descriptions, as `pipeline.refresh_fish_guesses`,
  `diagnostics.card_flags` and `lookalike.py` do. With `--follow` it answers
  `/api/follows` too, as `follow.py` does: the next frame, the boxes in it with the cards
  they're on now, and what a follow found.
- **`hosted/runtime.js`** connects it to the review and follow pages: it replaces `fetch`
  for `/api/` and cuts `/crop/<id>` and `/frame/<id>` pictures from the sheets (or the
  follow strips) into `blob:` images. A
  service worker would be tidier, but some browsers and managed school laptops block
  them. This way works in any recent browser.
  - **Pictures are made ahead.** The review page lists the open card's frames, nearest
    the one shown first, and then the next card's pictures (`BRUV_preload`). On the follow
    page, the runtime itself makes the 30 frames on each side of the one shown. Two are
    made at a time, in that order. A picture that's ready shows at once. One that isn't
    keeps the previous picture on screen, dimmed, rather than going blank. The last 600
    pictures are kept.
  - **Pictures are WebP.** Chrome encodes JPEGs only while the page is idle, which held
    each frame up by about a second. WebP took a tenth of that. Safari, which can't encode
    WebP, gets JPEGs.
  - **Measured on 28 Oct** (local copy of `trial`, this laptop, browser pane hidden):

    | Stepping with ↓ | Before | After |
    |---|---|---|
    | Card of 272 sightings (55 in its filmstrip), 2 s after opening | 1.2 s a step; 63 s for all 55 | 7 ms a step (90% under 0.11 s); 1.9 s for all 55 |
    | Card of 23 sightings, straight after opening | 1.1 s a step; 24 s in all | 6 ms a step; 0.2 s in all |
    | Follow page, the next frame after a mark | 1.1–1.5 s | mostly 30–60 ms, at most 0.4 s |

    Over the internet, the first frames of a card also wait for their strips to download;
    after that, the frames ahead are ready.
- **What's saved** is an event log per volunteer and video (D1 table `events`). Kinds:
  `label`, `undo`, `split`, `move`, `unmerge`, `suggestions`, and for follows
  `follow_start` (with the sighting, or the drawn box, it starts from and its name),
  `follow_mark`, `follow_rest` (gone for good: the frames it marked), `follow_undo`,
  `follow_delete` and `box_fix` (what's wrong with a box). Each has the fields its replay
  needs and the site's time. New cards are numbered after the highest published card.
- **`hosted/functions/sync/[[path]].js`** is the only server code (sign-in, events,
  feedback, admin). `bruv_app/site.py` has the same API on SQLite for `site serve` and the
  tests. `_routes.json` keeps every other request a free static one.
- **`bruv_app/pull.py`** replays one volunteer's log onto a project with the store's own
  methods, and follows with `follow.py`'s, in order: undo takes back the latest mark.
- **Checking that the browser matches the server** (`tests/test_hosted_js.py`, needs
  Node.js; skipped without it). It publishes the synthetic test project, with one card
  made of two different fish. `tests/hosted_parity.js` then runs `core.js` in Node and
  makes, through the same requests the page sends:
  - an answer on that card (turned down for its warnings, then confirmed), a look-alike
    batch and its undo;
  - "multiple fish", a split, a split of the split-off card, and a move;
  - other answers and an equipment answer;
  - follows (the video is published with `--follow`): one followed to the end, with a box
    on another card, spots with and without a box, and an undo; a second of the same fish,
    which finds the first one's boxes taken and shares its card; one discarded; one asking
    for less than a second;
  - requests the site turns down.

  The test then checks three things:
  - opening the page again replays to exactly the same state;
  - after the events are pulled into the project, the Python server gives the same card
    list, card details, look-alikes, counts and follows (each one's view and summary),
    field by field;
  - the published size of each sighting keeps look-alike groups in the server's order
    when two fish are the same size.

  The same comparison was made by hand in a browser on the demo project, with the same
  result.

## Measured on the 44-minute test video

`publish` on a copy of `bruv_project_full_v2` (1,145 published cards), on the CPU-only
laptop:

| | Size |
|---|---|
| Time to publish | 8.5 minutes |
| Files | 1,153 (1,149 sheets + manifest, card data, descriptions, still-object sheet): 6% of a site |
| Pictures (`sheets/`) | 233 MB, about 0.2 MB a card |
| `emb.bin` (descriptions) | 17.8 MB: the largest file, against the 25 MiB limit |
| `cards.json` | 13.5 MB (26,278 sightings) |
| Whole video | about 263 MB |
| With `--follow` (27 Oct, 1,149 cards) | 21.5 minutes; 1,681 files (8% of a site), 679 MB. The follow part: 526 strips, 389 MB; `boxes.json` 3.2 MB, `frames.json` 0.1 MB |

- A volunteer's browser downloads `cards.json` and `emb.bin` (about 31 MB) once when the video
  opens, then each card's sheet as they reach it.
- About 15 videos of this length fit in one site, or 11 published with `--follow`.
- Following a fish downloads a strip of 20 frames (about 0.75 MB) every 5 seconds of
  video it steps through, and the boxes file once.
- A video about 60% longer than this one would put `emb.bin` over 25 MiB. It would then
  have to be split into parts; publish doesn't do that yet.
