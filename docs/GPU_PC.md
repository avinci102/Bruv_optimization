# Processing on the GPU PC

The PC with the NVIDIA graphics card processes new BRUV videos and puts them on the review
site. Claude Code on that PC does the installing and the running. You copy the videos,
answer a few prompts, and say when to put them online.

The laptop doesn't need to be on, and neither PC needs a copy of the other's files. The
two PCs keep in step through the live site itself (see "Two PCs, one site" below).

## What you do

### Once (about 30 minutes, mostly waiting)

1. **Install the Claude desktop app** on the GPU PC from <https://claude.ai/download>, and
   sign in.
2. **Make a folder** `C:\BRUV`. In the Claude app, open the **Code** tab, choose `C:\BRUV`
   as the folder, and paste this:

   > Set up this PC to process BRUV videos for the review site. Install Git if it's
   > missing (`winget install -e --id Git.Git`). Clone
   > https://github.com/avinci102/Bruv_optimization into `C:\BRUV\Bruv_optimization`, check
   > out the branch `claude/hosted-alpha`, then follow docs/GPU_PC.md, "For Claude on the
   > GPU PC". Ask me before signing in to anything or putting anything online.

3. **Allow** what it asks for:
   - Windows asks to allow the installers (Python and Node.js).
   - A browser page asks you to let wrangler, Cloudflare's tool, into the Cloudflare
     account the site runs on. Sign in with that account, then click **Allow**.
   - When Claude asks to run a command, you can allow that kind of command for the rest of
     the session, which saves clicks later.

### Each batch of videos

4. **Copy the videos** into `C:\BRUV\videos`, one folder per deployment (one drop of the
   rig). Put all of that deployment's video files in it, in any order. The folder's name is
   the video's name on the site, for example `A1`; use letters, digits, `-` and `_`.
   Put the Excel viewing template once in `C:\BRUV\videos` itself.
   ```
   C:\BRUV\videos\
       BRUV viewing template v.2024.xlsx
       A1\  DJI_0001.MP4  DJI_0002.MP4
       A2\  DJI_0003.MP4
   ```
5. **Tell Claude:** "Process the new videos and get them ready for the site." Leave the PC
   on. The app keeps it awake while it works. Claude tells you when it's done, and what
   each video came to.
6. **Say "put them online"** when you're ready. Claude checks that nothing already online
   would be lost, then deploys.
7. **Tell the architecture chat on the laptop** who should review each new video. It
   updates the volunteer list there; this PC never needs that list.

Processing a video that's already been processed just resumes it. A video that's already
published isn't published again.

## Two PCs, one site

A deploy replaces the whole live site with the folder it uploads. A PC that deploys without
every video already online would take the missing ones off the site. So every deploy goes
through `sitesync`, which works like this:

- Each deploy puts a list of every video's files online, at `/videos/index.json`, with
  the code's commit.
- Before deploying, a PC downloads the videos it lacks from the live site (`mirror`, plain
  downloads, free).
- The deploy is refused if it would:
  - take a live video off the site (unless you name it with `--drop`);
  - put back an older copy of a video another PC re-published;
  - run code older than what's online.

  The refusal says what to do instead.
- Each site folder remembers what it last saw online (`.sync.json`, never uploaded). That's
  how it tells a video re-published here, which goes up, from one changed online, which
  comes down.

Rules for both PCs:

- **Deploy only with `sitesync deploy`** (or `scripts\deploy_site.bat`), never with plain
  `wrangler pages deploy`. A plain deploy leaves the list out of date.
- **One deploy at a time.** If two PCs deploy at once, the later one wins. The next
  `mirror` notices, and you deploy again.

**Before the GPU PC's first deploy**, the live site needs its list. The laptop, which has
every live video, deploys once with `--first` (from a checkout that has `sitesync`):
```
python -m bruv_app.sitesync deploy --site ..\bruv_site --first
```
Until then, the GPU PC can process and publish, but its deploy waits.

## For Claude on the GPU PC

Run everything from `C:\BRUV\Bruv_optimization`. Ask the user before signing in to
anything and before deploying. Run long steps in the background and report as they finish.
Never commit videos, model weights or project folders.

1. **Set up** (Python 3.13, Node.js, the app with PyTorch for the card, the site folder):
   ```
   powershell -ExecutionPolicy Bypass -File scripts\gpu_pc_setup.ps1
   ```
   - It's safe to run again: it skips what's done. Add `-Check` to only report.
   - If it says a new window is needed to see Python or Node, run it again in a new shell.
   - Check its `check-setup` lines. You want:
     - `Processing will run on: cuda`;
     - `PyTorch … (CUDA …): runs on <the card>`.

     If the card is found but PyTorch runs on the CPU, follow the advice printed (usually
     updating the NVIDIA driver, then deleting `.venv` and running the script again).
   - The Cloudflare step runs `wrangler login`: tell the user a browser page will open
     first. Then `sitesync init` writes `C:\BRUV\site\wrangler.toml` with the database's
     id and downloads the live videos.
     - If it says the live site has no index yet, the laptop hasn't made its `--first`
       deploy. Processing can go ahead anyway.

2. **Process and publish** every deployment folder in `C:\BRUV\videos`. Update the code
   first, then run the batch in the background:
   ```
   git pull --ff-only
   .venv\Scripts\python.exe -m bruv_app.batch --videos C:\BRUV\videos --site C:\BRUV\site
   ```
   - It processes each folder as `quickstart` does, at the app's default 4 frames per
     second, into `<folder>\bruv_project`. A crash is retried twice and resumes where it
     stopped.
   - It downloads the live videos, then publishes each new video under its folder's name
     with `--follow`.
   - It ends by saying what a deploy would do.
   - Options:
     - `--only A1 A2`: just those folders;
     - `--no-follow`: smaller; fish can't be followed online;
     - `--republish A1`: after re-processing A1.
   - The user can do the same by double-clicking `scripts\process_new_videos.bat`.
   - Report to the user, per video: the processing time, the cards, and the files and MB
     published. These are the first measurements on this PC.

3. **Review window, if the user gives one** (when the rig settled, when it was lifted).
   Cards outside it aren't published.
   ```
   .venv\Scripts\python.exe -m bruv_app review-window --project C:\BRUV\videos\A1\bruv_project --start 1:30 --end 58:10
   ```
   Then publish again with `--only A1 --republish A1`.

4. **Deploy, once the user says so:**
   ```
   .venv\Scripts\python.exe -m bruv_app.sitesync mirror --site C:\BRUV\site
   .venv\Scripts\python.exe -m bruv_app.sitesync deploy --site C:\BRUV\site --yes
   ```
   - `--yes` skips its own question, since the user has already said yes in the chat.
   - If it refuses, tell the user what it says. Don't reach for `--drop`, `--overwrite` or
     `--force-code` without their OK.
   - After it deploys, give the user the site's address and the list of videos now live.
     Remind them to have the laptop's architecture chat assign the new videos to
     volunteers.

5. **Leave the projects where they are.** `C:\BRUV\videos\<name>\bruv_project` is where
   volunteers' answers get pulled in and exported (below).

## Where the answers go

Volunteers' answers are kept in the site's database. To pull them into a project and export
to Excel, do it on whichever PC holds that project: here, or the laptop after copying
`C:\BRUV\videos\<name>\bruv_project` across.
- On the start page, signed in as an admin, click **Download answers: A1**.
- Then: `python -m bruv_app pull --project C:\BRUV\videos\A1\bruv_project --answers answers-A1.json --volunteer Ana`.
- Details are in [HOSTED.md](HOSTED.md), step 8.

## Sizes and limits

Measured on the 44-minute test video (`trial`):

| | Per 44-minute video |
|---|---|
| Source video (2.7K HEVC) | 17.2 GB |
| Project folder (frames at 4 fps, database) | 1.4 GB |
| Published with `--follow` | 1,695 files, 0.68 GB uploaded |
| Published without `--follow` | about 1,150 files, 0.26 GB |

- **20,000 files per site** (free plan): about 11 videos of that length with `--follow`, or
  15 without. `sitesync check` counts them before every deploy. When a site is full, make
  a second site ([HOSTED.md](HOSTED.md)).
- **25 MiB per file:** a video's `emb.bin` grows with its cards. At 17.8 MB for 44 minutes,
  a review window much past 60 minutes may go over. The deploy refuses it and names the
  file. Splitting it isn't built yet: tell the architecture chat.
- **Disk:** keep about 20 GB free per video for its source files, project and published
  copy. Once a video's answers are pulled and exported, its source files can move to an
  external drive.
- **Processing time** on this PC isn't measured yet. On the CPU-only laptop it took hours
  per video. Report the first video's time.
