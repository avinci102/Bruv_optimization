"""SQLite project store, so a long video can be processed and reviewed over several sessions.

A project is a folder: project.db, settings.json, frames/ (review JPEGs), exports/, and
snapshots/ (frozen, read-only copies of the database, see Project.snapshot).
"""

from __future__ import annotations

import json
import os
import sqlite3
import stat
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import Settings
from .detect import Detection

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS chapters (
    idx INTEGER PRIMARY KEY, path TEXT, name TEXT, start_s REAL, duration_s REAL, fps REAL);
CREATE TABLE IF NOT EXISTS frames (
    id INTEGER PRIMARY KEY, seq INTEGER UNIQUE, t REAL, chapter INTEGER, local_t REAL,
    path TEXT, width INTEGER, height INTEGER);
-- The source video frame each sampled frame is (video.source_index). Its own table, so the
-- frames table keeps the columns older versions read, and they can still open the project.
CREATE TABLE IF NOT EXISTS frame_sources (frame_id INTEGER PRIMARY KEY, src_index INTEGER);
CREATE VIEW IF NOT EXISTS frames_src AS
    SELECT f.id, f.seq, f.t, f.chapter, f.local_t, f.path, f.width, f.height, s.src_index
    FROM frames f LEFT JOIN frame_sources s ON s.frame_id=f.id;
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY, frame_id INTEGER REFERENCES frames(id),
    x1 REAL, y1 REAL, x2 REAL, y2 REAL, score REAL, quality REAL, app BLOB,
    tracklet_id INTEGER, embedding BLOB,
    link_cost REAL, link_margin REAL, history_sim REAL);
CREATE TABLE IF NOT EXISTS tracklets (
    id INTEGER PRIMARY KEY, fish_id INTEGER, start_t REAL, end_t REAL, embedding BLOB);
CREATE TABLE IF NOT EXISTS fish (
    id INTEGER PRIMARY KEY, start_t REAL, end_t REAL, guesses TEXT, best_det INTEGER,
    status TEXT NOT NULL DEFAULT 'pending', species TEXT, type TEXT, sex TEXT,
    behavior TEXT, comment TEXT, reviewed_at TEXT);
CREATE TABLE IF NOT EXISTS stages (name TEXT PRIMARY KEY, done_at TEXT);
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY, at TEXT, action TEXT, fish_id INTEGER, details TEXT);
-- One row per stage run: the settings, code and models it used, and what it made.
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY, stage TEXT, started_at TEXT, finished_at TEXT,
    settings TEXT, code TEXT, info TEXT);
-- The cards as the machine built them, before any review: tracker and merge decisions
-- stay scoreable after a reviewer splits or merges cards.
CREATE TABLE IF NOT EXISTS original_groups (
    run_id INTEGER, det_id INTEGER, tracklet_id INTEGER, fish_id INTEGER,
    PRIMARY KEY (run_id, det_id));
-- Every answer given, with the detections the card held when it was given (its scope): a
-- label never spreads to detections the reviewer didn't see on the card. details: the
-- card's warnings when it was answered (diagnostics.py) and that the reviewer confirmed them.
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY, at TEXT, fish_id INTEGER, status TEXT, species TEXT, scope TEXT,
    details TEXT);
-- Equipment a reviewer marked (equipment.py): the card, its usual box and its colours.
CREATE TABLE IF NOT EXISTS rig_refs (
    id INTEGER PRIMARY KEY, at TEXT, fish_id INTEGER, x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    colour BLOB, active INTEGER NOT NULL DEFAULT 1);
-- Every join the merge weighed between two pieces (tracklets), with its evidence, and
-- whether it was made: "joined", "lost" (a better join took one end), or the check it failed.
CREATE TABLE IF NOT EXISTS merge_edges (
    run_id INTEGER, a INTEGER, b INTEGER, gap REAL, similarity REAL, colour REAL,
    size_ratio REAL, decision TEXT, PRIMARY KEY (run_id, a, b));
-- How much of each detection's box differs from the scene's background (still.py): the
-- bait arm and canister barely differ, fish do. Its own table, like frame_sources.
CREATE TABLE IF NOT EXISTS det_change (det_id INTEGER PRIMARY KEY, change REAL);
-- Look-alikes shown in the "answer together" list (lookalike.py) and whether the reviewer
-- left each one ticked: the evidence an auto-accept threshold would be set from.
CREATE TABLE IF NOT EXISTS suggestion_log (
    id INTEGER PRIMARY KEY, at TEXT, fish_id INTEGER, source INTEGER, similarity REAL,
    decision TEXT, batch TEXT);
-- Fish followed frame by frame (follow.py): clicked ground truth for continuity. A mark per
-- frame looked at: the fish's box, "nobox" (where it was, if clicked) or "gone".
CREATE TABLE IF NOT EXISTS follows (
    id INTEGER PRIMARY KEY, at TEXT, card INTEGER, start_det INTEGER, gone_s REAL, done_at TEXT);
CREATE TABLE IF NOT EXISTS follow_marks (
    follow_id INTEGER, frame_id INTEGER, state TEXT, det_id INTEGER, x REAL, y REAL, at TEXT,
    PRIMARY KEY (follow_id, frame_id));
CREATE INDEX IF NOT EXISTS det_frame ON detections(frame_id);
CREATE INDEX IF NOT EXISTS det_track ON detections(tracklet_id);
CREATE INDEX IF NOT EXISTS track_fish ON tracklets(fish_id);
CREATE INDEX IF NOT EXISTS review_fish ON reviews(fish_id);
CREATE INDEX IF NOT EXISTS correction_fish ON corrections(fish_id);
"""

# Review outcomes. "species" means a species was picked.
STATUSES = ("pending", "species", "unknown", "not_fish", "skipped")
REVIEWED = ("species", "unknown", "not_fish")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_blob(x: np.ndarray | None) -> bytes | None:
    return None if x is None else np.asarray(x, dtype=np.float32).tobytes()


def from_blob(b: bytes | None) -> np.ndarray | None:
    return None if b is None else np.frombuffer(b, dtype=np.float32)


def box_px(d) -> float:
    """A detection's size: its box's longer side, in pixels of the saved frame."""
    return max((d["x2"] - d["x1"]) * d["width"], (d["y2"] - d["y1"]) * d["height"])


@dataclass
class FrameRow:
    id: int
    seq: int
    t: float
    chapter: int
    local_t: float
    path: str
    width: int
    height: int
    src_index: int | None = None  # the source video frame (video.source_index)


class Project:
    def __init__(self, root: Path, db: str = "project.db"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "frames").mkdir(exist_ok=True)
        # A rule trial (trials.py) is another database in the folder, sharing its frames.
        self.db_path = self.root / db
        self.db = sqlite3.connect(self.db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self._migrate()
        self.db.commit()

    @classmethod
    def read_only(cls, db_path: Path) -> "Project":
        """Open a database without changing it, e.g. a snapshot (snapshots/x.db: its project
        folder, for frames and settings, is the one above)."""
        db_path = Path(db_path)
        self = cls.__new__(cls)
        self.root = db_path.parent.parent if db_path.parent.name == "snapshots" else db_path.parent
        self.db_path = None  # its settings are the project's
        self.db = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        return self

    def _migrate(self) -> None:
        """Bring a project made by an older version up to date (columns are only added)."""
        # Never add columns to frames: older versions build a record from all of them.
        added = {
            "detections": {"link_cost": "REAL", "link_margin": "REAL", "history_sim": "REAL"},
            # family: an answer's family (families.py), kept apart from the species, which
            # is optional (docs/SPEC.md §15).
            "reviews": {"details": "TEXT", "family": "TEXT"},
        }
        for table, columns in added.items():
            have = {r["name"] for r in self.db.execute(f"PRAGMA table_info({table})")}
            for name, kind in columns.items():
                if name not in have:
                    self.db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")

    # --- settings & meta -------------------------------------------------------------

    @property
    def settings_path(self) -> Path:
        """settings.json; a rule trial's own are beside its database (trials/<name>.settings.json)."""
        db_path = getattr(self, "db_path", None)
        if db_path is None or db_path.name == "project.db":
            return self.root / "settings.json"
        return db_path.with_suffix(".settings.json")

    def load_settings(self) -> Settings:
        return Settings.load(self.settings_path) if self.settings_path.exists() else Settings()

    def save_settings(self, settings: Settings) -> None:
        settings.save(self.settings_path)

    def get_meta(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set_meta(self, key: str, value) -> None:
        self.db.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )
        self.db.commit()

    # --- run records & snapshots -------------------------------------------------------

    def start_run(self, stage: str, settings: Settings, **info) -> int:
        """Record a stage starting: the settings and code it runs with, plus `info` (models,
        inputs). finish_run adds what it made."""
        from .provenance import code_revision  # noqa: PLC0415

        cur = self.db.execute(
            "INSERT INTO runs(stage, started_at, settings, code, info) VALUES(?, ?, ?, ?, ?)",
            (stage, _now(), json.dumps(asdict(settings)), code_revision(), json.dumps(info)),
        )
        self.db.commit()
        return cur.lastrowid

    def finish_run(self, run_id: int, **info) -> None:
        row = self.db.execute("SELECT info FROM runs WHERE id=?", (run_id,)).fetchone()
        merged = {**json.loads(row["info"] or "{}"), **info}
        self.db.execute(
            "UPDATE runs SET finished_at=?, info=? WHERE id=?", (_now(), json.dumps(merged), run_id)
        )
        self.db.commit()

    def runs(self, stage: str | None = None) -> list[sqlite3.Row]:
        if stage is None:
            return self.db.execute("SELECT * FROM runs ORDER BY id").fetchall()
        return self.db.execute("SELECT * FROM runs WHERE stage=? ORDER BY id", (stage,)).fetchall()

    def freeze_groups(self, run_id: int) -> int:
        """Keep the cards as the machine built them (detection -> tracklet -> card)."""
        cur = self.db.execute(
            """INSERT OR REPLACE INTO original_groups(run_id, det_id, tracklet_id, fish_id)
               SELECT ?, d.id, d.tracklet_id, t.fish_id FROM detections d
               JOIN tracklets t ON t.id=d.tracklet_id""",
            (run_id,),
        )
        self.db.commit()
        return cur.rowcount

    def save_merge_edges(self, run_id: int, edges) -> None:
        self.db.executemany(
            """INSERT OR REPLACE INTO merge_edges(run_id, a, b, gap, similarity, colour, size_ratio, decision)
               VALUES(?, ?, ?, ?, ?, ?, ?, ?)""",
            [(run_id, e.a, e.b, e.gap, e.similarity, e.colour, e.size_ratio, e.decision) for e in edges],
        )
        self.db.commit()

    def merge_edges(self, run_id: int | None = None) -> list[sqlite3.Row]:
        """The joins the merge weighed: by the given merge run, or the latest one."""
        if run_id is None:
            run_id = self.db.execute("SELECT MAX(run_id) FROM merge_edges").fetchone()[0]
        return self.db.execute("SELECT * FROM merge_edges WHERE run_id=? ORDER BY a, b", (run_id,)).fetchall()

    def original_groups(self, run_id: int | None = None) -> dict[int, tuple[int, int]]:
        """Detection id -> (tracklet id, card id) as first built: by the given merge run, or
        the latest one."""
        if run_id is None:
            run_id = self.db.execute("SELECT MAX(run_id) FROM original_groups").fetchone()[0]
        rows = self.db.execute(
            "SELECT det_id, tracklet_id, fish_id FROM original_groups WHERE run_id=?", (run_id,)
        )
        return {r[0]: (r[1], r[2]) for r in rows}

    def snapshot(self, label: str) -> Path:
        """Freeze the project's database as it is now: a compact, read-only copy in snapshots/,
        with a manifest saying what made it. Results scored from a snapshot can be
        reproduced, whatever is reviewed or re-run afterwards."""
        from .provenance import code_revision, fingerprint  # noqa: PLC0415

        self.fill_source_indices()
        self.db.commit()
        folder = self.root / "snapshots"
        folder.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in label) or "snapshot"
        path = folder / f"{stamp}-{safe}.db"
        n = 1
        while path.exists():
            n += 1
            path = folder / f"{stamp}-{safe}-{n}.db"
        self.db.execute("VACUUM INTO ?", (str(path),))
        counts = {
            table: self.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608
            for table in ("frames", "detections", "tracklets", "fish", "corrections")
        }
        manifest = {
            "label": label,
            "made_at": _now(),
            "project": str(self.root),
            "deployment": self.get_meta("deployment"),
            "videos": self.get_meta("video_fingerprints"),
            "code": code_revision(),
            "settings": asdict(self.load_settings()),
            "runs": [dict(r) for r in self.runs()],
            "counts": counts,
            "reviewed": self.reviewed_count(),
            "database": fingerprint(path, quick=False),
        }
        path.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
        for p in (path, path.with_suffix(".json")):
            os.chmod(p, stat.S_IREAD)
        return path

    def stage_done(self, name: str) -> bool:
        return self.db.execute("SELECT 1 FROM stages WHERE name=?", (name,)).fetchone() is not None

    def mark_stage(self, name: str, done: bool = True) -> None:
        if done:
            self.db.execute("INSERT OR REPLACE INTO stages VALUES(?, ?)", (name, _now()))
        else:
            self.db.execute("DELETE FROM stages WHERE name=?", (name,))
        self.db.commit()

    # --- chapters & frames ---------------------------------------------------------------

    def save_chapters(self, chapters) -> None:
        self.db.execute("DELETE FROM chapters")
        self.db.executemany(
            "INSERT INTO chapters VALUES(?, ?, ?, ?, ?, ?)",
            [(c.index, str(c.path), c.name, c.start_s, c.duration_s, c.fps) for c in chapters],
        )
        self.db.commit()

    def chapters(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM chapters ORDER BY idx").fetchall()

    def add_frame(self, seq, t, chapter, local_t, path, width, height, src_index=None) -> int:
        cur = self.db.execute(
            "INSERT INTO frames(seq, t, chapter, local_t, path, width, height) VALUES(?,?,?,?,?,?,?)",
            (seq, t, chapter, local_t, path, width, height),
        )
        if src_index is not None:
            self.db.execute("INSERT INTO frame_sources(frame_id, src_index) VALUES(?, ?)", (cur.lastrowid, src_index))
        return cur.lastrowid

    def add_detections(self, frame_id: int, dets: list[Detection], qualities: list[float]) -> None:
        self.db.executemany(
            "INSERT INTO detections(frame_id, x1, y1, x2, y2, score, quality, app) VALUES(?,?,?,?,?,?,?,?)",
            [(frame_id, d.x1, d.y1, d.x2, d.y2, d.score, q, to_blob(d.app)) for d, q in zip(dets, qualities)],
        )

    def commit(self) -> None:
        self.db.commit()

    def last_frame(self) -> FrameRow | None:
        row = self.db.execute("SELECT * FROM frames_src ORDER BY seq DESC LIMIT 1").fetchone()
        return FrameRow(**dict(row)) if row else None

    def frame(self, frame_id: int) -> FrameRow | None:
        row = self.db.execute("SELECT * FROM frames_src WHERE id=?", (frame_id,)).fetchone()
        return FrameRow(**dict(row)) if row else None

    def frame_count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM frames").fetchone()[0]

    def fill_source_indices(self) -> int:
        """Give frames saved before source frame numbers were kept a number worked out from
        their time. That is exact when detection ran in one go from the start of the video;
        older versions handed back frames one step late after a resume, so the project is
        flagged and anything that relies on the numbers checks them against the saved frames."""
        from .video import source_index  # noqa: PLC0415

        rows = self.db.execute(
            """SELECT f.id, f.local_t, c.fps FROM frames f JOIN chapters c ON c.idx=f.chapter
               LEFT JOIN frame_sources s ON s.frame_id=f.id WHERE s.frame_id IS NULL"""
        ).fetchall()
        if rows:
            self.db.executemany(
                "INSERT INTO frame_sources(frame_id, src_index) VALUES(?, ?)",
                [(r["id"], source_index(r["local_t"], r["fps"])) for r in rows],
            )
            self.set_meta("src_index_inferred", True)
        return len(rows)

    def frame_rate(self) -> float:
        """Sampled frames per second, from the stored frames' times (1.0 before any)."""
        times = [r[0] for r in self.db.execute("SELECT t FROM frames ORDER BY seq")]
        steps = [b - a for a, b in zip(times, times[1:]) if b > a]
        return 1.0 / float(np.median(steps)) if steps else 1.0

    def frames_with_detections(self, min_score: float = 0.0, exclude: set[int] | None = None):
        """Yield (FrameRow, [(det id, Detection)]) in time order: the detections scoring at
        least `min_score` (the rest are kept below the operating threshold, untracked), less
        any in `exclude` (e.g. still objects)."""
        frames = self.db.execute("SELECT * FROM frames_src ORDER BY seq").fetchall()
        dets: dict[int, list] = {}
        for d in self.db.execute("SELECT * FROM detections WHERE score >= ? ORDER BY id", (min_score,)):
            if exclude and d["id"] in exclude:
                continue
            dets.setdefault(d["frame_id"], []).append(
                (d["id"], Detection(d["x1"], d["y1"], d["x2"], d["y2"], d["score"], from_blob(d["app"])))
            )
        for f in frames:
            yield FrameRow(**dict(f)), dets.get(f["id"], [])

    # --- still objects (still.py) ----------------------------------------------------------

    def save_det_change(self, values: dict[int, float]) -> None:
        self.db.execute("DELETE FROM det_change")
        self.db.executemany("INSERT INTO det_change(det_id, change) VALUES(?, ?)", list(values.items()))
        self.db.commit()

    def det_changes(self) -> dict[int, float]:
        return dict(self.db.execute("SELECT det_id, change FROM det_change").fetchall())

    def still_ids(self, settings: Settings) -> set[int]:
        """Detections the still-object test hides (none if it's off or hasn't run)."""
        if not settings.still_filter:
            return set()
        rows = self.db.execute(
            """SELECT d.id FROM detections d JOIN det_change c ON c.det_id=d.id
               WHERE d.x2 - d.x1 >= ? AND c.change < ?""",
            (settings.still_min_width, settings.still_max_change),
        )
        return {r[0] for r in rows}

    def still_detections(self, settings: Settings, min_score: float) -> list[sqlite3.Row]:
        """The hidden still objects scoring at least `min_score`, in time order."""
        if not settings.still_filter:
            return []
        return self.db.execute(
            """SELECT d.*, c.change, f.t, f.path, f.width, f.height FROM detections d
               JOIN det_change c ON c.det_id=d.id JOIN frames f ON f.id=d.frame_id
               WHERE d.x2 - d.x1 >= ? AND c.change < ? AND d.score >= ? ORDER BY f.t, d.id""",
            (settings.still_min_width, settings.still_max_change, min_score),
        ).fetchall()

    # --- tracklets & fish ------------------------------------------------------------------

    def reset_tracking(self) -> None:
        # Crop embeddings stay: they don't depend on tracking, so a rerun can reuse them.
        self.db.execute(
            "UPDATE detections SET tracklet_id=NULL, link_cost=NULL, link_margin=NULL, history_sim=NULL"
        )
        self.db.execute("DELETE FROM tracklets")
        self.db.execute("DELETE FROM fish")
        self.db.commit()

    def clear_embeddings(self) -> None:
        self.db.execute("UPDATE detections SET embedding=NULL")
        self.db.commit()

    def save_tracklets(self, tracklets) -> None:
        for tr in tracklets:
            self.db.execute("INSERT INTO tracklets(id) VALUES(?)", (tr.id,))
            links = getattr(tr, "links", {})
            self.db.executemany(
                "UPDATE detections SET tracklet_id=?, link_cost=?, link_margin=?, history_sim=? WHERE id=?",
                [(tr.id, *links.get(key, (None, None, None)), key) for _, key in tr.members],
            )
        self.db.execute(
            """UPDATE tracklets SET
                 start_t=(SELECT MIN(f.t) FROM detections d JOIN frames f ON f.id=d.frame_id
                          WHERE d.tracklet_id=tracklets.id),
                 end_t=(SELECT MAX(f.t) FROM detections d JOIN frames f ON f.id=d.frame_id
                        WHERE d.tracklet_id=tracklets.id)"""
        )
        self.db.commit()

    def tracklet_detections(self, tracklet_id: int) -> list[sqlite3.Row]:
        return self.db.execute(
            """SELECT d.*, f.t, f.path, f.width, f.height FROM detections d
               JOIN frames f ON f.id=d.frame_id WHERE d.tracklet_id=? ORDER BY f.t""",
            (tracklet_id,),
        ).fetchall()

    def tracklet_ids(self) -> list[int]:
        return [r[0] for r in self.db.execute("SELECT id FROM tracklets ORDER BY start_t, id")]

    def set_detection_embedding(self, det_id: int, emb: np.ndarray) -> None:
        self.db.execute("UPDATE detections SET embedding=? WHERE id=?", (to_blob(emb), det_id))

    def set_tracklet_embedding(self, tracklet_id: int, emb: np.ndarray) -> None:
        self.db.execute("UPDATE tracklets SET embedding=? WHERE id=?", (to_blob(emb), tracklet_id))

    def tracklets(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM tracklets ORDER BY start_t, id").fetchall()

    def create_fish(self, tracklet_ids: list[int]) -> int:
        # A new card never takes the number of a card that's gone (merged away, or rebuilt):
        # answers and fixes are logged by card number, and an old card's answer must never
        # be read as the new card's.
        fish_id = self.db.execute(
            """SELECT MAX(COALESCE((SELECT MAX(id) FROM fish), 0),
                          COALESCE((SELECT MAX(fish_id) FROM reviews), 0),
                          COALESCE((SELECT MAX(fish_id) FROM corrections), 0),
                          COALESCE((SELECT MAX(fish_id) FROM rig_refs), 0)) + 1"""
        ).fetchone()[0]
        self.db.execute("INSERT INTO fish(id, status) VALUES(?, 'pending')", (fish_id,))
        self.db.executemany(
            "UPDATE tracklets SET fish_id=? WHERE id=?", [(fish_id, t) for t in tracklet_ids]
        )
        self._refresh_span(fish_id)
        return fish_id

    def _refresh_span(self, fish_id: int) -> None:
        self.db.execute(
            """UPDATE fish SET
                 start_t=(SELECT MIN(start_t) FROM tracklets WHERE fish_id=?),
                 end_t=(SELECT MAX(end_t) FROM tracklets WHERE fish_id=?)
               WHERE id=?""",
            (fish_id, fish_id, fish_id),
        )

    def fish_tracklets(self, fish_id: int) -> list[int]:
        return [
            r[0]
            for r in self.db.execute(
                "SELECT id FROM tracklets WHERE fish_id=? ORDER BY start_t, id", (fish_id,)
            )
        ]

    def fish_detections(self, fish_id: int) -> list[sqlite3.Row]:
        return self.db.execute(
            """SELECT d.*, f.t, f.chapter, f.local_t, f.path, f.width, f.height FROM detections d
               JOIN tracklets t ON t.id=d.tracklet_id JOIN frames f ON f.id=d.frame_id
               WHERE t.fish_id=? ORDER BY f.t, d.id""",
            (fish_id,),
        ).fetchall()

    def fish_sizes(self) -> dict[int, float]:
        """Each card's fish at its largest: the box's longer side, in pixels of the saved frames."""
        return dict(
            self.db.execute(
                """SELECT t.fish_id, MAX(MAX((d.x2-d.x1)*f.width, (d.y2-d.y1)*f.height)) FROM detections d
                   JOIN tracklets t ON t.id=d.tracklet_id JOIN frames f ON f.id=d.frame_id
                   WHERE t.fish_id IS NOT NULL GROUP BY t.fish_id"""
            ).fetchall()
        )

    def set_fish_guesses(self, fish_id: int, guesses: list, best_det: int | None) -> None:
        self.db.execute(
            "UPDATE fish SET guesses=?, best_det=? WHERE id=?",
            (json.dumps(guesses), best_det, fish_id),
        )

    def fish(self, fish_id: int) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM fish WHERE id=?", (fish_id,)).fetchone()

    def all_fish(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM fish ORDER BY start_t, id").fetchall()

    def label_fish(self, fish_id: int, status: str, details: dict | None = None, **fields) -> None:
        """Record an answer. `details` goes with it into the review log (e.g. the card's
        warnings, and that the reviewer confirmed them); so does `family` (an answer's
        family, which the fish table has no column for)."""
        if status not in STATUSES:
            raise ValueError(f"Bad status: {status}")
        species = fields.get("species") if status == "species" else None
        if status == "species" and not species:
            raise ValueError("A species is required")
        self.db.execute(
            """UPDATE fish SET status=?, species=?, type=?, sex=?, behavior=?, comment=?,
                 reviewed_at=? WHERE id=?""",
            (
                status,
                species,
                fields.get("type") or None,
                fields.get("sex") or None,
                fields.get("behavior") or None,
                fields.get("comment") or None,
                _now() if status != "pending" else None,
                fish_id,
            ),
        )
        scope = [d["id"] for d in self.fish_detections(fish_id)] if status != "pending" else []
        family = fields.get("family") if status == "species" else None
        self.db.execute(
            "INSERT INTO reviews(at, fish_id, status, species, scope, details, family) VALUES(?, ?, ?, ?, ?, ?, ?)",
            (_now(), fish_id, status, species, json.dumps(scope), json.dumps(details) if details else None, family),
        )
        self.db.commit()

    def log_suggestions(self, rows: list[tuple[int, int, float, str, str]]) -> None:
        """(card, source card, similarity, decision, batch) for look-alikes a reviewer saw."""
        self.db.executemany(
            "INSERT INTO suggestion_log(at, fish_id, source, similarity, decision, batch) VALUES(?, ?, ?, ?, ?, ?)",
            [(_now(), *row) for row in rows],
        )
        self.db.commit()

    def answer_family(self, fish_id: int) -> str | None:
        """The family recorded with a card's current answer (None before families were kept)."""
        review = self.last_review(fish_id)
        return review["family"] if review is not None and "family" in review.keys() else None

    # --- the review window: the part of the video that is reviewed and counted ----------

    def review_window(self) -> tuple[float | None, float | None]:
        """(start, end) in seconds of the combined video; None where it's open. Before the
        start the rig is being set down, after the end it is being lifted: divers, hands
        and drifting rope, no survey."""
        window = self.get_meta("review_window") or {}
        return window.get("start_s"), window.get("end_s")

    def set_review_window(self, start_s: float | None, end_s: float | None) -> None:
        if start_s is not None and end_s is not None and end_s <= start_s:
            raise ValueError("The review window must end after it starts")
        self.set_meta("review_window", {"start_s": start_s, "end_s": end_s})

    def in_window(self, t: float) -> bool:
        start, end = self.review_window()
        return (start is None or t >= start) and (end is None or t <= end)

    def outside_window(self) -> set[int]:
        """Cards wholly outside the review window: not reviewed and not counted."""
        start, end = self.review_window()
        if start is None and end is None:
            return set()
        rows = self.db.execute("SELECT id, start_t, end_t FROM fish").fetchall()
        return {
            r["id"] for r in rows
            if r["end_t"] is not None and ((start is not None and r["end_t"] < start) or (end is not None and r["start_t"] > end))
        }

    def add_rig_ref(self, fish_id: int, box, colour) -> int:
        cur = self.db.execute(
            "INSERT INTO rig_refs(at, fish_id, x1, y1, x2, y2, colour) VALUES(?, ?, ?, ?, ?, ?, ?)",
            (_now(), fish_id, box.x1, box.y1, box.x2, box.y2, to_blob(colour)),
        )
        self.db.commit()
        return cur.lastrowid

    def rig_refs(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM rig_refs WHERE active=1 ORDER BY id").fetchall()

    def forget_rig_ref(self, ref_id: int) -> None:
        """Stop queueing cards like this reference; answers already given stay."""
        self.db.execute("UPDATE rig_refs SET active=0 WHERE id=?", (ref_id,))
        self.db.commit()

    def reviews_in_batch(self, batch: str) -> list[int]:
        """Cards whose current answer came from the given batch (e.g. of equipment)."""
        rows = self.db.execute(
            """SELECT r.fish_id, r.details FROM reviews r
               JOIN (SELECT fish_id, MAX(id) AS last FROM reviews GROUP BY fish_id) l ON l.last=r.id
               WHERE r.details IS NOT NULL"""
        )
        return [r["fish_id"] for r in rows if json.loads(r["details"]).get("batch") == batch]

    def answer_details(self, fish_id: int) -> dict:
        """What was kept with the card's current answer (e.g. equipment, confirmed warnings)."""
        review = self.last_review(fish_id)
        return json.loads(review["details"]) if review and review["details"] else {}

    def last_review(self, fish_id: int) -> sqlite3.Row | None:
        return self.db.execute(
            "SELECT * FROM reviews WHERE fish_id=? ORDER BY id DESC LIMIT 1", (fish_id,)
        ).fetchone()

    def label_scopes(self) -> dict[int, set[int]]:
        """Card id -> the detections its current answer was given for (from its latest
        review). Cards answered before scopes were recorded aren't listed."""
        rows = self.db.execute(
            """SELECT r.fish_id, r.scope FROM reviews r
               JOIN (SELECT fish_id, MAX(id) AS last FROM reviews GROUP BY fish_id) l ON l.last=r.id"""
        )
        return {r["fish_id"]: set(json.loads(r["scope"])) for r in rows}

    def answers_by_detection(self) -> dict[int, dict]:
        """Detection id -> the answer it holds now: each answered card's answer, for the
        detections it was given for that are still on the card. What a rebuild carries over."""
        scopes = self.label_scopes()
        out: dict[int, dict] = {}
        q = f"SELECT * FROM fish WHERE status IN ({','.join('?' * len(REVIEWED))})"
        for f in self.db.execute(q, REVIEWED).fetchall():
            details = self.answer_details(f["id"])
            answer = {k: f[k] for k in ("status", "species", "type", "sex", "behavior", "comment")}
            answer.update(card=f["id"], equipment=bool(details.get("equipment")))
            scope = scopes.get(f["id"])
            for d in self.fish_detections(f["id"]):
                if scope is None or d["id"] in scope:
                    out[d["id"]] = answer
        return out

    def carry_answers(self, answers: dict[int, dict], hidden: set[int] = frozenset()) -> dict:
        """Give rebuilt cards the answers their detections held (answers_by_detection).

        A card gets an answer only when every one of its detections had that same answer,
        so nothing is answered for a sighting nobody saw. A card with no answered detections
        stays pending; one with some, or with different answers, stays pending and is
        counted. Detections now hidden as still objects aren't on any card; those answered
        as fish are counted, so a fish the still test hid can be found. Returns the counts."""
        report = {"carried": 0, "partly_answered": 0, "conflicting": 0, "new": 0,
                  "hidden_answered_fish": sum(1 for d in hidden if answers.get(d, {}).get("status") in ("species", "unknown"))}
        key = ("status", "species")
        for f in self.all_fish():
            dets = [d["id"] for d in self.fish_detections(f["id"])]
            held = [answers[d] for d in dets if d in answers]
            if not held:
                report["new"] += 1
                continue
            if len({tuple(a[k] for k in key) for a in held}) > 1:
                report["conflicting"] += 1
                continue
            if len(held) < len(dets):
                report["partly_answered"] += 1
                continue
            # The other fields from the old card that gave most of these detections.
            source = max({a["card"]: a for a in held}.values(), key=lambda a: sum(h["card"] == a["card"] for h in held))
            details = {"carried": True, "from": sorted({a["card"] for a in held})}
            if source["equipment"]:
                details["equipment"] = True
            fields = {k: source[k] for k in ("species", "type", "sex", "behavior", "comment")}
            self.label_fish(f["id"], source["status"], details, **fields)
            report["carried"] += 1
        return report

    def reviewed_count(self) -> int:
        q = f"SELECT COUNT(*) FROM fish WHERE status IN ({','.join('?' * len(REVIEWED))})"
        return self.db.execute(q, REVIEWED).fetchone()[0]

    def split_fish(self, fish_id: int, from_t: float) -> int:
        """Move every detection of a fish at or after `from_t` onto a new, unreviewed fish.

        Works inside a tracklet too (the tracker can swap identities), by cutting that
        tracklet in two. Returns the new fish id.
        """
        dets = self.fish_detections(fish_id)
        if not any(d["t"] < from_t for d in dets) or not any(d["t"] >= from_t for d in dets):
            raise ValueError("Split point must leave detections on both sides")
        new_id = self._move_detections(fish_id, {d["id"] for d in dets if d["t"] >= from_t})
        self.log_correction("split", fish_id, from_t=from_t, new_card=new_id)
        self.db.commit()
        return new_id

    def move_to_new_fish(self, fish_id: int, det_ids: list[int]) -> int:
        """Move some of a card's detections (the ticked sightings) onto a new, unreviewed
        card: e.g. the other fish's sightings after two fish crossed and the tracker swapped
        them. The card keeps its answer. Returns the new card's id."""
        dets = {d["id"] for d in self.fish_detections(fish_id)}
        chosen = dets & set(det_ids)
        if not chosen or chosen == dets:
            raise ValueError("Tick some of this card's sightings, but not all of them")
        new_id = self._move_detections(fish_id, chosen)
        self.log_correction("move_out", fish_id, detections=sorted(chosen), new_card=new_id)
        self.db.commit()
        return new_id

    def _move_detections(self, fish_id: int, chosen: set[int]) -> int:
        """Put the chosen detections of a card on a new card. A tracklet that loses only some
        of its detections is cut in two."""
        moved: list[int] = []
        for tid in self.fish_tracklets(fish_id):
            rows = self.tracklet_detections(tid)
            out = [r["id"] for r in rows if r["id"] in chosen]
            if not out:
                continue
            if len(out) == len(rows):
                moved.append(tid)
                continue
            new_tid = self.db.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM tracklets").fetchone()[0]
            self.db.execute("INSERT INTO tracklets(id) VALUES(?)", (new_tid,))
            self.db.executemany("UPDATE detections SET tracklet_id=? WHERE id=?", [(new_tid, i) for i in out])
            self._refresh_tracklet(tid)
            self._refresh_tracklet(new_tid)
            moved.append(new_tid)
        new_id = self.create_fish(moved)
        self._refresh_span(fish_id)
        return new_id

    def unmerge_fish(self, fish_id: int) -> list[int]:
        """The card mixes different fish: give each of its tracklets (the stretches the
        tracker followed frame by frame) its own unreviewed card. This card keeps the first
        one and goes back to pending. Returns the new card ids."""
        tracklets = self.fish_tracklets(fish_id)
        if len(tracklets) < 2:
            raise ValueError(
                "This card is one tracked stretch: select the sighting where the other fish "
                "starts and split there"
            )
        new_ids = [self.create_fish([tid]) for tid in tracklets[1:]]
        self._refresh_span(fish_id)
        self.db.execute(
            "UPDATE fish SET status='pending', species=NULL, reviewed_at=NULL WHERE id=?", (fish_id,)
        )
        self.log_correction("multiple_fish", fish_id, tracklets=tracklets, new_cards=new_ids)
        self.db.commit()
        return new_ids

    def log_correction(self, action: str, fish_id: int, **details) -> None:
        """Record a reviewer's fix to the cards (a split, merge or break-up): it shows where
        tracking and the continuity merge go wrong, for tuning them and for training."""
        self.db.execute(
            "INSERT INTO corrections(at, action, fish_id, details) VALUES(?, ?, ?, ?)",
            (_now(), action, fish_id, json.dumps(details)),
        )

    def corrections(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM corrections ORDER BY id").fetchall()

    def _refresh_tracklet(self, tid: int) -> None:
        rows = self.tracklet_detections(tid)
        embs = [from_blob(r["embedding"]) for r in rows if r["embedding"] is not None]
        emb = None
        if embs:
            mean = np.mean(embs, axis=0)
            emb = mean / max(float(np.linalg.norm(mean)), 1e-12)
        self.db.execute(
            "UPDATE tracklets SET start_t=?, end_t=?, embedding=? WHERE id=?",
            (rows[0]["t"], rows[-1]["t"], to_blob(emb), tid),
        )

    def merge_fish(self, keep_id: int, other_id: int) -> None:
        """Fold `other` into `keep`. The kept fish goes back to pending for re-review."""
        if keep_id == other_id:
            raise ValueError("Can't merge a fish with itself")
        self.db.execute("UPDATE tracklets SET fish_id=? WHERE fish_id=?", (keep_id, other_id))
        self.db.execute("DELETE FROM fish WHERE id=?", (other_id,))
        self._refresh_span(keep_id)
        self.db.execute(
            "UPDATE fish SET status='pending', species=NULL, reviewed_at=NULL WHERE id=?",
            (keep_id,),
        )
        self.log_correction("merge", keep_id, merged_card=other_id)
        self.db.commit()
