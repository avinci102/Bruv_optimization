"""SQLite project store, so a long video can be processed and reviewed over several sessions.

A project is a folder: project.db, settings.json, frames/ (review JPEGs), exports/.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
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
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY, frame_id INTEGER REFERENCES frames(id),
    x1 REAL, y1 REAL, x2 REAL, y2 REAL, score REAL, quality REAL, app BLOB,
    tracklet_id INTEGER, embedding BLOB);
CREATE TABLE IF NOT EXISTS tracklets (
    id INTEGER PRIMARY KEY, fish_id INTEGER, start_t REAL, end_t REAL, embedding BLOB);
CREATE TABLE IF NOT EXISTS fish (
    id INTEGER PRIMARY KEY, start_t REAL, end_t REAL, guesses TEXT, best_det INTEGER,
    status TEXT NOT NULL DEFAULT 'pending', species TEXT, type TEXT, sex TEXT,
    behavior TEXT, comment TEXT, reviewed_at TEXT);
CREATE TABLE IF NOT EXISTS stages (name TEXT PRIMARY KEY, done_at TEXT);
CREATE INDEX IF NOT EXISTS det_frame ON detections(frame_id);
CREATE INDEX IF NOT EXISTS det_track ON detections(tracklet_id);
CREATE INDEX IF NOT EXISTS track_fish ON tracklets(fish_id);
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


class Project:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "frames").mkdir(exist_ok=True)
        self.db = sqlite3.connect(self.root / "project.db", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    # --- settings & meta -------------------------------------------------------------

    @property
    def settings_path(self) -> Path:
        return self.root / "settings.json"

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

    def add_frame(self, seq, t, chapter, local_t, path, width, height) -> int:
        cur = self.db.execute(
            "INSERT INTO frames(seq, t, chapter, local_t, path, width, height) VALUES(?,?,?,?,?,?,?)",
            (seq, t, chapter, local_t, path, width, height),
        )
        return cur.lastrowid

    def add_detections(self, frame_id: int, dets: list[Detection], qualities: list[float]) -> None:
        self.db.executemany(
            "INSERT INTO detections(frame_id, x1, y1, x2, y2, score, quality, app) VALUES(?,?,?,?,?,?,?,?)",
            [(frame_id, d.x1, d.y1, d.x2, d.y2, d.score, q, to_blob(d.app)) for d, q in zip(dets, qualities)],
        )

    def commit(self) -> None:
        self.db.commit()

    def last_frame(self) -> FrameRow | None:
        row = self.db.execute("SELECT * FROM frames ORDER BY seq DESC LIMIT 1").fetchone()
        return FrameRow(**dict(row)) if row else None

    def frame(self, frame_id: int) -> FrameRow | None:
        row = self.db.execute("SELECT * FROM frames WHERE id=?", (frame_id,)).fetchone()
        return FrameRow(**dict(row)) if row else None

    def frame_count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM frames").fetchone()[0]

    def frames_with_detections(self):
        """Yield (FrameRow, [(det id, Detection)]) in time order."""
        frames = self.db.execute("SELECT * FROM frames ORDER BY seq").fetchall()
        dets: dict[int, list] = {}
        for d in self.db.execute("SELECT * FROM detections ORDER BY id"):
            dets.setdefault(d["frame_id"], []).append(
                (d["id"], Detection(d["x1"], d["y1"], d["x2"], d["y2"], d["score"], from_blob(d["app"])))
            )
        for f in frames:
            yield FrameRow(**dict(f)), dets.get(f["id"], [])

    # --- tracklets & fish ------------------------------------------------------------------

    def reset_tracking(self) -> None:
        # Crop embeddings stay: they don't depend on tracking, so a rerun can reuse them.
        self.db.execute("UPDATE detections SET tracklet_id=NULL")
        self.db.execute("DELETE FROM tracklets")
        self.db.execute("DELETE FROM fish")
        self.db.commit()

    def clear_embeddings(self) -> None:
        self.db.execute("UPDATE detections SET embedding=NULL")
        self.db.commit()

    def save_tracklets(self, tracklets) -> None:
        for tr in tracklets:
            self.db.execute("INSERT INTO tracklets(id) VALUES(?)", (tr.id,))
            self.db.executemany(
                "UPDATE detections SET tracklet_id=? WHERE id=?",
                [(tr.id, key) for _, key in tr.members],
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
        cur = self.db.execute("INSERT INTO fish(status) VALUES('pending')")
        fish_id = cur.lastrowid
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

    def set_fish_guesses(self, fish_id: int, guesses: list, best_det: int | None) -> None:
        self.db.execute(
            "UPDATE fish SET guesses=?, best_det=? WHERE id=?",
            (json.dumps(guesses), best_det, fish_id),
        )

    def fish(self, fish_id: int) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM fish WHERE id=?", (fish_id,)).fetchone()

    def all_fish(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM fish ORDER BY start_t, id").fetchall()

    def label_fish(self, fish_id: int, status: str, **fields) -> None:
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
        self.db.commit()

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
        moved: list[int] = []
        for tid in self.fish_tracklets(fish_id):
            rows = self.tracklet_detections(tid)
            later = [r["id"] for r in rows if r["t"] >= from_t]
            if not later:
                continue
            if len(later) == len(rows):
                moved.append(tid)
                continue
            new_tid = self.db.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM tracklets").fetchone()[0]
            self.db.execute("INSERT INTO tracklets(id) VALUES(?)", (new_tid,))
            self.db.executemany(
                "UPDATE detections SET tracklet_id=? WHERE id=?", [(new_tid, i) for i in later]
            )
            self._refresh_tracklet(tid)
            self._refresh_tracklet(new_tid)
            moved.append(new_tid)
        new_id = self.create_fish(moved)
        self._refresh_span(fish_id)
        self.db.commit()
        return new_id

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
        self.db.commit()
