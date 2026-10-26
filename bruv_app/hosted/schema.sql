-- The hosted review site's database (Cloudflare D1). Volunteers, what each reviews, and an
-- append-only log of their answers and fixes (events), replayed by the review page and
-- pulled to the coordinator's PC with `bruv_app pull` (docs/SPEC.md §16).
CREATE TABLE IF NOT EXISTS volunteers (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT, pass TEXT NOT NULL UNIQUE,
    role TEXT NOT NULL DEFAULT 'volunteer', active INTEGER NOT NULL DEFAULT 1);
-- role: 'own' (the volunteer's own video, the review of record) or 'calibration' (a
-- shared check: cards, a JSON list of card ids, or NULL for the whole video).
CREATE TABLE IF NOT EXISTS assignments (
    volunteer_id INTEGER NOT NULL, video TEXT NOT NULL, role TEXT NOT NULL DEFAULT 'own', cards TEXT,
    PRIMARY KEY (volunteer_id, video));
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, volunteer_id INTEGER NOT NULL, video TEXT NOT NULL,
    at TEXT NOT NULL, kind TEXT NOT NULL, body TEXT NOT NULL, app TEXT);
CREATE INDEX IF NOT EXISTS events_by ON events (volunteer_id, video, id);
CREATE INDEX IF NOT EXISTS events_video ON events (video, id);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT, volunteer_id INTEGER NOT NULL, video TEXT, at TEXT NOT NULL,
    rating INTEGER, text TEXT);
