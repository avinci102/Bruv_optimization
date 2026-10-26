"""The hosted review site's other half, on the coordinator's PC (docs/SPEC.md §16):
volunteers and their passes, and a local copy of the site for trying it out.

- `volunteers_sql`: the volunteer list (a CSV) as SQL for the site's D1 database, with a
  short access pass made for each volunteer who has none.
- `create_dev_app`: the site served from this PC, with the same /sync API as the Cloudflare
  function (hosted/functions/sync) on a local SQLite file, so a school can try the whole
  flow before deploying, and the tests can run it.
"""

# No `from __future__ import annotations` here: FastAPI reads the endpoint's `Request`
# annotation at runtime, and that name only exists inside create_dev_app.
import base64
import csv
import hashlib
import hmac
import json
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

HOSTED_DIR = Path(__file__).parent / "hosted"
PASS_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O or 1/I/L: read aloud or typed easily
DAYS = 30


def new_pass() -> str:
    pick = lambda n: "".join(secrets.choice(PASS_LETTERS) for _ in range(n))  # noqa: E731
    return f"BRUV-{pick(4)}-{pick(4)}"


@dataclass
class Volunteer:
    name: str
    email: str | None
    passcode: str
    role: str  # "volunteer" or "admin"
    own: list[str]  # videos that are theirs to review (the review of record)
    calibration: dict[str, list[int] | None]  # shared checks: video -> card ids (None: all)


def _sql(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def read_volunteers(path: Path) -> list[Volunteer]:
    """A CSV with the columns name, email, pass, role, videos, calibration. videos: the
    volunteer's own videos ("A1 A2"); calibration: shared checks, a video alone (all its
    cards) or with card ids ("A3=12;15;99 A4", or with commas in a quoted cell). An empty
    pass gets a new one."""
    out = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            if not row.get("name"):
                continue
            cal: dict[str, list[int] | None] = {}
            for item in row.get("calibration", "").split():
                video, _, cards = item.partition("=")
                cal[video] = [int(c) for c in cards.replace(";", ",").split(",") if c.strip()] if cards else None
            out.append(Volunteer(
                row["name"], row.get("email") or None, (row.get("pass") or new_pass()).upper(),
                row.get("role") or "volunteer", row.get("videos", "").split(), cal,
            ))
    return out


def volunteers_sql(volunteers: list[Volunteer]) -> str:
    """SQL that sets the site's volunteers and assignments to exactly this list. A volunteer
    is known by their pass, so their saved answers stay theirs whatever the row order; one
    left off the list can no longer sign in, and their answers stay in the database."""
    lines = ["DELETE FROM assignments;", "UPDATE volunteers SET active = 0;"]
    for v in volunteers:
        lines.append(
            "INSERT INTO volunteers (name, email, pass, role, active) VALUES "
            f"({_sql(v.name)}, {_sql(v.email)}, {_sql(v.passcode)}, {_sql(v.role)}, 1) "
            "ON CONFLICT(pass) DO UPDATE SET name=excluded.name, email=excluded.email, role=excluded.role, active=1;"
        )
        who = f"(SELECT id FROM volunteers WHERE pass={_sql(v.passcode)})"
        for video in v.own:
            lines.append(f"INSERT INTO assignments (volunteer_id, video, role, cards) VALUES ({who}, {_sql(video)}, 'own', NULL);")
        for video, cards in v.calibration.items():
            lines.append(
                f"INSERT INTO assignments (volunteer_id, video, role, cards) VALUES ({who}, {_sql(video)}, 'calibration', "
                f"{_sql(json.dumps(cards) if cards else None)});"
            )
    return "\n".join(lines) + "\n"


def write_volunteers(volunteers: list[Volunteer], path: Path) -> None:
    """The list back as CSV, with the passes made: keep it to send the invitations."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "email", "pass", "role", "videos", "calibration"])
        for v in volunteers:
            cal = " ".join(f"{k}={','.join(map(str, c))}" if c else k for k, c in v.calibration.items())
            w.writerow([v.name, v.email or "", v.passcode, v.role, " ".join(v.own), cal])


# --- the site, served from this PC ------------------------------------------------------


def _sign(secret: str, text: str) -> str:
    mac = hmac.new(secret.encode(), text.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac).decode().rstrip("=")


def create_dev_app(site: Path, db_path: Path, secret: str = "dev-secret"):
    """The hosted site from this PC: public/ as static files plus the /sync API (the same as
    hosted/functions/sync/[[path]].js), on SQLite at `db_path`."""
    from fastapi import FastAPI, Request  # noqa: PLC0415
    from fastapi.responses import JSONResponse  # noqa: PLC0415
    from fastapi.staticfiles import StaticFiles  # noqa: PLC0415

    db = sqlite3.connect(db_path, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.executescript((HOSTED_DIR / "schema.sql").read_text(encoding="utf-8"))
    lock = threading.Lock()
    app = FastAPI(title="BRUV review site (local)")

    def reply(status: int, data, cookie: str | None = None):
        res = JSONResponse(data, status_code=status, headers={"Cache-Control": "no-store"})
        if cookie is not None:
            res.headers["Set-Cookie"] = cookie
        return res

    def cookie_for(vid: int) -> str:
        exp = int(time.time()) + DAYS * 86400
        value = f"{vid}.{exp}.{_sign(secret, f'{vid}.{exp}')}"
        return f"bruv={value}; Path=/; HttpOnly; SameSite=Lax; Max-Age={DAYS * 86400}"

    def session(request: Request):
        raw = request.cookies.get("bruv")
        if not raw:
            return None
        parts = raw.split(".")
        if len(parts) != 3 or not parts[1].isdigit() or int(parts[1]) < time.time():
            return None
        if not hmac.compare_digest(_sign(secret, f"{parts[0]}.{parts[1]}"), parts[2]):
            return None
        return db.execute("SELECT id, name, email, role FROM volunteers WHERE id=? AND active=1", (int(parts[0]),)).fetchone()

    def me(v) -> dict:
        rows = db.execute("SELECT video, role, cards FROM assignments WHERE volunteer_id=? ORDER BY video", (v["id"],))
        return {"id": v["id"], "name": v["name"], "role": v["role"],
                "videos": [{"video": r["video"], "role": r["role"], "cards": json.loads(r["cards"]) if r["cards"] else None} for r in rows]}

    def assigned(v, video) -> bool:
        if v["role"] == "admin":
            return True
        return db.execute("SELECT 1 FROM assignments WHERE volunteer_id=? AND video=?", (v["id"], video)).fetchone() is not None

    @app.api_route("/sync/{path:path}", methods=["GET", "POST"])
    async def sync(path: str, request: Request):
        body = None
        if request.method == "POST":
            try:
                body = await request.json()
            except ValueError:  # no body, or not JSON (as the function: request.json().catch)
                body = {}
        with lock:
            path = "/" + path
            if request.method == "POST" and path == "/login":
                who = str((body or {}).get("who") or "").strip()
                if not who:
                    return reply(400, {"detail": "Type your access pass or email address."})
                if "@" in who:
                    v = db.execute("SELECT id, name, email, role FROM volunteers WHERE lower(email)=lower(?) AND active=1", (who,)).fetchone()
                else:
                    v = db.execute("SELECT id, name, email, role FROM volunteers WHERE upper(pass)=? AND active=1",
                                   ("".join(who.upper().split()),)).fetchone()
                if not v:
                    return reply(401, {"detail": "That pass or email isn't on the volunteer list. Check your invitation, or ask the coordinator."})
                return reply(200, me(v), cookie_for(v["id"]))
            if request.method == "POST" and path == "/logout":
                return reply(200, {"ok": True}, "bruv=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0")
            v = session(request)
            if not v:
                return reply(401, {"detail": "Not signed in."})
            if request.method == "GET" and path == "/me":
                return reply(200, me(v))
            if path == "/events":
                video = request.query_params.get("video") if request.method == "GET" else (body or {}).get("video")
                if not video or not assigned(v, video):
                    return reply(403, {"detail": "This video isn't assigned to you."})
                if request.method == "GET":
                    rows = db.execute("SELECT body FROM events WHERE volunteer_id=? AND video=? ORDER BY id", (v["id"], video))
                    return reply(200, {"events": [json.loads(r["body"]) for r in rows]})
                events = body.get("events") or []
                if not events:
                    return reply(400, {"detail": "Nothing to save."})
                at = datetime.now(timezone.utc).isoformat()
                db.executemany(
                    "INSERT INTO events (volunteer_id, video, at, kind, body, app) VALUES (?, ?, ?, ?, ?, ?)",
                    [(v["id"], video, at, str(e.get("kind", "?")), json.dumps({**e, "at": at}), body.get("app")) for e in events],
                )
                db.commit()
                return reply(200, {"ok": True, "saved": len(events)})
            if request.method == "POST" and path == "/feedback":
                db.execute("INSERT INTO feedback (volunteer_id, video, at, rating, text) VALUES (?, ?, ?, ?, ?)",
                           (v["id"], body.get("video"), datetime.now(timezone.utc).isoformat(), body.get("rating"),
                            str(body.get("text") or "")[:5000]))
                db.commit()
                return reply(200, {"ok": True})
            if path.startswith("/admin/"):
                if v["role"] != "admin":
                    return reply(403, {"detail": "Coordinators only."})
                if path == "/admin/progress":
                    rows = db.execute(
                        """SELECT v.name, a.video, a.role,
                                  SUM(CASE WHEN e.kind='label' THEN 1 ELSE 0 END) AS answers,
                                  SUM(CASE WHEN e.kind IN ('split','move','unmerge') THEN 1 ELSE 0 END) AS fixes,
                                  MAX(e.at) AS last
                             FROM assignments a JOIN volunteers v ON v.id=a.volunteer_id
                             LEFT JOIN events e ON e.volunteer_id=a.volunteer_id AND e.video=a.video
                            GROUP BY a.volunteer_id, a.video ORDER BY a.video, v.name""").fetchall()
                    videos = [r[0] for r in db.execute("SELECT DISTINCT video FROM assignments ORDER BY video")]
                    return reply(200, {"rows": [dict(r) for r in rows], "videos": videos})
                if path == "/admin/export":
                    return reply(200, export(db, request.query_params.get("video")))
            return reply(404, {"detail": "Not Found"})

    @app.api_route("/api/{path:path}", methods=["GET", "POST"])
    def not_signed_in(path: str):
        return reply(409, {"detail": "Sign in first: open /start.html."})

    app.mount("/", StaticFiles(directory=site / "public", html=True), name="site")
    return app


def export(db: sqlite3.Connection, video: str) -> dict:
    """Every volunteer's events for one video: what the coordinator downloads and pulls."""
    db.row_factory = sqlite3.Row
    return {
        "format": 1, "video": video, "exported_at": datetime.now(timezone.utc).isoformat(),
        "volunteers": [dict(r) for r in db.execute("SELECT id, name, email, role FROM volunteers ORDER BY id")],
        "assignments": [
            {**dict(r), "cards": json.loads(r["cards"]) if r["cards"] else None}
            for r in db.execute("SELECT volunteer_id, video, role, cards FROM assignments WHERE video=?", (video,))
        ],
        "events": [
            {"id": r["id"], "volunteer_id": r["volunteer_id"], "at": r["at"], "app": r["app"], "event": json.loads(r["body"])}
            for r in db.execute("SELECT id, volunteer_id, at, body, app FROM events WHERE video=? ORDER BY id", (video,))
        ],
        "feedback": [dict(r) for r in db.execute(
            "SELECT volunteer_id, at, rating, text FROM feedback WHERE video=? OR video IS NULL ORDER BY id", (video,))],
    }


def load_volunteers_into(db_path: Path, sql: str) -> None:
    db = sqlite3.connect(db_path)
    db.executescript((HOSTED_DIR / "schema.sql").read_text(encoding="utf-8"))
    db.executescript(sql)
    db.commit()
    db.close()
