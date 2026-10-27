/* The hosted review site's only server code (a Cloudflare Pages Function, with D1):
 * sign-in, and storing each volunteer's answers and fixes as an event log.
 *
 *   POST /sync/login        {who}            access pass or email -> session cookie
 *   POST /sync/logout
 *   GET  /sync/me                            name, role, assigned videos
 *   GET  /sync/events?video=K                this volunteer's events for K, in order
 *   POST /sync/events       {video, events, app}
 *   POST /sync/feedback     {video, rating, text}
 *   GET  /sync/admin/progress                (coordinator) answers, fixes and follows per volunteer
 *   GET  /sync/admin/export?video=K          (coordinator) every volunteer's events for K
 *
 * Sign-in needs no password: the data isn't sensitive (docs/SPEC.md §16). A session is a
 * cookie signed with SESSION_SECRET (a Pages secret), valid for 30 days.
 */

const DAYS = 30;
const enc = new TextEncoder();

function json(status, data, headers = {}) {
  return new Response(JSON.stringify(data), { status, headers: { "Content-Type": "application/json", "Cache-Control": "no-store", ...headers } });
}

function b64url(bytes) {
  let s = "";
  for (const b of new Uint8Array(bytes)) s += String.fromCharCode(b);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

async function sign(secret, text) {
  const key = await crypto.subtle.importKey("raw", enc.encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  return b64url(await crypto.subtle.sign("HMAC", key, enc.encode(text)));
}

async function session(request, env) {
  const cookie = request.headers.get("Cookie") || "";
  const m = cookie.match(/(?:^|;\s*)bruv=([^;]+)/);
  if (!m) return null;
  const [id, exp, sig] = decodeURIComponent(m[1]).split(".");
  if (!id || !exp || !sig || Number(exp) < Date.now() / 1000) return null;
  if ((await sign(env.SESSION_SECRET, `${id}.${exp}`)) !== sig) return null;
  return env.DB.prepare("SELECT id, name, email, role FROM volunteers WHERE id = ? AND active = 1").bind(Number(id)).first();
}

async function cookieFor(env, id) {
  const exp = Math.floor(Date.now() / 1000) + DAYS * 86400;
  const value = `${id}.${exp}.${await sign(env.SESSION_SECRET, `${id}.${exp}`)}`;
  return `bruv=${encodeURIComponent(value)}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=${DAYS * 86400}`;
}

async function assignments(env, volunteerId) {
  const rows = await env.DB.prepare("SELECT video, role, cards FROM assignments WHERE volunteer_id = ? ORDER BY video")
    .bind(volunteerId).all();
  return rows.results.map((r) => ({ video: r.video, role: r.role, cards: r.cards ? JSON.parse(r.cards) : null }));
}

async function me(env, v) {
  return { id: v.id, name: v.name, role: v.role, videos: await assignments(env, v.id) };
}

async function assigned(env, v, video) {
  if (v.role === "admin") return true;
  const row = await env.DB.prepare("SELECT 1 AS ok FROM assignments WHERE volunteer_id = ? AND video = ?").bind(v.id, video).first();
  return Boolean(row);
}

export async function onRequest({ request, env, params }) {
  if (!env.DB || !env.SESSION_SECRET) return json(500, { detail: "The site isn't set up yet: it needs its D1 database and SESSION_SECRET." });
  const path = "/" + [].concat(params.path || []).join("/");
  const url = new URL(request.url);
  const method = request.method;
  const body = method === "POST" ? await request.json().catch(() => ({})) : null;

  if (method === "POST" && path === "/login") {
    const who = String((body && body.who) || "").trim();
    if (!who) return json(400, { detail: "Type your access pass or email address." });
    const v = who.includes("@")
      ? await env.DB.prepare("SELECT id, name, email, role FROM volunteers WHERE lower(email) = lower(?) AND active = 1").bind(who).first()
      : await env.DB.prepare("SELECT id, name, email, role FROM volunteers WHERE upper(pass) = ? AND active = 1")
        .bind(who.toUpperCase().replace(/\s+/g, "")).first();
    if (!v) return json(401, { detail: "That pass or email isn't on the volunteer list. Check your invitation, or ask the coordinator." });
    return json(200, await me(env, v), { "Set-Cookie": await cookieFor(env, v.id) });
  }
  if (method === "POST" && path === "/logout") {
    return json(200, { ok: true }, { "Set-Cookie": "bruv=; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=0" });
  }

  const v = await session(request, env);
  if (!v) return json(401, { detail: "Not signed in." });

  if (method === "GET" && path === "/me") return json(200, await me(env, v));

  if (path === "/events") {
    const video = method === "GET" ? url.searchParams.get("video") : body && body.video;
    if (!video || !(await assigned(env, v, video))) return json(403, { detail: "This video isn't assigned to you." });
    if (method === "GET") {
      const rows = await env.DB.prepare("SELECT body FROM events WHERE volunteer_id = ? AND video = ? ORDER BY id")
        .bind(v.id, video).all();
      return json(200, { events: rows.results.map((r) => JSON.parse(r.body)) });
    }
    const events = Array.isArray(body.events) ? body.events : [];
    if (!events.length) return json(400, { detail: "Nothing to save." });
    const at = new Date().toISOString();
    const insert = env.DB.prepare("INSERT INTO events (volunteer_id, video, at, kind, body, app) VALUES (?, ?, ?, ?, ?, ?)");
    await env.DB.batch(events.map((e) => insert.bind(v.id, video, at, String(e.kind || "?"), JSON.stringify({ ...e, at }), body.app || null)));
    return json(200, { ok: true, saved: events.length });
  }

  if (method === "POST" && path === "/feedback") {
    await env.DB.prepare("INSERT INTO feedback (volunteer_id, video, at, rating, text) VALUES (?, ?, ?, ?, ?)")
      .bind(v.id, body.video || null, new Date().toISOString(), body.rating || null, String(body.text || "").slice(0, 5000)).run();
    return json(200, { ok: true });
  }

  if (path.startsWith("/admin/")) {
    if (v.role !== "admin") return json(403, { detail: "Coordinators only." });
    if (method === "GET" && path === "/admin/progress") {
      const rows = await env.DB.prepare(
        `SELECT v.name, a.video, a.role,
                SUM(CASE WHEN e.kind = 'label' THEN 1 ELSE 0 END) AS answers,
                SUM(CASE WHEN e.kind IN ('split', 'move', 'unmerge') THEN 1 ELSE 0 END) AS fixes,
                SUM(CASE WHEN e.kind = 'follow_start' THEN 1 ELSE 0 END) AS follows,
                MAX(e.at) AS last
           FROM assignments a JOIN volunteers v ON v.id = a.volunteer_id
           LEFT JOIN events e ON e.volunteer_id = a.volunteer_id AND e.video = a.video
          GROUP BY a.volunteer_id, a.video ORDER BY a.video, v.name`).all();
      const videos = await env.DB.prepare("SELECT DISTINCT video FROM assignments ORDER BY video").all();
      return json(200, { rows: rows.results, videos: videos.results.map((r) => r.video) });
    }
    if (method === "GET" && path === "/admin/export") {
      const video = url.searchParams.get("video");
      const volunteers = await env.DB.prepare("SELECT id, name, email, role FROM volunteers ORDER BY id").all();
      const assigns = await env.DB.prepare("SELECT volunteer_id, video, role, cards FROM assignments WHERE video = ?").bind(video).all();
      const events = await env.DB.prepare("SELECT id, volunteer_id, at, kind, body, app FROM events WHERE video = ? ORDER BY id").bind(video).all();
      const feedback = await env.DB.prepare("SELECT volunteer_id, at, rating, text FROM feedback WHERE video = ? OR video IS NULL ORDER BY id").bind(video).all();
      return json(200, {
        format: 1, video, exported_at: new Date().toISOString(), volunteers: volunteers.results,
        assignments: assigns.results.map((a) => ({ ...a, cards: a.cards ? JSON.parse(a.cards) : null })),
        events: events.results.map((e) => ({ id: e.id, volunteer_id: e.volunteer_id, at: e.at, app: e.app, event: JSON.parse(e.body) })),
        feedback: feedback.results,
      }, { "Content-Disposition": `attachment; filename="answers-${video}.json"` });
    }
  }
  return json(404, { detail: "Not Found" });
}
