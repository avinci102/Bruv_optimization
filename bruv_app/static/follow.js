// Follow one fish frame by frame (follow.py): click it in each frame and the page moves on.
// Ground truth for how well the cards keep to one fish (docs/SPEC.md §15). Also the place to
// find a fish to follow anywhere in the video, to look back and ahead while following (the
// frames under the picture), and to say what's wrong with a box. Works with a mouse and keys
// or by touch: every key has a button, a press and hold is a right-click, a drag draws a box.

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const AROUND = 15; // frames fetched each side of the one looked at
const SHOWN = 10; // of those, the frames shown each side under the picture
const state = {
  card: Number(params.get("card")) || null, // the card the fish was picked from
  id: Number(params.get("id")) || null, // the follow, once started
  mode: null, // home (find a fish), intro (a card's fish, before starting), follow, done
  view: null, // the server's latest answer: the follow, its next frame and that frame's boxes
  total: 0, // frames in the video
  frames: new Map(), // place in the video -> {id, t, clip, local_t, boxes, mark, gen}
  places: new Map(), // frame id -> place
  gen: 0, // bumped whenever the follow's marks change: frames fetched before are out of date
  looking: null, // the place looked at
  stepPlace: null, // the place of the next frame to mark
  shown: null, // the frame on screen: {id, boxes}
  pick: null, // what a new follow starts from: {card, det, frame, box} or {frame, drawn: {x, y, w, h}}
  cardBest: null, // the card's best view (intro), started from with {card}
  suggested: null, // the box offered in this step
  zoom: false,
  fixing: false, // "Fix a box": the next box clicked is fixed, not marked
  busy: false, // a mark is on its way: ignore clicks until the next frame is shown
  loading: false,
  wanted: null, // a place to fetch once the fetch under way is back
  aspect: 16 / 9, // the frame's width over its height, once one has loaded
};

async function api(path, body, method) {
  const opts = body === undefined && !method ? {} : {
    method: method || "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  };
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

function fmtTime(seconds) {
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
  const mm = String(m).padStart(2, "0"), ss = String(r).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

function fmtExact(seconds) {
  // Frames are a quarter of a second apart: show the fraction.
  const whole = Math.floor(seconds);
  return `${fmtTime(whole)}.${String(Math.round((seconds - whole) * 100)).padStart(2, "0")}`;
}

function message(text, kind = "") {
  const el = $("message");
  el.textContent = text;
  el.className = kind;
}

function backToReview() {
  location.href = state.card ? `/#card=${state.card}` : "/";
}

// The title goes back to the page this one was opened from (the review page, or the
// website's start page), or to the review.
function setBrand() {
  try {
    const from = new URL(document.referrer);
    if (from.origin === location.origin && from.pathname !== "/follow") $("brand").href = from.pathname + from.search + from.hash;
  } catch (e) { /* opened directly */ }
}

const atStep = () => state.mode === "follow" && state.looking !== null && state.looking === state.stepPlace;
const show = (id, on) => { $(id).hidden = !on; };

// --- close-ups: a box's part of a frame, zoomed by the browser ---------------------------

function closeUp(frameId, b, caption, onclick) {
  const W = 120, H = 90, A = state.aspect;
  const fig = document.createElement("figure");
  fig.className = "closeup";
  const box = document.createElement("div");
  box.className = "closeup-pic";
  const img = document.createElement("img");
  img.alt = "Close-up";
  img.draggable = false;
  // The box with a margin, fitted to the close-up: the whole frame drawn that much bigger.
  const pad = 0.3, bw = b.x2 - b.x1, bh = b.y2 - b.y1;
  const x1 = Math.max(0, b.x1 - pad * bw), x2 = Math.min(1, b.x2 + pad * bw);
  const y1 = Math.max(0, b.y1 - pad * bh), y2 = Math.min(1, b.y2 + pad * bh);
  const S = Math.min(W / Math.max(x2 - x1, 0.01), (H * A) / Math.max(y2 - y1, 0.01));
  Object.assign(img.style, {
    width: `${S}px`, height: `${S / A}px`,
    left: `${(W - (x2 - x1) * S) / 2 - x1 * S}px`, top: `${(H - ((y2 - y1) * S) / A) / 2 - (y1 * S) / A}px`,
  });
  img.src = `/frame/${frameId}`;
  box.append(img);
  fig.append(box);
  if (caption) {
    const c = document.createElement("figcaption");
    c.textContent = caption;
    fig.append(c);
  }
  if (onclick) {
    fig.classList.add("clickable");
    fig.onclick = onclick;
  }
  return fig;
}

// --- drawing a frame ----------------------------------------------------------------------

function place(el, b) {
  Object.assign(el.style, {
    left: `${b.x1 * 100}%`, top: `${b.y1 * 100}%`,
    width: `${(b.x2 - b.x1) * 100}%`, height: `${(b.y2 - b.y1) * 100}%`,
  });
}

const FIX_TEXT = { part: "part of a fish", not_fish: "not a fish", twice: "second box" };

function drawFrame(frameId, boxes, opts = {}) {
  state.shown = { id: frameId, boxes };
  $("follow-frame").src = `/frame/${frameId}`;
  const layer = $("boxes");
  layer.replaceChildren();
  // Biggest first, so a small box on top of a big one shows.
  const area = (b) => (b.x2 - b.x1) * (b.y2 - b.y1);
  for (const b of [...boxes].sort((a, c) => area(c) - area(a))) {
    const el = document.createElement("div");
    el.className = "fbox" + (b.faint ? " faint" : "") + (b.taken ? " taken" : "") + (b.fix ? " fixed" : "")
      + (opts.focus === b.det ? " focus" : "") + (opts.suggest === b.det ? " suggest" : "");
    el.title = (b.faint
      ? `A faint box (score ${b.score}), or one hidden as still: on no card`
      : `Score ${b.score}, on card #${b.card}` + (b.taken ? `; taken by follow ${b.taken}` : ""))
      + (b.fix ? `; you said: ${FIX_TEXT[b.fix]}` : "");
    place(el, b);
    const tag = b.fix ? FIX_TEXT[b.fix] : b.taken;
    if (tag) {
      const span = document.createElement("span");
      span.textContent = tag;
      el.append(span);
    }
    layer.append(el);
  }
  if (opts.drawn) {
    const el = document.createElement("div");
    el.className = "fbox drawn";
    place(el, opts.drawn);
    layer.append(el);
  }
  const ring = $("ring");
  ring.hidden = !opts.ring;
  if (opts.ring) Object.assign(ring.style, { left: `${opts.ring[0] * 100}%`, top: `${opts.ring[1] * 100}%` });
  applyZoom(opts.ring || (opts.drawn ? [(opts.drawn.x1 + opts.drawn.x2) / 2, (opts.drawn.y1 + opts.drawn.y2) / 2] : null));
}

function applyZoom(at) {
  const z = $("zoomer");
  $("t-zoom").setAttribute("aria-pressed", String(state.zoom));
  if (state.zoom && at) {
    z.style.transformOrigin = `${at[0] * 100}% ${at[1] * 100}%`;
    z.style.transform = "scale(2)";
  } else {
    z.style.transform = "";
  }
}

const corners = (d) => ({ x1: d.x - d.w / 2, y1: d.y - d.h / 2, x2: d.x + d.w / 2, y2: d.y + d.h / 2 });

// The box this step offers: the one nearest where the fish was last seen, if it's close.
function suggestion(view) {
  if (!view.last) return null;
  let best = null, dist = Infinity;
  for (const b of view.boxes) {
    if (b.fix || b.taken) continue;
    const d = Math.hypot(((b.x1 + b.x2) / 2 - view.last[0]) * state.aspect, (b.y1 + b.y2) / 2 - view.last[1]);
    if (d < dist) [best, dist] = [b, d];
  }
  return dist <= 0.25 ? best : null; // within a quarter of the frame's height
}

// The frame looked at: the step to mark, or any other (looking back or ahead, or finding a
// fish to follow), with the follow's mark in it.
function renderLooking() {
  const f = state.frames.get(state.looking);
  const v = state.view;
  show("looking", state.mode === "follow" && v && v.frame && state.looking !== null && !atStep());
  if (atStep() || (state.mode === "follow" && state.looking === null && v && v.frame)) {
    state.suggested = suggestion(v);
    drawFrame(v.frame.id, v.boxes, { ring: v.last, suggest: state.suggested && state.suggested.det });
    const fr = v.frame;
    $("where").textContent = `${fr.clip} ${fmtExact(fr.local_t)}` + (v.phase === "backward" ? " · going back in time" : "") +
      ` · ${v.boxes.length} box${v.boxes.length === 1 ? "" : "es"} in this frame`;
    renderSuggestion();
  } else if (f) {
    state.suggested = null;
    const m = f.mark;
    const opts = {};
    if (m && m.state === "box") opts.focus = m.det;
    if (m && m.state === "nobox" && m.w !== null) opts.drawn = corners(m);
    else if (m && m.state === "nobox" && m.x !== null) opts.ring = [m.x, m.y];
    if (state.pick && state.pick.frame === f.id) {
      if (state.pick.det) opts.focus = state.pick.det;
      if (state.pick.drawn) opts.drawn = corners(state.pick.drawn);
    }
    drawFrame(f.id, f.boxes, opts);
    const said = !m ? "" : m.state === "gone" ? " · you said it wasn't there" : m.state === "nobox" ? " · there, without a box"
      : " · its box";
    $("where").textContent = `${f.clip} ${fmtExact(f.local_t)} · ${f.boxes.length} box${f.boxes.length === 1 ? "" : "es"}${said}`;
    if (state.mode === "follow" && state.stepPlace !== null) {
      const n = state.looking - state.stepPlace;
      $("looking-text").textContent = `Looking ${Math.abs(n)} frame${Math.abs(n) === 1 ? "" : "s"} ${n < 0 ? "back" : "ahead"} ` +
        `(${f.clip} ${fmtExact(f.local_t)}): nothing changes here.`;
    }
    renderSuggestion();
  }
  $("t-accept").disabled = !(atStep() && state.suggested);
  renderCarousel();
}

function renderSuggestion() {
  const b = atStep() ? state.suggested : null;
  show("suggest", Boolean(b));
  if (b) $("suggest-closeup").replaceChildren(closeUp(state.view.frame.id, b, b.faint ? "a faint box" : `card #${b.card}`));
}

// --- the frames under the picture ---------------------------------------------------------

function renderCarousel() {
  const strip = $("carousel");
  strip.replaceChildren();
  if (state.looking === null) return;
  const scrub = $("scrub");
  scrub.hidden = state.total < 2;
  scrub.max = String(Math.max(0, state.total - 1));
  scrub.value = String(state.looking);
  const startId = state.view ? state.view.start.id : null;
  let selected = null;
  for (let p = Math.max(0, state.looking - SHOWN); p <= Math.min(state.total - 1, state.looking + SHOWN); p++) {
    const f = state.frames.get(p);
    const fig = document.createElement("figure");
    fig.onclick = () => look(p);
    if (f) {
      const img = document.createElement("img");
      img.alt = "";
      img.draggable = false;
      img.src = `/frame/${f.id}`;
      const cap = document.createElement("figcaption");
      cap.textContent = fmtExact(f.local_t);
      fig.append(img, cap);
      if (f.mark) fig.classList.add(`m-${f.mark.state}`);
      if (f.id === startId) fig.classList.add("is-start");
      fig.title = `${f.clip} ${fmtExact(f.local_t)}` + (f.mark ? ` · ${f.mark.state === "gone" ? "not there" : f.mark.state === "box" ? "its box" : "there, no box"}` : "");
    } else {
      fig.classList.add("pending");
    }
    if (p === state.stepPlace && state.mode === "follow") fig.classList.add("is-step");
    if (p === state.looking) {
      fig.classList.add("selected");
      selected = fig;
    }
    strip.append(fig);
  }
  if (selected) strip.scrollLeft = selected.offsetLeft - strip.clientWidth / 2 + selected.offsetWidth / 2;
}

// The frames around a place (or a frame), with their boxes and the follow's marks.
async function fetchAround(where) {
  const q = new URLSearchParams({ n: String(AROUND), ...where });
  if (state.id) q.set("follow", String(state.id));
  const gen = state.gen;
  const data = await api(`/api/frames?${q}`);
  state.total = data.total;
  data.frames.forEach((f, k) => {
    state.frames.set(data.first + k, { ...f, gen });
    state.places.set(f.id, data.first + k);
  });
  return data.index;
}

// Fetches the frames around a place unless they're here and up to date.
async function ensure(p) {
  const lo = Math.max(0, p - SHOWN), hi = Math.min(state.total - 1, p + SHOWN);
  let fresh = true;
  for (let k = lo; k <= hi; k++) {
    const f = state.frames.get(k);
    if (!f || f.gen !== state.gen) fresh = false;
  }
  if (fresh) return;
  if (state.loading) {
    state.wanted = p;
    return;
  }
  state.loading = true;
  try {
    await fetchAround({ i: String(p) });
  } catch (e) {
    message(e.message, "error");
  } finally {
    state.loading = false;
  }
  if (state.looking !== null) renderLooking();
  const next = state.wanted;
  state.wanted = null;
  if (next !== null && next !== p) ensure(next);
}

function look(p) {
  if (state.total === 0 || p === null || p === undefined) return;
  state.looking = Math.max(0, Math.min(state.total - 1, p));
  renderLooking();
  ensure(state.looking);
}

async function lookAt(frameId) {
  const p = state.places.get(frameId);
  if (p !== undefined) return look(p);
  try {
    look(await fetchAround({ at: String(frameId) }));
  } catch (e) {
    message(e.message, "error");
  }
}

// --- following ----------------------------------------------------------------------------

function render(view) {
  state.view = view;
  state.id = view.follow.id;
  state.card = view.follow.card;
  state.gen++;
  if (params.get("id") !== String(state.id)) history.replaceState(null, "", `/follow?id=${state.id}`);
  $("back").href = state.card ? `/#card=${state.card}` : "/";
  show("intro", false);
  show("home", false);
  show("gone-wrap", false);
  const f = view.follow;
  $("who").textContent = f.name + (f.card ? ` · from card #${f.card}` : "");
  $("counts").textContent = `${f.boxes} with a box · ${f.nobox} without · ${f.gone} not there`;
  const phase = $("phase");
  phase.hidden = false;
  renderRecent(view);
  if (view.phase === "done") {
    state.mode = "done";
    phase.textContent = "Done";
    phase.className = "phase done";
    show("keys", false);
    show("tools", false);
    renderDone(view);
    state.stepPlace = null;
    lookAt(view.start.id);
    state.busy = false;
    return;
  }
  state.mode = "follow";
  show("done", false);
  show("keys", true);
  show("tools", true);
  const from = `${view.start.clip} ${fmtTime(view.start.local_t)}`;
  phase.textContent = view.phase === "forward" ? `Forward from ${from}` : `Back from ${from}`;
  phase.className = `phase ${view.phase}`;
  // The step is drawn now; the frames around it (and its place) follow.
  const known = state.places.get(view.frame.id);
  state.stepPlace = known === undefined ? null : known;
  state.looking = state.stepPlace;
  renderLooking();
  fetchAround({ at: String(view.frame.id) }).then((p) => {
    if (state.view !== view) return;
    state.stepPlace = p;
    if (state.looking === null || state.looking === p) state.looking = p;
    renderLooking();
  }).catch((e) => message(e.message, "error"));
  state.busy = false;
}

function renderRecent(view) {
  const el = $("recent");
  el.replaceChildren(...view.recent.map((r) => closeUp(r.id, r, fmtExact(r.local_t), () => lookAt(r.id))));
  if (!view.recent.length) el.textContent = "—";
}

function renderDone(view) {
  const f = view.follow;
  show("done", true);
  const el = $("done-text");
  el.replaceChildren();
  const p = (html) => { const x = document.createElement("p"); x.innerHTML = html; el.append(x); return x; };
  if (view.seen) {
    const { from, to } = view.seen;
    p(`Seen from ${from.clip} ${fmtExact(from.local_t)} to ${to.clip === from.clip ? "" : `${to.clip} `}${fmtExact(to.local_t)}: ` +
      `its box in ${f.boxes} frame${f.boxes === 1 ? "" : "s"}, there without a box in ${f.nobox}.`);
  }
  const cards = Object.entries(f.cards).sort((a, b) => b[1] - a[1]);
  if (cards.length) {
    const x = p(`Its boxes are on ${cards.length} card${cards.length === 1 ? "" : "s"}: `);
    cards.forEach(([card, n], i) => {
      const a = document.createElement("a");
      a.href = `/#card=${card}`;
      a.textContent = `#${card}`;
      x.append(i ? ", " : "", a, ` (${n} frame${n === 1 ? "" : "s"})`);
    });
    x.append(cards.length > 1 ? ". One fish, several cards: look-alikes or P can bring them together." : ".");
  }
  if (f.faint) p(`${f.faint} of its boxes are faint or were hidden as still, so they're on no card.`);
  if (view.breaks.length) {
    p("Its card changes at: " + view.breaks.map((b) => `${b.clip} ${fmtExact(b.local_t)} (#${b.from} → #${b.to})`).join(", ") + ".");
  }
  const others = Object.entries(f.others);
  if (others.length) {
    p("⚠ " + others.map(([card, n]) => `card #${card} has a box that isn't this fish in ${n} frame${n === 1 ? "" : "s"} you checked`)
      .join("; ") + ": it holds another fish (check it, and split it with X) or a second box on this one.").className = "warn";
  }
  const shared = Object.entries(f.shared);
  if (shared.length) {
    p("⚠ " + shared.map(([card, names]) => `card #${card} also holds ${names.join(", ")}`).join("; ") +
      ": that card mixes fish, or two follows took the same fish.").className = "warn";
  }
  p("Look back through it with ↑ ↓ or the frames under the picture.").className = "hint";
}

async function send(promise) {
  state.busy = true;
  try {
    render(await promise);
    message("");
  } catch (e) {
    state.busy = false;
    message(e.message, "error");
  }
}

function canMark() {
  return !state.busy && state.view && state.view.frame && atStep();
}

function mark(body) {
  if (!canMark()) return;
  send(api(`/api/follows/${state.id}/mark`, { frame_id: state.view.frame.id, ...body }));
}

const markBox = (b, via) => mark({ state: "box", det_id: b.det, via: via || null });
const markSpot = (p) => mark({ state: "nobox", x: p[0], y: p[1] });
const markDrawn = (d) => mark({ state: "nobox", ...d });
const markNoBox = () => mark({ state: "nobox" });
const markGone = () => mark({ state: "gone" });
const markRest = () => mark({ state: "gone", rest: true });

function accept() {
  if (state.suggested && canMark()) markBox(state.suggested, "suggested");
}

function undo() {
  if (state.busy || !state.id) return;
  send(api(`/api/follows/${state.id}/undo`, {}));
}

async function discard() {
  if (!state.id || !confirm("Discard this follow and all its clicks?")) return;
  try {
    await api(`/api/follows/${state.id}`, undefined, "DELETE");
    backToReview();
  } catch (e) {
    message(e.message, "error");
  }
}

// --- finding a fish to follow -------------------------------------------------------------

function goneSeconds() {
  const gone = Number($("gone").value) || 5;
  try { localStorage.setItem("follow-gone", String(gone)); } catch (e) { /* not kept */ }
  return gone;
}

function begin() {
  const pick = state.pick;
  if (state.id || state.busy || !pick) return;
  const body = { gone_s: goneSeconds() };
  if (pick.card && state.cardBest && pick.det === state.cardBest.det_id) body.card = pick.card;
  else if (pick.det) body.det_id = pick.det;
  else Object.assign(body, { frame_id: pick.frame, ...pick.drawn });
  send(api("/api/follows", body));
}

function setPick(pick) {
  state.pick = pick;
  if (state.mode === "intro" && !(pick.card && state.cardBest && pick.det === state.cardBest.det_id)) {
    // Another fish than the card's: the intro's picture no longer fits.
    state.mode = "home";
    show("intro", false);
    show("home", true);
    listFollows();
  }
  if (state.mode === "home") {
    show("pick", true);
    $("pick-start").before($("gone-wrap"));
    show("gone-wrap", true);
    const f = state.frames.get(state.looking);
    const box = pick.drawn ? corners(pick.drawn) : pick.box;
    $("pick-closeup").replaceChildren(closeUp(pick.frame, box, f ? `${f.clip} ${fmtExact(f.local_t)}` : ""));
    $("pick-text").textContent = pick.drawn ? "A box you drew: no box was found on this fish here."
      : pick.box.faint ? "A faint box: on no card." : `On card #${pick.box.card}.`;
  }
  renderLooking();
}

// The follows made so far, latest first, to look at again or carry on with.
async function listFollows() {
  try {
    const list = await api("/api/follows");
    const el = $("follow-list");
    el.replaceChildren();
    if (!list.length) el.textContent = "None yet.";
    for (const f of [...list].reverse()) {
      const a = document.createElement("a");
      a.href = `/follow?id=${f.id}`;
      a.textContent = `${f.name}` + (f.card ? ` · card #${f.card}` : "") +
        ` · ${f.boxes} frame${f.boxes === 1 ? "" : "s"} with its box` + (f.done ? " · done" : " · not finished");
      el.append(a);
    }
  } catch (e) {
    message(e.message, "error");
  }
}

async function showHome() {
  state.mode = "home";
  show("home", true);
  listFollows();
  const at = Number(params.get("at")) || null;
  try {
    look(at ? await fetchAround({ at: String(at) }) : await fetchAround({ i: String(Number(params.get("i")) || 0) }));
  } catch (e) {
    message(e.message, "error");
  }
}

async function showIntro() {
  state.mode = "intro";
  const d = await api(`/api/fish/${state.card}`);
  $("back").href = `/#card=${state.card}`;
  $("who").textContent = `card #${state.card}`;
  show("intro", true);
  if (!d.best) {
    message("This card has no sightings to follow.", "error");
    return;
  }
  state.cardBest = d.best;
  $("begin").before($("gone-wrap"));
  show("gone-wrap", true);
  $("start-crop").src = `/crop/${d.best.det_id}`;
  $("start-caption").textContent = `The fish to follow, at ${d.best.clip} ${fmtExact(d.best.local_t)}`;
  state.pick = { card: state.card, det: d.best.det_id, frame: d.best.frame_id, box: { ...d.best.box, card: state.card } };
  look(await fetchAround({ at: String(d.best.frame_id) }));
}

// --- what's wrong with a box --------------------------------------------------------------

function closeMenu() {
  $("menu").hidden = true;
  $("menu").replaceChildren();
}

function openMenu(x, y, items) {
  const menu = $("menu");
  menu.replaceChildren();
  for (const item of items) {
    if (item.node) {
      menu.append(item.node);
      continue;
    }
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = item.label;
    if (item.primary) b.className = "primary";
    b.onclick = () => { closeMenu(); if (item.run) item.run(); };
    menu.append(b);
  }
  menu.hidden = false;
  const r = menu.getBoundingClientRect();
  menu.style.left = `${Math.max(8, Math.min(x, innerWidth - r.width - 8))}px`;
  menu.style.top = `${Math.max(8, Math.min(y, innerHeight - r.height - 8))}px`;
}

function openFix(b, x, y) {
  const fix = (kind) => async () => {
    try {
      await api(`/api/boxes/${b.det}/fix`, { kind, follow: state.id });
      // The box as it is now, wherever it's drawn from.
      const same = (list) => (list || []).filter((c) => c.det === b.det).forEach((c) => { c.fix = kind; });
      if (state.view) same(state.view.boxes);
      for (const f of state.frames.values()) same(f.boxes);
      setFixing(false);
      renderLooking();
      message(kind ? `Noted: ${FIX_TEXT[kind]}. Thanks, this helps the fish finder.` : "Noted: that box is right.", "ok");
    } catch (e) {
      message(e.message, "error");
    }
  };
  const items = [
    { node: closeUp(state.shown.id, b, "What's wrong with this box?") },
    { label: "Part of a fish (a tail, a fin…)", run: fix("part") },
    { label: "Not a fish", run: fix("not_fish") },
    { label: "A second box on the same fish", run: fix("twice") },
  ];
  if (b.fix) items.push({ label: "The box is right after all", run: fix(null) });
  items.push({ label: "Cancel", run: () => setFixing(false) });
  openMenu(x, y, items);
}

function setFixing(on) {
  state.fixing = on;
  $("t-fix").setAttribute("aria-pressed", String(on));
  $("stage").classList.toggle("fixing", on);
  if (on) message("Click (tap) the box that's wrong.");
}

// --- clicks, taps, drags and presses on the picture ---------------------------------------

function norm(e) {
  const r = $("zoomer").getBoundingClientRect();
  return [Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)), Math.max(0, Math.min(1, (e.clientY - r.top) / r.height))];
}

// The boxes under a point, smallest first (a little margin makes small ones easier to hit).
function hits(p) {
  if (!state.shown) return [];
  const r = $("zoomer").getBoundingClientRect();
  const mx = 6 / Math.max(r.width, 1), my = 6 / Math.max(r.height, 1);
  const area = (b) => (b.x2 - b.x1) * (b.y2 - b.y1);
  return state.shown.boxes.filter((b) => p[0] >= b.x1 - mx && p[0] <= b.x2 + mx && p[1] >= b.y1 - my && p[1] <= b.y2 + my)
    .sort((a, b) => area(a) - area(b));
}

// One of the boxes under a finger: overlapping boxes are hard to tap, so they're offered
// as close-ups to choose from. With a mouse, the smallest (the one drawn on top) is taken.
function choose(found, pointer, x, y, then, spot) {
  if (found.length === 1 || pointer !== "touch") return then(found[0]);
  const items = found.map((b) => ({ node: closeUp(state.shown.id, b, b.faint ? "faint box" : `card #${b.card}`, () => { closeMenu(); then(b); }) }));
  if (spot) items.push({ label: "None of these: it's here without a box", run: spot });
  items.push({ label: "Cancel" });
  openMenu(x, y, items);
}

const canDraw = () => !state.fixing && (state.mode === "home" || state.mode === "intro" || canMark());

function drawn(a, b) {
  const x1 = Math.min(a[0], b[0]), x2 = Math.max(a[0], b[0]), y1 = Math.min(a[1], b[1]), y2 = Math.max(a[1], b[1]);
  return { x: (x1 + x2) / 2, y: (y1 + y2) / 2, w: x2 - x1, h: y2 - y1 };
}

function onTap(p, pointer, x, y) {
  const found = hits(p);
  if (state.fixing) {
    if (found.length) choose(found, pointer, x, y, (b) => openFix(b, x, y));
    return;
  }
  if (state.mode === "follow") {
    if (!atStep()) {
      if (!state.busy) message("You're looking back (or ahead): press Space or “Back to my step” to carry on.");
      return;
    }
    if (!found.length) return markSpot(p);
    return choose(found, pointer, x, y, (b) => markBox(b), () => markSpot(p));
  }
  if (state.mode === "home" || state.mode === "intro") {
    if (!found.length) return message("No box there: drag one around the fish to follow it.");
    const id = state.shown.id;
    choose(found, pointer, x, y, (b) => setPick({ card: b.card, det: b.det, frame: id, box: b }));
  }
}

function onDraw(d) {
  if (d.w < 0.01 || d.h < 0.01) return;
  if (state.mode === "follow") return markDrawn(d);
  if (state.shown) setPick({ frame: state.shown.id, drawn: d });
}

let press = null; // the pointer pressed on the picture
const stage = $("stage");
stage.addEventListener("pointerdown", (e) => {
  if (e.button !== 0) return;
  closeMenu();
  const p = norm(e);
  press = { x: e.clientX, y: e.clientY, p, pointer: e.pointerType, moved: false, held: false, timer: null };
  stage.setPointerCapture(e.pointerId);
  const hit = hits(p)[0];
  if (hit) {
    press.timer = setTimeout(() => {
      if (!press) return;
      press.held = true;
      openFix(hit, press.x, press.y);
    }, 550);
  }
});
stage.addEventListener("pointermove", (e) => {
  if (!press) return;
  if (!press.moved && Math.hypot(e.clientX - press.x, e.clientY - press.y) > 8) {
    press.moved = true;
    clearTimeout(press.timer);
  }
  if (press.moved && canDraw()) {
    const d = drawn(press.p, norm(e));
    const el = $("drawing");
    el.hidden = false;
    place(el, corners(d));
  }
});
stage.addEventListener("pointerup", (e) => {
  if (!press) return;
  const was = press;
  press = null;
  clearTimeout(was.timer);
  $("drawing").hidden = true;
  if (was.held) return;
  if (was.moved) {
    if (canDraw()) onDraw(drawn(was.p, norm(e)));
    return;
  }
  onTap(was.p, was.pointer, e.clientX, e.clientY);
});
stage.addEventListener("pointercancel", () => {
  if (press) clearTimeout(press.timer);
  press = null;
  $("drawing").hidden = true;
});
stage.addEventListener("contextmenu", (e) => {
  e.preventDefault();
  if (press) {
    clearTimeout(press.timer);
    if (press.held) return; // a press and hold already opened it
    press.held = true;
  }
  const hit = hits(norm(e))[0];
  if (hit) openFix(hit, e.clientX, e.clientY);
});
document.addEventListener("pointerdown", (e) => {
  if (!$("menu").hidden && !$("menu").contains(e.target) && !stage.contains(e.target)) closeMenu();
});
$("follow-frame").addEventListener("load", (e) => {
  const img = e.target;
  if (img.naturalWidth > 1 && img.naturalHeight > 1) {
    const changed = state.aspect !== img.naturalWidth / img.naturalHeight;
    state.aspect = img.naturalWidth / img.naturalHeight;
    document.documentElement.style.setProperty("--aspect", String(state.aspect)); // the frame's shape (styles.css)
    if (changed) renderLooking(); // the suggestion weighs distances by it
  }
});

// --- buttons and keys ---------------------------------------------------------------------

$("begin").onclick = begin;
$("pick-start").onclick = begin;
$("done-back").onclick = backToReview;
$("done-undo").onclick = undo;
$("discard").onclick = discard;
$("to-step").onclick = () => look(state.stepPlace);
$("t-accept").onclick = accept;
$("t-gone").onclick = markGone;
$("t-nobox").onclick = markNoBox;
$("t-rest").onclick = markRest;
$("t-undo").onclick = undo;
$("t-zoom").onclick = () => { state.zoom = !state.zoom; renderLooking(); };
$("t-fix").onclick = () => setFixing(!state.fixing);
$("scrub").addEventListener("input", (e) => look(Number(e.target.value)));

document.addEventListener("keydown", (e) => {
  const tag = e.target.tagName;
  if (tag === "SELECT" || tag === "TEXTAREA" || (tag === "INPUT" && e.target.type !== "range")) return;
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  const arrows = tag === "INPUT" ? [] : ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"]; // the slider moves itself
  if (e.key === "Escape") {
    if (!$("menu").hidden) closeMenu();
    else if (state.fixing) setFixing(false);
    else backToReview();
  } else if (arrows.includes(e.key)) {
    const back = e.key === "ArrowUp" || e.key === "ArrowLeft";
    look(state.looking + (back ? -1 : 1) * (e.shiftKey ? 10 : 1));
  } else if (e.key === " " && state.mode === "follow") {
    look(state.stepPlace);
  } else if (e.key === "Enter") {
    if (state.mode === "follow") accept();
    else if (state.mode === "home" || state.mode === "intro") begin();
  } else if (k === "b") markNoBox();
  else if (k === "o") markGone();
  else if (k === "g") markRest();
  else if (e.key === "Backspace") undo();
  else if (k === "z") { state.zoom = !state.zoom; renderLooking(); }
  else if (k === "f") setFixing(!state.fixing);
  else return;
  e.preventDefault();
});

// --- start --------------------------------------------------------------------------------

async function init() {
  setBrand();
  if (window.BRUV_preload) show("data-note", true); // the website's runtime: pictures come over the network
  try { $("gone").value = localStorage.getItem("follow-gone") || "5"; } catch (e) { /* not kept */ }
  if (state.id) return render(await api(`/api/follows/${state.id}`));
  if (state.card) return showIntro();
  return showHome();
}

init().catch((e) => message(e.message, "error"));
