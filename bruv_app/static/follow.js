// Follow one fish frame by frame (follow.py): click it in each frame and the page moves on.
// Ground truth for how well the cards keep to one fish (docs/SPEC.md §15).

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const state = {
  card: Number(params.get("card")) || null,  // the card the fish was picked from
  id: Number(params.get("id")) || null,  // the follow, once started
  view: null,  // the server's latest answer: the follow, its next frame and that frame's boxes
  intro: null,  // the card's best view, shown before starting
  zoom: false,
  busy: false,  // a mark is on its way: ignore clicks until the next frame is shown
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

// --- drawing a frame ----------------------------------------------------------------------

function place(el, b) {
  Object.assign(el.style, {
    left: `${b.x1 * 100}%`, top: `${b.y1 * 100}%`,
    width: `${(b.x2 - b.x1) * 100}%`, height: `${(b.y2 - b.y1) * 100}%`,
  });
}

function drawFrame(frameId, boxes, opts = {}) {
  $("follow-frame").src = `/frame/${frameId}`;
  const layer = $("boxes");
  layer.replaceChildren();
  // Biggest first, so a small box on top of a big one can still be clicked.
  const area = (b) => (b.x2 - b.x1) * (b.y2 - b.y1);
  for (const b of [...boxes].sort((a, c) => area(c) - area(a))) {
    const el = document.createElement("button");
    el.type = "button";
    el.className = "fbox" + (b.faint ? " faint" : "") + (b.taken ? " taken" : "") + (opts.focus === b.det ? " focus" : "");
    el.title = b.faint
      ? `A faint box (score ${b.score}), or one hidden as still: on no card`
      : `Score ${b.score}, on card #${b.card}` + (b.taken ? `; taken by follow ${b.taken}` : "");
    place(el, b);
    if (b.taken) {
      const tag = document.createElement("span");
      tag.textContent = b.taken;
      el.append(tag);
    }
    if (opts.clickable) el.onclick = (e) => { e.stopPropagation(); markBox(b); };
    layer.append(el);
  }
  const ring = $("ring");
  ring.hidden = !opts.last;
  if (opts.last) Object.assign(ring.style, { left: `${opts.last[0] * 100}%`, top: `${opts.last[1] * 100}%` });
  applyZoom(opts.last);
}

function applyZoom(at) {
  const z = $("zoomer");
  if (state.zoom && at) {
    z.style.transformOrigin = `${at[0] * 100}% ${at[1] * 100}%`;
    z.style.transform = "scale(2)";
  } else {
    z.style.transform = "";
  }
}

// --- the steps ----------------------------------------------------------------------------

function render(view) {
  state.view = view;
  state.id = view.follow.id;
  state.card = view.follow.card;
  if (!params.get("id")) history.replaceState(null, "", `/follow?id=${state.id}`);
  $("back").href = `/#card=${state.card}`;
  $("intro").hidden = true;
  const f = view.follow;
  $("who").textContent = `${f.name} · from card #${f.card}`;
  $("counts").textContent = `${f.boxes} with a box · ${f.nobox} without · ${f.gone} not there`;
  const phase = $("phase");
  phase.hidden = false;
  if (view.phase === "done") {
    phase.textContent = "Done";
    phase.className = "phase done";
    $("keys").hidden = true;
    renderDone(view);
    // Show the start frame, so the page isn't blank.
    drawFrame(view.start.id, [{ det: view.start.det, faint: false, score: "", card: f.card, ...view.start.box }], { focus: view.start.det });
    $("where").textContent = `The fish at ${view.start.clip} ${fmtExact(view.start.local_t)}, where the follow started.`;
    return;
  }
  $("done").hidden = true;
  $("keys").hidden = false;
  const from = `${view.start.clip} ${fmtTime(view.start.local_t)}`;
  phase.textContent = view.phase === "forward" ? `Forward from ${from}` : `Back from ${from}`;
  phase.className = `phase ${view.phase}`;
  const fr = view.frame;
  $("where").textContent = `${fr.clip} ${fmtExact(fr.local_t)}` +
    (view.phase === "backward" ? " · going back in time" : "") +
    ` · ${view.boxes.length} box${view.boxes.length === 1 ? "" : "es"} in this frame`;
  drawFrame(fr.id, view.boxes, { clickable: true, last: view.last });
  state.busy = false;
}

function renderDone(view) {
  const f = view.follow;
  $("done").hidden = false;
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

function markBox(b) {
  if (state.busy || !state.view || !state.view.frame) return;
  send(api(`/api/follows/${state.id}/mark`, { frame_id: state.view.frame.id, state: "box", det_id: b.det }));
}

function markSpot(e) {
  if (state.busy || !state.view || !state.view.frame) return;
  // Where on the frame: the rectangle is the zoomed one when zoomed in.
  const r = $("zoomer").getBoundingClientRect();
  const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
  if (x < 0 || x > 1 || y < 0 || y > 1) return;
  send(api(`/api/follows/${state.id}/mark`, { frame_id: state.view.frame.id, state: "nobox", x, y }));
}

function markNoBox() {
  if (state.busy || !state.view || !state.view.frame) return;
  send(api(`/api/follows/${state.id}/mark`, { frame_id: state.view.frame.id, state: "nobox" }));
}

function markGone() {
  if (state.busy || !state.view || !state.view.frame) return;
  send(api(`/api/follows/${state.id}/mark`, { frame_id: state.view.frame.id, state: "gone" }));
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

async function begin() {
  if (state.id || state.busy) return;
  const gone = Number($("gone").value) || 5;
  try { localStorage.setItem("follow-gone", String(gone)); } catch (e) { /* not kept */ }
  send(api("/api/follows", { card: state.card, gone_s: gone }));
}

// --- start --------------------------------------------------------------------------------

async function init() {
  try { $("gone").value = localStorage.getItem("follow-gone") || "5"; } catch (e) { /* not kept */ }
  if (state.id) return render(await api(`/api/follows/${state.id}`));
  if (!state.card) {
    message("Open a card on the review page and pick “Follow this fish”.", "error");
    return;
  }
  const d = await api(`/api/fish/${state.card}`);
  $("back").href = `/#card=${state.card}`;
  $("who").textContent = `card #${state.card}`;
  $("intro").hidden = false;
  if (!d.best) {
    message("This card has no sightings to follow.", "error");
    return;
  }
  state.intro = d.best;
  $("start-crop").src = `/crop/${d.best.det_id}`;
  $("start-caption").textContent = `The fish to follow, at ${d.best.clip} ${fmtExact(d.best.local_t)}`;
  drawFrame(d.best.frame_id, [{ det: d.best.det_id, faint: false, score: d.best.box.score, card: state.card, ...d.best.box }],
    { focus: d.best.det_id });
  $("where").textContent = `${d.best.clip} ${fmtExact(d.best.local_t)} · card #${state.card}'s best view`;
}

$("stage").addEventListener("click", markSpot);
$("begin").onclick = begin;
$("done-back").onclick = backToReview;
$("done-undo").onclick = undo;
$("discard").onclick = discard;
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "SELECT") return;
  const k = e.key.toLowerCase();
  if (e.key === "Enter" && !state.id) begin();
  else if (k === "b") markNoBox();
  else if (k === "o") markGone();
  else if (e.key === "Backspace") { e.preventDefault(); undo(); }
  else if (k === "z") { state.zoom = !state.zoom; applyZoom(state.view && state.view.last); }
  else if (e.key === "Escape") backToReview();
});

init().catch((e) => message(e.message, "error"));
