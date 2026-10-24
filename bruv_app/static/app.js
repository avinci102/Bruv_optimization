"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  summary: null, cards: [], index: 0, detail: null, selected: null, selectedIndex: -1,
  template: new Set(), other: null, tinyPx: 40, ticked: new Set(),
  rig: new Map(), lastBatch: null,  // cards queued as likely equipment; the last batch marked
  familyOpen: null,  // the family picked in "by family first", while its species are shown
  lastAnswer: null,  // your last answer here (R repeats it when no look-alike was answered)
  likeSource: null,  // the card you just answered and its unanswered look-alikes (L)
  likeBatch: null,  // the last look-alikes answered together, for undo
  grid: null,  // the picture grid shown over the page (look-alikes, or hidden still objects)
};

function pct(p) {
  return `${Math.round(p * 100)}%`;
}

// "By family first" or "by species", remembered on this computer.
function loadGuessMode() {
  try {
    const saved = localStorage.getItem("guess-mode");
    if (saved === "family" || saved === "species") $("guess-mode").value = saved;
  } catch (e) { /* no storage here: keep the default */ }
}

const NOT_IN_TEMPLATE = "not in template";
const EXTRA_HINT = "Not in the template's species list: exported as its 'other' entry, with this name in the comment.";

const STATUS_TEXT = {
  pending: "Not reviewed yet",
  species: "Recorded",
  unknown: "Recorded: Unknown fish",
  not_fish: "Recorded: Not a fish",
  skipped: "Skipped",
};

async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
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

function message(text, kind = "") {
  const el = $("message");
  el.textContent = text;
  el.className = kind;
}

function fillSelect(el, values) {
  el.replaceChildren(new Option("—", ""), ...values.map((v) => new Option(v, v)));
}

// Rank (1-3) of the first of the template's own species among a card's guesses, 0 if none.
function templateRank(labels) {
  const i = (labels || []).findIndex((g) => state.template.has(g));
  return i < 0 ? 0 : i + 1;
}

function statusGroup(status) {
  return status === "pending" || status === "skipped" ? status : "reviewed";
}

// The cards the header lets through (Status, Show: the rank a template species needs among
// the guesses, or "all"; and whether tiny fish are shown), in the order Sort asks for. Biggest
// fish first puts the clearest fish, the ones the counts rest on, first. The server sends the
// cards in time order, which the sort keeps for equal sizes.
function visibleCards(status = $("status").value, rule = $("filter").value, showTiny = $("show-tiny").checked) {
  const list = state.cards.filter((c) => {
    if (status !== "all" && statusGroup(c.status) !== status) return false;
    if (!showTiny && c.max_px < state.tinyPx) return false;
    if (rule === "all") return true;
    if (rule === "rig") return state.rig.has(c.id);
    const rank = templateRank(c.guesses);
    return rank > 0 && rank <= Number(rule);
  });
  const sort = $("sort").value;
  if (sort === "size") return list.sort((a, b) => b.max_px - a.max_px);
  if (sort === "group") {
    // Look-alike groups (the server's), biggest fish's group first, each in time order; so
    // a run of the same kind of fish comes together and R answers it quickly.
    const byId = new Map(state.cards.map((c) => [c.id, c]));
    const lead = (c) => (c.group == null ? -1 : (byId.get(c.group) || c).max_px);
    return list.sort((a, b) => lead(b) - lead(a) || (a.group ?? 0) - (b.group ?? 0) || a.start_t - b.start_t);
  }
  return list;
}

// How an answer reads: the species, or "not a fish" / "unknown fish".
function answerName(a) {
  return a.status === "species" ? a.species : a.status === "not_fish" ? "not a fish" : "unknown fish";
}

// "Same fish as the card before it": the card just before it in the video, skipping hidden
// tiny fish, whatever Status, Show and Sort are set to.
function previousInTime(card) {
  for (let i = state.cards.findIndex((c) => c.id === card.id) - 1; i >= 0; i--) {
    if ($("show-tiny").checked || state.cards[i].max_px >= state.tinyPx) return state.cards[i];
  }
  return null;
}

// Tiny-fish cards that Status and Show let through: shown only when ticked.
function tinyCount() {
  return visibleCards(undefined, undefined, true).length - visibleCards(undefined, undefined, false).length;
}

// Each choice shows how many cards it would give, with the other controls as they are.
function updateFilterCounts() {
  const count = (opt, n) => {
    if (!opt.dataset.label) opt.dataset.label = opt.textContent;
    opt.textContent = `${opt.dataset.label} (${n})`;
  };
  for (const opt of $("status").options) count(opt, visibleCards(opt.value, undefined).length);
  for (const opt of $("filter").options) count(opt, visibleCards(undefined, opt.value).length);
  $("tiny-label").textContent = `Show tiny fish (${tinyCount()})`;
}

async function loadSummary() {
  state.summary = await api("/api/summary");
  $("project").textContent = state.summary.project;
  const still = state.summary.still_hidden || 0;
  $("still-note").hidden = !still;
  $("still-note").textContent = `${still} still-object boxes hidden`;
  showProgress();
}

async function loadCards() {
  state.cards = await api("/api/fish");
  showProgress();
}

// Cards that sit where a card marked as equipment sat, and look like it (equipment.py).
async function loadRig() {
  const r = await api("/api/rig").catch(() => ({ candidates: {} }));  // (an older server has none)
  state.rig = new Map(Object.entries(r.candidates).map(([id, m]) => [Number(id), m]));
}

function updateRigBar() {
  const queue = $("filter").value === "rig";
  const list = queue ? visibleCards() : [];
  const whole = list.filter((c) => !(state.rig.get(c.id) || {}).odd).length;
  $("rig-batch").hidden = !queue && !state.lastBatch;
  $("rig-all").hidden = !whole;
  $("rig-all").textContent = `Mark the ${whole} that match throughout as equipment` +
    (list.length > whole ? ` (${list.length - whole} to check one by one)` : "");
  $("rig-undo").hidden = !state.lastBatch;
  if (state.lastBatch) $("rig-undo").textContent = `Undo: the ${state.lastBatch.marked.length} just marked`;
}

// The bar follows the cards shown by default. Tiny fish are hidden until asked for, but an
// untouched tiny card is still unreviewed: they're counted apart, never as done.
function showProgress() {
  const answered = (c) => c.status !== "pending" && c.status !== "skipped";
  const main = state.cards.filter((c) => c.max_px >= state.tinyPx);
  const tiny = state.cards.filter((c) => c.max_px < state.tinyPx);
  const done = main.filter(answered).length;
  $("progress-text").textContent = `${done} of ${main.length} reviewed` +
    (tiny.length ? ` · tiny fish: ${tiny.filter(answered).length} of ${tiny.length}` : "");
  $("bar").style.width = main.length ? `${(100 * done) / main.length}%` : "0";
}

async function init() {
  await Promise.all([loadSummary(), loadCards(), loadRig()]);
  const dd = state.summary.dropdowns;
  state.template = new Set(dd.species || []);
  state.other = (dd.species || []).find((s) => s.trim().toLowerCase().startsWith("other")) || null;
  state.tinyPx = state.summary.tiny_fish_px ?? state.tinyPx;
  showProgress();
  $("tiny-toggle").title =
    `Tiny fish never reach ${state.tinyPx} px across, too small to identify. They're kept off ` +
    "other cards and hidden; tick to review them. Untouched, they stay out of the counts, and " +
    "the export marks its counts provisional.";
  // Open on the pending cards, or the skipped ones once nothing is pending.
  for (const status of ["pending", "skipped", "all"]) {
    $("status").value = status;
    if (visibleCards().length) break;
  }
  updateFilterCounts();
  $("species-list").replaceChildren(
    ...(dd.species || []).map((v) => new Option(v)),
    // Extra names aren't in the template: the export writes its "other" entry plus the name.
    ...(state.summary.extra_species || []).map((v) => Object.assign(new Option(v), { label: NOT_IN_TEMPLATE })),
  );
  fillSelect($("f-type"), dd.type || []);
  // When the species list depends on type (=INDIRECT), the export fills type from the species.
  $("type-field").hidden = Object.values(state.summary.dependent || {}).includes("type");
  fillSelect($("f-sex"), dd.sex || []);
  fillSelect($("f-behavior"), dd.behavior || []);
  if (!state.cards.length) {
    document.querySelector("main").hidden = true;
    $("empty").hidden = false;
    return;
  }
  await show(0);
}

async function show(index) {
  const list = visibleCards();
  updateRigBar();
  if (!list.length) {
    const tiny = $("show-tiny").checked ? 0 : tinyCount();
    const allPending = $("status").value === "pending" && $("filter").value === "all";
    message(tiny
      ? `Only tiny fish left here (${tiny}). Tick “Show tiny fish” to review them, or export.`
      : allPending
        ? "No pending cards left. Pick “Skipped” or “Reviewed” under Status to revisit, or export."
        : "No cards match. Change Status or Show.", "ok");
    return;
  }
  state.index = Math.max(0, Math.min(index, list.length - 1));
  const card = list[state.index];
  const d = await api(`/api/fish/${card.id}`);
  // (A server started before warnings existed sends none.)
  d.pieces ??= [];
  d.flags ??= [];
  d.closest_calls ??= [];
  d.rig ??= null;
  d.families ??= [];
  d.answered_like ??= null;
  d.lookalikes ??= 0;
  d.earlier ??= null;
  state.detail = d;
  state.selected = null;

  $("card-pos").textContent = `Card ${state.index + 1} of ${list.length}  ·  #${d.id}`;
  $("best-crop").src = d.best ? `/crop/${d.best.det_id}` : "";
  showFrame(d.best);
  $("meta").textContent =
    `First seen ${d.clip_in} ${fmtTime(d.time_in)} · last seen ${d.clip_out} ${fmtTime(d.time_out)} · ` +
    `${d.n_detections} detections in ${d.tracklets.length} track${d.tracklets.length === 1 ? "" : "s"} · ` +
    `${Math.round(card.max_px)} px across at its largest${card.max_px < state.tinyPx ? " (tiny fish)" : ""}`;
  // Where the tracker had the closest call (its next-best option nearly as good): a place
  // to look if the card might have jumped to another fish.
  const call = d.closest_calls[0];
  if (call) {
    const link = document.createElement("span");
    link.className = "call";
    link.textContent = `tracker's closest call: ${call.clip} ${fmtTime(call.local_t)}`;
    link.title = `The next-best option was ${call.margin} worse here (0 or less: a toss-up). Click to see it.`;
    link.onclick = () => showFrame(call);
    $("meta").append(" · ", link);
  }

  // Status: only the user's own recorded answer is ever highlighted.
  const badge = $("status-badge");
  badge.textContent = d.status === "species" ? `Recorded: ${d.species}`
    : d.status === "not_fish" && d.equipment ? "Recorded: Not a fish (equipment)" : STATUS_TEXT[d.status];
  badge.className = "badge " + (d.status === "pending" ? "" : d.status === "skipped" ? "skipped" : "done");
  const rank = templateRank(d.guesses.map((g) => g[0]));
  $("template-hint").hidden = !rank;
  $("template-hint").textContent = rank ? `★ Template species in the guesses: ${d.guesses[rank - 1][0]} (guess ${rank})` : "";

  renderGuesses(d);

  $("species-input").value = "";
  $("f-type").value = d.type || "";
  $("f-sex").value = d.sex || "";
  $("f-behavior").value = d.behavior || "";
  $("f-comment").value = d.comment || "";

  renderFilmstrip(d);
  renderPieces(d);
  renderFlags(d);
  renderLike(d);
  updateLikeBar();
  const rig = d.rig;
  $("rig-hint").hidden = !rig;
  if (rig) {
    $("rig-hint").textContent = `⚙ Sits where card #${rig.card} (marked as equipment) sat` +
      ` (overlap ${rig.overlap}${rig.colour === null ? "" : `, colours ${rig.colour} alike`}). Equipment: E. A fish: answer as usual.` +
      (rig.odd ? ` ⚠ ${rig.odd} of its sightings ${rig.odd === 1 ? "is elsewhere or looks" : "are elsewhere or look"} ` +
        "different: a fish may be joined on. Check them (tick and move them with X) before answering." : "");
  }
  updateRigBar();
  const prev = previousInTime(d);
  $("merge-prev").disabled = !prev;
  $("merge-prev").innerHTML = prev ? `Same fish as #${prev.id}, the card before it <kbd>P</kbd>` : "Same fish as the card before it <kbd>P</kbd>";
  $("merge-prev").title = prev ? `Fold this card into #${prev.id}, the card just before it in the video` : "No card before this one";
  message("");
}

function showFrame(item) {
  const box = $("box");
  if (!item) {
    $("frame").removeAttribute("src");
    box.hidden = true;
    return;
  }
  $("frame").src = `/frame/${item.frame_id}`;
  const b = item.box;
  Object.assign(box.style, {
    left: `${b.x1 * 100}%`,
    top: `${b.y1 * 100}%`,
    width: `${(b.x2 - b.x1) * 100}%`,
    height: `${(b.y2 - b.y1) * 100}%`,
  });
  box.hidden = false;
  $("frame-caption").textContent = `${item.clip} ${fmtTime(item.local_t)} · detector score ${b.score}`;
}

function renderFilmstrip(d) {
  const strip = $("filmstrip");
  strip.replaceChildren();
  $("strip-count").textContent =
    d.filmstrip.length < d.n_detections ? `(showing ${d.filmstrip.length} of ${d.n_detections})` : `(${d.n_detections})`;
  let lastTrack = null, lastClip = null;
  d.filmstrip.forEach((item, i) => {
    const fig = document.createElement("figure");
    if (lastTrack !== null && item.tracklet !== lastTrack) fig.classList.add("new-tracklet");
    lastTrack = item.tracklet;
    // Name the clip on the first thumbnail and wherever the chapter changes.
    const clip = item.clip !== lastClip ? `${item.clip} ` : "";
    lastClip = item.clip;
    fig.innerHTML =
      `<span class="tick" title="Tick (Space or Ctrl+click). X moves ticked sightings to a new card."></span>` +
      `<img loading="lazy" src="/crop/${item.det_id}" alt=""><figcaption>${clip}${fmtTime(item.local_t)}</figcaption>`;
    fig.title = `${item.clip} ${fmtTime(item.local_t)} (track ${item.tracklet})`;
    fig.onclick = (e) => {
      if (e.ctrlKey || e.metaKey) toggleTick(i);
      else if (e.shiftKey) tickRange(state.selectedIndex < 0 ? i : state.selectedIndex, i);
      else selectSighting(i);
    };
    fig.querySelector(".tick").onclick = (e) => {
      e.stopPropagation();
      toggleTick(i);
    };
    strip.append(fig);
  });
  state.selectedIndex = -1;
  state.ticked = new Set();
  updateSplitButton();
}

function guessButton(key, label, p, recorded, note = "") {
  const b = document.createElement("button");
  const extra = new Set(state.summary.extra_species || []);
  const tag = extra.has(label) ? `<span class="tag" title="${EXTRA_HINT}">${NOT_IN_TEMPLATE}</span>` : "";
  b.innerHTML = `<span><kbd>${key}</kbd> <span class="name"></span>${tag}<span class="of"></span></span><span class="pct">${pct(p)}</span>`;
  b.querySelector(".name").textContent = label;
  b.querySelector(".of").textContent = note;
  if (recorded) b.classList.add("recorded");
  return b;
}

// The model's guesses: its top three names, or ("by family first") its top three families,
// each scored with the sum of its names' scores. Picking a family with one name records it;
// otherwise its names follow, best first, and F records just the family.
function renderGuesses(d) {
  const box = $("guesses");
  box.replaceChildren();
  state.familyOpen = null;
  $("family-step").hidden = true;
  const byFamily = $("guess-mode").value === "family" && d.families.length;
  if (!byFamily && !d.guesses.length) {
    box.innerHTML = '<p class="hint">No model guesses for this card — pick from the list.</p>';
    return;
  }
  if (byFamily) {
    // Ranked by each family's best name, as the species are; the number is that name's score.
    d.families.forEach((f, i) => {
      const lead = f.members[0];
      const note = f.members.length > 1 ? ` · ${lead[0]}` : "";
      const b = guessButton(i + 1, f.name, f.best ?? lead[1], d.status === "species" && f.members.some(([n]) => n === d.species), note);
      b.title = f.members.length > 1 ? `${f.members.length} names; all of them together ${pct(f.p)}` : "";
      b.onclick = () => pickFamily(f);
      box.append(b);
    });
    return;
  }
  d.guesses.forEach(([label, p], i) => {
    const b = guessButton(i + 1, label, p, d.status === "species" && d.species === label);
    b.onclick = () => record("species", label);
    box.append(b);
  });
}

// The names to pick from in an open family: all but its own family-level entry (that's F),
// best first; past the first three, only names the model gives 1% or more. The full list is
// in the search box.
function familyNames(f) {
  return f.members.filter(([n]) => n !== f.entry).filter(([, p], i) => i < 3 || p >= 0.01).slice(0, 9);
}

function pickFamily(f) {
  if (f.members.length === 1) return record("species", f.members[0][0]);
  state.familyOpen = f;
  const step = $("family-step");
  step.replaceChildren();
  const head = document.createElement("p");
  head.className = "hint";
  head.textContent = `${f.name}: pick the species${f.entry ? ", or F for just the family" : ""}. Esc goes back.`;
  step.append(head);
  const d = state.detail;
  familyNames(f).forEach(([name, p], i) => {
    const b = guessButton(i + 1, name, p, d.status === "species" && d.species === name);
    b.onclick = () => record("species", name);
    step.append(b);
  });
  if (f.entry) {
    const b = guessButton("F", f.entry, f.members.find(([n]) => n === f.entry)?.[1] ?? 0, d.status === "species" && d.species === f.entry, " · species not known");
    b.onclick = () => record("species", f.entry);
    step.append(b);
  }
  step.hidden = false;
}

function closeFamily() {
  state.familyOpen = null;
  $("family-step").hidden = true;
}

// A card's tracked stretches side by side, each with its own guess: what to compare before
// one answer goes on all of them.
function renderPieces(d) {
  const many = d.pieces.length > 1;
  $("pieces-head").hidden = $("pieces").hidden = !many;
  $("pieces").replaceChildren();
  if (!many) return;
  for (const p of d.pieces) {
    const fig = document.createElement("figure");
    const guess = p.guesses.length ? `${p.guesses[0][0]} ${Math.round(p.guesses[0][1] * 100)}%` : "no guess";
    fig.innerHTML = `<img loading="lazy" src="/crop/${p.best_det}" alt=""><figcaption></figcaption>`;
    fig.querySelector("figcaption").textContent = `${p.clip} ${fmtTime(p.local_t)} · ${guess}`;
    fig.title = `Track ${p.tracklet}: ${p.detections} sightings over ${Math.round(p.seconds)} s`;
    fig.onclick = () => {
      const i = d.filmstrip.findIndex((s) => s.tracklet === p.tracklet);
      if (i >= 0) selectSighting(i);
    };
    $("pieces").append(fig);
  }
}

// Warnings from the evidence (pieces unlike each other, pieces favouring different species,
// two boxes at once): answering then asks to confirm it's one fish.
function renderFlags(d) {
  const box = $("flags");
  box.replaceChildren();
  box.hidden = !d.flags.length;
  if (!d.flags.length) return;
  const head = document.createElement("strong");
  head.textContent = "⚠ Check this card before answering";
  box.append(head);
  for (const f of d.flags) {
    const p = document.createElement("p");
    p.textContent = f.text;
    box.append(p);
  }
  const hint = document.createElement("p");
  hint.className = "hint";
  hint.textContent = "Compare the pieces under the photo. One fish: answer as usual and confirm. " +
    "Different fish: Multiple fish (M), or tick and move one fish's sightings (X).";
  box.append(hint);
}

// What R would answer, best evidence first: what most of this card's own sightings were
// answered before the cards were rebuilt; else the answered card it looks most like
// (lookalike.py); else your last answer. Nothing is recorded until you press it.
function repeatable(d) {
  if (d.status !== "pending" && d.status !== "skipped") return null;
  if (d.earlier && d.earlier.answers === 1) return { kind: "earlier", ...d.earlier };
  if (d.answered_like) return { kind: "like", ...d.answered_like };
  return state.lastAnswer ? { kind: "last", ...state.lastAnswer } : null;
}

function renderLike(d) {
  const hint = $("like-hint");
  const a = repeatable(d);
  const mixed = (d.status === "pending" || d.status === "skipped") && d.earlier && d.earlier.answers > 1;
  hint.hidden = !a && !mixed;
  if (mixed) {
    hint.innerHTML = `⚠ Before the cards were rebuilt, this card's sightings had ${d.earlier.answers} different answers ` +
      "(most of them <strong></strong>): check it, and use Multiple fish (M) if it mixes fish.";
    hint.querySelector("strong").textContent = answerName(d.earlier);
  } else if (a && a.kind === "earlier") {
    hint.innerHTML = `Before the cards were rebuilt you answered <strong></strong> for ${a.sightings} of this card's ` +
      `${a.of} sightings. <kbd>R</kbd> gives the whole card that answer.`;
    hint.querySelector("strong").textContent = answerName(a);
  } else if (a && a.kind === "like") {
    hint.innerHTML = `Looks like <a href="#"></a>, which you answered <strong></strong> ` +
      `<span class="muted">(${pct(a.similarity)} alike)</span>. <kbd>R</kbd> gives it the same answer.`;
    hint.querySelector("a").textContent = `#${a.card}`;
    hint.querySelector("a").onclick = (e) => { e.preventDefault(); goToCard(a.card); };
    hint.querySelector("strong").textContent = answerName(a);
  } else if (a) {
    hint.innerHTML = `<span class="muted"><kbd>R</kbd>: same as your last answer, <strong></strong>.</span>`;
    hint.querySelector("strong").textContent = answerName(a);
  }
}

async function goToCard(id) {
  $("status").value = "all";
  updateFilterCounts();
  const i = visibleCards().findIndex((c) => c.id === id);
  if (i >= 0) await show(i);
}

function repeatAnswer() {
  const d = state.detail;
  if (!d) return;
  const a = repeatable(d) || (d.status === "pending" || d.status === "skipped" ? null : state.lastAnswer);
  if (!a) {
    message("Nothing to repeat yet: answer a card first.", "error");
    return;
  }
  record(a.status, a.status === "species" ? a.species : null);
}

// After an answer: the unanswered cards that look like that card, to answer together (L).
async function loadLikeSource(id, status, species) {
  try {
    const cards = await api(`/api/fish/${id}/lookalikes`);
    state.likeSource = cards.length ? { id, status, species, cards } : null;
  } catch (e) {
    state.likeSource = null;  // (an older server has no look-alikes)
  }
  updateLikeBar();
}

function updateLikeBar() {
  const s = state.likeSource;
  $("like-batch").hidden = !s && !state.likeBatch;
  $("like-open").hidden = !s;
  if (s) {
    const n = s.cards.length;
    $("like-open").innerHTML = `${n} unanswered card${n === 1 ? "" : "s"} look${n === 1 ? "s" : ""} like #${s.id} ` +
      `(<span class="name"></span>): answer together <kbd>L</kbd>`;
    $("like-open").querySelector(".name").textContent = answerName(s);
  }
  $("like-undo").hidden = !state.likeBatch;
  if (state.likeBatch) $("like-undo").textContent = `Undo: the ${state.likeBatch.answered.length} look-alikes just answered`;
}

// A grid of pictures over the page. Tickable: click to untick a card that differs.
function openGrid({ title, hint, items, tickable = false, okText = null, onOk = null }) {
  state.grid = { items, onOk, okText, ticked: new Set(tickable ? items.filter((i) => !i.flagged).map((i) => i.id) : []) };
  $("grid-title").textContent = title;
  $("grid-hint").textContent = hint;
  const grid = $("grid");
  grid.replaceChildren();
  for (const it of items) {
    const fig = document.createElement("figure");
    fig.innerHTML = `<img loading="lazy" src="/crop/${it.det}" alt=""><figcaption></figcaption>`;
    fig.querySelector("figcaption").textContent = it.caption;
    if (it.flagged) {
      fig.classList.add("flagged");
      fig.title = "This card has warnings: answer it on its own";
    } else if (tickable) {
      fig.classList.toggle("ticked", state.grid.ticked.has(it.id));
      fig.onclick = () => {
        const t = state.grid.ticked;
        if (t.has(it.id)) t.delete(it.id);
        else t.add(it.id);
        fig.classList.toggle("ticked", t.has(it.id));
        updateGridOk();
      };
    }
    grid.append(fig);
  }
  $("grid-ok").hidden = !onOk;
  updateGridOk();
  $("grid-panel").hidden = false;
}

function updateGridOk() {
  const g = state.grid;
  if (!g || !g.onOk) return;
  $("grid-ok").textContent = g.okText(g.ticked.size);
  $("grid-ok").disabled = !g.ticked.size;
}

function closeGrid() {
  $("grid-panel").hidden = true;
  state.grid = null;
}

function openLikeGrid() {
  const s = state.likeSource;
  if (!s) {
    message("Answer a card first: its unanswered look-alikes can then be answered together.", "error");
    return;
  }
  const name = answerName(s);
  openGrid({
    title: `Cards that look like #${s.id}, which you answered ${name}`,
    hint: "Most alike first. Click a picture to untick a card that isn't the same kind of fish; " +
      "the ticked ones get the same answer, and you can undo it. Cards with warnings (orange) are left to answer one by one.",
    items: s.cards.map((c) => ({
      id: c.id, det: c.best_det, flagged: c.flagged,
      caption: `#${c.id} · ${c.clip} ${fmtTime(c.local_t)} · ${pct(c.similarity)} alike`,
    })),
    tickable: true,
    okText: (n) => `Answer the ${n} ticked as ${name}`,
    onOk: answerLikeGrid,
  });
}

async function answerLikeGrid() {
  const s = state.likeSource;
  const ids = [...state.grid.ticked];
  try {
    const res = await api("/api/lookalikes/answer", { source: s.id, ids, status: s.status, species: s.species });
    for (const id of res.answered) Object.assign(state.cards.find((c) => c.id === id) || {}, { status: s.status, species: s.species });
    state.likeBatch = res;
    state.likeSource = null;
    closeGrid();
    updateFilterCounts();
    await loadSummary();
    await show(state.index);
    message(`Answered ${res.answered.length} look-alike card${res.answered.length === 1 ? "" : "s"} as ${answerName(s)}` +
      (res.left.length ? `; ${res.left.length} left to answer one by one.` : "."), "ok");
  } catch (e) {
    message(e.message, "error");
  }
}

async function undoLikeBatch() {
  const b = state.likeBatch;
  if (!b) return;
  try {
    const res = await api("/api/batch/undo", { batch: b.batch });
    for (const id of res.undone) Object.assign(state.cards.find((c) => c.id === id) || {}, { status: "pending", species: null });
    state.likeBatch = null;
    updateFilterCounts();
    await loadSummary();
    await show(state.index);
    message(`Undone: ${res.undone.length} card${res.undone.length === 1 ? " is" : "s are"} pending again.`, "ok");
  } catch (e) {
    message(e.message, "error");
  }
}

// Boxes hidden as still objects (still.py), a sample over the video, to check none is a fish.
async function openStill() {
  try {
    const r = await api("/api/still");
    openGrid({
      title: `${r.count} boxes hidden as still objects`,
      hint: "These matched the scene's background for the minute either side (the bait arm, canister, rope, rocks), " +
        "so they make no cards and aren't counted. A fish here would have stayed motionless for over a minute. " +
        "If you see one, note its time: the still filter can be switched off for a rebuild.",
      items: r.sample.map((x) => ({ id: x.det_id, det: x.det_id, caption: `${x.clip} ${fmtTime(x.local_t)} · ${pct(x.change)} changed` })),
    });
  } catch (e) {
    message(e.message, "error");
  }
}

// Show one sighting of the card (click it, or ↑ and ↓); splitting starts there.
function selectSighting(i) {
  const d = state.detail;
  if (!d || !d.filmstrip.length) return;
  i = Math.max(0, Math.min(i, d.filmstrip.length - 1));
  const figs = $("filmstrip").querySelectorAll("figure");
  figs.forEach((f) => f.classList.remove("selected"));
  figs[i].classList.add("selected");
  figs[i].scrollIntoView({ block: "nearest", inline: "nearest" });
  state.selectedIndex = i;
  state.selected = d.filmstrip[i];
  showFrame(state.selected);
  updateSplitButton();
}

// Ticked sightings move to a new card together (e.g. the other fish, after two fish crossed).
function toggleTick(i) {
  if (state.ticked.has(i)) state.ticked.delete(i);
  else state.ticked.add(i);
  $("filmstrip").querySelectorAll("figure")[i].classList.toggle("ticked", state.ticked.has(i));
  updateSplitButton();
}

function tickRange(from, to) {
  const figs = $("filmstrip").querySelectorAll("figure");
  for (let i = Math.min(from, to); i <= Math.max(from, to); i++) {
    state.ticked.add(i);
    figs[i].classList.add("ticked");
  }
  selectSighting(to);
}

function clearTicks() {
  state.ticked.clear();
  $("filmstrip").querySelectorAll("figure.ticked").forEach((f) => f.classList.remove("ticked"));
  updateSplitButton();
}

// X moves the ticked sightings if there are any, else splits from the selected sighting on.
function updateSplitButton() {
  const n = state.ticked.size;
  $("split").innerHTML = n ? `Move ${n} ticked to a new card <kbd>X</kbd>` : "Split from selected <kbd>X</kbd>";
  $("split").disabled = !n && state.selectedIndex <= 0;
}

function optionalFields() {
  return {
    type: $("f-type").value || null,
    sex: $("f-sex").value || null,
    behavior: $("f-behavior").value || null,
    comment: $("f-comment").value.trim() || null,
  };
}

async function record(status, species = null) {
  const d = state.detail;
  if (!d) return;
  // The template's catch-all ("other …") is only useful with the fish's name in the comment.
  if (species && species === state.other && !$("f-comment").value.trim()) {
    message(`Type the fish's name in Comment, then record “${species}” again.`, "error");
    $("f-comment").focus();
    return;
  }
  // A flagged card gets one answer for all its pieces only once you confirm they're one fish.
  const answering = status === "species" || status === "unknown" || status === "not_fish";
  let confirmed = false;
  if (answering && d.flags.length) {
    const what = species || STATUS_TEXT[status].replace("Recorded: ", "");
    confirmed = confirm(
      `This card may hold more than one fish:\n\n• ${d.flags.map((f) => f.text).join("\n• ")}\n\n` +
      `Are all ${d.pieces.length} pieces one fish? OK records “${what}” for all of them. ` +
      "Cancel to look again (Multiple fish, M, gives each piece its own card).",
    );
    if (!confirmed) return;
  }
  try {
    await api(`/api/fish/${d.id}/label`, { status, species, ...optionalFields(), confirm: confirmed });
  } catch (e) {
    message(e.message, "error");
    return;
  }
  const card = state.cards.find((c) => c.id === d.id);
  card.status = status;
  card.species = species;
  if (answering) {
    state.lastAnswer = { status, species };
    loadLikeSource(d.id, status, species);  // not awaited: the next card shows meanwhile
  }
  updateFilterCounts();
  await loadSummary();
  // Advance: a card that no longer matches Status (e.g. "Pending") drops out, so the same index is next.
  const list = visibleCards();
  const stays = list.some((c) => c.id === d.id);
  await show(stays ? state.index + 1 : state.index);
}

function recordTyped() {
  const value = $("species-input").value.trim();
  const options = [...(state.summary.dropdowns.species || []), ...(state.summary.extra_species || [])];
  const match = options.find((o) => o.toLowerCase() === value.toLowerCase());
  if (!match) {
    message(value ? `“${value}” isn't in the template's species list.` : "Type or pick a species first.", "error");
    return;
  }
  $("species-input").blur();  // so the next card's shortcuts work straight away
  record("species", match);
}

async function split() {
  const d = state.detail, sel = state.selected;
  if (!d) return;
  if (state.ticked.size) return moveTicked();
  if (!sel || state.selectedIndex <= 0) {
    message("Pick the sighting where the other fish starts (click it, or use ↑ and ↓), then press X.", "error");
    return;
  }
  if (!confirm(`Split card #${d.id}: sightings from ${fmtTime(sel.local_t)} onward become a new card?`)) return;
  try {
    const res = await api(`/api/fish/${d.id}/split`, { from_t: sel.t });
    await Promise.all([loadCards(), loadSummary()]);
    updateFilterCounts();
    const list = visibleCards();
    await show(Math.max(0, list.findIndex((c) => c.id === d.id)));
    message(`Split off card #${res.new_id}.`, "ok");
  } catch (e) {
    message(e.message, "error");
  }
}

async function moveTicked() {
  const d = state.detail;
  const n = state.ticked.size;
  const ids = [...state.ticked].flatMap((i) => d.filmstrip[i].covers);
  if (!confirm(`Move the ${n} ticked sighting${n === 1 ? "" : "s"} of card #${d.id} to a new card?`)) return;
  try {
    const res = await api(`/api/fish/${d.id}/move`, { det_ids: ids });
    await Promise.all([loadCards(), loadSummary()]);
    updateFilterCounts();
    const list = visibleCards();
    await show(Math.max(0, list.findIndex((c) => c.id === d.id)));
    message(`Moved ${n} sighting${n === 1 ? "" : "s"} to card #${res.new_id}.`, "ok");
  } catch (e) {
    message(e.message, "error");
  }
}

async function mergePrev() {
  const d = state.detail;
  const prev = d && previousInTime(d);
  if (!prev) return;
  if (!confirm(`Same fish as #${prev.id}, the card before it in the video? Card #${d.id} is merged into it, which then needs re-reviewing.`)) return;
  try {
    await api(`/api/fish/${prev.id}/merge`, { other_id: d.id });
    await Promise.all([loadCards(), loadSummary()]);
    updateFilterCounts();
    const list = visibleCards();
    await show(Math.max(0, list.findIndex((c) => c.id === prev.id)));
    message(`Merged into card #${prev.id}.`, "ok");
  } catch (e) {
    message(e.message, "error");
  }
}

// The card mixes different fish: each tracked piece becomes its own card to review.
async function multipleFish() {
  const d = state.detail;
  if (!d) return;
  if (d.tracklets.length < 2) {
    message("This card is one tracked stretch: pick the sighting where the other fish starts (↑ and ↓) and press X.", "error");
    return;
  }
  if (!confirm(`Card #${d.id} shows more than one fish: break it into its ${d.tracklets.length} tracked pieces, each its own card?`)) return;
  try {
    const res = await api(`/api/fish/${d.id}/unmerge`, {});
    await Promise.all([loadCards(), loadSummary()]);
    updateFilterCounts();
    const list = visibleCards();
    await show(Math.max(0, list.findIndex((c) => c.id === d.id)));
    message(`Card #${d.id} is now ${res.new_ids.length + 1} cards.`, "ok");
  } catch (e) {
    message(e.message, "error");
  }
}

// Equipment (bait arm, canister, rope): answered "not a fish", and it becomes a reference, so
// cards in the same spot that look the same are queued under Show → likely equipment.
async function markEquipment() {
  const d = state.detail;
  if (!d) return;
  let confirmed = false;
  if (d.flags.length) {
    confirmed = confirm(
      `This card may hold more than one thing:\n\n• ${d.flags.map((f) => f.text).join("\n• ")}\n\n` +
      `Are all ${d.pieces.length} pieces equipment?`,
    );
    if (!confirmed) return;
  }
  try {
    await api(`/api/fish/${d.id}/equipment`, { confirm: confirmed });
  } catch (e) {
    message(e.message, "error");
    return;
  }
  const card = state.cards.find((c) => c.id === d.id);
  card.status = "not_fish";
  card.species = null;
  await Promise.all([loadSummary(), loadRig()]);
  updateFilterCounts();
  const stays = visibleCards().some((c) => c.id === d.id);
  await show(stays ? state.index + 1 : state.index);
  const n = state.rig.size;
  message(`Recorded as equipment. ${n} card${n === 1 ? "" : "s"} like it queued under Show → likely equipment.`, "ok");
}

async function markAllEquipment() {
  const list = visibleCards();
  const whole = list.filter((c) => !(state.rig.get(c.id) || {}).odd);
  if (!whole.length) return;
  if (!confirm(
    `Answer ${whole.length} cards as equipment (not a fish)? Every sighting on them sits where a card you ` +
    "marked as equipment sat, and looks like it. The rest of the queue, and any card with warnings, is left " +
    "for you to check one by one. You can undo this.",
  )) return;
  try {
    const res = await api("/api/rig/mark", { ids: whole.map((c) => c.id) });
    state.lastBatch = res;
    for (const id of res.marked) Object.assign(state.cards.find((c) => c.id === id), { status: "not_fish", species: null });
    await Promise.all([loadSummary(), loadRig()]);
    updateFilterCounts();
    await show(0);
    message(`Marked ${res.marked.length} as equipment` +
      (res.left.length ? `; ${res.left.length} left to check one by one (still under likely equipment).` : "."), "ok");
  } catch (e) {
    message(e.message, "error");
  }
}

async function undoBatch() {
  if (!state.lastBatch) return;
  try {
    const res = await api("/api/rig/undo", { batch: state.lastBatch.batch });
    for (const id of res.undone) Object.assign(state.cards.find((c) => c.id === id), { status: "pending", species: null });
    state.lastBatch = null;
    await Promise.all([loadSummary(), loadRig()]);
    updateFilterCounts();
    await show(state.index);
    message(`Undone: ${res.undone.length} card${res.undone.length === 1 ? " is" : "s are"} pending again.`, "ok");
  } catch (e) {
    message(e.message, "error");
  }
}

async function exportExcel() {
  message("Exporting…");
  try {
    const res = await api("/api/export", {});
    await loadSummary();
    message(`Saved: ${res.path}`, "ok");
  } catch (e) {
    message(e.message, "error");
  }
}

document.addEventListener("keydown", (e) => {
  const typing = ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName);
  if (typing) {
    if (e.key === "Enter" && document.activeElement.id === "species-input") recordTyped();
    if (e.key === "Escape") document.activeElement.blur();
    return;
  }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (!$("grid-panel").hidden) {  // the picture grid is open: Esc closes it, Enter confirms
    if (e.key === "Escape") closeGrid();
    else if (e.key === "Enter" && state.grid && state.grid.onOk && state.grid.ticked.size) state.grid.onOk();
    else return;
    e.preventDefault();
    return;
  }
  const d = state.detail;
  const guesses = d ? d.guesses : [];
  const byFamily = d && $("guess-mode").value === "family" && d.families.length;
  const open = state.familyOpen;
  if (open && /^[1-9]$/.test(e.key)) {
    const pick = familyNames(open)[Number(e.key) - 1];
    if (!pick) return;
    record("species", pick[0]);
  } else if (open && (e.key === "f" || e.key === "F") && open.entry) {
    record("species", open.entry);
  } else if (open && e.key === "Escape") {
    closeFamily();
  } else if (byFamily && ["1", "2", "3"].includes(e.key) && d.families[Number(e.key) - 1]) {
    pickFamily(d.families[Number(e.key) - 1]);
  } else if (!byFamily && ["1", "2", "3"].includes(e.key) && guesses[Number(e.key) - 1]) {
    record("species", guesses[Number(e.key) - 1][0]);
  } else if (e.key === "u" || e.key === "U") {
    record("unknown");
  } else if (e.key === "n" || e.key === "N") {
    record("not_fish");
  } else if (e.key === "s" || e.key === "S") {
    record("skipped");
  } else if (e.key === "e" || e.key === "E") {
    markEquipment();
  } else if (e.key === "m" || e.key === "M") {
    multipleFish();
  } else if (e.key === "x" || e.key === "X") {
    split();
  } else if (e.key === "p" || e.key === "P") {
    mergePrev();
  } else if (e.key === "r" || e.key === "R") {
    repeatAnswer();
  } else if (e.key === "l" || e.key === "L") {
    openLikeGrid();
  } else if (e.key === "ArrowUp") {
    selectSighting(state.selectedIndex < 0 ? 0 : state.selectedIndex - 1);
  } else if (e.key === "ArrowDown") {
    selectSighting(state.selectedIndex < 0 ? 1 : state.selectedIndex + 1);
  } else if (e.key === " ") {
    if (state.selectedIndex < 0) selectSighting(0);
    if (state.selectedIndex >= 0) toggleTick(state.selectedIndex);
  } else if (e.key === "Escape") {
    clearTicks();
  } else if (e.key === "ArrowRight") {
    show(state.index + 1);
  } else if (e.key === "ArrowLeft") {
    show(state.index - 1);
  } else if (e.key === "/") {
    e.preventDefault();
    $("species-input").focus();
  } else {
    return;
  }
  e.preventDefault();
});

$("species-save").onclick = recordTyped;
document.querySelectorAll(".other-buttons button[data-status]").forEach((b) => (b.onclick = () => record(b.dataset.status)));
document.querySelector('.other-buttons [data-action="multiple"]').onclick = multipleFish;
document.querySelector('.other-buttons [data-action="equipment"]').onclick = markEquipment;
$("rig-all").onclick = markAllEquipment;
$("rig-undo").onclick = undoBatch;
$("like-open").onclick = openLikeGrid;
$("like-undo").onclick = undoLikeBatch;
$("grid-cancel").onclick = closeGrid;
$("grid-ok").onclick = () => state.grid && state.grid.onOk && state.grid.onOk();
$("still-note").onclick = openStill;
loadGuessMode();
$("guess-mode").onchange = (e) => {
  e.target.blur();  // so the review keys work straight away
  try { localStorage.setItem("guess-mode", e.target.value); } catch (err) { /* not kept */ }
  if (state.detail) renderGuesses(state.detail);
};
$("prev").onclick = () => show(state.index - 1);
$("next").onclick = () => show(state.index + 1);
$("split").onclick = split;
$("merge-prev").onclick = mergePrev;
$("export").onclick = exportExcel;
for (const id of ["status", "filter", "show-tiny", "sort"]) {
  $(id).onchange = (e) => {
    e.target.blur();  // so the review keys work straight away
    updateFilterCounts();
    show(0);
  };
}

init().catch((e) => message(e.message, "error"));
