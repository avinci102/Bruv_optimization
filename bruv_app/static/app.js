"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  summary: null, cards: [], index: 0, detail: null, selected: null, selectedIndex: -1,
  template: new Set(), other: null, tinyPx: 40, ticked: new Set(),
};

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
    const rank = templateRank(c.guesses);
    return rank > 0 && rank <= Number(rule);
  });
  return $("sort").value === "size" ? list.sort((a, b) => b.max_px - a.max_px) : list;
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
  showProgress();
}

async function loadCards() {
  state.cards = await api("/api/fish");
  showProgress();
}

// Tiny fish are optional, so they count towards progress only once reviewed or skipped.
function showProgress() {
  const counted = state.cards.filter((c) => c.max_px >= state.tinyPx || c.status !== "pending");
  const done = counted.filter((c) => c.status !== "pending" && c.status !== "skipped").length;
  $("progress-text").textContent = `${done} of ${counted.length} reviewed`;
  $("bar").style.width = counted.length ? `${(100 * done) / counted.length}%` : "0";
}

async function init() {
  await Promise.all([loadSummary(), loadCards()]);
  const dd = state.summary.dropdowns;
  state.template = new Set(dd.species || []);
  state.other = (dd.species || []).find((s) => s.trim().toLowerCase().startsWith("other")) || null;
  state.tinyPx = state.summary.tiny_fish_px ?? state.tinyPx;
  showProgress();
  $("tiny-toggle").title =
    `Tiny fish never reach ${state.tinyPx} px across, too small to identify. They're kept off ` +
    "other cards and hidden; tick to review them. Untouched, they stay out of the export.";
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
  state.detail = d;
  state.selected = null;

  $("card-pos").textContent = `Card ${state.index + 1} of ${list.length}  ·  #${d.id}`;
  $("best-crop").src = d.best ? `/crop/${d.best.det_id}` : "";
  showFrame(d.best);
  $("meta").textContent =
    `First seen ${d.clip_in} ${fmtTime(d.time_in)} · last seen ${d.clip_out} ${fmtTime(d.time_out)} · ` +
    `${d.n_detections} detections in ${d.tracklets.length} track${d.tracklets.length === 1 ? "" : "s"} · ` +
    `${Math.round(card.max_px)} px across at its largest${card.max_px < state.tinyPx ? " (tiny fish)" : ""}`;

  // Status: only the user's own recorded answer is ever highlighted.
  const badge = $("status-badge");
  badge.textContent = d.status === "species" ? `Recorded: ${d.species}` : STATUS_TEXT[d.status];
  badge.className = "badge " + (d.status === "pending" ? "" : d.status === "skipped" ? "skipped" : "done");
  const rank = templateRank(d.guesses.map((g) => g[0]));
  $("template-hint").hidden = !rank;
  $("template-hint").textContent = rank ? `★ Template species in the guesses: ${d.guesses[rank - 1][0]} (guess ${rank})` : "";

  const guesses = $("guesses");
  guesses.replaceChildren();
  if (!d.guesses.length) {
    guesses.innerHTML = '<p class="hint">No model guesses for this card — pick from the list.</p>';
  }
  const extra = new Set(state.summary.extra_species || []);
  d.guesses.forEach(([label, p], i) => {
    const b = document.createElement("button");
    const tag = extra.has(label) ? `<span class="tag" title="${EXTRA_HINT}">${NOT_IN_TEMPLATE}</span>` : "";
    b.innerHTML = `<span><kbd>${i + 1}</kbd> ${label}${tag}</span><span class="pct">${Math.round(p * 100)}%</span>`;
    if (d.status === "species" && d.species === label) b.classList.add("recorded");
    b.onclick = () => record("species", label);
    guesses.append(b);
  });

  $("species-input").value = "";
  $("f-type").value = d.type || "";
  $("f-sex").value = d.sex || "";
  $("f-behavior").value = d.behavior || "";
  $("f-comment").value = d.comment || "";

  renderFilmstrip(d);
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
  try {
    await api(`/api/fish/${d.id}/label`, { status, species, ...optionalFields() });
  } catch (e) {
    message(e.message, "error");
    return;
  }
  const card = state.cards.find((c) => c.id === d.id);
  card.status = status;
  card.species = species;
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
  const guesses = state.detail ? state.detail.guesses : [];
  if (["1", "2", "3"].includes(e.key) && guesses[Number(e.key) - 1]) {
    record("species", guesses[Number(e.key) - 1][0]);
  } else if (e.key === "u" || e.key === "U") {
    record("unknown");
  } else if (e.key === "n" || e.key === "N") {
    record("not_fish");
  } else if (e.key === "s" || e.key === "S") {
    record("skipped");
  } else if (e.key === "m" || e.key === "M") {
    multipleFish();
  } else if (e.key === "x" || e.key === "X") {
    split();
  } else if (e.key === "p" || e.key === "P") {
    mergePrev();
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
