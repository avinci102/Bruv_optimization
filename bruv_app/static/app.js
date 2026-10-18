"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  summary: null, cards: [], index: 0, detail: null, selected: null, selectedIndex: -1,
  template: new Set(), other: null, otherPending: false,
};

const NOT_IN_TEMPLATE = "not in template";
const EXTRA_HINT = "Not in the template's species list: exported as its 'other' entry, with this name in the comment.";
const FILTER_TEXT = {
  all: "all cards",
  1: "★ template species 1st",
  2: "★ template species in top 2",
  3: "★ template species in top 3",
};

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

// A filter is "<rule>" or "<rule>/<px>": the rank a template species needs among the guesses
// ("all" for any card), then optionally the size the fish must reach at its largest.
function visibleCards(filter = $("filter").value) {
  const [rule, minPx = 0] = filter.split("/");
  const pendingOnly = $("pending-only").checked;
  return state.cards.filter((c) => {
    if (pendingOnly && c.status !== "pending" && c.status !== "skipped") return false;
    if (c.max_px < Number(minPx)) return false;
    if (rule === "all") return true;
    const rank = templateRank(c.guesses);
    return rank > 0 && rank <= Number(rule);
  });
}

function updateFilterCounts() {
  for (const opt of $("filter").options) {
    const [rule, minPx] = opt.value.split("/");
    const size = minPx ? `, fish ≥ ${minPx} px` : "";
    opt.textContent = `${FILTER_TEXT[rule]}${size} (${visibleCards(opt.value).length})`;
  }
}

async function loadSummary() {
  state.summary = await api("/api/summary");
  const s = state.summary;
  $("project").textContent = s.project;
  $("progress-text").textContent = `${s.reviewed} of ${s.total} reviewed`;
  $("bar").style.width = s.total ? `${(100 * s.reviewed) / s.total}%` : "0";
}

async function loadCards() {
  state.cards = await api("/api/fish");
}

async function init() {
  await Promise.all([loadSummary(), loadCards()]);
  const dd = state.summary.dropdowns;
  state.template = new Set(dd.species || []);
  state.other = (dd.species || []).find((s) => s.trim().toLowerCase().startsWith("other")) || null;
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
  const list = visibleCards();
  if (!list.length) {
    $("pending-only").checked = false;
  }
  await show(0);
}

async function show(index) {
  const list = visibleCards();
  if (!list.length) {
    message($("filter").value === "all"
      ? "All cards reviewed. Untick “Pending only” to revisit, or export."
      : "No cards match. Change “Show” or untick “Pending only”.", "ok");
    return;
  }
  state.index = Math.max(0, Math.min(index, list.length - 1));
  const card = list[state.index];
  const d = await api(`/api/fish/${card.id}`);
  state.detail = d;
  state.selected = null;
  state.otherPending = false;

  $("card-pos").textContent = `Card ${state.index + 1} of ${list.length}  ·  #${d.id}`;
  $("best-crop").src = d.best ? `/crop/${d.best.det_id}` : "";
  showFrame(d.best);
  $("meta").textContent =
    `First seen ${d.clip_in} ${fmtTime(d.time_in)} · last seen ${d.clip_out} ${fmtTime(d.time_out)} · ` +
    `${d.n_detections} detections in ${d.tracklets.length} track${d.tracklets.length === 1 ? "" : "s"} · ` +
    `${Math.round(card.max_px)} px across at its largest`;

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
  $("merge-prev").disabled = state.cards.findIndex((c) => c.id === d.id) <= 0;
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
    fig.innerHTML = `<img loading="lazy" src="/crop/${item.det_id}" alt=""><figcaption>${clip}${fmtTime(item.local_t)}</figcaption>`;
    fig.title = `${item.clip} ${fmtTime(item.local_t)} (track ${item.tracklet})`;
    fig.onclick = () => selectSighting(i);
    strip.append(fig);
  });
  state.selectedIndex = -1;
  $("split").disabled = true;
}

// Show one sighting of the card (click it, or [ and ]); splitting starts there.
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
  $("split").disabled = i === 0;
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
  // Advance: with "Pending only" the reviewed card drops out, so the same index is next.
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
  if (!sel || state.selectedIndex <= 0) {
    message("Pick the sighting where the other fish starts (click it, or use [ and ]), then press X.", "error");
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

async function mergePrev() {
  const d = state.detail;
  const pos = state.cards.findIndex((c) => c.id === d.id);
  if (pos <= 0) return;
  const prev = state.cards[pos - 1];
  if (!confirm(`Same fish as the previous card #${prev.id}? Card #${d.id} is merged into it, which then needs re-reviewing.`)) return;
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
    message("This card is one tracked stretch: pick the sighting where the other fish starts ([ and ]) and press X.", "error");
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

// A fish on neither list: record the template's "other" entry with its name as the comment.
function otherFish() {
  if (!state.other) {
    message("The template has no “other” species: use Unknown fish (U) and write the name in the comment.", "error");
    return;
  }
  state.otherPending = true;
  $("f-comment").focus();
  message(`Type the fish's name, then Enter to record it as “${state.other}”. Esc cancels.`);
}

function recordOther() {
  if (!$("f-comment").value.trim()) {
    message("Type the fish's name first.", "error");
    return;
  }
  state.otherPending = false;
  $("f-comment").blur();
  record("species", state.other);
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
    if (e.key === "Enter" && document.activeElement.id === "f-comment" && state.otherPending) recordOther();
    if (e.key === "Escape") {
      if (state.otherPending) message("");
      state.otherPending = false;
      document.activeElement.blur();
    }
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
  } else if (e.key === "o" || e.key === "O") {
    otherFish();
  } else if (e.key === "x" || e.key === "X") {
    split();
  } else if (e.key === "p" || e.key === "P") {
    mergePrev();
  } else if (e.key === "[") {
    selectSighting(state.selectedIndex < 0 ? 0 : state.selectedIndex - 1);
  } else if (e.key === "]") {
    selectSighting(state.selectedIndex < 0 ? 1 : state.selectedIndex + 1);
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
document.querySelector('.other-buttons [data-action="other"]').onclick = otherFish;
document.querySelector('.other-buttons [data-action="multiple"]').onclick = multipleFish;
$("prev").onclick = () => show(state.index - 1);
$("next").onclick = () => show(state.index + 1);
$("split").onclick = split;
$("merge-prev").onclick = mergePrev;
$("export").onclick = exportExcel;
$("pending-only").onchange = () => {
  updateFilterCounts();
  show(0);
};
$("filter").onchange = () => show(0);

init().catch((e) => message(e.message, "error"));
