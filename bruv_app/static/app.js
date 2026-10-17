"use strict";

const $ = (id) => document.getElementById(id);
const state = { summary: null, cards: [], index: 0, detail: null, selected: null };

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

function visibleCards() {
  if (!$("pending-only").checked) return state.cards;
  return state.cards.filter((c) => c.status === "pending" || c.status === "skipped");
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
  $("species-list").replaceChildren(...(dd.species || []).map((v) => new Option(v)));
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
    message("All cards reviewed. Untick “Pending only” to revisit, or export.", "ok");
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
    `${d.n_detections} detections in ${d.tracklets.length} track${d.tracklets.length === 1 ? "" : "s"}`;

  // Status: only the user's own recorded answer is ever highlighted.
  const badge = $("status-badge");
  badge.textContent = d.status === "species" ? `Recorded: ${d.species}` : STATUS_TEXT[d.status];
  badge.className = "badge " + (d.status === "pending" ? "" : d.status === "skipped" ? "skipped" : "done");

  const guesses = $("guesses");
  guesses.replaceChildren();
  if (!d.guesses.length) {
    guesses.innerHTML = '<p class="hint">No model guesses for this card — pick from the list.</p>';
  }
  d.guesses.forEach(([label, p], i) => {
    const b = document.createElement("button");
    b.innerHTML = `<span><kbd>${i + 1}</kbd> ${label}</span><span class="pct">${Math.round(p * 100)}%</span>`;
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
    fig.onclick = () => {
      strip.querySelectorAll("figure").forEach((f) => f.classList.remove("selected"));
      fig.classList.add("selected");
      state.selected = item;
      showFrame(item);
      $("split").disabled = i === 0;
    };
    strip.append(fig);
  });
  $("split").disabled = true;
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
  await loadSummary();
  // Advance: with "Pending only" the reviewed card drops out, so the same index is next.
  const list = visibleCards();
  const stays = list.some((c) => c.id === d.id);
  await show(stays ? state.index + 1 : state.index);
}

function recordTyped() {
  const value = $("species-input").value.trim();
  const options = state.summary.dropdowns.species || [];
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
  if (!d || !sel) return;
  if (!confirm(`Split card #${d.id}: sightings from ${fmtTime(sel.local_t)} onward become a new card?`)) return;
  try {
    const res = await api(`/api/fish/${d.id}/split`, { from_t: sel.t });
    await Promise.all([loadCards(), loadSummary()]);
    message(`Split off card #${res.new_id}.`, "ok");
    const list = visibleCards();
    await show(Math.max(0, list.findIndex((c) => c.id === d.id)));
  } catch (e) {
    message(e.message, "error");
  }
}

async function mergePrev() {
  const d = state.detail;
  const pos = state.cards.findIndex((c) => c.id === d.id);
  if (pos <= 0) return;
  const prev = state.cards[pos - 1];
  if (!confirm(`Merge card #${d.id} into the previous card #${prev.id}? It will need re-reviewing.`)) return;
  try {
    await api(`/api/fish/${prev.id}/merge`, { other_id: d.id });
    await Promise.all([loadCards(), loadSummary()]);
    const list = visibleCards();
    await show(Math.max(0, list.findIndex((c) => c.id === prev.id)));
    message(`Merged into card #${prev.id}.`, "ok");
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
document.querySelectorAll(".other-buttons button").forEach((b) => (b.onclick = () => record(b.dataset.status)));
$("prev").onclick = () => show(state.index - 1);
$("next").onclick = () => show(state.index + 1);
$("split").onclick = split;
$("merge-prev").onclick = mergePrev;
$("export").onclick = exportExcel;
$("pending-only").onchange = () => show(0);

init().catch((e) => message(e.message, "error"));
