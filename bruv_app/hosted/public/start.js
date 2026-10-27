/* Start page of the hosted review site: sign in (access pass or email), pick a video,
 * leave feedback; the coordinator also sees progress and downloads the answers. */
"use strict";
const $ = (id) => document.getElementById(id);

function message(text, kind = "") {
  $("msg").textContent = text;
  $("msg").className = kind;
}

async function call(path, body) {
  const opts = body === undefined ? { credentials: "same-origin" } : {
    method: "POST", credentials: "same-origin", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  };
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(data.detail || res.statusText);
    err.status = res.status;
    throw err;
  }
  return data;
}

// The review page reads which video to show from here (hosted/runtime.js).
function remember(video) {
  try {
    if (video) localStorage.setItem("bruv-video", video);
    else localStorage.removeItem("bruv-video");
    return true;
  } catch (e) {
    return false;
  }
}

function open(video) {
  if (!remember(video)) {
    message("This browser is blocking site storage, which the review page needs. Allow it for this site, or use a normal (not private) window.", "error");
    return;
  }
  message(`Opening ${video}…`);
  location.href = "/";
}

function showVideos(me) {
  $("signin").hidden = true;
  $("pick").hidden = false;
  $("feedback").hidden = false;
  $("hello").textContent = `Signed in as ${me.name}.`;
  const box = $("videos");
  box.replaceChildren();
  $("fb-video").replaceChildren(new Option("—", ""));
  if (!me.videos.length) box.textContent = "No videos are assigned to you yet.";
  for (const v of me.videos) {
    const b = document.createElement("button");
    b.className = "primary";
    b.textContent = v.role === "calibration" ? `${v.video} (shared check: ${v.cards ? v.cards.length : "all"} cards)` : v.video;
    b.onclick = () => open(v.video);
    box.append(b);
    $("fb-video").append(new Option(v.video, v.video));
  }
  if (me.role === "admin") showAdmin();
}

async function showAdmin() {
  $("admin").hidden = false;
  try {
    const p = await call("/sync/admin/progress");
    const table = document.createElement("table");
    table.innerHTML = "<tr><th>Volunteer</th><th>Video</th><th>Role</th><th>Answers</th><th>Fixes</th><th>Follows</th><th>Last active</th></tr>";
    for (const r of p.rows) {
      const tr = document.createElement("tr");
      for (const v of [r.name, r.video, r.role, r.answers, r.fixes, r.follows ?? 0, r.last || "—"]) {
        const td = document.createElement("td");
        td.textContent = v;
        tr.append(td);
      }
      table.append(tr);
    }
    $("progress").replaceChildren(table);
    const ex = $("exports");
    ex.replaceChildren();
    for (const video of p.videos) {
      const a = document.createElement("a");
      a.href = `/sync/admin/export?video=${encodeURIComponent(video)}`;
      a.textContent = `Download answers: ${video}`;
      a.download = `answers-${video}.json`;
      ex.append(a);
    }
  } catch (e) {
    $("progress").textContent = e.message;
  }
}

$("signin-form").onsubmit = async (e) => {
  e.preventDefault();
  try {
    const me = await call("/sync/login", { who: $("who").value });
    remember(null);
    message("");
    showVideos(me);
  } catch (err) {
    message(err.message, "error");
  }
};

$("signout").onclick = async () => {
  await call("/sync/logout", {}).catch(() => {});
  remember(null);
  location.reload();
};

$("feedback-form").onsubmit = async (e) => {
  e.preventDefault();
  try {
    await call("/sync/feedback", { video: $("fb-video").value || null, rating: Number($("rating").value) || null, text: $("fb-text").value });
    $("fb-text").value = "";
    message("Thanks: feedback sent.", "ok");
  } catch (err) {
    message(err.message, "error");
  }
};

(async () => {
  try {
    showVideos(await call("/sync/me"));
  } catch (e) {
    $("signin").hidden = false;
    if (e.status !== 401) message(e.message, "error");
  }
})();
