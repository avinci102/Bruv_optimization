/* The hosted review page's server, running in the page itself (docs/SPEC.md §16).
 *
 * Loaded before the review page's own script (publish adds it to the hosted copy of
 * index.html only). It answers the page's requests without changing the page:
 *   - fetch() to /api/... is answered by hosted/core.js from the published files and the
 *     volunteer's saved answers and fixes; new answers and fixes go to the site's database
 *     (/sync) first, so nothing shows as done that wasn't saved;
 *   - pictures at /crop/<sighting> and /frame/<frame> are cut from the card's published
 *     sheet and shown as blob: images.
 * A service worker would do this more quietly, but some browsers block those (private
 * windows, managed school laptops); this works in any recent browser.
 */
"use strict";
(function () {
  const video = (() => { try { return localStorage.getItem("bruv-video"); } catch (e) { return null; } })();
  if (!video) {
    location.replace("/start.html");
    return;
  }
  const realFetch = window.fetch.bind(window);

  async function getJson(url, opts) {
    const res = await realFetch(url, { credentials: "same-origin", ...(opts || {}) });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      if (res.status === 401) location.replace("/start.html");
      throw new BRUV.HttpError(res.status, data.detail || `${url}: ${res.status}`);
    }
    return res.json();
  }

  // The published video with this volunteer's saved answers and fixes replayed on top.
  const ready = (async () => {
    const base = `/videos/${video}/`;
    const [manifest, cards, emb, mine, me] = await Promise.all([
      getJson(base + "manifest.json"),
      getJson(base + "cards.json"),
      realFetch(base + "emb.bin").then((r) => {
        if (!r.ok) throw new BRUV.HttpError(r.status, "This video's data is missing from the site.");
        return r.arrayBuffer();
      }),
      getJson(`/sync/events?video=${encodeURIComponent(video)}`),
      getJson("/sync/me"),
    ]);
    const state = new BRUV.Video(video, manifest, cards, emb);
    const assignment = (me.videos || []).find((v) => v.video === video);
    if (assignment && assignment.cards) state.restrict(assignment.cards);
    for (const e of mine.events) state.apply(e);
    return state;
  })();

  let writing = Promise.resolve();
  const sync = {
    send(events) {
      const run = writing.then(() => getJson("/sync/events", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video, events, app: "hosted-1" }),
      }));
      writing = run.catch(() => {});
      return run;
    },
  };

  function reply(status, data) {
    return new Response(JSON.stringify(data), { status, headers: { "Content-Type": "application/json" } });
  }

  window.fetch = async function (input, init) {
    const url = new URL(typeof input === "string" ? input : input.url, location.href);
    if (url.origin !== location.origin || !url.pathname.startsWith("/api/")) return realFetch(input, init);
    try {
      const state = await ready;
      const method = ((init && init.method) || "GET").toUpperCase();
      const body = init && init.body ? JSON.parse(init.body) : null;
      const [status, data] = await BRUV.handle(state, sync, method, url.pathname, url.searchParams, body);
      return reply(status, data);
    } catch (err) {
      if (err instanceof BRUV.HttpError) return reply(err.status, { detail: err.detail });
      return reply(500, { detail: String((err && err.message) || err) });
    }
  };

  // --- pictures ----------------------------------------------------------------------------

  const sheets = new Map(); // sheet url -> Promise<ImageBitmap>, the last 40 used
  const made = new Map(); // "/crop/12" -> Promise<blob url>

  function sheet(url) {
    if (sheets.has(url)) {
      const hit = sheets.get(url);
      sheets.delete(url);
      sheets.set(url, hit);
      return hit;
    }
    const p = realFetch("/" + url).then((r) => {
      if (!r.ok) throw new Error(`missing ${url}`);
      return r.blob();
    }).then((b) => createImageBitmap(b));
    p.catch(() => sheets.delete(url));
    sheets.set(url, p);
    while (sheets.size > 40) sheets.delete(sheets.keys().next().value);
    return p;
  }

  function canvas(w, h) {
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    return c;
  }

  async function cut(entry) {
    const [x, y, w, h] = entry.rect;
    const img = await sheet(entry.sheet);
    const c = canvas(w, h);
    c.getContext("2d").drawImage(img, x, y, w, h, 0, 0, w, h);
    return c;
  }

  function toUrl(c) {
    return new Promise((resolve) => c.toBlob((b) => resolve(URL.createObjectURL(b)), "image/jpeg", 0.9));
  }

  async function crop(state, det) {
    let entry = state.sheets.get(det);
    if (!entry) {
      const near = state.nearestTiled(det);
      entry = near === null ? null : state.sheets.get(near);
    }
    if (!entry) throw new Error("No picture for this sighting");
    return toUrl(await cut(entry));
  }

  // A scene that wasn't published whole: the nearest published scene of the same card,
  // dimmed, with this moment's close-ups where the fish were, and a note saying so.
  async function frame(state, frameId) {
    const entry = state.frameSheets.get(frameId);
    if (entry) return toUrl(await cut(entry));
    const near = state.nearestFrame(frameId);
    const W = 640, H = Math.round(640 * state.H / state.W);
    const c = canvas(W, H);
    const ctx = c.getContext("2d");
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, W, H);
    if (near) {
      ctx.globalAlpha = 0.35;
      ctx.drawImage(await cut(state.frameSheets.get(near.frame)), 0, 0, W, H);
      ctx.globalAlpha = 1;
    }
    for (const det of state.frameDets.get(frameId) || []) {
      const tile = state.sheets.get(det);
      const d = state.det(det);
      if (!tile || !d) continue;
      const [left, top, right, bottom] = state.padded(d);
      ctx.drawImage(await cut(tile), left * W, top * H, (right - left) * W, (bottom - top) * H);
    }
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.fillRect(0, H - 22, W, 22);
    ctx.fillStyle = "#fff";
    ctx.font = "13px sans-serif";
    ctx.fillText(near ? `Scene from ${Math.abs(near.dt).toFixed(1)} s ${near.dt < 0 ? "earlier" : "later"} (dimmed); the fish as seen at this moment`
      : "Whole scene not published for this moment; the fish as seen", 8, H - 7);
    return toUrl(c);
  }

  function picture(path) {
    if (!made.has(path)) {
      const p = ready.then((state) => {
        let m;
        if ((m = path.match(/^\/crop\/(\d+)$/))) return crop(state, Number(m[1]));
        if ((m = path.match(/^\/frame\/(\d+)$/))) return frame(state, Number(m[1]));
        throw new Error("not a picture");
      });
      p.catch(() => made.delete(path));
      made.set(path, p);
    }
    return made.get(path);
  }

  const BLANK = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7";
  const desc = Object.getOwnPropertyDescriptor(HTMLImageElement.prototype, "src");
  const PICTURE = /^\/(crop|frame)\/\d+$/;

  function pathOf(value) {
    try {
      const u = new URL(value, location.href);
      return u.origin === location.origin && PICTURE.test(u.pathname) ? u.pathname : null;
    } catch (e) {
      return null;
    }
  }

  function show(img, path) {
    img.dataset.picture = path;
    desc.set.call(img, BLANK);
    picture(path).then((url) => {
      if (img.dataset.picture === path) desc.set.call(img, url);
    }).catch(() => {});
  }

  Object.defineProperty(HTMLImageElement.prototype, "src", {
    configurable: true,
    enumerable: desc.enumerable,
    get() {
      return this.dataset && this.dataset.picture ? this.dataset.picture : desc.get.call(this);
    },
    set(value) {
      const path = pathOf(value);
      if (path) show(this, path);
      else {
        if (this.dataset) delete this.dataset.picture;
        desc.set.call(this, value);
      }
    },
  });

  // Pictures written into the page as HTML (innerHTML) set the attribute directly.
  function scan(root) {
    const imgs = root.tagName === "IMG" ? [root] : root.querySelectorAll ? root.querySelectorAll("img[src]") : [];
    for (const img of imgs) {
      const path = pathOf(img.getAttribute("src"));
      if (path && img.dataset.picture !== path) show(img, path);
    }
  }
  new MutationObserver((records) => {
    for (const r of records) {
      if (r.type === "attributes" && r.target.tagName === "IMG") {
        const path = pathOf(r.target.getAttribute("src"));
        if (path && r.target.dataset.picture !== path) show(r.target, path);
      }
      for (const n of r.addedNodes || []) if (n.nodeType === 1) scan(n);
    }
  }).observe(document.documentElement, { childList: true, subtree: true, attributes: true, attributeFilter: ["src"] });

  // A small link back to the start page: switch video, or say how the session went.
  document.addEventListener("DOMContentLoaded", () => {
    const a = document.createElement("a");
    a.href = "/start.html";
    a.textContent = `${video} · videos · feedback`;
    a.title = "Back to the start page: switch video, or tell us how this session went";
    a.style.cssText = "position:fixed;right:12px;bottom:10px;z-index:20;padding:4px 10px;border-radius:6px;" +
      "background:rgba(0,0,0,.55);color:#fff;font:13px sans-serif;text-decoration:none";
    document.body.appendChild(a);
    ready.catch((e) => {
      const msg = document.getElementById("message");
      if (msg) {
        msg.textContent = `Couldn't load ${video}: ${e.message}`;
        msg.className = "error";
      }
    });
  });
})();
