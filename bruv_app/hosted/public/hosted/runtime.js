/* The hosted review page's server, running in the page itself (docs/SPEC.md §16).
 *
 * Loaded before the page's own script (publish adds it to the hosted copies of index.html
 * and follow.html). It answers the page's requests without changing the page:
 *   - fetch() to /api/... is answered by hosted/core.js from the published files and the
 *     volunteer's saved answers and fixes; new answers and fixes go to the site's database
 *     (/sync) first, so nothing shows as done that wasn't saved;
 *   - pictures at /crop/<sighting> and /frame/<frame> are cut from the card's published
 *     sheet, or for a video published with --follow from its strips of every frame, and
 *     shown as blob: images; those wanted next (the page lists them with BRUV_preload, and
 *     the frames around the one being followed) are made ahead.
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
    state.volunteer = me.id;
    if (manifest.follow) {
      // Every frame in order: following a fish steps through them, and any moment's scene
      // can be shown whole. The boxes in them are loaded when a follow is opened.
      state.setFollowFrames(await getJson(base + manifest.follow.frames));
      state.boxLoader = () => getJson(base + manifest.follow.boxes);
    }
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

  const sheets = new Map(); // card sheet url -> Promise<ImageBitmap>, the last 40 used
  const strips = new Map(); // strip of frames -> Promise<ImageBitmap>, the last 4 (each is large)
  const made = new Map(); // "/crop/12" -> Promise<blob url>, the last KEEP used
  const done = new Map(); // "/crop/12" -> blob url, once made: shown at once
  const KEEP = 600; // pictures kept made (tens of KB each); older ones are freed

  function sheet(url, cache = sheets, keep = 40) {
    if (cache.has(url)) {
      const hit = cache.get(url);
      cache.delete(url);
      cache.set(url, hit);
      return hit;
    }
    const p = realFetch("/" + url).then((r) => {
      if (!r.ok) throw new Error(`missing ${url}`);
      return r.blob();
    }).then((b) => createImageBitmap(b));
    p.catch(() => cache.delete(url));
    cache.set(url, p);
    while (cache.size > keep) cache.delete(cache.keys().next().value);
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
    const img = await (entry.strip ? sheet(entry.sheet, strips, 4) : sheet(entry.sheet));
    const c = canvas(w, h);
    c.getContext("2d").drawImage(img, x, y, w, h, 0, 0, w, h);
    return c;
  }

  // WebP where the browser can make it: Chrome makes JPEGs only while the page is idle, which
  // held each frame up by about a second (WebP took a tenth of that). Safari can't make WebP
  // (it gives a PNG instead), so it gets JPEGs from then on.
  let pictureType = "image/webp";
  function toUrl(c) {
    return new Promise((resolve) => c.toBlob((b) => {
      if (b.type !== pictureType) pictureType = "image/jpeg";
      resolve(URL.createObjectURL(b));
    }, pictureType, 0.9));
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

  // Where a frame is in the strips of every frame (videos published with --follow).
  function stripOf(state, frameId) {
    const f = state.m.follow;
    const i = state.frameIndex.get(frameId);
    if (!f || i === undefined) return null;
    const k = Math.floor(i / f.strip), cell = i % f.strip;
    return { sheet: `videos/${state.key}/follow/${k}.webp`, strip: true,
      rect: [(cell % f.cols) * f.w, Math.floor(cell / f.cols) * f.h, f.w, f.h] };
  }

  // A scene: from the card's sheet, or the strips of every frame. Otherwise, the nearest
  // published scene of the same card, dimmed, with this moment's close-ups where the fish
  // were, and a note saying so.
  async function frame(state, frameId) {
    const entry = state.frameSheets.get(frameId) || stripOf(state, frameId);
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
    if (made.has(path)) {
      const hit = made.get(path);
      made.delete(path);
      made.set(path, hit);
      return hit;
    }
    const p = ready.then((state) => {
      let m;
      if ((m = path.match(/^\/crop\/(\d+)$/))) return crop(state, Number(m[1]));
      if ((m = path.match(/^\/frame\/(\d+)$/))) return frame(state, Number(m[1]));
      throw new Error("not a picture");
    }).then((url) => {
      done.set(path, url);
      return url;
    });
    p.catch(() => made.delete(path));
    made.set(path, p);
    trim();
    return p;
  }

  // Frees the least recently used pictures past KEEP, except any on screen or still being made.
  function trim() {
    for (const key of made.keys()) {
      if (made.size <= KEEP) break;
      const url = done.get(key);
      if (!url || document.querySelector(`img[data-picture="${key}"]`)) continue;
      made.delete(key);
      done.delete(key);
      URL.revokeObjectURL(url);
    }
  }

  // Pictures wanted soon (the open card's frames, the frames around the one being followed),
  // made two at a time in the order given, nearest first, so frames that share a strip are
  // cut while it's at hand. A new list replaces the old one: what's wanted now comes first.
  let queue = [];
  let working = 0;
  function preload(paths) {
    queue = paths.filter((p) => PICTURE.test(p) && !made.has(p));
    pump();
  }
  function pump() {
    while (working < 2 && queue.length) {
      const path = queue.shift();
      if (made.has(path)) continue;
      working++;
      picture(path).catch(() => {}).finally(() => {
        working--;
        pump();
      });
    }
  }
  window.BRUV_preload = preload;

  // On the follow page the next frame is the one just after or before the one shown (a
  // follow goes forward, then back): the frames around it are made ahead, nearest first.
  const AROUND = 30;
  function around(path) {
    const m = path.match(/^\/frame\/(\d+)$/);
    if (!m || !location.pathname.startsWith("/follow")) return;
    ready.then((state) => {
      const i = state.frameIndex.get(Number(m[1]));
      if (i === undefined) return;
      const paths = [];
      for (let k = 1; k <= AROUND; k++) {
        for (const j of [i + k, i - k]) if (j >= 0 && j < state.frameIds.length) paths.push(`/frame/${state.frameIds[j]}`);
      }
      preload(paths);
    }).catch(() => {});
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

  // A picture made already is shown at once. Otherwise the picture keeps what it showed,
  // dimmed, until the new one is made (so stepping through frames never flashes blank); a
  // picture that showed nothing yet starts blank.
  function show(img, path) {
    img.dataset.picture = path;
    around(path);
    const url = done.get(path);
    if (url) {
      picture(path); // (now the most recently used)
      img.classList.remove("bruv-wait");
      desc.set.call(img, url);
      return;
    }
    if (!String(desc.get.call(img)).startsWith("blob:")) desc.set.call(img, BLANK);
    img.classList.add("bruv-wait");
    picture(path).then((made) => {
      if (img.dataset.picture !== path) return;
      img.classList.remove("bruv-wait");
      desc.set.call(img, made);
    }).catch(() => {
      if (img.dataset.picture === path) img.classList.remove("bruv-wait");
    });
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

  // A small link back to the start page: switch video, or say how the session went. The
  // Excel export is the coordinator's, after pulling, so it's hidden; so is "Follow this
  // fish" unless the video was published with --follow.
  document.addEventListener("DOMContentLoaded", () => {
    const style = document.createElement("style");
    style.textContent = "#export { display: none !important; } img.bruv-wait { opacity: 0.5; }";
    document.head.appendChild(style);
    const noFollow = document.createElement("style");
    noFollow.textContent = "#follow { display: none !important; }";
    document.head.appendChild(noFollow);
    ready.then((state) => {
      if (state.followable) noFollow.remove();
    }).catch(() => {});
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
