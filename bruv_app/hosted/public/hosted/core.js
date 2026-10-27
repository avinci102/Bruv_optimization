/* The review server's answers, worked out in the volunteer's browser (docs/SPEC.md §16).
 *
 * The hosted site has no server: `bruv_app publish` writes what the local review server
 * would answer for every card, and this file answers the same /api requests from those
 * files, so the review page itself is unchanged (one page, one API). What a volunteer does
 * is kept as an event log (their answers and fixes), sent to the site's database and
 * replayed here on top of the published cards. A split or move makes new cards; their
 * guesses, warnings, pieces and look-alikes are worked out here the way the server works
 * them out (pipeline.refresh_fish_guesses, diagnostics.card_flags, lookalike.py), from the
 * published descriptions (BioCLIP embeddings, float16). Following a fish (follow.py) works
 * the same way, for videos published with --follow: each follow is kept as events too.
 *
 * Plain script (no modules), so it loads with a classic <script> tag (hosted/runtime.js
 * wires it to the page) and in Node for tests. Everything is under the global BRUV.
 */
"use strict";

var BRUV = (typeof self !== "undefined" ? self : globalThis).BRUV || {};
(function (B) {
  const STATUSES = ["pending", "species", "unknown", "not_fish", "skipped"];
  const REVIEWED = ["species", "unknown", "not_fish"];
  const OPEN = ["pending", "skipped"];

  class HttpError extends Error {
    constructor(status, detail) {
      super(detail);
      this.status = status;
      this.detail = detail;
    }
  }
  B.HttpError = HttpError;

  // --- numbers ---------------------------------------------------------------------------

  // Python's round(): halves to even, on the decimal value.
  function round(x, n) {
    if (x === null || x === undefined || Number.isNaN(x)) return x;
    const f = Math.pow(10, n);
    const v = x * f;
    const r = Math.round(v);
    return (Math.abs(v % 1) === 0.5 ? (r % 2 === 0 ? r : r - 1) : r) / f;
  }
  B.round = round;

  function halfToFloat(h) {
    const s = (h & 0x8000) ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const f = h & 0x3ff;
    if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
    if (e === 31) return f ? NaN : s * Infinity;
    return s * Math.pow(2, e - 15) * (1 + f / 1024);
  }

  function decodeF16(buffer) {
    const half = new Uint16Array(buffer);
    const out = new Float32Array(half.length);
    for (let i = 0; i < half.length; i++) out[i] = halfToFloat(half[i]);
    return out;
  }
  B.decodeF16 = decodeF16;

  function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  function normalize(v) {
    let n = 0;
    for (let i = 0; i < v.length; i++) n += v[i] * v[i];
    n = Math.max(Math.sqrt(n), 1e-12);
    const out = new Float32Array(v.length);
    for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
    return out;
  }

  function meanOf(rows, dim) {
    const out = new Float32Array(dim);
    for (const r of rows) for (let i = 0; i < dim; i++) out[i] += r[i];
    for (let i = 0; i < dim; i++) out[i] /= rows.length;
    return out;
  }

  function softmax(x) {
    let m = -Infinity;
    for (const v of x) m = Math.max(m, v);
    const e = x.map((v) => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }

  // Sort that keeps the order of equal items, as Python's sorted() does.
  function stableSort(items, key) {
    return items.map((it, i) => [it, i]).sort((a, b) => key(a[0], b[0]) || a[1] - b[1]).map((p) => p[0]);
  }

  function utcStamp(prefix) {
    const d = new Date();
    const p = (n, w = 2) => String(n).padStart(w, "0");
    return `${prefix}-${d.getUTCFullYear()}${p(d.getUTCMonth() + 1)}${p(d.getUTCDate())}T${p(d.getUTCHours())}` +
      `${p(d.getUTCMinutes())}${p(d.getUTCSeconds())}${p(d.getUTCMilliseconds(), 3)}000`;
  }

  // views.view_score: how good a sighting is to show a person (published per sighting).
  function bestView(dets) {
    let best = null;
    for (const d of dets) if (best === null || d.view > best.view) best = d;
    return best;
  }

  function box(d) {
    return { x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2, score: round(d.score, 3) };
  }

  // --- one published video, with one volunteer's answers and fixes on top ----------------

  class Video {
    constructor(key, manifest, cardsJson, embBuffer) {
      this.key = key;
      this.m = manifest;
      this.s = manifest.settings;
      this.version = 0;
      const e = manifest.emb;
      this.dim = e.dim;
      const all = decodeF16(embBuffer);
      const block = (start) => (i) => all.subarray((start + i) * e.dim, (start + i + 1) * e.dim);
      this.labelRow = block(0);
      this.cropRow = block(e.labels);
      this.pieceRow = block(e.labels + e.crops);
      this.lookRow = block(e.labels + e.crops + e.pieces);
      this.text = [];
      for (let i = 0; i < e.labels; i++) this.text.push(this.labelRow(i));
      this.names = new Set(manifest.names || []);
      this.familyKeys = new Set(Object.values(manifest.name_family || {}));
      const [W, H] = manifest.frame_size || [960, 540];
      this.W = W;
      this.H = H;

      this.cards = new Map();
      this.sheets = new Map(); // det or frame id -> {sheet, rect}
      this.frameSheets = new Map();
      this.frameDets = new Map(); // frame id -> det ids on it (published cards)
      this.frameTime = new Map(); // frame id -> its time
      this.detById = new Map();
      this.detCard = new Map(); // det id -> the published card it came with (its sheet)
      this.maxCard = 0;
      this.maxTracklet = 0;
      for (const c of cardsJson.cards) {
        const dets = c.dets.map((r) => ({
          id: r[0], frame: r[1], t: r[2], x1: r[3], y1: r[4], x2: r[5], y2: r[6], score: r[7], quality: r[8],
          tracklet: r[9], view: r[10], emb: r[11], margin: r[12], history: r[13], px: r.length > 14 ? r[14] : null,
        }));
        const tracklets = new Map();
        for (const [tid, row] of c.pieces) {
          tracklets.set(tid, row >= 0 ? this.pieceRow(row) : null);
          this.maxTracklet = Math.max(this.maxTracklet, tid);
        }
        const card = {
          id: c.id, base: c, dets, tracklets, changed: false, group: c.list ? c.list.group : null,
          look: c.look >= 0 ? this.lookRow(c.look) : null,
          status: "pending", species: null, type: null, sex: null, behavior: null, comment: null, reviews: [],
        };
        this.cards.set(c.id, card);
        this.maxCard = Math.max(this.maxCard, c.id);
        const sheet = `videos/${key}/${c.sheet}`;
        for (const [det, rect] of Object.entries(c.crops)) this.sheets.set(Number(det), { sheet, rect });
        for (const [frame, rect] of Object.entries(c.frames)) this.frameSheets.set(Number(frame), { sheet, rect });
        for (const d of dets) {
          if (!this.frameDets.has(d.frame)) this.frameDets.set(d.frame, []);
          this.frameDets.get(d.frame).push(d.id);
          this.frameTime.set(d.frame, d.t);
          this.detById.set(d.id, d);
          this.detCard.set(d.id, c);
        }
      }
      const still = manifest.still || {};
      for (const [det, rect] of Object.entries(still.crops || {})) {
        this.sheets.set(Number(det), { sheet: `videos/${key}/${still.sheet}`, rect });
      }
      this.cache = {};

      // Following a fish (follow.py), for videos published with --follow: every frame in
      // order (setFollowFrames) and, once a follow is opened, every box in them (boxLoader).
      this.followable = Boolean(manifest.follow);
      this.frameIds = [];
      this.frameT = [];
      this.frameIndex = new Map(); // frame id -> its place in the video
      this.boxLoader = null; // async () -> the published boxes
      this.boxes = null; // frame id -> its boxes, highest score first (follow.frame_boxes)
      this.boxById = null;
      this.follows = new Map();
      this.followSeq = 0; // the highest follow id given; ids aren't reused
      this.markSeq = 0; // marks in the order they were made (follow_marks' rowid)
      this.volunteer = null; // their id on the site: follows are named F<id>-v<volunteer>
    }

    // ---- small helpers

    // A calibration assignment shows only the cards chosen for it.
    restrict(ids) {
      const keep = new Set(ids.map(Number));
      for (const id of [...this.cards.keys()]) if (!keep.has(id)) this.cards.delete(id);
      this.version++;
    }

    det(id) {
      return this.detById.get(id) || null;
    }

    // The sighting's close-up area in the frame (imaging.crop_box's padding), 0-1.
    padded(d) {
      const pad = this.s.crop_pad;
      const w = d.x2 - d.x1, h = d.y2 - d.y1;
      return [Math.max(0, d.x1 - pad * w), Math.max(0, d.y1 - pad * h), Math.min(1, d.x2 + pad * w), Math.min(1, d.y2 + pad * h)];
    }

    // For a sighting with no close-up of its own: the nearest one on its card that has one.
    nearestTiled(detId) {
      const d = this.det(detId);
      const card = this.detCard.get(detId);
      if (!d || !card) return null;
      let best = null;
      for (const k of Object.keys(card.crops)) {
        const other = this.det(Number(k));
        if (other && (best === null || Math.abs(other.t - d.t) < Math.abs(this.det(best).t - d.t))) best = other.id;
      }
      return best;
    }

    // For a frame that wasn't published whole: the nearest published scene of a card with a
    // sighting on it, and how far away in time it is.
    nearestFrame(frameId) {
      const t = this.frameTime.get(frameId);
      if (t === undefined) return null;
      let best = null;
      for (const det of this.frameDets.get(frameId) || []) {
        const card = this.detCard.get(det);
        for (const k of Object.keys(card.frames)) {
          const f = Number(k);
          const dt = this.frameTime.get(f) - t;
          if (best === null || Math.abs(dt) < Math.abs(best.dt)) best = { frame: f, dt };
        }
      }
      return best;
    }

    get(id) {
      const card = this.cards.get(Number(id));
      if (!card) throw new HttpError(404, "No such fish");
      return card;
    }

    locate(t) {
      const ch = this.m.chapters;
      for (let i = ch.length - 1; i >= 0; i--) {
        if (t >= ch[i].start_s - 1e-6) return [ch[i].name, Math.max(0, t - ch[i].start_s)];
      }
      return [ch[0].name, 0];
    }

    when(t) {
      const [clip, local] = this.locate(t);
      const s = round(local, 0);
      return `${clip} ${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
    }

    span(card) {
      let a = Infinity, b = -Infinity;
      for (const d of card.dets) {
        a = Math.min(a, d.t);
        b = Math.max(b, d.t);
      }
      return [a, b];
    }

    // store.fish_sizes: the fish at its largest, from each sighting's published size.
    maxPx(card) {
      let m = 0;
      for (const d of card.dets) m = Math.max(m, d.px ?? Math.max((d.x2 - d.x1) * this.W, (d.y2 - d.y1) * this.H));
      return m;
    }

    // The card's tracked pieces in order (start, id), each with its sightings.
    pieceList(card) {
      const by = new Map();
      for (const d of card.dets) {
        if (!by.has(d.tracklet)) by.set(d.tracklet, []);
        by.get(d.tracklet).push(d);
      }
      const out = [];
      for (const [tid, dets] of by) {
        out.push({ tracklet: tid, dets, start_t: Math.min(...dets.map((d) => d.t)), end_t: Math.max(...dets.map((d) => d.t)),
          emb: card.tracklets.get(tid) || null });
      }
      return out.sort((a, b) => a.start_t - b.start_t || a.tracklet - b.tracklet);
    }

    lastReview(card) {
      return card.reviews.length ? card.reviews[card.reviews.length - 1] : null;
    }

    lastDetails(card) {
      const r = this.lastReview(card);
      return (r && r.details) || {};
    }

    // ---- the species model's scores, from the published descriptions

    scores(emb) {
      if (!emb || !this.text.length) return null;
      const scale = this.m.logit_scale;
      return softmax(this.text.map((t) => scale * dot(t, emb)));
    }

    topGuesses(probs, k) {
      if (!probs) return [];
      const order = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, k);
      return order.map(([p, i]) => [this.m.labels[i], round(p, 4)]);
    }

    // pipeline.card_scores: the mean description of the card's best embedded crops.
    cardEmbedding(card) {
      const ranked = stableSort(card.dets, (a, b) => b.quality - a.quality);
      const rows = ranked.filter((d) => d.emb >= 0).slice(0, this.s.crops_per_fish).map((d) => this.cropRow(d.emb));
      return rows.length ? normalize(meanOf(rows, this.dim)) : null;
    }

    guesses(card) {
      if (!card.changed) return card.base.detail.guesses;
      return this.topGuesses(this.scores(this.cardEmbedding(card)), this.s.top_k);
    }

    look(card) {
      if (!card.changed) return card.look;
      const key = `look:${card.id}:${this.version}`;
      if (!(key in this.cache)) {
        const ranked = stableSort(card.dets.filter((d) => d.emb >= 0), (a, b) => b.quality - a.quality);
        const rows = ranked.slice(0, this.s.crops_per_fish).map((d) => this.cropRow(d.emb));
        this.cache[key] = rows.length ? normalize(meanOf(rows, this.dim)) : null;
      }
      return this.cache[key];
    }

    families(card) {
      if (!card.changed) return card.base.detail.families;
      const probs = this.scores(this.cardEmbedding(card));
      if (!probs) return [];
      const by = new Map();
      this.m.labels.forEach((label, i) => {
        const key = this.m.label_family[label] || label;
        if (!by.has(key)) by.set(key, []);
        by.get(key).push([label, probs[i]]);
      });
      const out = [];
      for (const [key, members] of by) {
        members.sort((a, b) => b[1] - a[1]);
        const info = this.m.families[key] || { name: key.replace(/_/g, " "), entry: null };
        out.push({ key, name: info.name, p: members.reduce((s, m) => s + m[1], 0), best: members[0][1], entry: info.entry, members });
      }
      out.sort((a, b) => b.best - a.best);
      return out.slice(0, 3).map((f) => ({
        key: f.key, name: f.name, p: round(f.p, 4), best: round(f.best, 4), entry: f.entry,
        members: f.members.map(([n, p]) => [n, round(p, 4)]),
      }));
    }

    // diagnostics.pieces
    pieces(card) {
      return this.pieceList(card).map((p) => {
        const best = stableSort(p.dets, (a, b) => b.quality - a.quality)[0];
        return {
          tracklet: p.tracklet, start_t: p.start_t, end_t: p.end_t, detections: p.dets.length, best_det: best ? best.id : null,
          guesses: p.emb ? this.topGuesses(this.scores(p.emb), 3) : [], embedding: p.emb, dets: p.dets,
        };
      });
    }

    // diagnostics.card_flags
    flags(card) {
      if (!card.changed) return card.base.detail.flags;
      const key = `flags:${card.id}:${this.version}`;
      if (key in this.cache) return this.cache[key];
      const parts = this.pieces(card);
      const out = [];
      const embedded = parts.filter((p) => p.embedding);
      let worst = null;
      for (let i = 0; i < embedded.length; i++) {
        for (let j = i + 1; j < embedded.length; j++) {
          const sim = dot(embedded[i].embedding, embedded[j].embedding);
          if (worst === null || sim < worst[0]) worst = [sim, embedded[i], embedded[j]];
        }
      }
      const minSim = this.s.merge_min_similarity;
      if (worst !== null && worst[0] < minSim) {
        const [sim, a, b] = worst;
        out.push({ kind: "pieces look different",
          text: `The pieces at ${this.when(a.start_t)} and ${this.when(b.start_t)} are less alike (${sim.toFixed(3)}) ` +
            `than the merge asks of a join (${minSim}).` });
      }
      const sure = parts.filter((p) => p.guesses.length && p.guesses[0][1] >= this.s.piece_sure);
      const top3 = new Map(sure.map((p) => [p.tracklet, new Set(p.guesses.slice(0, 3).map((g) => g[0]))]));
      let clash = null;
      for (let i = 0; i < sure.length && !clash; i++) {
        for (let j = i + 1; j < sure.length && !clash; j++) {
          const a = sure[i], b = sure[j];
          if (!top3.get(b.tracklet).has(a.guesses[0][0]) && !top3.get(a.tracklet).has(b.guesses[0][0])) clash = [a, b];
        }
      }
      if (clash) {
        const [a, b] = clash;
        const pc = (x) => `${Math.round(x * 100)}%`;
        out.push({ kind: "pieces disagree on species",
          text: `The piece at ${this.when(a.start_t)} favours ${a.guesses[0][0]} (${pc(a.guesses[0][1])}), the one at ` +
            `${this.when(b.start_t)} ${b.guesses[0][0]} (${pc(b.guesses[0][1])}); neither has the other's ` +
            "species among its three best guesses." });
      }
      const perFrame = new Map();
      for (const d of card.dets) {
        if (!perFrame.has(d.frame)) perFrame.set(d.frame, []);
        perFrame.get(d.frame).push(d.t);
      }
      const twice = [...perFrame.values()].filter((ts) => ts.length > 1).map((ts) => Math.min(...ts)).sort((a, b) => a - b);
      if (twice.length) {
        out.push({ kind: "two boxes at once",
          text: `${twice.length} frame${twice.length > 1 ? "s" : ""} hold two of its boxes, the first at ` +
            `${this.when(twice[0])}: two fish, or one boxed twice.` });
      }
      this.cache[key] = out;
      return out;
    }

    // ---- look-alikes (lookalike.py), from the published neighbour lists and the answers

    neighbours(card) {
      const key = `near:${card.id}:${this.version}`;
      if (key in this.cache) return this.cache[key];
      const thr = this.s.lookalike;
      let out;
      const mine = this.look(card);
      if (!mine) {
        out = [];
      } else if (!card.changed) {
        out = card.base.neighbours.filter(([id]) => this.cards.has(id) && !this.cards.get(id).changed);
        for (const other of this.cards.values()) {
          if (!other.changed || other.id === card.id) continue;
          const theirs = this.look(other);
          if (!theirs) continue;
          const sim = dot(mine, theirs);
          if (sim >= thr) out.push([other.id, round(sim, 4)]);
        }
        out = [...out].sort((a, b) => b[1] - a[1]);
      } else {
        out = [];
        for (const other of this.cards.values()) {
          if (other.id === card.id) continue;
          const theirs = this.look(other);
          if (!theirs) continue;
          const sim = dot(mine, theirs);
          if (sim >= thr) out.push([other.id, round(sim, 4)]);
        }
        out.sort((a, b) => b[1] - a[1]);
      }
      this.cache[key] = out;
      return out;
    }

    answeredLike(card) {
      for (const [id, sim] of this.neighbours(card)) {
        const other = this.cards.get(id);
        if (REVIEWED.includes(other.status)) return { card: id, status: other.status, species: other.species, similarity: sim };
      }
      return null;
    }

    unansweredLike(card) {
      return this.neighbours(card).filter(([id]) => OPEN.includes(this.cards.get(id).status));
    }

    // lookalike.card_pieces: each piece's shown sighting and how alike it is to `ref`
    cardPieces(card, ref) {
      return this.pieceList(card).map((p) => ({
        tracklet: p.tracklet, det: bestView(p.dets).id, similarity: p.emb && ref ? dot(p.emb, ref) : null,
        start_t: p.start_t, seconds: p.end_t - p.start_t,
      }));
    }

    everyPieceAlike(parts) {
      return parts.length > 0 && parts.every((p) => p.similarity !== null && p.similarity >= this.s.lookalike);
    }

    answerFamily(card) {
      if (card.status !== "species") return null;
      const r = this.lastReview(card);
      if (r && r.family) return r.family;
      return card.species ? (this.m.name_family[card.species] || this.m.label_family[card.species] || null) : null;
    }

    likeView(like) {
      if (!like) return null;
      const card = this.cards.get(like.card);
      const view = bestView(card.dets);
      return { ...like, similarity: round(like.similarity, 3), family: this.answerFamily(card), det: view ? view.id : null };
    }

    suggestionFor(card) {
      const guesses = this.guesses(card);
      const like = this.likeView(this.answeredLike(card));
      return {
        like: like ? { card: like.card, status: like.status, species: like.species, family: like.family, similarity: like.similarity } : null,
        model: guesses.length ? guesses[0] : null,
      };
    }

    // families.resolve_answer
    resolve(status, species, family) {
      if (status !== "species") return [null, null];
      const names = this.m.names || [];
      if (!names.length) return [species, family || species];
      const fam = this.m.name_family;
      if (species && species in fam) {
        if (family && family !== fam[species]) throw new HttpError(400, `${species} isn't in the family ${family}`);
        return [species, fam[species]];
      }
      if (species && !family && this.familyKeys.has(species)) family = species;
      if (species && species !== family) throw new HttpError(400, "Pick a species from the list");
      if (!family || !this.familyKeys.has(family)) throw new HttpError(400, "Pick a family from the list");
      return [(this.m.name_entries || {})[family] || family, family];
    }

    // ---- the page's requests ------------------------------------------------------------

    summary() {
      const counts = Object.fromEntries(STATUSES.map((s) => [s, 0]));
      for (const c of this.cards.values()) counts[c.status] += 1;
      return { ...this.m.summary, total: this.cards.size, reviewed: REVIEWED.reduce((n, s) => n + counts[s], 0), counts,
        outside_window: 0, hosted: true };
    }

    ordered() {
      const rows = [...this.cards.values()].map((c) => [c, this.span(c)[0]]);
      rows.sort((a, b) => a[1] - b[1] || a[0].id - b[0].id);
      return rows.map((r) => r[0]);
    }

    // lookalike.groups (Sort: look-alikes together): biggest fish first, each card joins the
    // first group leader it looks like or leads a new group. The server works them out again
    // after every fix; until then the published groups stand.
    groups() {
      if (![...this.cards.values()].some((c) => c.changed)) return null;
      const key = `groups:${this.version}`;
      if (!(key in this.cache)) {
        const size = new Map([...this.cards.values()].map((c) => [c.id, this.maxPx(c)]));
        const order = [...this.cards.values()].sort((a, b) => size.get(b.id) - size.get(a.id) || a.id - b.id);
        const leaders = [];
        const out = new Map();
        for (const card of order) {
          const look = this.look(card);
          if (!look) continue;
          let best = null;
          for (const lead of leaders) {
            const sim = dot(lead.look, look);
            if (!best || sim > best.sim) best = { id: lead.id, sim };
          }
          if (best && best.sim >= this.s.lookalike) out.set(card.id, best.id);
          else {
            leaders.push({ id: card.id, look });
            out.set(card.id, card.id);
          }
        }
        this.cache[key] = out;
      }
      return this.cache[key];
    }

    listFish() {
      const out = [];
      const groups = this.groups();
      for (const card of this.ordered()) {
        const base = card.base.list || {};
        const [start, end] = this.span(card);
        const [clip, local] = this.locate(start);
        const guesses = this.guesses(card);
        let like = null;
        if (OPEN.includes(card.status)) {
          const a = this.answeredLike(card);
          if (a) like = { card: a.card, sim: round(a.similarity, 3) };
        }
        out.push({
          ...base, id: card.id, start_t: start, end_t: end, clip, local_t: local, status: card.status, species: card.species,
          top: guesses.length ? guesses[0][0] : null, guesses: guesses.slice(0, 3).map((g) => g[0]),
          max_px: card.changed ? round(this.maxPx(card), 1) : base.max_px, group: groups ? groups.get(card.id) ?? null : card.group, like, outside: false,
        });
      }
      return out;
    }

    detail(id) {
      const card = this.get(id);
      let d;
      if (!card.changed) {
        d = { ...card.base.detail };
      } else {
        const dets = card.dets;
        const best = bestView(dets);
        const step = Math.max(1, Math.ceil(dets.length / this.s.filmstrip_max));
        const strip = [];
        for (let i = 0; i < dets.length; i += step) {
          const x = dets[i];
          const [clip, local] = this.locate(x.t);
          strip.push({ det_id: x.id, frame_id: x.frame, t: x.t, clip, local_t: local, tracklet: x.tracklet, box: box(x),
            covers: dets.slice(i, i + step).map((y) => y.id) });
        }
        const [start, end] = this.span(card);
        const [clipIn, tIn] = this.locate(start);
        const [clipOut, tOut] = this.locate(end);
        const parts = this.pieces(card);
        const calls = dets.filter((x) => x.margin !== null).sort((a, b) => a.margin - b.margin).slice(0, 3).map((x) => {
          const [clip, local] = this.locate(x.t);
          return { det_id: x.id, frame_id: x.frame, t: x.t, clip, local_t: local, box: box(x), margin: round(x.margin, 3),
            history: x.history === null ? null : round(x.history, 3) };
        });
        let shown = null;
        if (best) {
          const [bc, bl] = this.locate(best.t);
          shown = { det_id: best.id, frame_id: best.frame, t: best.t, clip: bc, local_t: bl, box: box(best) };
        }
        d = {
          id: card.id, guesses: this.guesses(card), clip_in: clipIn, time_in: tIn, clip_out: clipOut, time_out: tOut,
          n_detections: dets.length, tracklets: this.pieceList(card).map((p) => p.tracklet), best: shown, filmstrip: strip,
          pieces: parts.map((p) => {
            const [clip, local] = this.locate(p.start_t);
            const view = bestView(p.dets);
            return { tracklet: p.tracklet, clip, local_t: local, seconds: p.end_t - p.start_t, detections: p.detections,
              best_det: view ? view.id : p.best_det, guesses: p.guesses };
          }),
          flags: this.flags(card), closest_calls: calls, families: this.families(card),
        };
      }
      return {
        ...d, status: card.status, species: card.species, type: card.type, sex: card.sex, behavior: card.behavior,
        comment: card.comment, equipment: Boolean(this.lastDetails(card).equipment), family: this.answerFamily(card),
        outside: false, answered_like: this.likeView(this.answeredLike(card)), lookalikes: this.unansweredLike(card).length,
        earlier: null, rig: null,
      };
    }

    lookalikes(id, limit = 400) {
      const source = this.get(id);
      const ref = this.look(source);
      const cards = this.unansweredLike(source).slice(0, limit).map(([fid, sim]) => {
        const f = this.cards.get(fid);
        const [start, end] = this.span(f);
        const [clip, local] = this.locate(start);
        const parts = this.cardPieces(f, ref);
        return {
          id: fid, similarity: round(sim, 3), clip, local_t: local, seconds: end - start, n_pieces: parts.length,
          pieces: parts.slice(0, 3).map((p) => ({ det: p.det, similarity: p.similarity === null ? null : round(p.similarity, 3) })),
          mixed: !this.everyPieceAlike(parts), flagged: this.flags(f).length > 0,
        };
      });
      const view = bestView(source.dets);
      const [clip, local] = this.locate(this.span(source)[0]);
      return {
        source: { id: source.id, status: source.status, species: source.species, family: this.answerFamily(source),
          det: view ? view.id : null, clip, local_t: local },
        cards,
      };
    }

    // label: checks and the event to record (server.label); apply() makes it so.
    labelEvent(id, body) {
      const card = this.get(id);
      const status = body.status;
      const [species, family] = this.resolve(status, body.species, body.family);
      const flags = REVIEWED.includes(status) ? this.flags(card) : [];
      if (flags.length && !body.confirm) {
        throw new HttpError(409, "Check this card first: " + flags.map((x) => x.text).join(" ") +
          " Answer again to confirm it's one fish, or use Multiple fish.");
      }
      if (!STATUSES.includes(status)) throw new HttpError(400, `Bad status: ${status}`);
      if (status === "species" && !species) throw new HttpError(400, "A species is required");
      const details = flags.length ? { flags: flags.map((x) => x.kind), confirmed: true } : {};
      if (REVIEWED.includes(status)) {
        const offered = this.suggestionFor(card);
        details.suggestion = offered;
        details.matched = matched(offered.like, status, species, family);
        if (body.queue) details.queue = "quick";
      }
      if (body.seconds !== null && body.seconds !== undefined) details.seconds = round(Math.max(0, body.seconds), 1);
      return {
        kind: "label", card: card.id, status, species, family, type: body.type || null, sex: body.sex || null,
        behavior: body.behavior || null, comment: body.comment || null,
        details: Object.keys(details).length ? details : null, scope: status !== "pending" ? card.dets.map((x) => x.id) : [],
      };
    }

    equipmentEvent(id, body) {
      const card = this.get(id);
      const flags = this.flags(card);
      if (flags.length && !body.confirm) throw new HttpError(409, "Check this card first: " + flags.map((x) => x.text).join(" "));
      return { kind: "label", card: card.id, status: "not_fish", species: null, family: null, type: null, sex: null,
        behavior: null, comment: null, details: { equipment: true, rig: null }, scope: card.dets.map((x) => x.id) };
    }

    lookalikeEvents(body) {
      const source = this.get(body.source);
      if (!REVIEWED.includes(body.status)) throw new HttpError(400, "Pick an answer for them");
      const [species, family] = this.resolve(body.status, body.species, body.family);
      const alike = new Map(this.unansweredLike(source));
      const ref = this.look(source);
      const batch = utcStamp("like");
      const equipment = body.status === "not_fish" && Boolean(this.lastDetails(source).equipment);
      const ids = body.ids || [];
      const share = body.seconds !== null && body.seconds !== undefined ? round(body.seconds / Math.max(ids.length, 1), 1) : null;
      const events = [], answered = [], left = [];
      for (const fid of ids) {
        const f = this.cards.get(fid);
        if (!f || !alike.has(fid) || !this.everyPieceAlike(this.cardPieces(f, ref)) || this.flags(f).length) {
          left.push(fid);
          continue;
        }
        const details = { batch, like: source.id, similarity: round(alike.get(fid), 3) };
        if (share !== null) details.seconds = share;
        if (equipment) details.equipment = true;
        events.push({ kind: "label", card: fid, status: body.status, species: body.status === "species" ? species : null,
          family, type: null, sex: null, behavior: null, comment: null, details, scope: f.dets.map((x) => x.id) });
        answered.push(fid);
      }
      const ticked = new Set(ids), done = new Set(answered);
      const seen = [...new Set([...(body.shown || []), ...ids])];
      const rows = seen.filter((fid) => alike.has(fid)).map((fid) =>
        [fid, source.id, round(alike.get(fid), 3), done.has(fid) ? "answered" : ticked.has(fid) ? "left" : "unticked", batch]);
      if (rows.length) events.push({ kind: "suggestions", rows });
      return { events, result: { batch, answered, left } };
    }

    undoEvent(batch) {
      const ids = [];
      for (const c of this.cards.values()) {
        const d = this.lastDetails(c);
        if (d.batch === batch) ids.push(c.id);
      }
      return { kind: "undo", batch, cards: ids.sort((a, b) => a - b) };
    }

    // ---- fixes (store.split_fish / move_to_new_fish / unmerge_fish)

    splitEvent(id, fromT) {
      const card = this.get(id);
      if (!card.dets.some((d) => d.t < fromT) || !card.dets.some((d) => d.t >= fromT)) {
        throw new HttpError(400, "Split point must leave detections on both sides");
      }
      return { kind: "split", card: card.id, from_t: fromT, new_card: this.maxCard + 1 };
    }

    moveEvent(id, detIds) {
      const card = this.get(id);
      const mine = new Set(card.dets.map((d) => d.id));
      const chosen = [...new Set(detIds)].filter((d) => mine.has(d)).sort((a, b) => a - b);
      if (!chosen.length || chosen.length === mine.size) throw new HttpError(400, "Tick some of this card's sightings, but not all of them");
      return { kind: "move", card: card.id, detections: chosen, new_card: this.maxCard + 1 };
    }

    unmergeEvent(id) {
      const card = this.get(id);
      const parts = this.pieceList(card);
      if (parts.length < 2) {
        throw new HttpError(400, "This card is one tracked stretch: select the sighting where the other fish starts and split there");
      }
      return { kind: "unmerge", card: card.id, tracklets: parts.map((p) => p.tracklet),
        new_cards: parts.slice(1).map((_, i) => this.maxCard + 1 + i) };
    }

    // ---- following a fish (follow.py)

    setFollowFrames(frames) {
      this.frameIds = frames.ids;
      this.frameT = frames.t;
      this.frameIndex = new Map(frames.ids.map((id, i) => [id, i]));
    }

    async needFollow() {
      if (!this.followable) {
        throw new HttpError(400, "Following a fish isn't available for this video: it was published without --follow.");
      }
      if (!this.boxes) {
        const data = await this.boxLoader();
        const boxes = new Map(), byId = new Map();
        data.boxes.forEach((rows, i) => {
          const frame = this.frameIds[i];
          boxes.set(frame, rows.map((r) => {
            const b = { det: r[0], frame, x1: r[1], y1: r[2], x2: r[3], y2: r[4], score: r[5], card: r[6], faint: Boolean(r[7]) };
            byId.set(b.det, b);
            return b;
          }));
        });
        this.boxes = boxes;
        this.boxById = byId;
      }
    }

    knownFollow(id) {
      const f = this.follows.get(id);
      if (!f) throw new HttpError(404, "No such follow");
      return f;
    }

    // The card a box is on now: the volunteer's card after their fixes, else the published one.
    cardOfDet(det) {
      const key = `cardOf:${this.version}`;
      if (!(key in this.cache)) {
        const now = new Map();
        for (const c of this.cards.values()) for (const d of c.dets) now.set(d.id, c.id);
        this.cache[key] = now;
      }
      const now = this.cache[key].get(det);
      if (now !== undefined) return now;
      const b = this.boxById.get(det);
      return b ? b.card : null;
    }

    // follow._spot: the middle of a mark's box, or the spot clicked.
    spot(m) {
      if (!m) return null;
      if (m.state === "box") {
        const b = this.boxById.get(m.det);
        return b ? [(b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2] : null;
      }
      if (m.state === "nobox" && m.x !== null) return [m.x, m.y];
      return null;
    }

    // follow.next_step: forward from the start until the fish has been gone for gone_s, then
    // backward the same way; the first frame not yet marked is the next.
    followStep(f) {
      const ids = this.frameIds, T = this.frameT;
      const s = this.frameIndex.get(f.startFrame);
      for (const [phase, dir] of [["forward", 1], ["backward", -1]]) {
        let seen = T[s];
        let last = this.spot(f.marks.get(f.startFrame));
        for (let i = s + dir; i >= 0 && i < ids.length; i += dir) {
          if (Math.abs(T[i] - seen) > f.gone_s + 1e-6) break;
          const m = f.marks.get(ids[i]);
          if (!m) return { phase, frame: ids[i], last };
          if (m.state !== "gone") {
            seen = T[i];
            last = this.spot(m) || last;
          }
        }
      }
      return { phase: "done", frame: null, last: null };
    }

    // follow.frame_boxes
    frameBoxes(frame, followId) {
      const taken = new Map(); // det -> the latest other follow that clicked it
      for (const g of this.follows.values()) {
        const m = g.marks.get(frame);
        if (g.id === followId || !m || m.state !== "box") continue;
        const had = taken.get(m.det);
        if (!had || had.seq < m.seq) taken.set(m.det, { seq: m.seq, name: g.name });
      }
      return (this.boxes.get(frame) || []).map((b) => ({
        det: b.det, x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y2, score: b.score, card: this.cardOfDet(b.det), faint: b.faint,
        taken: taken.has(b.det) ? taken.get(b.det).name : null,
      }));
    }

    // follow.summary, against the cards as they are now
    followSummary(f, done) {
      const s = { id: f.id, name: f.name, card: f.card, done, gone_s: f.gone_s, boxes: 0, nobox: 0, gone: 0, first_t: null,
        last_t: null, cards: {}, faint: 0, breaks: [], shared: {}, others: {} };
      const marks = [...f.marks.entries()].filter(([frame]) => this.frameIndex.has(frame))
        .sort((a, b) => this.frameIndex.get(a[0]) - this.frameIndex.get(b[0]));
      let previous = null;
      for (const [frame, m] of marks) {
        const t = this.frameT[this.frameIndex.get(frame)];
        if (m.state === "gone") {
          s.gone += 1;
          continue;
        }
        if (s.first_t === null) s.first_t = t;
        s.last_t = t;
        if (m.state === "nobox") {
          s.nobox += 1;
          continue;
        }
        s.boxes += 1;
        const card = this.cardOfDet(m.det);
        if (card === null) {
          s.faint += 1;
          continue;
        }
        s.cards[card] = (s.cards[card] || 0) + 1;
        if (previous !== null && card !== previous) s.breaks.push([t, previous, card]);
        previous = card;
      }
      const mine = new Set(Object.keys(s.cards).map(Number));
      if (mine.size) {
        // Other follows (this volunteer's) with boxes on this follow's cards.
        for (const g of [...this.follows.values()].sort((a, b) => a.id - b.id)) {
          if (g.id === f.id) continue;
          const on = new Set();
          for (const m of g.marks.values()) {
            if (m.state !== "box") continue;
            const c = this.cardOfDet(m.det);
            if (mine.has(c)) on.add(c);
          }
          for (const c of on) (s.shared[c] = s.shared[c] || []).push(g.name);
        }
        // Frames this follow checked where one of its cards has a box that isn't the fish.
        const frames = new Map();
        for (const [frame, m] of f.marks) {
          for (const b of this.boxes.get(frame) || []) {
            const c = this.cardOfDet(b.det);
            if (b.faint || !mine.has(c) || (m.det !== null && b.det === m.det)) continue;
            if (!frames.has(c)) frames.set(c, new Set());
            frames.get(c).add(frame);
          }
        }
        for (const [c, set] of frames) s.others[c] = set.size;
      }
      return s;
    }

    // server.follow_view: the follow, what it has found, and the next frame with its boxes.
    followView(id) {
      const f = this.knownFollow(id);
      const step = this.followStep(f);
      const s = this.followSummary(f, step.phase === "done");
      const at = (t) => {
        const [clip, local] = this.locate(t);
        return { clip, local_t: round(local, 3) };
      };
      const where = (frame) => {
        const t = this.frameT[this.frameIndex.get(frame)];
        return { id: frame, t: round(t, 3), ...at(t) };
      };
      const b = this.boxById.get(f.startDet) || {};
      const out = {
        follow: s,
        seen: s.first_t === null ? null : { from: at(s.first_t), to: at(s.last_t) },
        breaks: s.breaks.map(([t, a, c]) => ({ ...at(t), from: a, to: c })),
        gone_s: f.gone_s,
        start: { ...where(f.startFrame), det: f.startDet, box: { x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y2 } },
        phase: step.phase, last: step.last, frame: null, boxes: [],
      };
      if (step.frame !== null) {
        out.frame = where(step.frame);
        out.boxes = this.frameBoxes(step.frame, f.id);
      }
      return out;
    }

    followList() {
      return [...this.follows.values()].sort((a, b) => a.id - b.id)
        .map((f) => this.followSummary(f, this.followStep(f).phase === "done"));
    }

    // follow.start, from the card's best view
    followStartEvent(body) {
      const card = this.cards.get(body.card);
      if (!card) throw new HttpError(404, "No such card");
      const best = bestView(card.dets);
      if (!best) throw new HttpError(400, `Card #${card.id} has no sightings`);
      const gone = Math.max(1, Math.min(body.gone_s ?? 5, 30));
      const id = this.followSeq + 1;
      return { kind: "follow_start", follow: id, card: card.id, det: best.id, frame: best.frame, gone_s: gone,
        name: `F${id}-v${this.volunteer}` };
    }

    // follow.mark's checks
    followMarkEvent(id, body) {
      this.knownFollow(id);
      const state = body.state;
      if (!["box", "nobox", "gone"].includes(state)) throw new HttpError(400, "A mark is one of box, nobox, gone");
      let det = null, x = null, y = null;
      if (state === "box") {
        const b = this.boxById.get(body.det_id);
        if (!b || b.frame !== body.frame_id) throw new HttpError(400, "That box isn't in this frame");
        det = b.det;
      } else if (state === "nobox") {
        x = body.x ?? null;
        y = body.y ?? null;
      }
      return { kind: "follow_mark", follow: id, frame: body.frame_id, state, det, x, y };
    }

    // The mark undo takes back: the latest, never the start's.
    latestMark(f) {
      let latest = null;
      for (const [frame, m] of f.marks) if (frame !== f.startFrame && (!latest || m.seq > latest.m.seq)) latest = { frame, m };
      return latest;
    }

    applyFollow(e) {
      if (e.kind === "follow_start") {
        const f = { id: e.follow, card: e.card, startDet: e.det, startFrame: e.frame, gone_s: e.gone_s, name: e.name, marks: new Map() };
        f.marks.set(e.frame, { state: "box", det: e.det, x: null, y: null, seq: ++this.markSeq });
        this.follows.set(f.id, f);
        this.followSeq = Math.max(this.followSeq, f.id);
        return;
      }
      const f = this.follows.get(e.follow);
      if (!f) return;
      if (e.kind === "follow_mark") {
        f.marks.delete(e.frame); // a mark replaces the frame's earlier one and is the latest
        f.marks.set(e.frame, { state: e.state, det: e.det ?? null, x: e.x ?? null, y: e.y ?? null, seq: ++this.markSeq });
      } else if (e.kind === "follow_undo") {
        const latest = this.latestMark(f);
        if (latest) f.marks.delete(latest.frame);
      } else if (e.kind === "follow_delete") {
        this.follows.delete(f.id);
      }
    }

    // ---- applying events (new answers, and the log replayed after a reload)

    apply(event) {
      if (event.kind === "label") return this.applyLabel(event);
      if (event.kind.startsWith("follow_")) return this.applyFollow(event);
      if (event.kind === "undo") {
        for (const id of event.cards) {
          if (this.cards.has(id)) this.applyLabel({ card: id, status: "pending", details: null, scope: [] });
        }
        return;
      }
      if (event.kind === "split") {
        const card = this.get(event.card);
        return this.moveDets(card, new Set(card.dets.filter((d) => d.t >= event.from_t).map((d) => d.id)), event.new_card);
      }
      if (event.kind === "move") return this.moveDets(this.get(event.card), new Set(event.detections), event.new_card);
      if (event.kind === "unmerge") {
        const card = this.get(event.card);
        const tracklets = this.pieceList(card).map((p) => p.tracklet);
        tracklets.slice(1).forEach((tid, i) => {
          const dets = card.dets.filter((d) => d.tracklet === tid);
          this.newCard(card, event.new_cards[i], dets, [tid]);
        });
        card.dets = card.dets.filter((d) => d.tracklet === tracklets[0]);
        for (const tid of [...card.tracklets.keys()]) if (tid !== tracklets[0]) card.tracklets.delete(tid);
        card.changed = true;
        card.status = "pending";
        card.species = null;
        this.version++;
        return;
      }
      // "suggestions", "feedback": logged only
    }

    applyLabel(e) {
      const card = this.get(e.card);
      card.status = e.status;
      card.species = e.status === "species" ? e.species : null;
      for (const k of ["type", "sex", "behavior", "comment"]) card[k] = e[k] || null;
      card.reviews.push({ status: e.status, species: card.species, family: e.family || null, details: e.details || null,
        scope: e.scope || [] });
    }

    // _move_detections: the chosen sightings go to a new card; a piece losing only some is cut.
    moveDets(card, chosen, newId) {
      const moved = [];
      for (const p of this.pieceList(card)) {
        const out = p.dets.filter((d) => chosen.has(d.id));
        if (!out.length) continue;
        if (out.length === p.dets.length) {
          moved.push(p.tracklet);
          continue;
        }
        const tid = ++this.maxTracklet;
        for (const d of out) d.tracklet = tid;
        card.tracklets.set(tid, null);
        moved.push(tid);
        // A cut piece's description is its embedded crops' mean (store._refresh_tracklet).
        for (const t of [p.tracklet, tid]) {
          const rows = card.dets.filter((d) => d.tracklet === t && d.emb >= 0).map((d) => this.cropRow(d.emb));
          card.tracklets.set(t, rows.length ? normalize(meanOf(rows, this.dim)) : null);
        }
      }
      const movedSet = new Set(moved);
      const dets = card.dets.filter((d) => movedSet.has(d.tracklet));
      card.dets = card.dets.filter((d) => !movedSet.has(d.tracklet));
      this.newCard(card, newId, dets, moved);
      for (const t of moved) card.tracklets.delete(t);
      card.changed = true;
      this.version++;
    }

    newCard(parent, id, dets, tracklets) {
      const card = {
        id, base: { list: null, detail: null, neighbours: [] }, dets, tracklets: new Map(), changed: true, group: parent.group,
        look: null, status: "pending", species: null, type: null, sex: null, behavior: null, comment: null, reviews: [],
      };
      for (const t of tracklets) card.tracklets.set(t, parent.tracklets.get(t) || null);
      this.cards.set(id, card);
      this.maxCard = Math.max(this.maxCard, id);
      this.version++;
      return card;
    }
  }
  B.Video = Video;

  function matched(suggested, status, species, family) {
    if (!suggested) return null;
    if (suggested.status !== status) return "different";
    if (status !== "species" || suggested.species === species) return "same";
    return suggested.family && suggested.family === family ? "same family" : "different";
  }

  // --- the page's requests, routed -------------------------------------------------------
  //
  // handle(video, sync, method, path, body) -> [status, json]. `sync.send(events)` stores
  // events in the site's database before they're applied, so nothing is shown as answered
  // that wasn't saved.

  // The same checks the server's request models make (server.py LabelIn, UndoIn, ...).
  const isText = (v) => typeof v === "string" && v.length > 0;
  const isIds = (v) => Array.isArray(v) && v.every(Number.isInteger);
  const isNumberOrNull = (v) => v === undefined || v === null || Number.isFinite(v);
  const isIdOrNull = (v) => v === undefined || v === null || Number.isInteger(v);
  function need(body, name, ok) {
    if (!body || !ok(body[name])) throw new HttpError(422, `Request is missing a valid ${name}`);
    return body[name];
  }

  // Following a fish (server.py's /api/follows): each change is saved, then applied.
  async function handleFollow(video, sync, method, path, body) {
    await video.needFollow();
    const save = async (e) => {
      await sync.send([e]);
      video.apply(e);
      return e;
    };
    let r;
    if (path === "/api/follows" && method === "GET") return [200, video.followList()];
    if (path === "/api/follows" && method === "POST") {
      need(body, "card", Number.isInteger);
      need(body, "gone_s", isNumberOrNull);
      const e = await save(video.followStartEvent(body));
      return [200, video.followView(e.follow)];
    }
    if ((r = path.match(/^\/api\/follows\/(\d+)$/))) {
      const id = Number(r[1]);
      if (method === "GET") return [200, video.followView(id)];
      if (method === "DELETE") {
        video.knownFollow(id);
        await save({ kind: "follow_delete", follow: id });
        return [200, { ok: true }];
      }
    }
    if (method === "POST" && (r = path.match(/^\/api\/follows\/(\d+)\/mark$/))) {
      need(body, "frame_id", Number.isInteger);
      need(body, "state", (v) => typeof v === "string");
      need(body, "det_id", isIdOrNull);
      need(body, "x", isNumberOrNull);
      need(body, "y", isNumberOrNull);
      const id = Number(r[1]);
      await save(video.followMarkEvent(id, body));
      return [200, video.followView(id)];
    }
    if (method === "POST" && (r = path.match(/^\/api\/follows\/(\d+)\/undo$/))) {
      const id = Number(r[1]);
      const f = video.knownFollow(id);
      if (video.latestMark(f)) await save({ kind: "follow_undo", follow: id });
      return [200, video.followView(id)];
    }
    return [404, { detail: "Not Found" }];
  }

  async function handle(video, sync, method, path, query, body) {
    const m = (re) => path.match(re);
    let r;
    try {
      if (method === "POST") {
        if (/\/label$/.test(path)) need(body, "status", isText);
        if (path === "/api/batch/undo" || path === "/api/rig/undo") need(body, "batch", isText);
        if (/\/split$/.test(path)) need(body, "from_t", Number.isFinite);
        if (/\/move$/.test(path)) need(body, "det_ids", isIds);
        if (path === "/api/lookalikes/answer") {
          need(body, "source", Number.isInteger);
          need(body, "ids", isIds);
          need(body, "status", isText);
        }
      }
      if (path === "/api/follows" || path.startsWith("/api/follows/")) return await handleFollow(video, sync, method, path, body);
      if (method === "GET" && path === "/api/summary") return [200, video.summary()];
      if (method === "GET" && path === "/api/fish") return [200, video.listFish()];
      if (method === "GET" && (r = m(/^\/api\/fish\/(\d+)$/))) return [200, video.detail(Number(r[1]))];
      if (method === "GET" && (r = m(/^\/api\/fish\/(\d+)\/lookalikes$/))) {
        return [200, video.lookalikes(Number(r[1]), Number(query.get("limit") || 400))];
      }
      if (method === "GET" && path === "/api/rig") return [200, { refs: [], candidates: {} }];
      if (method === "GET" && path === "/api/still") {
        const s = video.m.still || { count: 0, sample: [] };
        return [200, { count: s.count, sample: s.sample }];
      }
      if (method === "POST" && (r = m(/^\/api\/fish\/(\d+)\/label$/))) {
        const e = video.labelEvent(Number(r[1]), body);
        await sync.send([e]);
        video.apply(e);
        return [200, { ok: true }];
      }
      if (method === "POST" && (r = m(/^\/api\/fish\/(\d+)\/equipment$/))) {
        const e = video.equipmentEvent(Number(r[1]), body);
        await sync.send([e]);
        video.apply(e);
        return [200, { ref: null }];
      }
      if (method === "POST" && path === "/api/lookalikes/answer") {
        const { events, result } = video.lookalikeEvents(body);
        if (events.length) await sync.send(events);
        events.forEach((e) => video.apply(e));
        return [200, result];
      }
      if (method === "POST" && (path === "/api/batch/undo" || path === "/api/rig/undo")) {
        const e = video.undoEvent(body.batch);
        await sync.send([e]);
        video.apply(e);
        return [200, { undone: e.cards }];
      }
      if (method === "POST" && path === "/api/rig/mark") return [200, { batch: utcStamp("rig"), marked: [], left: [] }];
      if (method === "POST" && (r = m(/^\/api\/rig\/(\d+)\/forget$/))) return [200, { ok: true }];
      if (method === "POST" && (r = m(/^\/api\/fish\/(\d+)\/split$/))) {
        const e = video.splitEvent(Number(r[1]), body.from_t);
        await sync.send([e]);
        video.apply(e);
        return [200, { ok: true, new_id: e.new_card }];
      }
      if (method === "POST" && (r = m(/^\/api\/fish\/(\d+)\/move$/))) {
        const e = video.moveEvent(Number(r[1]), body.det_ids || []);
        await sync.send([e]);
        video.apply(e);
        return [200, { ok: true, new_id: e.new_card }];
      }
      if (method === "POST" && (r = m(/^\/api\/fish\/(\d+)\/unmerge$/))) {
        const e = video.unmergeEvent(Number(r[1]));
        await sync.send([e]);
        video.apply(e);
        return [200, { ok: true, new_ids: e.new_cards }];
      }
      if (method === "POST" && (r = m(/^\/api\/fish\/(\d+)\/merge$/))) {
        throw new HttpError(400, "Joining two cards isn't available online yet. Note the card numbers in a comment instead.");
      }
      if (method === "POST" && path === "/api/export") {
        throw new HttpError(400, "On the website, answers are saved as you go; the coordinator makes the Excel export.");
      }
      return [404, { detail: "Not Found" }];
    } catch (err) {
      if (err instanceof HttpError) return [err.status, { detail: err.detail }];
      return [500, { detail: String((err && err.message) || err) }];
    }
  }
  B.handle = handle;
})(BRUV);
if (typeof self !== "undefined") self.BRUV = BRUV;
