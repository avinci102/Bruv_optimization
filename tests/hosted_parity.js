/* Drives hosted/core.js in Node for tests/test_hosted_js.py: answers and fixes on a
 * published video, made through BRUV.handle as the review page makes them, then everything
 * the page can ask for afterwards. The test pulls the saved events into the project and
 * compares the Python review server's answers with these.
 *
 *   node hosted_parity.js <core.js> <published video folder> <video key> <out.json>
 */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const [, , coreJs, folder, key, outPath] = process.argv;
vm.runInThisContext(fs.readFileSync(coreJs, "utf8"), { filename: coreJs });
const read = (name) => JSON.parse(fs.readFileSync(path.join(folder, name), "utf8"));
const manifest = read("manifest.json");
const cardsJson = read("cards.json");
const raw = fs.readFileSync(path.join(folder, "emb.bin"));
const emb = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
const fresh = () => new BRUV.Video(key, manifest, cardsJson, emb);

// The site's database: it keeps each event with its own clock.
const saved = [];
const sync = {
  async send(events) {
    for (const e of events) saved.push({ ...e, at: `2026-11-01T10:00:${String(saved.length).padStart(2, "0")}+00:00` });
    return { ok: true, saved: events.length };
  },
};

async function call(video, method, p, body) {
  const url = new URL(p, "http://site");
  const [status, data] = await BRUV.handle(video, sync, method, url.pathname, url.searchParams, body === undefined ? null : body);
  return { status, data };
}

async function snapshot(video) {
  const list = (await call(video, "GET", "/api/fish")).data;
  const out = { summary: (await call(video, "GET", "/api/summary")).data, list, details: {}, looks: {} };
  for (const c of list) {
    out.details[c.id] = (await call(video, "GET", `/api/fish/${c.id}`)).data;
    out.looks[c.id] = (await call(video, "GET", `/api/fish/${c.id}/lookalikes`)).data;
  }
  return out;
}

async function main() {
  const v = fresh();
  const steps = {};
  const step = async (name, method, p, body) => {
    steps[name] = await call(v, method, p, body);
    return steps[name];
  };
  const detail = async (id) => (await call(v, "GET", `/api/fish/${id}`)).data;
  const ids = (await call(v, "GET", "/api/fish")).data.map((c) => c.id);
  const details = {};
  for (const id of ids) details[id] = await detail(id);
  const times = (d) => [...new Set(d.filmstrip.map((s) => s.t))].sort((a, b) => a - b);

  // An answer with the model's first guess; a card with warnings needs confirming first.
  const first = ids.find((id) => details[id].flags.length && details[id].guesses.length)
    || ids.find((id) => details[id].guesses.length) || ids[0];
  const top = details[first].guesses.length ? details[first].guesses[0][0] : null;
  const answer = { status: "species", species: top, type: "individual", seconds: 4.2 };
  if ((await step("label", "POST", `/api/fish/${first}/label`, answer)).status === 409) {
    await step("label confirmed", "POST", `/api/fish/${first}/label`, { ...answer, confirm: true });
  }
  // Its look-alikes, answered the same way as a batch, and the batch undone.
  const looks = (await call(v, "GET", `/api/fish/${first}/lookalikes`)).data;
  const offered = looks.cards.map((c) => c.id);
  const batch = await step("lookalikes", "POST", "/api/lookalikes/answer",
    { source: first, ids: offered.slice(0, 3), status: "species", species: top, seconds: 3, shown: offered });
  if (batch.status === 200 && batch.data.batch) await step("undo", "POST", "/api/batch/undo", { batch: batch.data.batch });
  // A second look-alike batch, kept.
  if (offered.length > 3) {
    await step("lookalikes kept", "POST", "/api/lookalikes/answer",
      { source: first, ids: offered.slice(3, 5), status: "species", species: top, seconds: 2, shown: offered.slice(3) });
  }
  // Fixes: "multiple fish" (the answered card, if it has pieces), a split, a split of the
  // split-off card, and a move of ticked sightings.
  const others = ids.filter((id) => id !== first);
  const mixed = details[first].pieces.length >= 2 ? first : others.find((id) => details[id].pieces.length >= 2);
  if (mixed) await step("multiple fish", "POST", `/api/fish/${mixed}/unmerge`, {});
  const splittable = others.find((id) => id !== mixed && times(details[id]).length >= 3);
  if (splittable) {
    const ts = times(details[splittable]);
    const res = await step("split", "POST", `/api/fish/${splittable}/split`, { from_t: ts[Math.floor(ts.length / 2)] });
    const made = await detail(res.data.new_id);
    const ts2 = times(made);
    if (ts2.length >= 2) await step("split again", "POST", `/api/fish/${res.data.new_id}/split`, { from_t: ts2[1] });
  }
  const movable = others.find((id) => ![mixed, splittable].includes(id) && details[id].filmstrip.length >= 4);
  if (movable) {
    const strip = details[movable].filmstrip;
    await step("move", "POST", `/api/fish/${movable}/move`, { det_ids: strip.slice(0, 2).map((s) => s.det_id) });
  }
  // Other answers, an equipment answer, and requests the site turns down.
  const rest = others.filter((id) => ![splittable, movable, mixed].includes(id));
  if (rest[0]) await step("unknown", "POST", `/api/fish/${rest[0]}/label`, { status: "unknown", comment: "too dark", confirm: true });
  if (rest[1]) await step("not fish", "POST", `/api/fish/${rest[1]}/label`, { status: "not_fish", confirm: true });
  if (rest[2]) await step("equipment", "POST", `/api/fish/${rest[2]}/equipment`, { confirm: true });
  if (rest[3]) await step("skip", "POST", `/api/fish/${rest[3]}/label`, { status: "skipped" });
  await step("undo without a batch", "POST", "/api/batch/undo", {});
  await step("join", "POST", `/api/fish/${others[0]}/merge`, { other_id: first });
  await step("unknown card", "GET", "/api/fish/999999");

  const live = await snapshot(v);
  // The page opened again: the published video with the saved events replayed on top.
  const again = fresh();
  for (const e of saved) again.apply(e);
  const replayed = await snapshot(again);
  fs.writeFileSync(outPath, JSON.stringify({
    steps, events: saved, live, replayed_same: JSON.stringify(live) === JSON.stringify(replayed),
  }));
}

main().catch((err) => {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
