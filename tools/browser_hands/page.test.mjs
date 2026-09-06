// Execute the production page script with explicit DOM/Worker/fetch test doubles.
// These are lifecycle tests, not browser/GPU performance measurements.
import test from "node:test";
import assert from "node:assert/strict";
import vm from "node:vm";
import {readFileSync} from "node:fs";
const source = readFileSync(new URL("./page.mjs", import.meta.url), "utf8");
const flush = () => new Promise(resolve => setImmediate(resolve));
function harness() {
  const requests = [], timers = new Map(), events = {}, elements = {};
  let peer, time = 0, held = false, unblock;
  class Worker {
    constructor() { peer = this; this.terminated = 0; }
    postMessage(value) { this.initial = value; }
    terminate() { this.terminated++; }
  }
  const context = {Worker, location: {hash: "#test-token", pathname: "/"},
    history: {replaceState() {}}, crypto: {randomUUID: () => "test-client"},
    performance: {now: () => time},
    document: {visibilityState: "visible", getElementById: k => (elements[k] ??= {}),
      addEventListener: (k, f) => { events[k] = f; }},
    window: {addEventListener: (k, f) => { events[k] = f; }},
    setInterval: f => { timers.set(1, f); return 1; }, clearInterval: id => timers.delete(id),
    fetch: (path, options) => {
      requests.push({path, body: JSON.parse(options.body)});
      if (held && path === "/rpc/progress") return new Promise(resolve => { unblock = resolve; });
      return Promise.resolve({ok: true});
    }};
  vm.runInNewContext(source, context, {filename: "production-page.mjs"});
  return {requests, timers, events, elements, context, peer,
    advance: n => { time += n; }, hold: () => { held = true; },
    release: () => { held = false; unblock({ok: true}); }};
}
test("page heartbeat reports last worker phase without waiting for the worker", async () => {
  const h = harness();
  h.peer.onmessage({data: {status: "connected"}});
  await flush();
  h.peer.onmessage({data: {id: 7, phase: "detect"}});
  h.advance(3000);
  h.timers.get(1)();
  await flush();
  const r = h.requests.filter(x => x.path === "/rpc/progress").at(-1).body;
  assert.equal(r.worker.id, 7); assert.equal(r.worker.phase, "detect");
  assert.equal(r.phase_age_ms, 3000); assert.equal(r.visible, true);
  assert.equal(h.peer.terminated, 0);
});
test("one heartbeat remains outstanding during a slow network", async () => {
  const h = harness(); h.hold();
  h.peer.onmessage({data: {status: "connected"}});
  h.timers.get(1)(); h.timers.get(1)();
  assert.equal(h.requests.filter(x => x.path === "/rpc/progress").length, 1);
  h.release(); await flush(); h.timers.get(1)(); await flush();
  assert.equal(h.requests.filter(x => x.path === "/rpc/progress").length, 2);
});
test("worker errors notify the server once and stop without restart", async () => {
  const h = harness(); h.peer.onmessage({data: {status: "connected"}}); await flush();
  h.peer.onerror({message: "private token must not be sent"});
  h.peer.onerror({}); await flush();
  const faults = h.requests.filter(x => x.path === "/rpc/fault");
  assert.equal(faults.length, 1); assert.equal(faults[0].body.code, "browser_worker_error");
  assert.equal(h.peer.terminated, 1); assert.equal(h.timers.size, 0);
  assert.ok(!JSON.stringify(faults).includes("private token"));
});
test("intentional stop does not become a page-close failure", async () => {
  const h = harness(); h.peer.onmessage({data: {status: "connected"}}); await flush();
  h.peer.onmessage({data: {status: "stopped"}}); h.events.pagehide(); await flush();
  assert.equal(h.requests.filter(x => x.path === "/rpc/fault").length, 0);
  assert.equal(h.timers.size, 0);
});
