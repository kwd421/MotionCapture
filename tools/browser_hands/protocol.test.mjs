import test from "node:test";
import assert from "node:assert/strict";
import {unpackMessage, rgbToRgba, checkGpuRenderer} from "./protocol.mjs";
function message(meta, pixels = []) {
  const json = new TextEncoder().encode(JSON.stringify(meta));
  const out = new Uint8Array(4 + json.length + pixels.length);
  new DataView(out.buffer).setUint32(0, json.length, true);
  out.set(json, 4); out.set(pixels, 4+json.length); return out.buffer;
}
test("source timestamp and original RGB bytes survive framing", () => {
  const {meta,pixels} = unpackMessage(message({id:1,op:"detect",timestamp_ms:33,width:2,height:1},
    [0,127,255,9,23,42]));
  assert.equal(meta.timestamp_ms, 33);
  assert.deepEqual([...pixels],[0,127,255,9,23,42]);
});
test("RGB conversion is exact, current-frame only, with alpha 255", () => {
  const src = new Uint8Array([0,127,255,9,23,42]);
  const first = rgbToRgba(src,2,1);
  assert.deepEqual([...first],[0,127,255,255,9,23,42,255]);
  const second = rgbToRgba(new Uint8Array(6).fill(1),2,1,first);
  assert.equal(second,first);
  assert.deepEqual([...second],[1,1,1,255,1,1,1,255]);
  assert.deepEqual([...src],[0,127,255,9,23,42]);
});
test("bad metadata never makes a plausible frame", () => {
  for(const meta of [{id:1,op:"detect",timestamp_ms:0,width:1,height:1},
    {id:1,op:"detect",timestamp_ms:-1,width:0,height:0},{id:1,op:"invalid"},{id:-1,op:"close"}])
    assert.throws(()=>unpackMessage(message(meta)));
  assert.throws(()=>unpackMessage(new ArrayBuffer(2)));
});
test("software and unidentified renderers are not hardware benchmarks", () => {
  for(const r of [null,"","unknown","WebKit WebGL","ANGLE (SwiftShader)","llvmpipe"])
    assert.throws(()=>checkGpuRenderer(r));
  assert.equal(checkGpuRenderer("ANGLE (Apple, Apple M5, Metal)"),"ANGLE (Apple, Apple M5, Metal)");
});
