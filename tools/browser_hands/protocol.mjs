// Lossless, bounded single-frame transport. No camera, codec or inference here.
export function unpackMessage(buffer) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 4) throw Error("bad_frame_header");
  const length = new DataView(buffer).getUint32(0, true);
  if (length < 2 || length > 16384 || length + 4 > buffer.byteLength) throw Error("bad_metadata_size");
  const meta = JSON.parse(new TextDecoder("utf-8", {fatal: true}).decode(
    new Uint8Array(buffer, 4, length)));
  if (!Number.isSafeInteger(meta.id) || meta.id <= 0 || !["open", "detect", "close"].includes(meta.op))
    throw Error("bad_request");
  const pixels = new Uint8Array(buffer, length + 4);
  if (meta.op === "detect") {
    const {width, height, timestamp_ms} = meta;
    if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width <= 0 || height <= 0 ||
        width * height > 4096 * 4096 || pixels.length !== width * height * 3 ||
        !Number.isSafeInteger(timestamp_ms) || timestamp_ms < 0) throw Error("bad_image_or_time");
  } else if (pixels.length !== 0) throw Error("unexpected_pixels");
  return {meta, pixels};
}

export function rgbToRgba(rgb, width, height, previous) {
  if (!(rgb instanceof Uint8Array) || rgb.length !== width * height * 3) throw Error("bad_rgb");
  const size = width * height * 4;
  const rgba = previous?.length === size ? previous : new Uint8ClampedArray(size);
  for (let s = 0, d = 0; s < rgb.length; s += 3, d += 4) {
    rgba[d] = rgb[s]; rgba[d+1] = rgb[s+1]; rgba[d+2] = rgb[s+2]; rgba[d+3] = 255;
  }
  return rgba;
}

export function checkGpuRenderer(renderer) {
  if (typeof renderer !== "string" || !renderer.trim() ||
      /swiftshader|llvmpipe|softpipe|lavapipe|software|unknown/i.test(renderer) ||
      /^(WebKit WebGL|WebGL)$/i.test(renderer.trim())) throw Error("unverified_or_software_gpu");
  return renderer;
}
