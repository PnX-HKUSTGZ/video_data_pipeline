"use strict";

// Keep speculative work small: one background decode, two requests in total, four images.
// The key includes the server's file version, never an annotation revision.
class ImageLoader {
  constructor(byteLimit = 96 * 1024 * 1024, entryLimit = 4) {
    this.byteLimit = byteLimit; this.entryLimit = entryLimit;
    this.entries = new Map(); this.queue = []; this.active = 0; this.bytes = 0;
  }
  key(endpoint, sample) {
    const query = new URLSearchParams();
    if (sample.image_version) query.set("version", sample.image_version);
    if (sample.key) query.set("sample_key", sample.key);
    return `${endpoint}/samples/${sample.id}/image${query.size ? `?${query}` : ""}`;
  }
  acquire(endpoint, sample, priority = "foreground") {
    const key = this.key(endpoint, sample);
    if (priority === "foreground") this.cancelPending(key);
    const existing = this.entries.get(key);
    if (existing) {
      this.entries.delete(key); this.entries.set(key, existing);
      if (priority === "foreground") existing.priority = priority;
      this.pump(); return existing.promise;
    }
    const entry = {key, priority, controller: new AbortController(), image: null, bytes: 0};
    entry.promise = new Promise((resolve, reject) => { entry.resolve = resolve; entry.reject = reject; });
    this.entries.set(key, entry); this.queue.push(entry); this.pump();
    return entry.promise;
  }
  pump() {
    this.queue.sort((a, b) => Number(a.priority !== "foreground") - Number(b.priority !== "foreground"));
    while (this.active < 2 && this.queue.length) {
      const entry = this.queue[0];
      if (entry.priority === "prefetch" && this.active > 0) return;
      this.queue.shift(); this.active++;
      this.load(entry).then(entry.resolve, entry.reject).finally(() => { this.active--; this.pump(); });
    }
  }
  async load(entry) {
    let objectURL;
    try {
      const response = await fetch(entry.key, {signal: entry.controller.signal, priority: entry.priority === "foreground" ? "high" : "low"});
      if (!response.ok) {
        let detail;
        try { detail = (await response.json()).detail; } catch { detail = response.statusText; }
        throw new Error(typeof detail === "string" ? detail : "图片无法加载");
      }
      const blob = await response.blob();
      entry.controller.signal.throwIfAborted();
      const image = new Image(); image.decoding = "async";
      objectURL = URL.createObjectURL(blob); image.src = objectURL;
      await image.decode(); entry.controller.signal.throwIfAborted();
      entry.image = image; entry.bytes = blob.size + image.naturalWidth * image.naturalHeight * 4;
      if (this.entries.get(entry.key) === entry) {
        this.bytes += entry.bytes; this.trim();
      }
      return image;
    } catch (error) {
      if (this.entries.get(entry.key) === entry) this.entries.delete(entry.key);
      throw error;
    } finally { if (objectURL) URL.revokeObjectURL(objectURL); }
  }
  trim() {
    for (const [key, entry] of this.entries) {
      const completed = [...this.entries.values()].filter((value) => value.image).length;
      if (this.bytes <= this.byteLimit && completed <= this.entryLimit) break;
      if (entry.image) { this.entries.delete(key); this.bytes -= entry.bytes; }
    }
  }
  cancelPending(except = null, prefetchOnly = false) {
    for (const [key, entry] of this.entries) {
      if (key === except || entry.image || prefetchOnly && entry.priority !== "prefetch") continue;
      this.entries.delete(key); entry.controller.abort();
      if (this.queue.includes(entry)) {
        this.queue = this.queue.filter((value) => value !== entry);
        entry.reject(new DOMException("请求已取消", "AbortError"));
      }
    }
  }
  clear() {
    this.cancelPending(); this.entries.clear(); this.bytes = 0;
  }
  stats() {
    return {bytes: this.bytes, cached: [...this.entries.values()].filter((entry) => entry.image).length,
      active: this.active, queued: this.queue.length};
  }
}

const loading = {
  images: new ImageLoader(), detailAbort: null, refreshAbort: null, neighborAbort: null,
  neighborTimer: null,
  stopNeighbors() {
    clearTimeout(this.neighborTimer); this.neighborTimer = null;
    this.neighborAbort?.abort(); this.neighborAbort = null;
    this.images.cancelPending(null, true);
  },
  stopDetail() {
    this.detailAbort?.abort(); this.detailAbort = null;
    this.stopNeighbors(); this.images.cancelPending();
  },
  reset() {
    this.stopDetail(); this.refreshAbort?.abort(); this.refreshAbort = null;
    this.images.clear();
  },
};
window.addEventListener("pagehide", () => loading.reset());
