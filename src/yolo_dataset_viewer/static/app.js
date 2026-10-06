"use strict";

const $ = (id) => document.getElementById(id);
const labels = {valid: "有效标签", empty: "空标签", missing: "缺失标签", invalid: "异常标签"};
const tasks = {detect: "Detection", segment: "Segmentation", obb: "OBB", pose: "Pose"};
const colors = ["#198268", "#db9241", "#5779bc", "#be6579", "#7d6ab0", "#3e98a1", "#a98436"];
const color = (id) => colors[id % colors.length];
const state = {session: null, names: {}, offset: 0, limit: 48, total: 0, items: [],
  openVersion: 0, viewVersion: 0, detailVersion: 0, request: null, position: 0, detail: null};
const canvas = $("image-canvas");
const ctx = canvas.getContext("2d");
const view = {image: null, scale: 1, x: 0, y: 0, selected: -1, width: 0, height: 0};
const thumbnails = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    if (!entry.isIntersecting) continue;
    const img = entry.target;
    if (img.dataset.src) img.src = img.dataset.src;
    thumbnails.unobserve(img);
  }
}, {rootMargin: "320px"});

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = String(text);
  return node;
}

function notice(text = "") { $("notice").textContent = text; $("notice").hidden = !text; }
function message(text) { $("image-message").textContent = text; }
function report(error) { notice(error.message || String(error)); }
async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail;
    try { detail = (await response.json()).detail; } catch { detail = response.statusText; }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return response.json();
}
const base = () => `/api/datasets/${state.session}`;
function filters() {
  const params = new URLSearchParams();
  for (const [key, id] of [["split", "split-filter"], ["class_id", "class-filter"], ["state", "state-filter"], ["edit_state", "edit-filter"], ["search", "search"]]) {
    const value = $(id).value.trim();
    if (value) params.set(key, value);
  }
  return params;
}

function enableFilters(enabled) {
  for (const id of ["split-filter", "class-filter", "state-filter", "search"]) $(id).disabled = !enabled;
}

function resetGallery() {
  thumbnails.disconnect();
  state.items = []; state.offset = 0; state.total = 0;
  $("gallery").replaceChildren(); $("pagination").hidden = true;
  $("empty-state").hidden = false;
  $("empty-state").querySelector("h3").textContent = "正在读取数据集";
  $("empty-state").querySelector("p").textContent = "索引完成后显示图片与统计";
  $("result-count").textContent = "等待索引";
  for (const id of ["metric-images", "metric-instances", "metric-empty", "metric-invalid"]) $(id).textContent = "—";
  $("class-stats").replaceChildren(); $("split-stats").replaceChildren();
}

async function openDataset(request, force = false) {
  if (!(await editorBeforeNavigation())) return;
  loading.reset(); editorReset();
  const version = ++state.openVersion;
  ++state.viewVersion; ++state.detailVersion;
  if ($("image-dialog").open) $("image-dialog").close();
  view.image = null; state.detail = null; state.session = null;
  state.request = {...request};
  $("dataset-path").value = request.path; $("task").value = request.task;
  $("root-path").value = request.root || "";
  $("task").dispatchEvent(new Event("change"));
  notice(); resetGallery(); enableFilters(false);
  $("rescan").disabled = true;
  $("candidates").hidden = true; $("index-progress").hidden = false;
  $("index-progress").removeAttribute("value");
  $("status-text").textContent = "正在打开…"; $("footer-state").textContent = "建立索引中";
  try {
    let result = null;
    if (!force) {
      try {
        const previous = JSON.parse(localStorage.getItem("yolo-viewer-session") || "null");
        if (previous && JSON.stringify(previous.request) === JSON.stringify(request)) result = await api(`/api/datasets/${previous.id}`);
      } catch { /* A stopped server or replaced session is reopened below. */ }
    }
    if (!result) result = await api("/api/datasets", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(request)});
    if (version !== state.openVersion) return;
    state.session = result.id;
    localStorage.setItem("yolo-viewer-session", JSON.stringify({id: result.id, request}));
    while (result.state === "indexing") {
      $("status-text").textContent = `${result.phase} · ${result.scanned}${result.total ? ` / ${result.total}` : ""}`;
      if (result.total) { $("index-progress").max = result.total; $("index-progress").value = result.scanned; }
      await new Promise((resolve) => setTimeout(resolve, 250));
      if (version !== state.openVersion) return;
      result = await api(`${base()}`);
      if (version !== state.openVersion) return;
    }
    $("index-progress").hidden = true;
    if (result.state === "error") {
      if (result.recovery_available) await editorRecoveryAvailable();
      if (result.candidates.length) {
        $("yaml-choice").replaceChildren(...result.candidates.map((path) => new Option(path, path)));
        $("candidates").hidden = false;
      }
      throw new Error(result.error);
    }
    state.names = result.names;
    $("split-filter").replaceChildren(new Option("全部划分", ""), ...result.splits.map((s) => new Option(s, s)));
    $("class-filter").replaceChildren(new Option("全部类别", ""), ...Object.entries(result.names).map(([id, name]) => new Option(`${id} · ${name}`, id)));
    $("state-filter").value = ""; $("search").value = "";
    $("status-text").textContent = `索引完成 · ${result.total.toLocaleString()} 张图片`;
    $("footer-state").textContent = "索引完成";
    $("dataset-caption").textContent = request.path;
    $("task-badge").textContent = tasks[result.task]; $("task-badge").hidden = false;
    $("rescan").disabled = false; enableFilters(true);
    notice(result.warnings.join("\n"));
    await editorOpened(result);
    await refresh();
  } catch (error) {
    if (version !== state.openVersion) return;
    report(error); $("status-text").textContent = "打开失败，请检查路径或标注设置";
    $("footer-state").textContent = "打开失败"; $("index-progress").hidden = true;
    $("rescan").disabled = !state.request;
    $("empty-state").querySelector("h3").textContent = "暂时无法打开";
    $("empty-state").querySelector("p").textContent = "根据上方提示调整路径、配置或任务类型";
  }
}

function renderStats(stats) {
  $("metric-images").textContent = stats.images.toLocaleString();
  $("unique-images").textContent = `${stats.unique_images.toLocaleString()} 张唯一图片`;
  $("metric-instances").textContent = stats.instances.toLocaleString();
  $("metric-empty").textContent = `${stats.states.empty} / ${stats.states.missing}`;
  $("metric-invalid").textContent = stats.states.invalid.toLocaleString();
  $("invalid-rows").textContent = `${stats.invalid_rows} 条标注诊断`;
  const maximum = Math.max(1, ...stats.classes.map((c) => c.instances));
  $("class-stats").replaceChildren(...stats.classes.map((c) => {
    const row = element("div", "class-row"), top = element("div", "class-row-top");
    const name = element("div", "class-name");
    const dot = element("i", "class-dot"); dot.style.background = color(c.id);
    name.append(dot, element("span", "", `${c.id} · ${c.name}`)); name.title = c.name;
    top.append(name, element("span", "class-count", `${c.instances} / ${c.images}`));
    const track = element("div", "class-track"), fill = element("div", "class-fill");
    fill.style.width = `${100 * c.instances / maximum}%`; fill.style.background = color(c.id);
    track.append(fill); row.append(top, track); return row;
  }));
  if (!stats.classes.length) $("class-stats").append(element("p", "muted", "暂无类别"));
  $("split-stats").replaceChildren(...Object.entries(stats.splits).map(([split, count]) => {
    const row = element("div", "split-row"); row.append(element("span", "", split), element("span", "", count.toLocaleString())); return row;
  }));
}

// Preserve decoded thumbnails and focus when edits change only counters or labels.
function renderGallery(items, endpoint) {
  const gallery = $("gallery"), existing = new Map([...gallery.children].map((node) => [node.dataset.sampleKey, node]));
  const retained = new Set();
  items.forEach((sample, index) => {
    const key = `${endpoint}:${sample.id}:${sample.key || ""}`;
    let card = existing.get(key);
    if (!card) {
      card = element("button", "sample-card"); card.dataset.sampleKey = key;
      const wrap = element("div", "thumb-wrap"), img = element("img");
      img.loading = "lazy"; img.decoding = "async";
      img.addEventListener("error", () => {
        img.hidden = true;
        if (!wrap.querySelector(".issue")) wrap.append(element("span", "issue", "图片无法读取"));
      });
      wrap.append(img, element("span", "edit-tag"), element("span", "thumb-tag"));
      const meta = element("div", "card-meta"), subtitle = element("div", "card-subtitle");
      subtitle.append(element("span", "sample-split"), element("span", "sample-number"));
      meta.append(element("span", "card-name"), subtitle); card.append(wrap, meta);
      card.addEventListener("click", () => showAt(Number(card.dataset.position)).catch(report));
    }
    retained.add(card); card.dataset.position = state.offset + index;
    card.className = `sample-card ${sample.state}`; card.title = sample.path;
    card.setAttribute("aria-label", `查看 ${sample.path}`);
    const img = card.querySelector("img"), src = `${endpoint}/samples/${sample.id}/image?thumbnail=true`;
    if (img.dataset.src !== src || img.dataset.version !== (sample.image_version || "")) {
      img.dataset.version = sample.image_version || "";
      img.dataset.src = src;
      img.hidden = false; card.querySelector(".issue")?.remove();
      thumbnails.observe(img);
    }
    img.alt = sample.path;
    const editTag = card.querySelector(".edit-tag"); editTag.hidden = !sample.modified && !sample.deleted;
    editTag.textContent = sample.deleted ? "已删除" : "已修改";
    card.querySelector(".thumb-tag").textContent = sample.state === "valid" ? `${sample.instances} 个目标` : labels[sample.state];
    card.querySelector(".card-name").textContent = sample.path.split("/").pop();
    card.querySelector(".sample-split").textContent = sample.split;
    card.querySelector(".sample-number").textContent = `#${sample.id + 1}`;
    if (gallery.children[index] !== card) gallery.insertBefore(card, gallery.children[index] || null);
  });
  for (const card of [...gallery.children]) if (!retained.has(card)) {
    thumbnails.unobserve(card.querySelector("img")); card.remove();
  }
}

async function refresh() {
  if (!state.session) return;
  const version = ++state.viewVersion, endpoint = base();
  loading.refreshAbort?.abort(); loading.stopNeighbors();
  const controller = new AbortController(); loading.refreshAbort = controller;
  $("gallery").setAttribute("aria-busy", "true");
  try {
    let page, stats, consistent = false;
    for (let attempt = 0; attempt < 3; attempt++) {
      const query = filters(), pageQuery = new URLSearchParams(query);
      pageQuery.set("offset", state.offset); pageQuery.set("limit", state.limit);
      const reply = await api(`${endpoint}/view?${pageQuery}`, {signal: controller.signal});
      if (version !== state.viewVersion || endpoint !== base()) return;
      ({page, stats} = reply);
      if (reply.draft && editor.draft && reply.draft.revision < editor.draft.revision) continue;
      if (reply.draft) { editor.draft = reply.draft; editorRenderStatus(); }
      if (query.toString() !== filters().toString()) continue;
      if (page.total && state.offset >= page.total) {
        state.offset = Math.floor((page.total - 1) / state.limit) * state.limit; continue;
      }
      consistent = true; break;
    }
    if (!consistent) throw new Error("筛选条件已变化，请重试。");
    state.items = page.items; state.total = page.total;
    renderGallery(page.items, endpoint);
    $("result-count").textContent = `${page.total.toLocaleString()} 张图片`;
    $("empty-state").hidden = page.total !== 0;
    $("empty-state").querySelector("h3").textContent = "没有匹配的图片";
    $("empty-state").querySelector("p").textContent = "调整筛选条件，或检查数据集图片目录";
    $("pagination").hidden = page.total === 0;
    $("page-info").textContent = `${Math.floor(state.offset / state.limit) + 1} / ${Math.max(1, Math.ceil(page.total / state.limit))}`;
    $("page-prev").disabled = state.offset === 0;
    $("page-next").disabled = state.offset + state.limit >= page.total;
    renderStats(stats);
  } catch (error) {
    if (error.name !== "AbortError" && version === state.viewVersion) report(error);
  } finally {
    if (version === state.viewVersion) $("gallery").setAttribute("aria-busy", "false");
  }
}

async function showAt(position) {
  if (!(await editorBeforeNavigation())) return;
  if (position < 0 || position >= state.total) return;
  const version = ++state.detailVersion, endpoint = base();
  loading.detailAbort?.abort();
  clearTimeout(loading.neighborTimer); loading.neighborAbort?.abort();
  const controller = new AbortController(); loading.detailAbort = controller;
  const current = () => version === state.detailVersion && endpoint === base() && !controller.signal.aborted;
  state.position = position; state.detail = null; view.image = null; view.selected = -1;
  if (!$("image-dialog").open) $("image-dialog").showModal();
  $("detail-title").textContent = "加载中…";
  $("annotation-list").replaceChildren(); $("selected-annotation").hidden = true;
  $("object-editor").hidden = true;
  $("detail-meta").textContent = ""; $("detail-issues").replaceChildren();
  $("annotation-count").textContent = "0";
  $("image-prev").disabled = position === 0; $("image-next").disabled = position + 1 >= state.total;
  $("detail-position").textContent = `${position + 1} / ${state.total}`;
  draw(); message("加载图片与标注…");
  try {
    let sample = state.items[position - state.offset];
    if (!sample) {
      const query = filters(); query.set("offset", position); query.set("limit", 1);
      sample = (await api(`${endpoint}/samples?${query}`, {signal: controller.signal})).items[0];
    }
    if (!current()) return;
    if (!sample) throw new Error("筛选结果已变化，请刷新图片列表。");
    // Start image transfer immediately; metadata still validates the current draft and file version.
    const imageResult = loading.images.acquire(endpoint, sample).then((image) => ({image}), (error) => ({error}));
    const detail = await editorFetchDetail(endpoint, sample.id, {signal: controller.signal});
    if (!current()) return;
    state.detail = detail; renderDetail(detail);
    if (detail.image_error) { loading.images.cancelPending(); message(detail.image_error); return; }
    let result;
    if (detail.image_version !== sample.image_version || detail.key !== sample.key) {
      result = {image: await loading.images.acquire(endpoint, detail)};
    } else result = await imageResult;
    if (!current()) return;
    if (result.error) throw result.error;
    view.image = result.image; message(""); fit();
    scheduleNeighbors(position, endpoint, version);
  } catch (error) {
    if (error.name !== "AbortError" && current()) message(error.message);
  }
}

function scheduleNeighbors(position, endpoint, detailVersion) {
  loading.stopNeighbors();
  if (navigator.connection?.saveData || /(^|-)2g$/.test(navigator.connection?.effectiveType || "")) return;
  const query = filters(), filterKey = query.toString();
  loading.neighborTimer = setTimeout(async () => {
    const controller = new AbortController(); loading.neighborAbort = controller;
    const current = () => !controller.signal.aborted && state.detailVersion === detailVersion &&
      endpoint === base() && $("image-dialog").open && filters().toString() === filterKey;
    try {
      // Look ahead at most one sample in either direction, including across gallery pages.
      const positions = [position + 1, position - 1].filter((p) => p >= 0 && p < state.total);
      for (const neighbor of positions) {
        if (!current()) return;
        let sample = state.items[neighbor - state.offset];
        if (!sample) {
          const request = new URLSearchParams(query); request.set("offset", neighbor); request.set("limit", 1);
          sample = (await api(`${endpoint}/samples?${request}`, {signal: controller.signal})).items[0];
        }
        if (!current() || !sample) return;
        await loading.images.acquire(endpoint, sample, "prefetch");
      }
    } catch { /* Speculative errors never replace the current image or its diagnostics. */ }
  }, 180);
}

function renderDetail(detail) {
    $("detail-title").textContent = detail.path;
    $("detail-meta").textContent = `${detail.split} · ${detail.width ?? "?"} × ${detail.height ?? "?"} px\n${labels[detail.state]}\n${detail.label_path}`;
    $("annotation-count").textContent = detail.annotations.length;
    $("annotation-list").replaceChildren(...detail.annotations.map((annotation, index) => {
      const row = element("button", "annotation-row"), dot = element("i", "class-dot");
      dot.style.background = color(annotation.class_id);
      row.append(dot, element("span", "", state.names[annotation.class_id] || `class_${annotation.class_id}`), element("small", "", `行 ${annotation.line}`));
      row.addEventListener("click", () => selectAnnotation(index)); return row;
    }));
    if (!detail.annotations.length) $("annotation-list").append(element("p", "muted", "此图片没有有效标注"));
    $("detail-issues").replaceChildren(...detail.issues.map((issue) => element("div", "issue", `${issue.file}${issue.line ? `:${issue.line}` : ""}\n${issue.message}${issue.raw !== null && issue.raw !== undefined ? `\n原始行：${issue.raw}` : ""}`)));
    editorDetail(detail);
}

function fit() {
  if (!view.image) return;
  view.scale = Math.min(view.width / view.image.width, view.height / view.image.height) * .92;
  view.x = (view.width - view.image.width * view.scale) / 2;
  view.y = (view.height - view.image.height * view.scale) / 2;
  draw();
}
function zoom(factor, x = view.width / 2, y = view.height / 2) {
  if (!view.image) return;
  const next = Math.max(.001, Math.min(30, view.scale * factor)), ratio = next / view.scale;
  view.x = x - (x - view.x) * ratio; view.y = y - (y - view.y) * ratio; view.scale = next; draw();
}

function shape(annotation, width, height) {
  if (annotation.box) {
    const b = annotation.box;
    return [[b.cx - b.width / 2, b.cy - b.height / 2], [b.cx + b.width / 2, b.cy - b.height / 2],
      [b.cx + b.width / 2, b.cy + b.height / 2], [b.cx - b.width / 2, b.cy + b.height / 2]].map(([x,y]) => [x * width,y * height]);
  }
  return annotation.points.map((p) => [p.x * width, p.y * height]);
}

let drawFrame = null;
function draw() {
  if (drawFrame !== null) return;
  drawFrame = requestAnimationFrame(() => { drawFrame = null; drawNow(); });
}
function drawNow() {
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, view.width, view.height);
  if (!view.image) return;
  const width = view.image.width, height = view.image.height;
  ctx.save(); ctx.translate(view.x, view.y); ctx.scale(view.scale, view.scale);
  ctx.drawImage(view.image, 0, 0);
  if ($("show-annotations").checked && state.detail) {
    state.detail.annotations.forEach((annotation, index) => {
      const points = shape(annotation, width, height), tint = color(annotation.class_id);
      ctx.beginPath(); points.forEach(([x,y], i) => i ? ctx.lineTo(x,y) : ctx.moveTo(x,y)); ctx.closePath();
      ctx.strokeStyle = tint; ctx.lineWidth = (index === view.selected ? 3.5 : 1.8) / view.scale;
      if (annotation.kind === "segment") { ctx.fillStyle = `${tint}35`; ctx.fill(); }
      if (index === view.selected) { ctx.fillStyle = `${tint}25`; ctx.fill(); }
      ctx.stroke();
      if (annotation.keypoints) annotation.keypoints.forEach((p, i) => {
        if (!p.visibility) return;
        ctx.beginPath(); ctx.arc(p.x * width, p.y * height, 4 / view.scale, 0, Math.PI * 2);
        ctx.fillStyle = p.visibility === 1 ? "#fff" : tint; ctx.fill();
        ctx.strokeStyle = tint; ctx.lineWidth = 1.5 / view.scale; ctx.stroke();
        if ($("show-names").checked) drawText(String(i), p.x * width + 6 / view.scale, p.y * height, tint);
      });
      if ($("show-names").checked) drawText(state.names[annotation.class_id] || `class_${annotation.class_id}`, points[0][0], Math.max(0, points[0][1] - 20 / view.scale), tint);
    });
  }
  editorDraw(ctx, width, height);
  ctx.restore(); $("zoom-level").textContent = `${Math.round(view.scale * 100)}%`;
}
function drawText(text, x, y, tint) {
  ctx.font = `${11 / view.scale}px sans-serif`; ctx.textBaseline = "top";
  const padding = 4 / view.scale, height = 18 / view.scale;
  ctx.fillStyle = tint; ctx.fillRect(x, y, ctx.measureText(text).width + padding * 2, height);
  ctx.fillStyle = "white"; ctx.fillText(text, x + padding, y + 3 / view.scale);
}
function inside(x, y, points) {
  let hit = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const [xi, yi] = points[i], [xj, yj] = points[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) hit = !hit;
  }
  return hit;
}
function selectAnnotation(index) {
  view.selected = index;
  [...$("annotation-list").children].forEach((node, i) => node.classList.toggle("selected", i === index));
  $("selected-annotation").hidden = index < 0;
  if (index >= 0 && state.detail) {
    const annotation = state.detail.annotations[index];
    $("raw-label").textContent = annotation.raw;
    const {raw, line, ...coordinates} = annotation;
    $("annotation-coordinates").textContent = JSON.stringify(coordinates, null, 2);
  }
  editorSelection();
  draw();
}

canvas.addEventListener("pointerdown", (event) => editorPointerDown(event));
canvas.addEventListener("pointermove", (event) => editorPointerMove(event));
canvas.addEventListener("pointerup", (event) => editorPointerUp(event));
canvas.addEventListener("pointercancel", (event) => editorPointerCancel(event));
canvas.addEventListener("wheel", (event) => { event.preventDefault(); zoom(Math.exp(-event.deltaY * .001), event.offsetX, event.offsetY); }, {passive: false});
new ResizeObserver(() => {
  const rect = $("canvas-stage").getBoundingClientRect();
  view.width = rect.width; view.height = rect.height;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr); canvas.height = Math.round(rect.height * dpr);
  if (view.image) fit(); else draw();
}).observe($("canvas-stage"));

$("open-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const request = {path: $("dataset-path").value.trim(), task: $("task").value};
  if ($("root-path").value.trim()) request.root = $("root-path").value.trim();
  if (request.task === "pose" && $("kpt-count").value) request.kpt_shape = [Number($("kpt-count").value), Number($("kpt-dims").value)];
  openDataset(request);
});
$("task").addEventListener("change", () => {
  $("pose-options").hidden = $("task").value !== "pose";
  $("task-hint").textContent = {detect: "矩形框与类别标签", segment: "闭合多边形与半透明填充", obb: "四点旋转框；与四点分割格式相同，需正确选择", pose: "关键点与可见性；从 YAML 读取 kpt_shape"}[$("task").value];
});
$("choose-yaml").addEventListener("click", () => { $("dataset-path").value = $("yaml-choice").value; $("open-form").requestSubmit(); });
$("rescan").addEventListener("click", () => { if (state.request) openDataset(state.request, true); });
for (const id of ["split-filter", "class-filter", "state-filter", "edit-filter"]) $(id).addEventListener("change", () => { state.offset = 0; refresh(); });
let searchTimer;
$("search").addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => { state.offset = 0; refresh(); }, 200); });
$("page-prev").addEventListener("click", () => { state.offset = Math.max(0, state.offset - state.limit); refresh(); });
$("page-next").addEventListener("click", () => { state.offset += state.limit; refresh(); });
$("close-detail").addEventListener("click", async () => { if (await editorBeforeNavigation()) $("image-dialog").close(); });
$("image-dialog").addEventListener("cancel", (event) => { event.preventDefault(); $("close-detail").click(); });
$("image-dialog").addEventListener("close", () => { loading.stopDetail(); editorPointerCancel(); ++state.detailVersion; view.image = null; state.detail = null; });
$("image-prev").addEventListener("click", () => showAt(state.position - 1).catch(report));
$("image-next").addEventListener("click", () => showAt(state.position + (state.detail?.deleted && $("edit-filter").value !== "deleted" ? 0 : 1)).catch(report));
$("fit-image").addEventListener("click", fit);
$("zoom-in").addEventListener("click", () => zoom(1.25));
$("zoom-out").addEventListener("click", () => zoom(.8));
for (const id of ["show-annotations", "show-names"]) $(id).addEventListener("change", draw);
document.addEventListener("keydown", (event) => {
  if (!$("image-dialog").open || ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) return;
  if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
    event.preventDefault(); showAt(state.position + (event.key === "ArrowLeft" ? -1 : 1)).catch(report);
  }
});
