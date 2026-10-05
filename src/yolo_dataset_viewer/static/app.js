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
  for (const [key, id] of [["split", "split-filter"], ["class_id", "class-filter"], ["state", "state-filter"], ["search", "search"]]) {
    const value = $(id).value.trim();
    if (value) params.set(key, value);
  }
  return params;
}

function enableFilters(enabled) {
  for (const id of ["split-filter", "class-filter", "state-filter", "search"]) $(id).disabled = !enabled;
}

function resetGallery() {
  state.items = []; state.offset = 0; state.total = 0;
  $("gallery").replaceChildren(); $("pagination").hidden = true;
  $("empty-state").hidden = false;
  $("empty-state").querySelector("h3").textContent = "正在读取数据集";
  $("empty-state").querySelector("p").textContent = "索引完成后显示图片与统计";
  $("result-count").textContent = "等待索引";
  for (const id of ["metric-images", "metric-instances", "metric-empty", "metric-invalid"]) $(id).textContent = "—";
  $("class-stats").replaceChildren(); $("split-stats").replaceChildren();
}

async function openDataset(request) {
  const version = ++state.openVersion;
  ++state.viewVersion; ++state.detailVersion;
  if ($("image-dialog").open) $("image-dialog").close();
  view.image = null; state.detail = null; state.session = null;
  state.request = {...request};
  notice(); resetGallery(); enableFilters(false);
  $("rescan").disabled = true;
  $("candidates").hidden = true; $("index-progress").hidden = false;
  $("index-progress").removeAttribute("value");
  $("status-text").textContent = "正在打开…"; $("footer-state").textContent = "建立索引中";
  try {
    let result = await api("/api/datasets", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(request)});
    if (version !== state.openVersion) return;
    state.session = result.id;
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

async function refresh() {
  if (!state.session) return;
  const version = ++state.viewVersion, endpoint = base();
  const query = filters(), pageQuery = new URLSearchParams(query);
  pageQuery.set("offset", state.offset); pageQuery.set("limit", state.limit);
  try {
    const [page, stats] = await Promise.all([api(`${endpoint}/samples?${pageQuery}`), api(`${endpoint}/stats?${query}`)]);
    if (version !== state.viewVersion) return;
    state.items = page.items; state.total = page.total;
    $("gallery").replaceChildren(...page.items.map((sample, index) => {
      const card = element("button", `sample-card ${sample.state}`);
      card.title = sample.path; card.setAttribute("aria-label", `查看 ${sample.path}`);
      const wrap = element("div", "thumb-wrap"), img = element("img");
      img.loading = "lazy"; img.decoding = "async"; img.alt = sample.path;
      img.src = `${endpoint}/samples/${sample.id}/image?thumbnail=true`;
      img.addEventListener("error", () => { img.hidden = true; wrap.append(element("span", "issue", "图片无法读取")); });
      wrap.append(img, element("span", "thumb-tag", sample.state === "valid" ? `${sample.instances} 个目标` : labels[sample.state]));
      const meta = element("div", "card-meta"); meta.append(element("span", "card-name", sample.path.split("/").pop()));
      const subtitle = element("div", "card-subtitle"); subtitle.append(element("span", "", sample.split), element("span", "", `#${sample.id + 1}`));
      meta.append(subtitle); card.append(wrap, meta);
      card.addEventListener("click", () => showAt(state.offset + index).catch(report));
      return card;
    }));
    $("result-count").textContent = `${page.total.toLocaleString()} 张图片`;
    $("empty-state").hidden = page.total !== 0;
    $("empty-state").querySelector("h3").textContent = "没有匹配的图片";
    $("empty-state").querySelector("p").textContent = "调整筛选条件，或检查数据集图片目录";
    $("pagination").hidden = page.total === 0;
    $("page-info").textContent = `${Math.floor(state.offset / state.limit) + 1} / ${Math.max(1, Math.ceil(page.total / state.limit))}`;
    $("page-prev").disabled = state.offset === 0;
    $("page-next").disabled = state.offset + state.limit >= page.total;
    renderStats(stats);
  } catch (error) { if (version === state.viewVersion) report(error); }
}

async function showAt(position) {
  if (position < 0 || position >= state.total) return;
  const version = ++state.detailVersion, endpoint = base();
  state.position = position; state.detail = null; view.image = null; view.selected = -1;
  if (!$("image-dialog").open) $("image-dialog").showModal();
  $("detail-title").textContent = "加载中…";
  $("annotation-list").replaceChildren(); $("selected-annotation").hidden = true;
  $("detail-meta").textContent = ""; $("detail-issues").replaceChildren();
  $("annotation-count").textContent = "0";
  $("image-prev").disabled = position === 0; $("image-next").disabled = position + 1 >= state.total;
  $("detail-position").textContent = `${position + 1} / ${state.total}`;
  draw(); message("加载图片与标注…");
  try {
    let sample = state.items[position - state.offset];
    if (!sample) {
      const query = filters(); query.set("offset", position); query.set("limit", 1);
      sample = (await api(`${endpoint}/samples?${query}`)).items[0];
    }
    if (version !== state.detailVersion) return;
    const detail = await api(`${endpoint}/samples/${sample.id}`);
    if (version !== state.detailVersion) return;
    state.detail = detail;
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
    $("detail-issues").replaceChildren(...detail.issues.map((issue) => element("div", "issue", `${issue.file}${issue.line ? `:${issue.line}` : ""}\n${issue.message}`)));
    if (detail.image_error) { message(detail.image_error); return; }
    const img = new Image();
    img.onload = () => { if (version !== state.detailVersion) return; view.image = img; message(""); fit(); };
    img.onerror = () => { if (version === state.detailVersion) message("图片无法解码，源文件可能已被移除或损坏"); };
    img.src = `${endpoint}/samples/${sample.id}/image`;
  } catch (error) { if (version === state.detailVersion) message(error.message); }
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

function draw() {
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
  draw();
}

let pointer = null;
canvas.addEventListener("pointerdown", (event) => {
  if (event.button !== 0) return;
  pointer = {x: event.offsetX, y: event.offsetY, startX: event.offsetX, startY: event.offsetY, moved: false};
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", (event) => {
  if (!pointer) return;
  view.x += event.offsetX - pointer.x; view.y += event.offsetY - pointer.y;
  pointer.moved ||= Math.hypot(event.offsetX - pointer.startX, event.offsetY - pointer.startY) > 4;
  pointer.x = event.offsetX; pointer.y = event.offsetY; draw();
});
canvas.addEventListener("pointerup", (event) => {
  if (pointer && !pointer.moved && view.image && state.detail && $("show-annotations").checked) {
    const x = (event.offsetX - view.x) / view.scale, y = (event.offsetY - view.y) / view.scale;
    let selected = -1;
    state.detail.annotations.forEach((a, i) => { if (inside(x, y, shape(a, view.image.width, view.image.height))) selected = i; });
    selectAnnotation(selected);
  }
  pointer = null;
});
canvas.addEventListener("pointercancel", () => { pointer = null; });
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
$("rescan").addEventListener("click", () => { if (state.request) openDataset(state.request); });
for (const id of ["split-filter", "class-filter", "state-filter"]) $(id).addEventListener("change", () => { state.offset = 0; refresh(); });
let searchTimer;
$("search").addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => { state.offset = 0; refresh(); }, 200); });
$("page-prev").addEventListener("click", () => { state.offset = Math.max(0, state.offset - state.limit); refresh(); });
$("page-next").addEventListener("click", () => { state.offset += state.limit; refresh(); });
$("close-detail").addEventListener("click", () => $("image-dialog").close());
$("image-dialog").addEventListener("close", () => { ++state.detailVersion; view.image = null; state.detail = null; pointer = null; });
$("image-prev").addEventListener("click", () => showAt(state.position - 1).catch(report));
$("image-next").addEventListener("click", () => showAt(state.position + 1).catch(report));
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
