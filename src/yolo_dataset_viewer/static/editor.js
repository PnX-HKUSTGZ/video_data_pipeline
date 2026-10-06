"use strict";

// Edits are serialized and acknowledged only after the SQLite transaction commits.
const editor = {draft: null, enabled: false, queue: [], saving: null, failure: null,
  tool: "browse", space: false, gesture: null, pointIndex: 0, preview: null, exporting: false,
  exportResult: null, classStats: [], repairLine: null, optionSchema: null, selectionSchema: null, reloadController: null, classForms: new Map()};

function editorError(error, target = "editor-error") {
  const text = error ? error.message || String(error) : "";
  $(target).textContent = text; $(target).hidden = !text;
}
function editorReset() {
  editor.reloadController?.abort(); editor.reloadController = null;
  editor.optionSchema = null; editor.selectionSchema = null; editor.classForms.clear();
  editor.draft = null; editor.enabled = false; editor.tool = "browse"; editor.gesture = null; editor.repairLine = null;
  $("draft-bar").hidden = true; $("edit-filter").hidden = true; $("edit-filter").value = "";
  $("edit-toolbar").hidden = true; $("object-editor").hidden = true;
  for (const id of ["classes-dialog", "export-dialog"]) if ($(id).open) $(id).close();
}
async function editorOpened(index) {
  editor.enabled = ["detect", "pose"].includes(index.task);
  sessionStorage.setItem("yolo-viewer-open", JSON.stringify(state.request));
  if (!editor.enabled) return;
  editor.draft = await api(`${base()}/draft`);
  $("draft-bar").hidden = false; $("edit-filter").hidden = false;
  editorRenderStatus();
}
function editorRenderStatus() {
  if (!editor.draft) return;
  const d = editor.draft;
  const busy = !!editor.saving || !!editor.queue.length || !!editor.gesture && editor.gesture.kind !== "pan";
  const blocked = busy || editor.exporting;
  document.querySelectorAll("#object-editor input, #object-editor select, #object-editor button, [data-tool], #new-object-class, #image-delete, .class-editor input, .class-editor button, #class-add-form input, #class-add-form button, .issue-list button").forEach((node) => {
    const disabled = blocked || node.dataset.edgeDisabled === "true";
    if (node.disabled !== disabled) node.disabled = disabled;
  });
  // Saving geometry does not change class choices; keep the existing DOM and focus.
  const optionSchema = JSON.stringify(d.classes.map((c) => [c.id, c.name]));
  if (optionSchema !== editor.optionSchema) {
    editor.optionSchema = optionSchema;
    state.names = Object.fromEntries(d.classes.map((c) => [c.id, c.name]));
    const selected = $("class-filter").value;
    $("class-filter").replaceChildren(new Option("全部类别", ""), ...d.classes.map((c) => new Option(`${c.id} · ${c.name}`, c.id)));
    if (d.classes.some((c) => String(c.id) === selected)) $("class-filter").value = selected;
    for (const id of ["new-object-class", "object-class"]) {
      const previous = $(id).value;
      $(id).replaceChildren(...d.classes.map((c) => new Option(c.name, c.id)));
      if (d.classes.some((c) => String(c.id) === previous)) $(id).value = previous;
    }
  }
  $("draft-summary").textContent = `修订 ${d.revision} · ${d.modified_images} 张修改 · ${d.deleted_images} 张删除`;
  const saveText = editor.failure ? "保存失败，待保存操作已保留" : busy ? "保存中…" : d.dirty ? "已保存到草稿" : "无待应用改动";
  $("save-status").textContent = saveText; $("detail-save-status").textContent = saveText;
  $("save-status").classList.toggle("danger", !!editor.failure);
  document.querySelectorAll(".pending-actions").forEach((node) => { node.hidden = !editor.failure; });
  $("save-retry").hidden = !editor.failure; $("draft-reload").hidden = !editor.failure && !d.conflicts.length;
  $("draft-undo").disabled = !d.can_undo || !!editor.failure || blocked;
  $("draft-redo").disabled = !d.can_redo || !!editor.failure || blocked;
  $("draft-discard").disabled = !d.dirty || blocked;
  $("export-open").disabled = blocked;
  $("manage-classes").disabled = blocked;
  editorError(editor.failure || (d.conflicts.length ? d.conflicts.map((v) => typeof v === "string" ? v : JSON.stringify(v)).join("\n") : ""), "draft-error");
  $("task-badge").textContent = tasks[d.task];
}
async function editorBeforeNavigation() {
  if (editor.exporting) { notice("导出正在进行，请等待完成后切换数据。"); return false; }
  if (editor.saving) await editor.saving;
  if (editor.queue.length) await editorDrain();
  if (editor.failure || editor.queue.length) {
    editorError("还有保存失败的操作，请重试保存，或读取最新草稿后继续。");
    return false;
  }
  return true;
}
function editorWritable({geometry = true} = {}) {
  if (!editor.enabled || !editor.draft || editor.exporting) return false;
  if (editor.saving || editor.queue.length) {
    editorError("正在保存上一项修改，请稍候。"); return false;
  }
  if (geometry && $("image-dialog").open && !view.image) {
    editorError("图片尚未加载或无法读取，暂不能修改坐标；仍可删除图片或异常标签。"); return false;
  }
  if ($("image-dialog").open && state.detail && state.detail.revision !== editor.draft.revision) {
    editorError("标注已有更新，正在读取最新版本，请检查后再编辑。"); editorReloadDetail(); return false;
  }
  if (editor.failure || editor.draft.conflicts.length) {
    editorError("编辑已暂停：请处理保存错误或源文件冲突，再继续操作。"); return false;
  }
  return true;
}
async function editorSubmit(operation) {
  if (!editorWritable({geometry: operation.kind === "annotation_upsert"})) return false;
  if (["class_add", "class_rename"].includes(operation.kind) && (!operation.name || editor.draft.classes.some((c) => c.name === operation.name && c.id !== operation.class_id))) {
    editorError("类别名称不能为空，且不能与其他类别同名。", "classes-error"); return false;
  }
  if (operation.kind === "keypoints_set" && (operation.keypoints.some((p) => !p.name) || new Set(operation.keypoints.map((p) => p.name)).size !== operation.keypoints.length)) {
    editorError("关键点名称不能为空，且同一类别内不能重复。", "classes-error"); return false;
  }
  editor.queue.push({path: "operations", operation: structuredClone(operation)});
  return editorDrain();
}
async function editorCommand(path) {
  if (!(await editorBeforeNavigation())) return false;
  editor.queue.push({path});
  return editorDrain();
}
async function editorDrain() {
  if (editor.saving) return editor.saving;
  if (editor.failure) return false;
  editor.saving = (async () => {
    do {
    while (editor.queue.length) {
      editorRenderStatus();
      const item = editor.queue[0];
      try {
        const body = {expected_revision: editor.draft.revision};
        if (item.operation) body.operation = item.operation;
        editor.draft = await api(`${base()}/draft/${item.path}`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
        editorAcknowledgeClassForm(item.operation);
        if (editor.draft.session_id && editor.draft.session_id !== state.session) {
          loading.reset(); editor.classForms.clear(); editor.selectionSchema = null;
          state.session = editor.draft.session_id;
          localStorage.setItem("yolo-viewer-session", JSON.stringify({id:state.session,request:state.request}));
          if ($("image-dialog").open) $("image-dialog").close();
          state.detail = null; view.image = null; state.offset = 0; ++state.detailVersion;
        }
        editor.queue.shift(); editorError(""); editorError("", "classes-error");
      } catch (error) {
        editor.failure = error;
        editorError(error); editorError(error, "classes-error");
        break;
      }
    }
    editorRenderStatus();
    if (!editor.failure) {
      await Promise.all([refresh(), editorReloadDetail()]);
      if ($("classes-dialog").open) await renderClassManager();
    }
    } while (editor.queue.length && !editor.failure);
    return !editor.failure;
  })();
  try { return await editor.saving; } finally { editor.saving = null; editorRenderStatus(); }
}
async function editorFetchDetail(endpoint, sampleId, {signal} = {}) {
  const version = state.detailVersion;
  const ensureCurrent = () => {
    if (signal?.aborted || endpoint !== base() || version !== state.detailVersion) {
      throw new DOMException("标注请求已被更新的浏览操作替代", "AbortError");
    }
  };
  for (let attempt = 0; attempt < 5; attempt++) {
    ensureCurrent();
    const detail = await api(`${endpoint}/samples/${sampleId}`, {signal});
    ensureCurrent();
    if (!editor.enabled || detail.revision === editor.draft?.revision) return detail;
    const draft = await api(`${endpoint}/draft`, {signal});
    ensureCurrent();
    if (detail.revision !== draft.revision || draft.revision < (editor.draft?.revision ?? 0)) continue;
    editor.draft = draft; editorRenderStatus(); return detail;
  }
  throw new Error("图片标注正在被另一页面修改，请重新打开。");
}
async function editorReloadDetail() {
  if (!state.detail || !$("image-dialog").open) return;
  editor.reloadController?.abort();
  const controller = new AbortController(); editor.reloadController = controller;
  const id = state.detail.id, key = state.detail.key, endpoint = base(), version = state.detailVersion;
  const imageVersion = state.detail.image_version;
  const uid = state.detail.annotations[view.selected]?.uid;
  try {
    const detail = await editorFetchDetail(endpoint, id, {signal: controller.signal});
    if (controller.signal.aborted || version !== state.detailVersion || endpoint !== base() || state.detail?.id !== id || state.detail.key !== key) return;
    if (detail.id !== id || detail.key !== key) {
      // A file stamp is not an image identity: different samples can have identical stamps.
      view.image = null; state.detail = null; editor.selectionSchema = null;
      $("image-dialog").close(); draw();
      notice("图片索引已变化，请从更新后的列表重新打开图片。");
      await refresh(); return;
    }
    if (imageVersion !== detail.image_version) {
      // The source image changed. Release stale pixels before restoring editable geometry.
      view.image = null; draw();
      const position = state.position;
      Promise.resolve(editor.saving).then(() => {
        if (version === state.detailVersion && endpoint === base() && $("image-dialog").open) return showAt(position);
      }).catch(report);
      return;
    }
    state.detail = detail; renderDetail(detail);
    const next = detail.annotations.findIndex((a) => a.uid === uid);
    selectAnnotation(next >= 0 ? next : Math.min(view.selected, detail.annotations.length - 1));
    editorRenderStatus();
    if (typeof syncInspectorPosition === "function") await syncInspectorPosition();
  } catch (error) { if (error.name !== "AbortError") editorError(error); }
  finally { if (editor.reloadController === controller) editor.reloadController = null; }
}

function editorDetail(detail) {
  $("edit-toolbar").hidden = !editor.enabled;
  $("object-editor").hidden = true; editorError("");
  if (!editor.enabled) return;
  $("image-delete").textContent = detail.deleted ? "恢复图片" : "删除图片";
  $("image-prev").disabled = state.position === 0;
  $("image-next").disabled = state.position + (detail.deleted && $("edit-filter").value !== "deleted" ? 0 : 1) >= state.total;
  $("detail-position").textContent = detail.deleted ? "已在草稿中删除" : `${state.position + 1} / ${state.total}`;
  $("detail-meta").textContent += detail.deleted ? "\n已在草稿中删除" : detail.modified ? "\n已在草稿中修改" : "";
  for (const [index, issue] of detail.issues.entries()) {
    if (!issue.line) continue;
    const row = $("detail-issues").children[index];
    const remove = element("button", "subtle danger", `删除异常行 ${issue.line}`);
    remove.type = "button";
    remove.addEventListener("click", async () => {
      if (confirm(`删除异常标签第 ${issue.line} 行？原行可通过撤销恢复。修正时可先新增正确目标，再删除此行。`)) {
        await editorSubmit({kind: "issue_delete", sample_id: detail.id, line: issue.line});
      }
    });
    const repair = element("button", "secondary", `替换异常行 ${issue.line}`);
    repair.type = "button";
    repair.addEventListener("click", () => {
      editorSetTool("draw"); editor.repairLine = issue.line;
      editorError(`请绘制正确目标，保存成功后将替换第 ${issue.line} 行。切换工具可取消。`);
    });
    row.append(remove, repair);
  }
  editorSetTool(editor.tool);
}
function editorSetTool(tool) {
  editor.tool = tool;
  if (tool !== "draw") { editor.repairLine = null; editorError(""); }
  document.querySelectorAll("[data-tool]").forEach((b) => b.classList.toggle("active", b.dataset.tool === tool));
  canvas.style.cursor = tool === "browse" ? "grab" : tool === "draw" || tool === "points" ? "crosshair" : "default";
  editorSelection(); draw();
}
function editorSelection() {
  const a = state.detail?.annotations[view.selected];
  if ($("object-editor").nextElementSibling !== $("selected-annotation")) $("selected-annotation").before($("object-editor"));
  $("object-editor").hidden = !editor.enabled || !a || state.detail.deleted;
  if (!editor.enabled || !a) { editor.selectionSchema = null; return; }
  const definitions = editor.draft.classes.find((c) => c.id === a.class_id)?.keypoints || [];
  const selectionSchema = JSON.stringify([state.session, state.detail.id, a.uid, a.class_id, a.box, a.keypoints, definitions]);
  if (selectionSchema === editor.selectionSchema) {
    $("point-editor").querySelectorAll(".point-select").forEach((node, index) => node.classList.toggle("active", editor.pointIndex === index));
    return;
  }
  editor.selectionSchema = selectionSchema;
  $("object-class").value = String(a.class_id);
  for (const key of ["cx", "cy", "width", "height"]) $("box-" + key).value = Number(a.box[key].toFixed(8));
  $("point-editor").replaceChildren();
  if (!definitions.length) { $("point-editor").append(element("p", "field-hint", "此类别尚无关键点。可在类别与关键点面板中新增。")); return; }
  $("point-editor").append(element("h3", "", "关键点 · 选点后在画布放置或拖动"));
  definitions.forEach((definition, index) => {
    const p = a.keypoints?.[index] || {x: 0, y: 0, visibility: 0};
    const row = element("div", "point-edit-row"); row.dataset.pointIndex = index;
    const select = element("button", "secondary point-select", `${index} · ${definition.name}`); select.type = "button";
    select.classList.toggle("active", editor.pointIndex === index);
    select.addEventListener("click", () => { editor.pointIndex = index; editorSetTool("points"); });
    const coordinates = element("div", "point-coordinates");
    const inputs = ["x", "y"].map((axis) => {
      const input = element("input"); input.type = "number"; input.min = "0"; input.max = "1"; input.step = "any";
      input.value = Number(p[axis].toFixed(8)); input.setAttribute("aria-label", `${definition.name} ${axis.toUpperCase()}`);
      coordinates.append(input); return input;
    });
    const visibility = element("select"); visibility.setAttribute("aria-label", `${definition.name} 可见性`);
    visibility.replaceChildren(new Option("未标注", 0), new Option("遮挡", 1), new Option("可见", 2)); visibility.value = p.visibility;
    const save = element("button", "secondary", "保存点"); save.type = "button";
    save.addEventListener("click", () => {
      const updated = structuredClone(state.detail.annotations[view.selected]);
      updated.keypoints ||= definitions.map(() => ({x: 0, y: 0, visibility: 0}));
      const v = Number(visibility.value);
      updated.keypoints[index] = v ? {x: Number(inputs[0].value), y: Number(inputs[1].value), visibility: v} : {x: 0, y: 0, visibility: 0};
      editorSaveAnnotation(updated);
    });
    row.append(select, coordinates, visibility, save); $("point-editor").append(row);
  });
}
function annotationOperation(annotation, annotationId = annotation.uid || null) {
  return {kind: "annotation_upsert", sample_id: state.detail.id, annotation_id: annotationId,
    class_id: annotation.class_id, box: annotation.box, keypoints: annotation.keypoints || null,
    issue_line: annotationId === null ? editor.repairLine : null};
}
async function editorSaveAnnotation(annotation, annotationId) {
  const b = annotation.box;
  if (![b.cx, b.cy, b.width, b.height].every(Number.isFinite) || b.width <= 0 || b.height <= 0 ||
      b.cx - b.width / 2 < -1e-9 || b.cy - b.height / 2 < -1e-9 || b.cx + b.width / 2 > 1 + 1e-9 || b.cy + b.height / 2 > 1 + 1e-9 ||
      annotation.keypoints?.some((p) => !Number.isFinite(p.x) || !Number.isFinite(p.y) || p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1)) {
    editorError("框与关键点必须位于图片内；框需有正面积，坐标必须为有限数值。"); return false;
  }
  const success = await editorSubmit(annotationOperation(annotation, annotationId));
  if (success && annotationId === null) { editor.repairLine = null; selectAnnotation(state.detail.annotations.length - 1); }
  return success;
}

// Form edits are local until their own confirmation; unrelated saves must preserve them.
function editorAcknowledgeClassForm(operation) {
  if (!operation) return;
  if (operation.kind === "class_delete") { editor.classForms.delete(operation.class_id); return; }
  const form = editor.classForms.get(operation.class_id);
  const saved = editor.draft.classes.find((c) => c.id === operation.class_id);
  if (!form || !saved) return;
  if (operation.kind === "class_rename") {
    form.name = saved.name; form.nameDirty = false;
  } else if (operation.kind === "keypoints_set") {
    form.keypoints = structuredClone(saved.keypoints); form.keypointsDirty = false;
  }
}

function editorClassForm(definition) {
  let form = editor.classForms.get(definition.id);
  if (!form) {
    form = {name: definition.name, keypoints: structuredClone(definition.keypoints), nameDirty: false, keypointsDirty: false};
    editor.classForms.set(definition.id, form);
  }
  if (!form.nameDirty) form.name = definition.name;
  if (!form.keypointsDirty) form.keypoints = structuredClone(definition.keypoints);
  return form;
}

// Stable IDs, rather than names or positions, preserve point coordinates during list edits.
async function renderClassManager() {
  if (!editor.draft) return;
  try { editor.classStats = (await api(`${base()}/stats`)).classes; } catch { editor.classStats = []; }
  const live = new Set(editor.draft.classes.map((c) => c.id));
  for (const id of editor.classForms.keys()) if (!live.has(id)) editor.classForms.delete(id);
  $("class-manager").replaceChildren(...editor.draft.classes.map((c) => {
    const form = editorClassForm(c);
    const card = element("section", "class-editor"); card.dataset.classId = c.id;
    const head = element("div", "button-row"), name = element("input");
    name.value = form.name; name.setAttribute("aria-label", `类别 ${c.id} 名称`);
    const pending = element("p", "field-hint class-form-pending", "有未保存的表单修改，请分别确认名称和关键点列表。");
    const updatePending = () => {
      form.nameDirty = form.name !== c.name;
      form.keypointsDirty = JSON.stringify(form.keypoints) !== JSON.stringify(c.keypoints);
      pending.hidden = !form.nameDirty && !form.keypointsDirty;
    };
    name.addEventListener("input", () => { form.name = name.value; updatePending(); });
    const rename = element("button", "secondary", "重命名"), remove = element("button", "subtle danger", "删除类别");
    rename.addEventListener("click", () => editorSubmit({kind: "class_rename", class_id: c.id, name: form.name.trim()}));
    remove.addEventListener("click", async () => {
      const counts = editor.classStats.find((s) => s.id === c.id);
      if (confirm(`删除类别“${c.name}”？将删除 ${counts?.instances ?? "全部"} 个目标，影响 ${counts?.images ?? "相关"} 张图片及其关键点。图片保留，操作可撤销。`)) await editorSubmit({kind: "class_delete", class_id: c.id});
    });
    head.append(element("span", "tiny-badge", c.id), name, rename, remove); card.append(head, pending);
    const items = form.keypoints;
    const list = element("div", "keypoint-definition-list");
    const render = () => {
      list.replaceChildren(...items.map((point, i) => {
        const row = element("div", "keypoint-definition-row"), input = element("input");
        input.value = point.name; input.setAttribute("aria-label", `${c.name} 关键点 ${i} 名称`);
        input.addEventListener("input", () => { point.name = input.value; updatePending(); });
        const up = element("button", "secondary", "↑"), down = element("button", "secondary", "↓"), del = element("button", "subtle danger", "移除");
        up.disabled = i === 0; down.disabled = i + 1 === items.length;
        up.dataset.edgeDisabled = String(up.disabled); down.dataset.edgeDisabled = String(down.disabled);
        up.setAttribute("aria-label", `上移关键点 ${i}`); down.setAttribute("aria-label", `下移关键点 ${i}`);
        up.addEventListener("click", () => { [items[i - 1], items[i]] = [items[i], items[i - 1]]; render(); });
        down.addEventListener("click", () => { [items[i + 1], items[i]] = [items[i], items[i + 1]]; render(); });
        del.addEventListener("click", () => { items.splice(i, 1); render(); });
        row.append(element("span", "muted", i), input, up, down, del); return row;
      }));
      updatePending();
    }; render();
    const buttons = element("div", "button-row"), add = element("button", "secondary", "增加关键点"), save = element("button", "primary", "保存关键点列表");
    add.addEventListener("click", () => {
      const names = new Set(items.map((point) => point.name.trim()));
      let index = 0; while (names.has(`keypoint_${index}`)) index++;
      items.push({name: `keypoint_${index}`}); render();
    });
    save.addEventListener("click", async () => {
      const removed = c.keypoints.filter((k) => !items.some((v) => v.id === k.id));
      const counts = editor.classStats.find((s) => s.id === c.id);
      if (removed.length && !confirm(`移除 ${removed.length} 个关键点定义，将移除 ${counts?.instances ?? "相关"} 个目标中对应的坐标（最多 ${(counts?.instances || 0) * removed.length} 个点），影响 ${counts?.images ?? "相关"} 张图片。确认保存？`)) return;
      await editorSubmit({kind: "keypoints_set", class_id: c.id, keypoints: items.map((p) => ({...p, name: p.name.trim()}))});
    });
    buttons.append(add, save); card.append(list, buttons); return card;
  }));
  editorRenderStatus();
}

// Gestures modify a preview only; pointer-up is a single reversible transaction.
function normalizedPoint(event) {
  return {x: (event.offsetX - view.x) / view.scale / view.image.width,
    y: (event.offsetY - view.y) / view.scale / view.image.height};
}
function clamp(value, min = 0, max = 1) { return Math.min(max, Math.max(min, value)); }
function boxHandles(b) {
  const x0 = b.cx - b.width / 2, y0 = b.cy - b.height / 2, x1 = b.cx + b.width / 2, y1 = b.cy + b.height / 2;
  return [[x0,y0],[b.cx,y0],[x1,y0],[x1,b.cy],[x1,y1],[b.cx,y1],[x0,y1],[x0,b.cy]];
}
function hitAnnotation(p) {
  let selected = -1;
  state.detail.annotations.forEach((a, i) => { if (inside(p.x * view.image.width, p.y * view.image.height, shape(a, view.image.width, view.image.height))) selected = i; });
  return selected;
}
function editorPointerDown(event) {
  if (![0, 1].includes(event.button) || !view.image || !state.detail) return;
  event.preventDefault(); canvas.setPointerCapture(event.pointerId);
  const start = normalizedPoint(event), a = state.detail.annotations[view.selected];
  const pan = event.button === 1 || editor.space || !editor.enabled || editor.tool === "browse";
  editor.gesture = {kind: "pan", start, x: event.offsetX, y: event.offsetY, startX: event.offsetX, startY: event.offsetY, moved: false};
  if (pan) return;
  if (!editorWritable() || state.detail.deleted || editor.queue.length) { editor.gesture = null; return; }
  if (editor.tool === "draw") {
    if (!editor.draft.classes.length) { editorError("请先新增类别。"); editor.gesture = null; return; }
    editor.gesture = {kind: "draw", start: {x: clamp(start.x), y: clamp(start.y)}, current: start};
  } else if (editor.tool === "points") {
    const defs = editor.draft.classes.find((c) => c.id === a?.class_id)?.keypoints || [];
    if (!a || !defs.length) { editorError("请先选择带有关键点定义的目标。"); editor.gesture = null; return; }
    const near = (a.keypoints || []).findIndex((p) => p.visibility && Math.hypot((p.x - start.x) * view.image.width, (p.y - start.y) * view.image.height) * view.scale < 10);
    const index = near >= 0 ? near : Math.min(editor.pointIndex, defs.length - 1);
    editor.pointIndex = index;
    editor.gesture = {kind: "point", start, index, original: structuredClone(a)};
    a.keypoints ||= defs.map(() => ({x: 0, y: 0, visibility: 0}));
    a.keypoints[index] = {x: clamp(start.x), y: clamp(start.y), visibility: a.keypoints[index]?.visibility || 2};
    draw();
  } else {
    let handle = -1;
    if (a) handle = boxHandles(a.box).findIndex(([x,y]) => Math.hypot((x-start.x)*view.image.width, (y-start.y)*view.image.height)*view.scale < 9);
    const selected = handle >= 0 ? view.selected : hitAnnotation(start);
    selectAnnotation(selected);
    if (selected < 0) { editor.gesture = null; return; }
    editor.gesture = {kind: handle >= 0 ? "resize" : "move", handle, start, original: structuredClone(state.detail.annotations[selected]), moved: false};
  }
  editorRenderStatus();
}
function editorPointerMove(event) {
  const g = editor.gesture; if (!g) return;
  if (!view.image || !state.detail) { editor.gesture = null; return; }
  const p = normalizedPoint(event);
  if (g.kind === "pan") {
    view.x += event.offsetX - g.x; view.y += event.offsetY - g.y;
    g.moved ||= Math.hypot(event.offsetX - g.startX, event.offsetY - g.startY) > 4;
    g.x = event.offsetX; g.y = event.offsetY;
  } else if (g.kind === "draw") g.current = {x: clamp(p.x), y: clamp(p.y)};
  else {
    const a = state.detail.annotations[view.selected];
    if (g.kind === "point") a.keypoints[g.index] = {x: clamp(p.x), y: clamp(p.y), visibility: a.keypoints[g.index].visibility || 2};
    if (g.kind === "move") {
      const b = g.original.box;
      a.box = {...b, cx: clamp(b.cx + p.x - g.start.x, b.width/2, 1-b.width/2), cy: clamp(b.cy + p.y - g.start.y, b.height/2, 1-b.height/2)};
    }
    if (g.kind === "resize") {
      const b = g.original.box;
      let x0 = b.cx-b.width/2, x1 = b.cx+b.width/2, y0 = b.cy-b.height/2, y1 = b.cy+b.height/2;
      if ([0,6,7].includes(g.handle)) x0 = clamp(p.x, 0, x1-1e-6);
      if ([2,3,4].includes(g.handle)) x1 = clamp(p.x, x0+1e-6, 1);
      if ([0,1,2].includes(g.handle)) y0 = clamp(p.y, 0, y1-1e-6);
      if ([4,5,6].includes(g.handle)) y1 = clamp(p.y, y0+1e-6, 1);
      a.box = {cx:(x0+x1)/2,cy:(y0+y1)/2,width:x1-x0,height:y1-y0};
    }
    g.moved = true;
  }
  draw();
}
async function editorPointerUp(event) {
  const g = editor.gesture; editor.gesture = null; if (!g) return;
  if (g.kind === "pan") {
    if (!g.moved && event.button === 0 && $("show-annotations").checked) selectAnnotation(hitAnnotation(normalizedPoint(event)));
  } else if (g.kind === "draw") {
    const end = g.current, w = Math.abs(end.x-g.start.x), h = Math.abs(end.y-g.start.y);
    if (w*view.image.width*view.scale > 3 && h*view.image.height*view.scale > 3) {
      await editorSaveAnnotation({class_id:Number($("new-object-class").value), box:{cx:(end.x+g.start.x)/2,cy:(end.y+g.start.y)/2,width:w,height:h}}, null);
    }
  } else if (g.kind === "point" || g.moved) {
    await editorSaveAnnotation(state.detail.annotations[view.selected]);
  }
  editorRenderStatus(); draw();
}
function editorPointerCancel() {
  const g = editor.gesture;
  if (g?.original && state.detail && view.selected >= 0) state.detail.annotations[view.selected] = g.original;
  editor.gesture = null; editorRenderStatus(); draw();
}
function editorDraw(context, width, height) {
  if (!editor.enabled) return;
  const a = state.detail?.annotations[view.selected];
  if (a?.box && editor.tool === "edit" && !state.detail.deleted) {
    context.strokeStyle = "#fff"; context.fillStyle = color(a.class_id); context.lineWidth = 1/view.scale;
    for (const [x,y] of boxHandles(a.box)) {
      context.fillRect(x*width-4/view.scale,y*height-4/view.scale,8/view.scale,8/view.scale);
      context.strokeRect(x*width-4/view.scale,y*height-4/view.scale,8/view.scale,8/view.scale);
    }
  }
  const g = editor.gesture;
  if (g?.kind === "draw") {
    context.strokeStyle = "#117b67"; context.lineWidth = 2/view.scale; context.setLineDash([5/view.scale]);
    context.strokeRect(g.start.x*width,g.start.y*height,(g.current.x-g.start.x)*width,(g.current.y-g.start.y)*height); context.setLineDash([]);
  }
}

function exportRequest() {
  const result = {expected_revision: editor.draft.revision, mode: $("export-mode").value};
  if (result.mode === "new") result.destination = $("export-destination").value.trim();
  return result;
}
function exportSummary(preview) {
  const container = $("export-preview"); container.replaceChildren(); container.hidden = false;
  container.append(element("h3", "", "导出预检"), element("p", "", `输出 ${preview.images} 张图片，删除 ${preview.deleted_images} 张图片、${preview.deleted_objects} 个目标。`));
  container.append(element("p", "", `任务：${preview.task} ${preview.kpt_shape ? `· kpt_shape: ${preview.kpt_shape.join(" × ")}` : ""}`));
  container.append(element("p", "path-text", `输出路径：${preview.output_path}`));
  container.append(element("p", "", `预计空间 ${(preview.required_bytes / 1024 / 1024).toFixed(1)} MiB / 可用 ${(preview.available_bytes / 1024 / 1024).toFixed(1)} MiB`));
  const classes = element("ul");
  for (const c of preview.classes) classes.append(element("li", "", `${c.old_id} → ${c.new_id} · ${c.name}${c.keypoints.length ? ` · [${c.keypoints.join(", ")}]` : ""}`));
  container.append(classes);
  for (const warning of preview.warnings) container.append(element("p", "notice", warning));
  for (const error of preview.errors) container.append(element("p", "issue", error));
  const files = element("details"), summary = element("summary", "", `文件操作清单（${preview.actions.length}）`);
  const list = element("pre", "export-actions");
  list.textContent = preview.actions.map((a) => `${{create:"新增",overwrite:"覆盖",delete:"删除"}[a.action]}  ${a.path}`).join("\n");
  files.append(summary, list); container.append(files);
  $("export-start").hidden = preview.errors.length !== 0;
  $("export-start").textContent = preview.mode === "inplace" ? "确认备份并应用到原数据集" : "确认导出新数据集";
}
async function runExport() {
  if (!editor.preview || !(await editorBeforeNavigation())) return;
  const request = exportRequest();
  if (JSON.stringify(request) !== JSON.stringify(editor.preview.request)) {
    editorError("导出设置或草稿已变化，请重新预检。", "export-error"); $("export-start").hidden = true; return;
  }
  editor.exporting = true; editorRenderStatus();
  $("export-start").disabled = true; $("export-preview-button").disabled = true;
  editorError("", "export-error");
  try {
    let result = await api(`${base()}/exports`, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(request)});
    const endpoint = `${base()}/exports/${result.id}`;
    while (result.state === "running") {
      $("export-status").textContent = `${result.phase} · ${result.completed} / ${result.total}`;
      await new Promise((resolve) => setTimeout(resolve, 300)); result = await api(endpoint);
    }
    if (result.state !== "complete") throw new Error(result.error || `导出${result.state}`);
    editor.exportResult = result;
    $("export-status").textContent = `导出完成\n${result.result.output_path}${result.result.backup_path ? `\n原文件备份：${result.result.backup_path}` : ""}`;
    $("export-start").hidden = true;
    $("export-view-result").hidden = false;
    if (request.mode === "inplace") {
      editor.exporting = false;
      // The server has reindexed the applied baseline; opening it also restores the current session.
      const reopen = result.reopen || {path:result.result.config_path,task:result.result.task};
      await openDataset(reopen);
      notice(`已应用回原数据集。原文件备份：${result.result.backup_path}`);
    }
  } catch (error) { editorError(error, "export-error"); }
  finally { editor.exporting = false; $("export-start").disabled = false; $("export-preview-button").disabled = false; editorRenderStatus(); }
}

$("manage-classes").addEventListener("click", async () => {
  if (!(await editorBeforeNavigation())) return;
  await renderClassManager(); $("classes-dialog").showModal();
});
$("close-classes").addEventListener("click", async () => { if (await editorBeforeNavigation()) $("classes-dialog").close(); });
$("classes-dialog").addEventListener("cancel", (e) => { e.preventDefault(); $("close-classes").click(); });
$("class-add-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (await editorSubmit({kind:"class_add",name:$("class-add-name").value.trim()})) $("class-add-name").value = "";
});
for (const kind of ["undo", "redo"]) $("draft-"+kind).addEventListener("click", () => editorCommand(kind));
$("draft-discard").addEventListener("click", async () => {
  if (confirm("放弃全部草稿修改？源数据保持原样，此操作不能撤销。")) await editorCommand("discard");
});
document.querySelectorAll(".retry-pending").forEach((b) => b.addEventListener("click", () => $("save-retry").click()));
document.querySelectorAll(".reload-pending").forEach((b) => b.addEventListener("click", () => $("draft-reload").click()));
$("save-retry").addEventListener("click", () => { editor.failure = null; editorDrain(); });
$("draft-reload").addEventListener("click", async () => {
  if (editor.queue.length && !confirm("读取最新草稿将放弃本页尚未保存的操作；已保存操作保持不变。继续？")) return;
  try {
    editor.queue = []; editor.failure = null; editor.selectionSchema = null; editor.classForms.clear();
    try { editor.draft = await api(`${base()}/draft`); }
    catch {
      // Rejoining another tab's session must reset sample IDs, pixels and pending loads together.
      await openDataset(state.request); return;
    }
    editorRenderStatus(); await refresh(); await editorReloadDetail();
    if ($("classes-dialog").open) await renderClassManager();
  } catch (error) { editorError(error); }
});
document.querySelectorAll("[data-tool]").forEach((b) => b.addEventListener("click", () => editorSetTool(b.dataset.tool)));
$("object-editor").addEventListener("submit", (e) => {
  e.preventDefault();
  const a = structuredClone(state.detail.annotations[view.selected]);
  for (const key of ["cx", "cy", "width", "height"]) a.box[key] = Number($("box-"+key).value);
  editorSaveAnnotation(a);
});
$("object-class").addEventListener("change", async () => {
  const a = structuredClone(state.detail.annotations[view.selected]);
  a.class_id = Number($("object-class").value);
  // Null asks the backend to migrate old points by definition name.
  a.keypoints = null; await editorSaveAnnotation(a);
});
$("object-delete").addEventListener("click", async () => {
  const a = state.detail?.annotations[view.selected];
  if (a) await editorSubmit({kind:"annotation_delete",sample_id:state.detail.id,annotation_id:a.uid});
});
$("image-delete").addEventListener("click", async () => {
  const d = state.detail;
  if (d && (d.deleted || confirm("从全部划分中删除此图片及标注？只保存到草稿，导出前源文件不变。"))) {
    await editorSubmit({kind:"sample_delete",sample_id:d.id,deleted:!d.deleted});
  }
});
$("export-open").addEventListener("click", async () => {
  if (!(await editorBeforeNavigation())) return;
  editor.preview = null; $("export-preview").hidden = true; $("export-start").hidden = true;
  $("export-view-result").hidden = true; $("export-status").textContent = ""; editorError("", "export-error");
  $("export-form").hidden = false;
  $("export-dialog").showModal();
  await renderRecoveries();
});
$("close-export").addEventListener("click", () => {
  if (editor.exporting) editorError("导出正在进行，请等待结束。", "export-error"); else $("export-dialog").close();
});
$("export-dialog").addEventListener("cancel", (e) => { e.preventDefault(); $("close-export").click(); });
$("export-mode").addEventListener("change", () => {
  const inplace = $("export-mode").value === "inplace";
  $("export-destination").disabled = inplace;
  $("export-help").textContent = inplace ? "将展示覆盖和删除清单。源文件先备份到相邻目录；含根目录外引用时不能原地应用。" : "原数据集与草稿继续保留。输出路径不能位于源数据集内。";
  $("export-start").hidden = true;
});
$("export-destination").addEventListener("input", () => { $("export-start").hidden = true; });
$("export-form").addEventListener("submit", async (e) => {
  e.preventDefault(); if (!(await editorBeforeNavigation())) return;
  $("export-preview-button").disabled = true; editorError("", "export-error");
  try {
    const request = exportRequest();
    const preview = await api(`${base()}/exports/preview`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(request)});
    editor.preview = {request,preview}; exportSummary(preview);
  } catch (error) { editorError(error, "export-error"); }
  finally { $("export-preview-button").disabled = false; }
});
$("export-start").addEventListener("click", runExport);
$("export-view-result").addEventListener("click", () => {
  const result = editor.exportResult;
  if (result) openDataset({path:result.result.config_path,task:result.result.task});
});
async function editorRecoveryAvailable() {
  $("export-form").hidden = true; $("export-start").hidden = true; $("export-preview").hidden = true;
  $("export-view-result").hidden = true;
  $("export-status").textContent = "发现未完成的原地应用，请恢复登记的备份后继续。";
  $("export-dialog").showModal(); await renderRecoveries();
}
async function renderRecoveries() {
  $("export-recoveries")?.remove();
  try {
    const recoveries = await api(`${base()}/exports/recoveries`);
    if (!recoveries.length) return;
    const container = element("section", "recovery-list"); container.id = "export-recoveries";
    container.append(element("h3", "", "未完成应用的恢复记录"));
    for (const recovery of recoveries) {
      const row = element("div", ""), button = element("button", "secondary", "恢复备份");
      row.append(element("p", "path-text", `${recovery.path} · ${recovery.state}`), button);
      button.addEventListener("click", async () => {
        if (!confirm("恢复此备份中登记的原始文件，并重新打开数据集？")) return;
        try {
          const recovered = await api(`${base()}/exports/recover`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({expected_revision:editor.draft?.revision || 0,backup_path:recovery.path})});
          await openDataset(recovered.request || state.request, true);
        } catch (error) { editorError(error, "export-error"); }
      }); container.append(row);
    }
    $("export-dialog").querySelector(".settings-body").append(container);
  } catch (error) { editorError(error, "export-error"); }
}
window.addEventListener("beforeunload", (e) => {
  if (editor.queue.length || editor.saving || editor.exporting) { e.preventDefault(); e.returnValue = ""; }
});
document.addEventListener("keydown", (e) => {
  if (["INPUT","TEXTAREA","SELECT"].includes(e.target.tagName)) return;
  if (e.code === "Space" && $("image-dialog").open) { editor.space = true; e.preventDefault(); }
});
document.addEventListener("keyup", (e) => { if (e.code === "Space") editor.space = false; });
window.addEventListener("blur", () => { editor.space = false; editorPointerCancel(); });
try {
  const previous = JSON.parse(sessionStorage.getItem("yolo-viewer-open") || "null");
  if (previous?.path) {
    $("dataset-path").value = previous.path; $("task").value = previous.task;
    $("root-path").value = previous.root || ""; $("task").dispatchEvent(new Event("change"));
    if (previous.kpt_shape) { $("kpt-count").value = previous.kpt_shape[0]; $("kpt-dims").value = previous.kpt_shape[1]; }
    openDataset(previous);
  }
} catch { sessionStorage.removeItem("yolo-viewer-open"); }
