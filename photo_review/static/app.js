const TOKEN = document.querySelector('meta[name="review-token"]').content;
const PAGE_SIZE = 20;
let state = null;
let statusMap = new Map();
let groupById = new Map();
let fileLookup = new Map();
let page = 0;
let saveTimer = null;
let pollTimer = null;
let lb = null;

const $ = (id) => document.getElementById(id);

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.classList.add("show");
  window.clearTimeout(toast.timer);
  toast.timer = window.setTimeout(() => el.classList.remove("show"), 3600);
}

async function api(path, body) {
  const options = { headers: { "X-Review-Token": TOKEN } };
  if (body !== undefined) {
    options.method = "POST";
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  let data = null;
  try {
    data = await response.json();
  } catch (err) {
    data = null;
  }
  if (!response.ok) throw new Error((data && data.error) || `Request failed (${response.status})`);
  return data;
}

function formatBytes(bytes) {
  const units = ["B", "KB", "MB", "GB"];
  let value = Number(bytes || 0);
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

const thumbUrl = (file) => `/thumb/${file.id}.jpg?v=${file.v}`;
const imageUrl = (file) => `/image/${file.id}?v=${file.v}`;

function groupProgress(group) {
  let keep = 0;
  let move = 0;
  group.files.forEach((file) => {
    const status = statusMap.get(file.id);
    if (status === "keep") keep += 1;
    if (status === "move") move += 1;
  });
  return { keep, move, undecided: group.files.length - keep - move };
}

// Untouched = no marks except the automatic Keep on the best shot.
function isUntouched(group) {
  return group.files.every((file) => !statusMap.has(file.id) || (file.id === group.suggested_id && statusMap.get(file.id) === "keep"));
}

function markCounts() {
  let keep = 0;
  let move = 0;
  statusMap.forEach((status, id) => {
    if (!fileLookup.has(id)) return;
    if (status === "keep") keep += 1;
    if (status === "move") move += 1;
  });
  return { keep, move };
}

function applyState(data) {
  state = data;
  statusMap = new Map(Object.entries(data.decisions || {}));
  groupById = new Map();
  fileLookup = new Map();
  data.groups.forEach((group, index) => {
    group.number = index + 1;
    groupById.set(group.id, group);
    group.files.forEach((file) => fileLookup.set(file.id, { file, group }));
  });
  if (document.activeElement !== $("folderInput")) $("folderInput").value = data.folder || "";
  fillSettings(data.options || {});
  if (lb && !groupById.has(lb.group.id)) closeLightbox();
  renderNotices();
  render();
  handleJob(data.job);
}

async function loadState() {
  applyState(await api("/api/state"));
}

function fillSettings(options) {
  document.querySelectorAll("[data-option]").forEach((input) => {
    const value = options[input.dataset.option];
    if (input.type === "checkbox") input.checked = Boolean(value);
    else if (value !== undefined) input.value = value;
  });
}

function readSettings() {
  const options = {};
  document.querySelectorAll("[data-option]").forEach((input) => {
    options[input.dataset.option] = input.type === "checkbox" ? input.checked : Number(input.value);
  });
  return options;
}

function updateStats() {
  if (!state) return;
  const summary = state.summary;
  const counts = markCounts();
  $("subtitle").textContent = state.scanned
    ? `${summary.group_count} groups to review in ${state.folder}`
    : "Choose a folder and press Scan to find duplicate and similar photos.";
  const stats = state.scanned ? [
    `${summary.image_count} photos scanned`,
    `${summary.group_count} groups`,
    `${summary.exact_group_count} exact`,
    `${summary.similar_group_count} similar`,
    `${summary.moment_group_count} same moment`,
    `${summary.people_group_count} with people`,
    `${counts.keep} marked keep`,
    `${counts.move} marked move`,
    `Moves go to: ${state.move_dir}`,
  ] : [];
  $("stats").innerHTML = stats.map((item) => `<span class="pill">${escapeHtml(item)}</span>`).join("");
  $("moveBtn").textContent = `Move marked (${counts.move})`;
  $("moveBtn").disabled = counts.move === 0;
  $("undoBtn").disabled = !state.can_undo;
}

function renderNotices() {
  const parts = [];
  if (state.scanned && !state.summary.faces_available) {
    parts.push(`<div class="notice">Face and posture ranking is off for this scan. ${escapeHtml(state.summary.faces_reason || "")}</div>`);
  }
  if (state.scanned && !state.summary.heif_available) {
    parts.push('<div class="notice">HEIC/HEIF photos are not scanned. Run <code>pip install pillow-heif</code> and rescan to include them.</div>');
  }
  if (state.errors && state.errors.length) {
    const items = state.errors.slice(0, 200).map((err) => `<li>${escapeHtml(err.rel_path)}: ${escapeHtml(err.error)}</li>`).join("");
    parts.push(`<details class="notice"><summary>${state.errors.length} files could not be read</summary><ul>${items}</ul></details>`);
  }
  $("notices").innerHTML = parts.join("");
}

function filteredGroups() {
  const query = $("searchBox").value.trim().toLowerCase();
  const type = $("typeFilter").value;
  const decision = $("decisionFilter").value;
  return state.groups.filter((group) => {
    if (query && !group.search_text.toLowerCase().includes(query)) return false;
    if (type === "exact" && group.kind !== "exact") return false;
    if (type === "similar" && group.kind !== "similar") return false;
    if (type === "moment" && group.kind !== "moment") return false;
    if (type === "people" && !group.has_faces) return false;
    const progress = groupProgress(group);
    const untouched = isUntouched(group);
    if (decision === "undecided" && !untouched) return false;
    if (decision === "progress" && (untouched || progress.undecided === 0)) return false;
    if (decision === "done" && progress.undecided > 0) return false;
    if (decision === "nokeep" && progress.move !== group.files.length) return false;
    return true;
  });
}

function render() {
  updateStats();
  const container = $("groups");
  if (!state.scanned) {
    container.innerHTML = '<div class="empty">No scan yet for this folder. Use <b>Choose folder...</b> or type a path, then press <b>Scan</b>.</div>';
    renderPager(0, 1);
    return;
  }
  const list = filteredGroups();
  const pages = Math.max(1, Math.ceil(list.length / PAGE_SIZE));
  page = Math.min(page, pages - 1);
  const slice = list.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  if (!state.groups.length) {
    container.innerHTML = '<div class="empty">No duplicate or similar-photo groups were found.</div>';
  } else if (!slice.length) {
    container.innerHTML = '<div class="empty">No groups match the current filters.</div>';
  } else {
    container.innerHTML = slice.map(groupHtml).join("");
  }
  renderPager(list.length, pages);
}

function renderPager(total, pages) {
  const html = total > PAGE_SIZE ? `
    <button type="button" data-page="prev" ${page === 0 ? "disabled" : ""}>&lsaquo; Prev</button>
    <span>Page ${page + 1} of ${pages} &middot; ${total} groups</span>
    <button type="button" data-page="next" ${page >= pages - 1 ? "disabled" : ""}>Next &rsaquo;</button>` : "";
  $("pagerTop").innerHTML = html;
  $("pagerBottom").innerHTML = html;
}

function groupHtml(group) {
  const progress = groupProgress(group);
  const tags = [
    { exact: '<span class="tag exact">Exact copies</span>', similar: '<span class="tag">Similar</span>', moment: '<span class="tag moment">Same moment</span>' }[group.kind],
    group.has_faces ? '<span class="tag people">People</span>' : "",
    progress.move === group.files.length ? '<span class="tag delete">Nothing kept</span>' : "",
  ].join("");
  const meta = `${group.files.length} photos | ${progress.keep} keep | ${progress.move} move | ${progress.undecided} not decided`;
  return `
    <section class="group" data-group-id="${group.id}">
      <div class="group-head">
        <div>
          <div class="group-title"><span>Group ${String(group.number).padStart(3, "0")}</span>${tags}</div>
          <div class="group-meta">${escapeHtml(meta)}</div>
        </div>
        <div class="group-tools">
          <button type="button" data-action="suggest" data-group-id="${group.id}" title="Keep the best shot and move the others">Keep best, move rest</button>
          <button type="button" data-action="move-rest" data-group-id="${group.id}" ${progress.keep ? "" : "disabled"} title="Move every photo not marked Keep">Move all not kept</button>
          <button type="button" data-action="keep-all" data-group-id="${group.id}">Keep all</button>
          <button type="button" data-action="clear" data-group-id="${group.id}">Clear</button>
        </div>
      </div>
      <div class="grid">${group.files.map((file) => cardHtml(file, group)).join("")}</div>
    </section>`;
}

function fileDetails(file, group) {
  const first = [`${file.width} x ${file.height}`, formatBytes(file.size), file.taken].filter(Boolean).join(" | ");
  const second = [
    group.has_faces ? `score ${file.rank.toFixed(1)} (photo ${file.score.toFixed(1)})` : `score ${file.score.toFixed(1)}`,
    `sharp ${file.sharpness.toFixed(0)}`,
    group.kind === "exact" ? "identical" : `match ${file.similarity}%`,
  ].join(" | ");
  return [first, second];
}

function notesHtml(file) {
  return file.notes.map((note) => `<span class="note ${note.kind}">${escapeHtml(note.text)}</span>`).join("");
}

function cardHtml(file, group) {
  const status = statusMap.get(file.id);
  const cls = status === "keep" ? "photo keep" : (status === "move" ? "photo marked" : "photo");
  const tags = [
    file.id === group.suggested_id ? '<span class="tag best">&starf; Best shot</span>' : "",
    status === "keep" ? '<span class="tag keep">Keep</span>' : "",
    status === "move" ? '<span class="tag delete">Move</span>' : "",
  ].join("");
  const [first, second] = fileDetails(file, group);
  const notes = notesHtml(file);
  return `
    <article class="${cls}">
      <img class="thumb" data-action="view" data-id="${file.id}" src="${thumbUrl(file)}" alt="${escapeHtml(file.name)}" loading="lazy">
      <div class="photo-body">
        <div class="tag-row">${tags}</div>
        <div class="name">${escapeHtml(file.name)}</div>
        <div class="detail">${escapeHtml(first)}</div>
        <div class="detail">${escapeHtml(second)}</div>
        ${notes ? `<div class="notes">${notes}</div>` : ""}
        <div class="detail">${escapeHtml(file.rel_path)}</div>
        <div class="choice-row">
          <button type="button" class="toggle ${status === "keep" ? "on-keep" : ""}" data-action="keep" data-id="${file.id}">&check; Keep</button>
          <button type="button" class="toggle ${status === "move" ? "on-move" : ""}" data-action="move" data-id="${file.id}">&rarr; Move</button>
        </div>
        <a class="open-link" href="${imageUrl(file)}" target="_blank" rel="noreferrer">Open full size</a>
      </div>
    </article>`;
}

function refreshGroup(group) {
  const el = document.querySelector(`section.group[data-group-id="${group.id}"]`);
  if (el) el.outerHTML = groupHtml(group);
}

function changed(groups) {
  groups.forEach(refreshGroup);
  updateStats();
  scheduleSave();
  if (lb) renderLightbox();
}

function toggleStatus(id, value) {
  if (statusMap.get(id) === value) statusMap.delete(id);
  else statusMap.set(id, value);
  const entry = fileLookup.get(id);
  changed(entry ? [entry.group] : []);
}

function groupAction(group, action) {
  group.files.forEach((file) => {
    if (action === "suggest") statusMap.set(file.id, file.id === group.suggested_id ? "keep" : "move");
    else if (action === "keep-all") statusMap.set(file.id, "keep");
    else if (action === "move-rest" && statusMap.get(file.id) !== "keep") statusMap.set(file.id, "move");
    else if (action === "clear") statusMap.delete(file.id);
  });
  changed([group]);
}

function scheduleSave() {
  window.clearTimeout(saveTimer);
  saveTimer = window.setTimeout(() => {
    saveTimer = null;
    saveDecisions().catch((err) => toast(`Could not save marks: ${err.message}`));
  }, 500);
}

async function saveDecisions() {
  if (!state || !state.folder) return;
  await api("/api/decisions", { folder: state.folder, status: Object.fromEntries(statusMap) });
}

async function flushSave() {
  if (!saveTimer) return;
  window.clearTimeout(saveTimer);
  saveTimer = null;
  await saveDecisions();
}

function handleJob(job) {
  const running = Boolean(job && job.running);
  $("progress").hidden = !running;
  $("scanBtn").disabled = running;
  $("browseBtn").disabled = running;
  $("moveBtn").disabled = running || markCounts().move === 0;
  $("undoBtn").disabled = running || !(state && state.can_undo);
  if (!running) return;
  const fill = $("progressFill");
  if (job.total) {
    fill.classList.remove("indeterminate");
    fill.style.width = `${Math.round((job.done / job.total) * 100)}%`;
    $("progressText").textContent = `${job.phase}: ${job.done} / ${job.total}`;
  } else {
    fill.classList.add("indeterminate");
    fill.style.width = "";
    $("progressText").textContent = `${job.phase}...`;
  }
  if (!pollTimer) pollTimer = window.setTimeout(pollJob, 800);
}

async function pollJob() {
  pollTimer = null;
  let job;
  try {
    job = await api("/api/job");
  } catch (err) {
    pollTimer = window.setTimeout(pollJob, 2000);
    return;
  }
  if (job.running) {
    handleJob(job);
    return;
  }
  handleJob(job);
  if (job.error) toast(`Scan failed: ${job.error}`);
  else if (job.message) toast(job.message);
  await loadState();
}

async function startScan() {
  const folder = $("folderInput").value.trim();
  if (!folder) {
    toast("Choose or type a folder first.");
    return;
  }
  await flushSave();
  const job = await api("/api/scan", { folder, force: $("forceScan").checked });
  $("forceScan").checked = false;
  handleJob(job);
}

async function browse() {
  $("browseBtn").disabled = true;
  try {
    toast("A folder picker opened. It may be behind the browser window.");
    const result = await api("/api/pick-folder", { initial: $("folderInput").value.trim() });
    if (result.path) {
      $("folderInput").value = result.path;
      await startScan();
    }
  } finally {
    $("browseBtn").disabled = false;
  }
}

async function applySettings() {
  await flushSave();
  applyState(await api("/api/settings", { options: readSettings() }));
  toast("Settings applied. Groups were rebuilt.");
}

function acceptExact() {
  const groups = state.groups.filter((group) => group.kind === "exact" && isUntouched(group));
  groups.forEach((group) => group.files.forEach((file) => statusMap.set(file.id, file.id === group.suggested_id ? "keep" : "move")));
  render();
  scheduleSave();
  toast(groups.length ? `Marked ${groups.length} exact-copy groups: one kept, copies to move.` : "No untouched exact-copy groups left.");
}

async function moveMarked() {
  const ids = Array.from(statusMap.entries()).filter(([id, value]) => value === "move" && fileLookup.has(id)).map(([id]) => id);
  if (!ids.length) return;
  const nothingKept = state.groups.filter((group) => group.files.every((file) => statusMap.get(file.id) === "move")).length;
  let message = `Move ${ids.length} photos into\n${state.move_dir}?\n\nNothing is deleted, and you can undo this move.`;
  if (nothingKept) message += `\n\nWarning: in ${nothingKept} group(s) every photo is marked Move, so none of them stays in the folder.`;
  if (!window.confirm(message)) return;
  await flushSave();
  const result = await api("/api/move", { ids });
  toast(`Moved ${result.moved.length} photos.${result.errors.length ? ` ${result.errors.length} could not be moved: ${result.errors[0].error}` : ""}`);
  await loadState();
}

async function undoMove() {
  await flushSave();
  const result = await api("/api/undo", {});
  toast(`Restored ${result.restored.length} photos.${result.errors.length ? ` ${result.errors.length} could not be restored: ${result.errors[0].error}` : ""}`);
  await loadState();
}

function openLightbox(id) {
  const entry = fileLookup.get(id);
  if (!entry) return;
  lb = { group: entry.group, index: entry.group.files.indexOf(entry.file) };
  $("lightbox").hidden = false;
  document.body.style.overflow = "hidden";
  renderLightbox();
}

function closeLightbox() {
  lb = null;
  $("lightbox").hidden = true;
  $("lbImg").removeAttribute("src");
  document.body.style.overflow = "";
}

function renderLightbox() {
  const group = lb.group;
  const file = group.files[lb.index];
  const status = statusMap.get(file.id);
  const url = imageUrl(file);
  if ($("lbImg").getAttribute("src") !== url) $("lbImg").src = url;
  const best = file.id === group.suggested_id ? " \u2605 Best shot" : "";
  $("lbTitle").textContent = `Group ${group.number} \u00b7 ${lb.index + 1} of ${group.files.length} \u00b7 ${file.name}${best}`;
  $("lbInfo").textContent = fileDetails(file, group).join(" | ");
  $("lbNotes").innerHTML = notesHtml(file);
  $("lbKeep").className = `toggle ${status === "keep" ? "on-keep" : ""}`;
  $("lbMove").className = `toggle ${status === "move" ? "on-move" : ""}`;
  const next = group.files[(lb.index + 1) % group.files.length];
  new Image().src = imageUrl(next);
}

function stepLightbox(delta) {
  const count = lb.group.files.length;
  lb.index = (lb.index + delta + count) % count;
  renderLightbox();
}

const report = (promise) => promise.catch((err) => toast(err.message));

$("groups").addEventListener("click", (event) => {
  const target = event.target.closest("[data-action]");
  if (!target) return;
  const action = target.dataset.action;
  if (action === "view") {
    openLightbox(target.dataset.id);
  } else if (action === "keep" || action === "move") {
    toggleStatus(target.dataset.id, action);
  } else {
    const group = groupById.get(target.dataset.groupId);
    if (group) groupAction(group, action);
  }
});

document.addEventListener("click", (event) => {
  const target = event.target.closest("[data-page]");
  if (!target) return;
  page += target.dataset.page === "next" ? 1 : -1;
  render();
  window.scrollTo({ top: 0 });
});

document.addEventListener("keydown", (event) => {
  if (!lb) return;
  const file = lb.group.files[lb.index];
  const key = event.key.toLowerCase();
  if (key === "arrowleft") stepLightbox(-1);
  else if (key === "arrowright") stepLightbox(1);
  else if (key === "escape") closeLightbox();
  else if (key === "k") toggleStatus(file.id, "keep");
  else if (key === "m") toggleStatus(file.id, "move");
  else return;
  event.preventDefault();
});

["searchBox", "typeFilter", "decisionFilter"].forEach((id) => {
  $(id).addEventListener(id === "searchBox" ? "input" : "change", () => {
    page = 0;
    render();
  });
});
$("folderInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter") report(startScan());
});
$("browseBtn").addEventListener("click", () => report(browse()));
$("scanBtn").addEventListener("click", () => report(startScan()));
$("settingsBtn").addEventListener("click", () => {
  $("settings").hidden = !$("settings").hidden;
  $("settingsBtn").setAttribute("aria-expanded", String(!$("settings").hidden));
});
$("applySettingsBtn").addEventListener("click", () => report(applySettings()));
$("acceptExactBtn").addEventListener("click", acceptExact);
$("moveBtn").addEventListener("click", () => report(moveMarked()));
$("undoBtn").addEventListener("click", () => report(undoMove()));
$("lbPrev").addEventListener("click", () => stepLightbox(-1));
$("lbNext").addEventListener("click", () => stepLightbox(1));
$("lbClose").addEventListener("click", closeLightbox);
$("lbKeep").addEventListener("click", () => toggleStatus(lb.group.files[lb.index].id, "keep"));
$("lbMove").addEventListener("click", () => toggleStatus(lb.group.files[lb.index].id, "move"));
$("lbStage").addEventListener("click", (event) => {
  if (event.target === $("lbStage")) closeLightbox();
});

window.addEventListener("beforeunload", () => {
  if (!saveTimer || !state) return;
  fetch("/api/decisions", {
    method: "POST",
    keepalive: true,
    headers: { "Content-Type": "application/json", "X-Review-Token": TOKEN },
    body: JSON.stringify({ folder: state.folder, status: Object.fromEntries(statusMap) }),
  });
});

loadState().catch((err) => {
  $("groups").innerHTML = `<div class="empty">${escapeHtml(err.message)}</div>`;
  $("subtitle").textContent = "Could not load groups.";
});
