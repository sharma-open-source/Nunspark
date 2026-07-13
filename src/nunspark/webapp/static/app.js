const $ = (id) => document.getElementById(id);
const api = (p, opts) => fetch(p, opts).then((r) => {
  if (!r.ok) return r.json().then((e) => { throw new Error(e.detail || r.statusText); });
  return r.json();
});

// Escape server/filesystem-derived strings (model names/paths, uploaded
// filenames, job fields) before interpolating into innerHTML, so a crafted
// filename can't inject markup.
const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

let uploaded = [];   // [{id, name}]
let activeES = null; // EventSource

async function loadModels() {
  const { models } = await api("/api/models");
  const packed = models.filter((m) => m.kind === "packed");
  const unpacked = models.filter((m) => m.kind === "unpacked");
  $("model").innerHTML = packed.map((m) => `<option value="${esc(m.path)}">${esc(m.name)}</option>`).join("");
  $("draft").innerHTML = `<option value="">none (plain)</option>` +
    unpacked.map((m) => `<option value="${esc(m.name)}">${esc(m.name)}</option>`).join("");
  $("unpacked").innerHTML = `<option value="">— pick cached —</option>` +
    unpacked.map((m) => `<option value="${esc(m.name)}">${esc(m.name)}</option>`).join("");
}

$("pack-btn").onclick = async () => {
  const source = $("pack-source").value.trim() || $("unpacked").value;
  if (!source) return;
  $("pack-status").textContent = "packing…";
  try {
    const r = await api("/api/pack", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source }) });
    $("pack-status").textContent = `packed ${r.num_layers} layers`;
    await loadModels();
  } catch (e) { $("pack-status").textContent = "error: " + e.message; }
};

// --- file upload ---
const dz = $("dropzone");
dz.onclick = () => $("files").click();
dz.ondragover = (e) => { e.preventDefault(); dz.classList.add("drag"); };
dz.ondragleave = () => dz.classList.remove("drag");
dz.ondrop = (e) => { e.preventDefault(); dz.classList.remove("drag"); upload(e.dataTransfer.files); };
$("files").onchange = (e) => upload(e.target.files);

async function upload(fileList) {
  const fd = new FormData();
  for (const f of fileList) fd.append("files", f);
  const { files } = await api("/api/files", { method: "POST", body: fd });
  uploaded = uploaded.concat(files);
  $("file-list").innerHTML = uploaded.map((f) => `<li>📄 ${esc(f.name)}</li>`).join("");
}

// --- run batch ---
$("run").onclick = async () => {
  if (!uploaded.length) return alert("Upload at least one file.");
  const advanced = {
    budget: $("budget").value || "4GB",
    kv_bits: $("kv-bits").value ? Number($("kv-bits").value) : null,
    accept_top_k: $("accept-top-k").value ? Number($("accept-top-k").value) : null,
    num_draft_tokens: $("num-draft-tokens").value ? Number($("num-draft-tokens").value) : null,
  };
  const body = {
    model: $("model").value,
    draft: $("draft").value ||  $("draft-source").value.trim() ,
    preset: $("preset").value,
    instruction: $("instruction").value,
    use_chat_template: $("chat-template").checked,
    max_tokens: Number($("max-tokens").value),
    temperature: Number($("temp").value),
    output_dir: $("output-dir").value,
    file_ids: uploaded.map((f) => f.id),
    advanced,
  };
  try {
    await api("/api/batch", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body) });
    refreshJobs();
  } catch (e) { alert(e.message); }
};

// --- jobs ---
async function refreshJobs() {
  const { jobs } = await api("/api/jobs");
  $("jobs").innerHTML = jobs.map((j) => `
    <li data-id="${esc(j.id)}" class="${j.id === currentJob ? "active" : ""}">
      <span>${esc(j.file_name)}</span>
      <span><span class="badge ${esc(j.status)}">${esc(j.status)}</span>
      ${j.status === "queued" || j.status === "running"
        ? `<button class="cancel" data-id="${esc(j.id)}" style="width:auto;padding:.1rem .4rem;">✕</button>` : ""}</span>
    </li>`).join("");
  document.querySelectorAll("#jobs li").forEach((li) =>
    li.onclick = (e) => { if (!e.target.classList.contains("cancel")) watch(li.dataset.id); });
  document.querySelectorAll(".cancel").forEach((b) =>
    b.onclick = () => api(`/api/jobs/${b.dataset.id}/cancel`, { method: "POST" }).then(refreshJobs));
}

let currentJob = null;
function watch(jobId) {
  currentJob = jobId;
  $("active-job").textContent = jobId.slice(0, 8);
  $("output").textContent = "";
  refreshJobs();
  if (activeES) activeES.close();
  activeES = new EventSource(`/api/jobs/${jobId}/events`);
  activeES.onmessage = (m) => {
    const ev = JSON.parse(m.data);
    if (ev.type === "token") $("output").textContent += ev.text;
    else if (ev.type === "metrics") renderMetrics(ev.metrics);
    else if (["done", "error", "cancelled"].includes(ev.type)) {
      if (ev.type === "error") $("output").textContent += `\n\n[error] ${ev.job.error}`;
      if (ev.job && ev.job.metrics) renderMetrics(ev.job.metrics);
      activeES.close(); refreshJobs();
    }
  };
}

$("metrics-toggle").onchange = (e) => $("metrics").classList.toggle("hidden", !e.target.checked);
function renderMetrics(m) {
  if (!m) return;
  const cells = [
    ["tok/s", (m.tok_per_s || 0).toFixed(2)],
    ["tokens", m.tokens || 0],
    ["peak mem", (m.peak_mem_gb || 0).toFixed(2) + " GB"],
    ["weight peak", (m.weight_peak_gb || 0).toFixed(2) + " GB"],
    ["kv peak", (m.kv_peak_gb || 0).toFixed(2) + " GB"],
  ];
  if (m.multiplier !== undefined) {
    cells.push(["multiplier", "×" + m.multiplier.toFixed(2)]);
    cells.push(["deviation", (m.deviation_rate || 0).toFixed(3)]);
  }
  $("metrics").innerHTML = cells.map(([k, v]) => `<div>${k}<br><b>${v}</b></div>`).join("");
}

loadModels();
setInterval(() => { if (document.querySelector("#jobs li")) refreshJobs(); }, 4000);
