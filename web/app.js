const sampleSelect    = document.getElementById("sampleSelect");
const openPdfBtn      = document.getElementById("openPdfBtn");
const openNodesBtn    = document.getElementById("openNodesBtn");
const pdfFile         = document.getElementById("pdfFile");
const uploadDropZone  = document.getElementById("uploadDropZone");
const uploadDropLabel = document.getElementById("uploadDropLabel");
const questionInput   = document.getElementById("questionInput");
const prepareBtn      = document.getElementById("prepareBtn");
const askBtn          = document.getElementById("askBtn");
const demoStatus      = document.getElementById("demoStatus");
const prepLabel       = document.getElementById("prepLabel");
const prepEta         = document.getElementById("prepEta");
const prepProgress    = document.getElementById("prepProgress");
const prepLoadingBlock = document.getElementById("prepLoadingBlock");
const prepReadyBlock   = document.getElementById("prepReadyBlock");
const prepReadyLabel   = document.getElementById("prepReadyLabel");

const emptyState      = document.getElementById("emptyState");
const resultBody      = document.getElementById("resultBody");
const statusBadge     = document.getElementById("statusBadge");
const docName         = document.getElementById("docName");
const shortAnswer     = document.getElementById("shortAnswer");
const reasonText      = document.getElementById("reasonText");
const clausesList     = document.getElementById("clausesList");
const sampleBoxes     = document.getElementById("sampleBoxes");
const sampleActiveLabel = document.getElementById("sampleActiveLabel");
const prepDetail      = document.getElementById("prepDetail");
const prepStageBadge  = document.getElementById("prepStageBadge");
const prepChunkStat   = document.getElementById("prepChunkStat");
const prepNodeStat    = document.getElementById("prepNodeStat");
const prepLLMStat     = document.getElementById("prepLLMStat");
const askLoading      = document.getElementById("askLoading");
const askLoadingLabel = document.getElementById("askLoadingLabel");
const stepVariants    = document.getElementById("stepVariants");
const stepRetrieve    = document.getElementById("stepRetrieve");
const stepTopology    = document.getElementById("stepTopology");
const stepGate        = document.getElementById("stepGate");
const stepAnswer      = document.getElementById("stepAnswer");

let preparedDocToken = null;
let prepareJobId     = null;
let preparePollTimer = null;
let savedDocs        = [];
let isPreparing      = false;

function updateLiveDebug({ phase, status, token, api, err } = {}) {
  void phase;
  void status;
  void token;
  void api;
  void err;
}

function stopAskRuntimeDebug() {
  return;
}

function startAskRuntimeDebug() {
  return;
}

async function loadRuntimeDebugSnapshot() {
  return;
}

function _getAccessCode() {
  return (localStorage.getItem("rag_access_code") || "").trim();
}

function _withAccessCode(url) {
  const code = _getAccessCode();
  if (!code) return url;
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}access_code=${encodeURIComponent(code)}`;
}

async function apiFetch(url, options = {}) {
  const headers = new Headers(options.headers || {});
  const code = _getAccessCode();
  if (code) headers.set("x-access-code", code);
  updateLiveDebug({ api: `${options.method || "GET"} ${url}` });
  return fetch(_withAccessCode(url), { ...options, headers });
}

async function apiFetchJson(url, options = {}) {
  const res = await apiFetch(url, options);
  const raw = await res.text();
  let data = null;
  if (raw) {
    try {
      data = JSON.parse(raw);
    } catch (_) {
      data = null;
    }
  }
  return { res, data, raw };
}

function debug(msg, level = "info") {
  void msg;
  void level;
}

function setAskEnabled(enabled) {
  askBtn.disabled    = !enabled;
}

function fmtSec(s) {
  const n = Math.max(0, Number(s || 0));
  if (n < 60) return `${Math.round(n)}s`;
  return `${Math.floor(n / 60)}m ${Math.round(n % 60)}s`;
}

function nextPaint() {
  return new Promise((resolve) => {
    // Safari/iOS may defer first paint unless we yield across 2 frames.
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        setTimeout(resolve, 0);
      });
    });
  });
}

/* ── Sample box state helpers ────────────────────────────── */
function _boxEl(name) {
  return sampleBoxes.querySelector(`.sample-box[data-name="${CSS.escape(name)}"]`);
}
function setBoxState(name, state) {
  // state: 'idle' | 'loading' | 'ready'
  const box = _boxEl(name);
  if (!box) return;
  box.classList.remove("sample-box-loading", "sample-box-ready");
  const sub = box.querySelector(".sample-box-sub");
  if (state === "loading") {
    box.classList.add("sample-box-loading");
    if (sub) sub.textContent = "Building knowledge graph…";
  } else if (state === "ready") {
    box.classList.add("sample-box-ready");
    if (sub) sub.textContent = "Ready for questions";
  } else {
    if (sub) sub.textContent = "Sample contract · PDF";
  }
}

/* ── Prep UI state ───────────────────────────────────────── */
function showPrepLoading(docName_) {
  updateLiveDebug({ phase: "preparing", status: `building ${docName_ || "document"}`, err: "none" });
  prepLoadingBlock.style.display = "block";
  prepReadyBlock.style.display   = "none";
  prepLabel.textContent = `Building knowledge graph for ${docName_ || "document"}…`;
  prepEta.textContent   = "";
  prepProgress.style.width = "3%";
}

function showPrepReady(docName_) {
  updateLiveDebug({ phase: "ready", status: `${docName_ || "Document"} ready` });
  prepLoadingBlock.style.display = "none";
  prepReadyBlock.style.display   = "flex";
  prepReadyLabel.textContent = `${docName_ || "Document"} is ready for questions`;
}

function hidePrepUI() {
  prepLoadingBlock.style.display = "none";
  prepReadyBlock.style.display   = "none";
  prepProgress.style.width = "0%";
}

function clearPreparedState() {
  debug("Reset prepared state.", "warn");
  preparedDocToken = null;
  updateLiveDebug({ token: "", status: "cleared selection" });
  isPreparing = false;
  setAskEnabled(false);
  hidePrepUI();
  // clear box sub-labels
  sampleBoxes.querySelectorAll(".sample-box").forEach(b => {
    b.classList.remove("sample-box-loading", "sample-box-ready");
    const sub = b.querySelector(".sample-box-sub");
    if (sub) sub.textContent = "Sample contract · PDF";
  });
  if (sampleActiveLabel) sampleActiveLabel.textContent = "";
}

/* ── Sample box selection ────────────────────────────────── */
function setSampleBox(name, triggerPrepare = true) {
  sampleSelect.value = name || "";
  if (sampleActiveLabel) sampleActiveLabel.textContent = name ? `Selected: ${name}` : "";
  sampleBoxes.querySelectorAll(".sample-box").forEach(b => {
    b.classList.toggle("sample-box-active", b.dataset.name === name);
  });
  clearPreparedState();
  if (name && triggerPrepare) {
    pdfFile.value = "";
    prepareDocument();
  }
}

/* ── Saved docs ──────────────────────────────────────────── */
function refreshBoxStates() {
  // Mark sample boxes as ready if they have a baked-in token
  sampleBoxes.querySelectorAll(".sample-box").forEach(box => {
    if (box.dataset.token && !box.classList.contains("sample-box-loading")) {
      box.classList.add("sample-box-ready");
      const sub = box.querySelector(".sample-box-sub");
      if (sub) sub.textContent = "Ready for questions";
    }
  });
}

async function loadSavedDocs() {
  debug("Loading saved document list...");
  updateLiveDebug({ phase: "loading", status: "loading saved docs" });
  try {
    const res  = await apiFetch("/api/saved-docs");
    const data = await res.json();
    savedDocs  = data.documents || [];
    debug(`Saved docs loaded: ${savedDocs.length}`);
    updateLiveDebug({ status: `saved docs ${savedDocs.length}` });
  } catch (_) {
    debug("Failed to load saved docs endpoint.", "err");
    updateLiveDebug({ err: "failed /api/saved-docs", status: "saved docs failed" });
  }
}

/* ── Sample docs grid ────────────────────────────────────── */
async function loadSamples(attempt = 0) {
  debug("Loading sample document list...");
  updateLiveDebug({ phase: "loading", status: `loading samples (${attempt + 1}/5)` });
  try {
    const res  = await apiFetch("/api/demo-docs");
    const data = await res.json();
    const docs = data.documents || [];
    sampleBoxes.innerHTML = "";
    if (!docs.length) {
      if (attempt < 4) {
        sampleBoxes.innerHTML = "<span style='color:#888;font-size:0.85rem;'>Loading sample docs…</span>";
        debug(`No sample docs yet (attempt ${attempt + 1}/5). Retrying...`, "warn");
        setTimeout(() => { void loadSamples(attempt + 1); }, 800);
        return;
      }
      sampleBoxes.innerHTML = "<span style='color:#888;font-size:0.85rem;'>Please refresh browser to load sample docs.</span>";
      updateLiveDebug({ status: "no sample docs found" });
      return;
    }
    updateLiveDebug({ status: `samples loaded: ${docs.length}` });

    sampleSelect.innerHTML = "<option value=''></option>" +
      docs.map(d => `<option value="${d.name}">${d.name}</option>`).join("");

    for (const d of docs) {
      const label = d.name.replace(/\.pdf$/i, "").replace(/_/g, " ");
      const box = document.createElement("button");
      box.type = "button";
      box.className = "sample-box";
      box.dataset.name  = d.name;
      box.dataset.token = d.token || "";
      box.innerHTML = `
        <span class="sample-box-icon">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
            <path d="M4 3a1 1 0 011-1h6l4 4v11a1 1 0 01-1 1H5a1 1 0 01-1-1V3z" fill="currentColor" opacity="0.15"/>
            <path d="M11 2v4h4M4 3a1 1 0 011-1h6l4 4v11a1 1 0 01-1 1H5a1 1 0 01-1-1V3z" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round" fill="none"/>
            <line x1="7" y1="9"  x2="13" y2="9"  stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>
            <line x1="7" y1="12" x2="13" y2="12" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>
            <line x1="7" y1="15" x2="10" y2="15" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>
          </svg>
        </span>
        <span class="sample-box-text">
          <span class="sample-box-name">${label}</span>
          <span class="sample-box-sub">Sample contract · PDF</span>
        </span>
        <span class="sample-box-status" aria-hidden="true"></span>
      `;
      box.addEventListener("click", () => {
        if (isPreparing) return;
        if (sampleSelect.value === d.name) return;

        // Token is baked in at load time from server, with no async dependency
        const token = box.dataset.token;
        if (token) {
          setSampleBox(d.name, false);
          preparedDocToken = token;
          updateLiveDebug({ token, phase: "ready", status: `${d.name} instant ready`, err: "none" });
          setAskEnabled(true);
          showPrepReady(d.name);
          setBoxState(d.name, "ready");
          demoStatus.textContent = "Document is ready. Ask a question.";
          debug(`Instant load: ${d.name} (cached token)`);
        } else {
          setSampleBox(d.name, true);           // triggers full prepare
        }
      });
      sampleBoxes.appendChild(box);
    }
  } catch (_) {
    if (attempt < 4) {
      debug(`Failed to load sample docs endpoint (attempt ${attempt + 1}/5). Retrying...`, "warn");
      sampleBoxes.innerHTML = "<span style='color:#888;font-size:0.85rem;'>Loading sample docs…</span>";
      setTimeout(() => { void loadSamples(attempt + 1); }, 800);
      return;
    }
    debug("Failed to load sample docs endpoint.", "err");
    sampleBoxes.innerHTML = "<span style='color:#f66;font-size:0.85rem;'>Failed to load samples</span>";
    updateLiveDebug({ err: "failed /api/demo-docs", status: "sample load failed" });
  }
}

/* ── Ask UI helpers ──────────────────────────────────────── */
function setAsking(isLoading) {
  updateLiveDebug({ phase: isLoading ? "asking" : "ready", status: isLoading ? "question running" : "idle" });
  askBtn.textContent    = isLoading ? "Running…" : "Ask question";
  setAskEnabled(!isLoading && !!preparedDocToken);
  if (!isLoading) demoStatus.textContent = "Ready.";
}

/* ── Prepare poll ────────────────────────────────────────── */
async function pollPrepareJob(jobId, sampleName) {
  updateLiveDebug({ phase: "preparing", status: `polling job ${jobId.slice(0, 8)}` });
  if (preparePollTimer) clearInterval(preparePollTimer);

  preparePollTimer = setInterval(async () => {
    try {
      const res  = await apiFetch(`/api/prepare-doc/${jobId}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Prepare job failed.");

      const modelPct   = Number(data.progress || 0);
      const chunkPct   = Number(data.chunk_progress || 0);
      const llmPct     = Number(data.llm_progress || 0);
      const blendedPct = Math.max(modelPct, chunkPct * 0.85 + llmPct * 0.15);
      const pct        = Math.round(blendedPct * 100);

      prepProgress.style.width = `${Math.max(pct, 5)}%`;

      if (data.remaining_sec != null) {
        const stage = String((data.metrics && data.metrics.stage) || "").toLowerCase();
        if (data.status === "running" && data.remaining_sec < 12 && !["embedding_nodes", "ready", "done"].includes(stage)) {
          prepEta.textContent = "Finalizing extraction...";
        } else {
          prepEta.textContent = `ETA ${fmtSec(data.remaining_sec)} (estimate only; may be shorter or longer)`;
        }
      } else if (data.over_eta_sec != null) {
        prepEta.textContent = `Running longer than initial estimate (${fmtSec(data.over_eta_sec)} so far)`;
      }

      if (data.metrics) {
        const m  = data.metrics;
        const tc = Number(m.total_chunks || 0);
        const ec = Number(m.extracted_chunks || 0);
        const nc = Number(m.nodes_created || 0);
        const lc = Number(m.llm_calls_actual || 0);
        const lt = Number(data.llm_calls_estimated_total || 0);
        if (data.status === "running" || data.status === "queued") {
          const stageLabel = (m.stage || "starting").replace(/_/g, " ");
          prepStageBadge.textContent  = stageLabel;
          prepChunkStat.textContent   = tc > 0 ? `chunks ${ec}/${tc}` : "chunks n/a";
          prepNodeStat.textContent    = `${nc} nodes`;
          prepLLMStat.textContent     = `${lc}${lt > lc ? `/${lt}` : ""} LLM calls`;
        }
        debug(`Prepare poll: status=${data.status} ${pct}% chunks=${ec}/${tc} nodes=${nc} llm=${lc}/${lt}`);
      }

      if (data.status === "done") {
        clearInterval(preparePollTimer);
        preparePollTimer = null;
        isPreparing = false;
        preparedDocToken = data.doc_token;
        updateLiveDebug({ token: data.doc_token || "", phase: "ready", status: "prepare done", err: "none" });
        prepProgress.style.width = "100%";

        setTimeout(() => {
          showPrepReady(data.document);
          if (sampleName) {
            setBoxState(sampleName, "ready");
            const box = _boxEl(sampleName);
            if (box && data.doc_token) box.dataset.token = data.doc_token;
          }
        }, 350);

        setAskEnabled(true);
        demoStatus.textContent = "Document is ready. Ask a question.";
        debug(`Prepare done. doc_token=${(data.doc_token || "").slice(0, 10)}…`);
        loadSavedDocs();
      }

      if (data.status === "error") {
        clearInterval(preparePollTimer);
        preparePollTimer = null;
        isPreparing = false;
        hidePrepUI();
        if (sampleName) setBoxState(sampleName, "idle");
        updateLiveDebug({ phase: "error", status: "prepare error", err: data.error || "prepare failed" });
        showError(data.error || "Failed to prepare document.");
      }
    } catch (err) {
      debug(`Prepare poll error: ${err.message}`, "err");
      clearInterval(preparePollTimer);
      preparePollTimer = null;
      isPreparing = false;
      hidePrepUI();
      if (sampleName) setBoxState(sampleName, "idle");
      updateLiveDebug({ phase: "error", status: "prepare poll failed", err: err.message || "poll error" });
      showError(err.message || "Failed to check preparation status.");
    }
  }, 1000);
}

/* ── Prepare document ────────────────────────────────────── */
async function prepareDocument() {
  if (isPreparing) return;
  isPreparing = true;

  const activeSample = sampleSelect.value;
  const activeFile   = pdfFile.files[0];

  if (!activeSample && !activeFile) {
    isPreparing = false;
    return;
  }

  debug(`Prepare start: sample="${activeSample || ""}" file="${activeFile?.name || ""}"`);
  updateLiveDebug({ phase: "preparing", status: `start ${activeSample || activeFile?.name || "document"}`, err: "none" });

  showPrepLoading(activeSample || activeFile?.name || "document");
  if (activeSample) setBoxState(activeSample, "loading");
  setAskEnabled(false);
  demoStatus.textContent = "Building knowledge graph…";

  const form = new FormData();
  if (activeSample) form.append("sample_name", activeSample);
  if (activeFile)   form.append("file", activeFile);

  try {
    // Ensure loader/progress UI paints before network + backend work begins.
    await nextPaint();
    const res  = await apiFetch("/api/prepare-doc", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Could not start document preparation.");

    debug(`Prepare job started: id=${data.job_id} eta=${data.estimate_sec}s`);
    updateLiveDebug({ status: `job started (${String(data.job_id || "").slice(0, 8)})` });
    prepareJobId = data.job_id;
    prepLabel.textContent = `Building knowledge graph for ${data.document}…`;
    if (data.estimate_sec) prepEta.textContent = `ETA ${fmtSec(data.estimate_sec)} (estimate only; may be shorter or longer)`;

    pollPrepareJob(prepareJobId, activeSample);
    loadSavedDocs();
  } catch (err) {
    debug(`Prepare start error: ${err.message}`, "err");
    isPreparing = false;
    hidePrepUI();
    if (activeSample) setBoxState(activeSample, "idle");
    updateLiveDebug({ phase: "error", status: "prepare start failed", err: err.message || "prepare start error" });
    showError(err.message || "Failed to prepare document.");
  }
}

/* ── Ask step animator ───────────────────────────────────── */
const ASK_STEPS = [
  { el: () => stepVariants,  label: "Generating query variants…",   ms: 4000 },
  { el: () => stepRetrieve,  label: "Retrieving relevant clauses…", ms: 1500 },
  { el: () => stepTopology,  label: "Scoring graph topology…",      ms:  300 },
  { el: () => stepGate,      label: "Running answerability gate…",  ms: 2500 },
  { el: () => stepAnswer,    label: "Generating answer…",           ms: null  },
];
let _askStepTimer = null;

function startAskSteps(isBatch = false) {
  if (!askLoading) return;
  emptyState.classList.add("hidden");
  resultBody.classList.add("hidden");
  askLoading.classList.remove("hidden");

  [stepVariants, stepRetrieve, stepTopology, stepGate, stepAnswer].forEach(el => {
    if (el) el.classList.remove("active", "done");
  });
  askLoading.querySelectorAll(".ask-step-connector").forEach(c => c.classList.remove("done"));

  let idx = 0;
  function advance() {
    if (idx > 0) {
      const prev = ASK_STEPS[idx - 1].el();
      if (prev) { prev.classList.remove("active"); prev.classList.add("done"); }
      const connectors = askLoading.querySelectorAll(".ask-step-connector");
      if (connectors[idx - 1]) connectors[idx - 1].classList.add("done");
    }
    if (idx >= ASK_STEPS.length) return;
    const step = ASK_STEPS[idx];
    const el   = step.el();
    if (el) el.classList.add("active");
    if (askLoadingLabel) askLoadingLabel.textContent = isBatch
      ? step.label.replace("answer", "answers") : step.label;
    idx++;
    if (step.ms !== null) _askStepTimer = setTimeout(advance, step.ms);
  }
  advance();
}

function stopAskSteps() {
  if (_askStepTimer) { clearTimeout(_askStepTimer); _askStepTimer = null; }
  if (askLoading) askLoading.classList.add("hidden");
  [stepVariants, stepRetrieve, stepTopology, stepGate, stepAnswer].forEach(el => {
    if (el) { el.classList.remove("active"); el.classList.add("done"); }
  });
  if (askLoading) askLoading.querySelectorAll(".ask-step-connector").forEach(c => c.classList.add("done"));
}

/* ── Result rendering ────────────────────────────────────── */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function showError(message) {
  debug(`UI error: ${message}`, "err");
  updateLiveDebug({ phase: "error", status: "ui error", err: message || "unknown error" });
  emptyState.classList.remove("hidden");
  resultBody.classList.add("hidden");
  emptyState.innerHTML = `<div class="empty-icon">!</div><p>${escapeHtml(message)}</p>`;
}

function renderResult(data) {
  debug(`Render result: answerable=${Boolean(data.answerable)} doc=${data.document || "n/a"}`);
  emptyState.classList.add("hidden");
  resultBody.classList.remove("hidden");

  const isAnswerable = !!data.answerable;
  statusBadge.textContent = isAnswerable ? "Answerable" : "Unanswerable";
  statusBadge.classList.toggle("ok",  isAnswerable);
  statusBadge.classList.toggle("bad", !isAnswerable);

  docName.textContent     = data.document ? `Document: ${data.document}` : "";
  shortAnswer.textContent = data.short_answer || (isAnswerable
    ? "The system found evidence but could not render a short answer."
    : "No direct answer is supported by this document.");
  reasonText.textContent  = data.reason || (isAnswerable
    ? "The retrieved clauses directly support this answer."
    : "The document does not contain enough direct evidence to answer safely.");

  clausesList.innerHTML = "";
  const clauses = data.supporting_clauses || [];
  if (!clauses.length) {
    clausesList.innerHTML = "<p class='hint'>No supporting text returned.</p>";
  } else {
    const initialCount = 3;
    const maxCount = Math.min(6, clauses.length);
    let expanded = false;

    function renderClauses() {
      clausesList.innerHTML = "";
      const count = expanded ? maxCount : Math.min(initialCount, maxCount);
      const visible = clauses.slice(0, count);
      for (const clause of visible) {
        const card = document.createElement("article");
        card.className = "clause";
        card.innerHTML = `
          <p>${escapeHtml(clause.text || "")}</p>
        `;
        clausesList.appendChild(card);
      }

      if (maxCount > initialCount) {
        const toggleBtn = document.createElement("button");
        toggleBtn.type = "button";
        toggleBtn.className = "btn btn-secondary";
        toggleBtn.style.marginTop = "0.4rem";
        toggleBtn.textContent = expanded ? "Show less" : "Show more";
        toggleBtn.addEventListener("click", () => {
          expanded = !expanded;
          renderClauses();
        });
        clausesList.appendChild(toggleBtn);
      }
    }

    renderClauses();
  }

}

/* ── Ask question ────────────────────────────────────────── */
async function askQuestion() {
  const question = questionInput.value.trim();
  if (!question)          { showError("Please enter a question first."); return; }
  if (!preparedDocToken)  { showError("Select and load a document first."); return; }

  const form = new FormData();
  form.append("question",  question);
  form.append("doc_token", preparedDocToken);
  debug(`Ask: q="${question.slice(0, 120)}"`);
  updateLiveDebug({ phase: "asking", status: "sending /api/ask", err: "none" });

  setAsking(true);
  startAskSteps(false);
  startAskRuntimeDebug();

  try {
    // Ensure ask loading animation is visible immediately on first run.
    await nextPaint();
    const { res, data, raw } = await apiFetchJson("/api/ask", { method: "POST", body: form });
    stopAskSteps();
    if (!res.ok) {
      const detail =
        (data && (data.detail || data.error || data.message))
        || (raw && raw.slice(0, 220))
        || `HTTP ${res.status}`;
      debug(`Ask failed: HTTP ${res.status} detail=${detail}`, "err");
      updateLiveDebug({ phase: "error", status: `ask failed ${res.status}`, err: detail });
      showError(`Request failed (${res.status}). ${detail}`);
      stopAskRuntimeDebug();
      return;
    }
    if (!data || typeof data !== "object") {
      const snippet = raw ? raw.slice(0, 220) : "empty response body";
      debug(`Ask failed: non-JSON success response: ${snippet}`, "err");
      updateLiveDebug({ phase: "error", status: "ask non-json", err: snippet });
      showError(`Server returned an unexpected response. ${snippet}`);
      stopAskRuntimeDebug();
      return;
    }
    updateLiveDebug({ phase: "ready", status: "answer received", err: "none" });
    renderResult(data);
  } catch (err) {
    stopAskSteps();
    const detail = err?.message ? String(err.message) : "Unknown network error";
    debug(`Ask network error: ${detail}`, "err");
    updateLiveDebug({ phase: "error", status: "network/server error", err: detail });
    showError(`Network or server error. ${detail}`);
  } finally {
    stopAskRuntimeDebug();
    setAsking(false);
  }
}

/* ── Event listeners ─────────────────────────────────────── */
prepareBtn.addEventListener("click", prepareDocument);
askBtn.addEventListener("click", askQuestion);

questionInput.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") askQuestion();
});

// File upload, auto-prepare (or instant load if already cached)
pdfFile.addEventListener("change", () => {
  if (!pdfFile.files[0]) return;
  const filename = pdfFile.files[0].name;
  uploadDropLabel.textContent = filename;
  setSampleBox("", false);

  const cached = savedDocs.find(s => s.name === filename);
  if (cached) {
    clearPreparedState();
    preparedDocToken = cached.token;
    updateLiveDebug({ token: cached.token || "", phase: "ready", status: "uploaded doc reused", err: "none" });
    setAskEnabled(true);
    showPrepReady(filename);
    demoStatus.textContent = "Document is ready. Ask a question.";
    debug(`Instant load: using cached token for uploaded ${filename}`);
    return;
  }

  clearPreparedState();
  prepareDocument();
});

// Drag & drop
uploadDropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadDropZone.classList.add("drag-over");
});
uploadDropZone.addEventListener("dragleave", () => uploadDropZone.classList.remove("drag-over"));
uploadDropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadDropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (!file || file.type !== "application/pdf") return;
  // Assign to file input via DataTransfer
  const dt = new DataTransfer();
  dt.items.add(file);
  pdfFile.files = dt.files;
  uploadDropLabel.textContent = file.name;
  setSampleBox("", false);
  clearPreparedState();
  prepareDocument();
});

async function _findReachableUrl(urls) {
  for (const url of urls) {
    try {
      const res = await apiFetch(url, { method: "HEAD" });
      if (res.ok) return url;
      if (res.status === 405) return url;
    } catch (_) {
      // try next candidate
    }
  }
  return null;
}

async function _openSelected(kind) {
  let tokenToOpen = (preparedDocToken || "").trim();
  const sampleName = (sampleSelect?.value || "").trim();
  if (sampleName) {
    const box = _boxEl(sampleName);
    tokenToOpen = tokenToOpen || String(box?.dataset?.token || "").trim();
    if (tokenToOpen) preparedDocToken = tokenToOpen;
  }

  const tokenCandidates = tokenToOpen
    ? [
        kind === "nodes"
          ? `/api/prepared-nodes/${encodeURIComponent(tokenToOpen)}`
          : `/api/prepared-doc/${encodeURIComponent(tokenToOpen)}`,
      ]
    : [];

  const sampleCandidates = sampleName
    ? [
        kind === "nodes"
          ? `/api/sample-nodes/${encodeURIComponent(sampleName)}`
          : `/api/sample-doc/${encodeURIComponent(sampleName)}`,
      ]
    : [];

  const url = await _findReachableUrl([...tokenCandidates, ...sampleCandidates]);
  if (!url) {
    updateLiveDebug({ phase: "error", status: `open ${kind} failed`, err: "no reachable url" });
    showError("Could not open this file. If you just updated code, restart the backend server and try again.");
    return;
  }

  window.open(_withAccessCode(url), "_blank", "noopener,noreferrer");
}

if (openPdfBtn) {
  openPdfBtn.addEventListener("click", () => { void _openSelected("pdf"); });
}
if (openNodesBtn) {
  openNodesBtn.addEventListener("click", () => { void _openSelected("nodes"); });
}

/* ── Boot ────────────────────────────────────────────────── */
setAskEnabled(false);
updateLiveDebug({ phase: "boot", status: "ui booting", token: "", api: "-", err: "none" });
loadSamples();
loadSavedDocs();
loadRuntimeDebugSnapshot();
