const sampleSelect = document.getElementById("sampleSelect");
const savedSelect = document.getElementById("savedSelect");
const useSavedBtn = document.getElementById("useSavedBtn");
const pdfFile = document.getElementById("pdfFile");
const questionInput = document.getElementById("questionInput");
const batchInput = document.getElementById("batchInput");
const prepareBtn = document.getElementById("prepareBtn");
const askBtn = document.getElementById("askBtn");
const askBatchBtn = document.getElementById("askBatchBtn");
const demoStatus = document.getElementById("demoStatus");
const prepLabel = document.getElementById("prepLabel");
const prepEta = document.getElementById("prepEta");
const prepProgress = document.getElementById("prepProgress");
const debugLog = document.getElementById("debugLog");
const clearDebugBtn = document.getElementById("clearDebugBtn");

const emptyState = document.getElementById("emptyState");
const resultBody = document.getElementById("resultBody");
const statusBadge = document.getElementById("statusBadge");
const docName = document.getElementById("docName");
const shortAnswer = document.getElementById("shortAnswer");
const reasonText = document.getElementById("reasonText");
const clausesList = document.getElementById("clausesList");
const metadataRow = document.getElementById("metadataRow");
const examples = document.getElementById("exampleQuestions");
const batchEmptyState = document.getElementById("batchEmptyState");
const batchBody = document.getElementById("batchBody");
const batchSummary = document.getElementById("batchSummary");
const batchDocName = document.getElementById("batchDocName");
const answerableList = document.getElementById("answerableList");
const unanswerableList = document.getElementById("unanswerableList");

let preparedDocToken = null;
let prepareJobId = null;
let preparePollTimer = null;
let savedDocs = [];

function debug(msg, level = "info") {
  if (!debugLog) return;
  const now = new Date();
  const ts = now.toLocaleTimeString();
  const line = document.createElement("div");
  line.className = `debug-line${level === "warn" ? " warn" : level === "err" ? " err" : ""}`;
  line.textContent = `[${ts}] ${msg}`;
  debugLog.appendChild(line);
  while (debugLog.children.length > 140) {
    debugLog.removeChild(debugLog.firstChild);
  }
  debugLog.scrollTop = debugLog.scrollHeight;
}

function setAskEnabled(enabled) {
  askBtn.disabled = !enabled;
  askBatchBtn.disabled = !enabled;
}

function clearPreparedState() {
  debug("Reset prepared state (doc selection changed).", "warn");
  preparedDocToken = null;
  setAskEnabled(false);
  prepLabel.textContent = "Document not prepared yet.";
  prepEta.textContent = "";
  prepProgress.style.width = "0%";
}

function setPreparingUI(isPreparing) {
  prepareBtn.disabled = isPreparing;
  prepareBtn.textContent = isPreparing ? "Preparing..." : "Prepare document";
}

function fmtSec(s) {
  const n = Math.max(0, Number(s || 0));
  if (n < 60) return `${Math.round(n)}s`;
  const m = Math.floor(n / 60);
  const r = Math.round(n % 60);
  return `${m}m ${r}s`;
}

async function loadSamples() {
  debug("Loading sample document list...");
  sampleSelect.innerHTML = "<option value=''>Loading samples...</option>";

  try {
    const res = await fetch("/api/demo-docs");
    const data = await res.json();
    const docs = data.documents || [];
    debug(`Sample docs response: count=${docs.length}`);

    if (!docs.length) {
      debug("No sample docs found in sample_docs/.", "warn");
      sampleSelect.innerHTML = "<option value=''>No sample docs found</option>";
      return;
    }

    const options = ["<option value=''>Select a sample...</option>"];
    for (const d of docs) {
      options.push(`<option value="${d.name}">${d.name}</option>`);
    }
    sampleSelect.innerHTML = options.join("");

    if (docs.length === 1) {
      sampleSelect.value = docs[0].name;
      debug(`Auto-selected only sample doc: ${docs[0].name}`);
    }
  } catch (_) {
    debug("Failed to load sample docs endpoint.", "err");
    sampleSelect.innerHTML = "<option value=''>Failed to load samples</option>";
  }
}

async function loadSavedDocs() {
  debug("Loading saved document list...");
  savedSelect.innerHTML = "<option value=''>Loading saved docs...</option>";
  try {
    const res = await fetch("/api/saved-docs");
    const data = await res.json();
    savedDocs = data.documents || [];
    if (!savedDocs.length) {
      savedSelect.innerHTML = "<option value=''>No saved docs yet</option>";
      debug("No saved docs found.");
      return;
    }
    const options = ["<option value=''>Select saved doc...</option>"];
    for (const d of savedDocs) {
      options.push(`<option value="${d.token}">${d.name}</option>`);
    }
    savedSelect.innerHTML = options.join("");
    debug(`Saved docs loaded: ${savedDocs.length}`);
  } catch (_) {
    savedSelect.innerHTML = "<option value=''>Failed to load saved docs</option>";
    debug("Failed to load saved docs endpoint.", "err");
  }
}

function setAsking(isLoading) {
  askBtn.textContent = isLoading ? "Running..." : "Submit question";
  askBatchBtn.textContent = isLoading ? "Running batch..." : "Run batch";
  if (!preparedDocToken) {
    setAskEnabled(false);
  } else {
    setAskEnabled(!isLoading);
  }
  demoStatus.textContent = isLoading
    ? "Running retrieval, answerability checks, and evidence selection..."
    : "Ready.";
}

function renderBatchResults(data) {
  const rows = Array.isArray(data.results) ? data.results : [];
  debug(`Render batch: total=${rows.length} answerable=${data.answerable_count} unanswerable=${data.unanswerable_count}`);
  batchEmptyState.classList.add("hidden");
  batchBody.classList.remove("hidden");
  batchSummary.textContent = `Total ${data.total} · Answerable ${data.answerable_count} · Unanswerable ${data.unanswerable_count}`;
  batchSummary.classList.remove("ok", "bad");
  batchDocName.textContent = data.document ? `Document: ${data.document}` : "";

  answerableList.innerHTML = "";
  unanswerableList.innerHTML = "";

  const mkCard = (r) => {
    const meta = r.metadata || {};
    const score = meta.topology_score ?? "-";
    const thr = meta.topology_threshold ?? "-";
    const conf = meta.confidence != null ? Number(meta.confidence).toFixed(2) : "-";
    const llmRaw = meta.llm_answerable_raw;
    const llmLabel = llmRaw === true ? "answerable" : llmRaw === false ? "unanswerable" : "unknown";
    const topoLabel = meta.topology_predicted || "unknown";
    const card = document.createElement("article");
    card.className = "batch-item";
    card.innerHTML = `
      <p class="batch-q">#${r.index} ${escapeHtml(r.question || "")}</p>
      <p class="batch-a">${escapeHtml(r.short_answer || "No direct answer returned.")}</p>
      <p class="batch-r">${escapeHtml(r.reason || "")}</p>
      <div class="batch-verdicts">LLM: ${llmLabel} · Topology: ${topoLabel}</div>
      <div class="batch-m">score=${score} threshold=${thr} confidence=${conf}</div>
    `;
    return card;
  };

  for (const r of rows) {
    if (r.answerable) {
      answerableList.appendChild(mkCard(r));
    } else {
      unanswerableList.appendChild(mkCard(r));
    }
  }

  if (!answerableList.children.length) {
    answerableList.innerHTML = "<p class='hint'>None in this run.</p>";
  }
  if (!unanswerableList.children.length) {
    unanswerableList.innerHTML = "<p class='hint'>None in this run.</p>";
  }
}

function showError(message) {
  debug(`UI error: ${message}`, "err");
  emptyState.classList.remove("hidden");
  resultBody.classList.add("hidden");
  emptyState.innerHTML = `<div class="empty-icon">!</div><p>${escapeHtml(message)}</p>`;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderResult(data) {
  debug(`Render result: answerable=${Boolean(data.answerable)} doc=${data.document || "n/a"}`);
  emptyState.classList.add("hidden");
  resultBody.classList.remove("hidden");

  const isAnswerable = !!data.answerable;
  statusBadge.textContent = isAnswerable ? "Answerable" : "Unanswerable";
  statusBadge.classList.toggle("ok", isAnswerable);
  statusBadge.classList.toggle("bad", !isAnswerable);

  docName.textContent = data.document ? `Document: ${data.document}` : "";
  shortAnswer.textContent = data.short_answer || "No direct answer returned.";
  reasonText.textContent = data.reason || "No reason provided.";

  clausesList.innerHTML = "";
  const clauses = data.supporting_clauses || [];

  if (!clauses.length) {
    clausesList.innerHTML = "<p class='hint'>No supporting clauses returned.</p>";
  } else {
    for (const clause of clauses) {
      const card = document.createElement("article");
      card.className = "clause";
      const sourceText = clause.source ? escapeHtml(clause.source) : "Unknown source";
      card.innerHTML = `
        <p>${escapeHtml(clause.text || "")}</p>
        <small>${sourceText}</small>
      `;
      clausesList.appendChild(card);
    }
  }

  metadataRow.innerHTML = "";
  const metadata = data.metadata || {};

  // Show confidence prominently if present
  const conf = metadata.confidence;
  if (conf != null && conf !== "") {
    const pill = document.createElement("span");
    pill.className = "meta-pill";
    pill.textContent = `confidence: ${Number(conf).toFixed(2)}`;
    metadataRow.appendChild(pill);
  }

  Object.entries(metadata).forEach(([key, value]) => {
    if (key === "confidence") return; // already shown above
    if (value === null || value === undefined || value === "") return;
    const pill = document.createElement("span");
    pill.className = "meta-pill";
    pill.textContent = `${key}: ${value}`;
    metadataRow.appendChild(pill);
  });
}

async function pollPrepareJob(jobId) {
  if (preparePollTimer) {
    clearInterval(preparePollTimer);
  }

  preparePollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/api/prepare-doc/${jobId}`);
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Prepare job failed.");
      }
      const modelPct = Number(data.progress || 0);
      const chunkPct = Number(data.chunk_progress || 0);
      const llmPct = Number(data.llm_progress || 0);
      const blendedPct = Math.max(modelPct, chunkPct * 0.85 + llmPct * 0.15);
      const pct = Math.round(blendedPct * 100);
      debug(
        `Prepare poll: status=${data.status} progress=${pct}% (model=${Math.round(modelPct * 100)} chunk=${Math.round(chunkPct * 100)} llm=${Math.round(llmPct * 100)}) remaining=${data.remaining_sec ?? "?"}s`
      );
      prepProgress.style.width = `${pct}%`;
      prepLabel.textContent = `${data.document || "Document"}: ${data.status}`;

      if (data.status === "running" || data.status === "queued") {
        if (data.remaining_sec !== null && data.remaining_sec !== undefined) {
          prepEta.textContent = `ETA ${fmtSec(data.remaining_sec)}`;
        } else if (data.over_eta_sec !== undefined) {
          prepEta.textContent = `Over ETA +${fmtSec(data.over_eta_sec)}`;
        }
      }
      if (data.note) {
        debug(`Prepare note: ${data.note}`, "warn");
      }
      if (data.warning) {
        debug(`Prepare warning: ${data.warning}`, "err");
      }
      if (data.metrics) {
        const m = data.metrics;
        const tc = Number(m.total_chunks || 0);
        const ec = Number(m.extracted_chunks || 0);
        const nc = Number(m.nodes_created || 0);
        const lc = Number(m.llm_calls_actual || 0);
        const lt = Number(data.llm_calls_estimated_total || 0);
        debug(`Live metrics: stage=${m.stage || "?"} chunks=${ec}/${tc} nodes=${nc} llm_calls=${lc}/${lt}`);
      }

      if (data.status === "done") {
        clearInterval(preparePollTimer);
        preparePollTimer = null;
        preparedDocToken = data.doc_token;
        prepProgress.style.width = "100%";
        prepLabel.textContent = `Prepared: ${data.document}`;
        prepEta.textContent = "Ready for questions";
        setAskEnabled(true);
        setPreparingUI(false);
        demoStatus.textContent = "Document prepared. Ask a question.";
        debug(`Prepare done. doc_token=${data.doc_token?.slice(0, 10) || "none"}...`);
        loadSavedDocs();
      }

      if (data.status === "error") {
        clearInterval(preparePollTimer);
        preparePollTimer = null;
        setPreparingUI(false);
        setAskEnabled(false);
        prepLabel.textContent = "Preparation failed.";
        prepEta.textContent = "";
        showError(data.error || "Failed to prepare document.");
      }
    } catch (err) {
      debug(`Prepare poll error: ${err.message || err}`, "err");
      clearInterval(preparePollTimer);
      preparePollTimer = null;
      setPreparingUI(false);
      setAskEnabled(false);
      prepLabel.textContent = "Preparation failed.";
      prepEta.textContent = "";
      showError(err.message || "Failed to check preparation status.");
    }
  }, 1000);
}

async function prepareDocument() {
  debug("Prepare clicked.");
  clearPreparedState();
  setPreparingUI(true);
  prepLabel.textContent = "Starting document preparation...";
  demoStatus.textContent = "Building index and retrieval graph for your document...";

  const form = new FormData();
  if (sampleSelect.value) {
    form.append("sample_name", sampleSelect.value);
    debug(`Prepare uses sample: ${sampleSelect.value}`);
  }
  if (pdfFile.files[0]) {
    form.append("file", pdfFile.files[0]);
    debug(`Prepare uses upload: ${pdfFile.files[0].name} (${pdfFile.files[0].size} bytes)`);
  }

  try {
    const res = await fetch("/api/prepare-doc", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Could not start document preparation.");
    }
    debug(`Prepare job started: id=${data.job_id} eta=${data.estimate_sec}s document=${data.document}`);

    prepareJobId = data.job_id;
    prepLabel.textContent = `Preparing: ${data.document}`;
    prepEta.textContent = data.estimate_sec ? `ETA ${fmtSec(data.estimate_sec)}` : "";
    prepProgress.style.width = "3%";
    pollPrepareJob(prepareJobId);
    loadSavedDocs();
  } catch (err) {
    debug(`Prepare start error: ${err.message || err}`, "err");
    setPreparingUI(false);
    showError(err.message || "Failed to prepare document.");
  }
}

async function askQuestion() {
  debug("Ask clicked.");
  const question = questionInput.value.trim();
  if (!question) {
    showError("Please enter a question first.");
    return;
  }
  if (!preparedDocToken) {
    showError("Prepare a document first.");
    return;
  }

  const form = new FormData();
  form.append("question", question);
  form.append("doc_token", preparedDocToken);
  debug(`Ask request: q="${question.slice(0, 120)}" token=${preparedDocToken.slice(0, 10)}...`);

  setAsking(true);

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      body: form,
    });

    const data = await res.json();
    debug(`Ask response: http=${res.status} answerable=${Boolean(data.answerable)}`);
    if (!res.ok) {
      showError(data.detail || "Request failed.");
      return;
    }

    renderResult(data);
  } catch (_) {
    debug("Ask network/server error.", "err");
    showError("Network or server error. Please try again.");
  } finally {
    setAsking(false);
  }
}

async function askBatchQuestions() {
  debug("Batch ask clicked.");
  if (!preparedDocToken) {
    showError("Prepare a document first.");
    return;
  }
  const lines = (batchInput.value || "")
    .split("\n")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  if (!lines.length) {
    showError("Enter batch questions (one per line).");
    return;
  }
  if (lines.length > 40) {
    showError("Batch limit is 40 questions.");
    return;
  }

  const form = new FormData();
  form.append("questions", lines.join("\n"));
  form.append("doc_token", preparedDocToken);
  debug(`Batch request: count=${lines.length} token=${preparedDocToken.slice(0, 10)}...`);

  setAsking(true);
  demoStatus.textContent = `Running ${lines.length} questions...`;
  try {
    const res = await fetch("/api/ask-batch", {
      method: "POST",
      body: form,
    });
    const data = await res.json();
    debug(`Batch response: http=${res.status} total=${data.total || 0}`);
    if (!res.ok) {
      showError(data.detail || "Batch request failed.");
      return;
    }
    renderBatchResults(data);
    demoStatus.textContent = `Batch complete: ${data.answerable_count} answerable, ${data.unanswerable_count} unanswerable.`;
  } catch (_) {
    debug("Batch ask network/server error.", "err");
    showError("Network or server error during batch run.");
  } finally {
    setAsking(false);
  }
}

prepareBtn.addEventListener("click", prepareDocument);
askBtn.addEventListener("click", askQuestion);
askBatchBtn.addEventListener("click", askBatchQuestions);
questionInput.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    askQuestion();
  }
});
examples.addEventListener("click", (event) => {
  if (event.target instanceof HTMLButtonElement) {
    questionInput.value = event.target.textContent || "";
    questionInput.focus();
    questionInput.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
});

sampleSelect.addEventListener("change", clearPreparedState);
pdfFile.addEventListener("change", clearPreparedState);
savedSelect.addEventListener("change", clearPreparedState);
useSavedBtn.addEventListener("click", () => {
  const token = (savedSelect.value || "").trim();
  if (!token) {
    showError("Select a saved document first.");
    return;
  }
  preparedDocToken = token;
  setAskEnabled(true);
  prepProgress.style.width = "100%";
  prepLabel.textContent = "Saved document selected.";
  prepEta.textContent = "Ready for questions";
  demoStatus.textContent = "Using saved document. Ask a question.";
  debug(`Using saved doc token=${token.slice(0, 10)}...`);
});
clearDebugBtn.addEventListener("click", () => {
  debugLog.textContent = "";
  debug("Debug log cleared.");
});

setAskEnabled(false);
debug("UI boot complete.");
loadSamples();
loadSavedDocs();
