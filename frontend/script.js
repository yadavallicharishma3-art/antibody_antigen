/* Zipped_ABAG frontend. All prediction values come from the existing API. */
(() => {
  "use strict";

  const HISTORY_KEY = "zipped-abag-analysis-history-v1";
  const panels = [...document.querySelectorAll("[data-view-panel]")];
  const subviews = [...document.querySelectorAll("[data-subview-panel]")];
  const aaFallback = "ACDEFGHIKLMNPQRSTVWY".split("");
  let currentView = "home";
  let currentSubview = "";
  let currentResult = null;
  let currentStructure = null;
  let viewer = null;
  let viewerModel = null;

  const byId = (id) => document.getElementById(id);
  const text = (id, value) => { const el = byId(id); if (el) el.textContent = value ?? "—"; };
  const show = (el, yes = true) => { if (el) el.hidden = !yes; };
  const chainsText = (value) => Array.isArray(value) && value.length ? value.join(", ") : "Not returned";
  const numberText = (value) => Number.isFinite(Number(value)) ? String(value) : "Not returned";
  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const homeHeroImage = document.querySelector(".interaction-hero-image");
  let lastScrollPosition = window.scrollY;
  let lastHeroBounce = -650;
  function bounceHomeImage() {
    if (currentView !== "home" || !homeHeroImage || performance.now() - lastHeroBounce < 650) return;
    lastHeroBounce = performance.now();
    homeHeroImage.classList.remove("is-bouncing");
    void homeHeroImage.offsetWidth;
    homeHeroImage.classList.add("is-bouncing");
    window.setTimeout(() => homeHeroImage.classList.remove("is-bouncing"), 700);
  }

  window.addEventListener("wheel", event => {
    if (event.deltaY < -2) bounceHomeImage();
  }, { passive: true });
  window.addEventListener("scroll", () => {
    if (window.scrollY < lastScrollPosition) bounceHomeImage();
    lastScrollPosition = window.scrollY;
  }, { passive: true });

  function setMessage(message, type = "info") {
    const box = byId("global-message");
    if (!box) return;
    box.textContent = message;
    box.className = `global-message ${type}`;
    show(box, Boolean(message));
  }

  function navigate(view, subview) {
    if (!panels.some(panel => panel.dataset.viewPanel === view)) return;
    currentView = view;
    panels.forEach(panel => show(panel, panel.dataset.viewPanel === view));
    document.querySelectorAll("[data-nav-view]").forEach(button => {
      button.classList.toggle("active", button.dataset.navView === view && button.classList.contains("nav-item"));
    });
    const defaultSub = document.querySelector(`[data-nav-view="${view}"][data-default-subview]`)?.dataset.defaultSubview;
    const target = subview || (view === "beginner" ? "beginner-overview" : view === "specialised" ? "specialised-overview" : defaultSub);
    if (target) navigateSubview(target);
    const label = byId("current-section-label");
    if (label) label.textContent = view.toUpperCase();
    byId("main-content")?.focus({ preventScroll: true });
    window.scrollTo({ top: 0, behavior: "smooth" });
    byId("sidebar")?.classList.remove("mobile-open");
    byId("mobile-menu")?.setAttribute("aria-expanded", "false");
  }

  function navigateSubview(name) {
    const target = subviews.find(panel => panel.dataset.subviewPanel === name);
    if (!target) return;
    currentSubview = name;
    subviews.filter(panel => panel.closest("[data-view-panel]")?.dataset.viewPanel === currentView)
      .forEach(panel => show(panel, panel === target));
    document.querySelectorAll("[data-nav-subview]").forEach(button => button.classList.toggle("active", button.dataset.navSubview === name));
    const group = name.startsWith("beginner-") ? "beginner" : name.startsWith("specialised-") ? "specialised" : "";
    document.querySelectorAll("[data-subnav-group]").forEach(nav => show(nav, nav.dataset.subnavGroup === group));
    const label = byId("current-section-label");
    if (label) label.textContent = name.replace("beginner-", "").replace("specialised-", "").replaceAll("-", " ").toUpperCase();
    if (name === "specialised-heatmap") drawHeatmap();
    if (name === "specialised-viewer") requestAnimationFrame(() => viewer?.resize());
  }

  document.addEventListener("click", event => {
    const viewButton = event.target.closest("[data-nav-view]");
    const subButton = event.target.closest("[data-nav-subview]");
    if (viewButton) { event.preventDefault(); navigate(viewButton.dataset.navView, viewButton.dataset.defaultSubview); }
    else if (subButton) { event.preventDefault(); navigateSubview(subButton.dataset.navSubview); }
  });

  byId("sidebar-toggle")?.addEventListener("click", () => {
    const sidebar = byId("sidebar");
    sidebar?.classList.toggle("collapsed");
    const button = byId("sidebar-toggle");
    button?.setAttribute("aria-expanded", String(!sidebar?.classList.contains("collapsed")));
  });
  byId("mobile-menu")?.addEventListener("click", () => {
    const sidebar = byId("sidebar");
    sidebar?.classList.toggle("mobile-open");
    byId("mobile-menu")?.setAttribute("aria-expanded", String(sidebar?.classList.contains("mobile-open")));
  });

  document.querySelectorAll("[data-analysis-form]").forEach(form => {
    const fileInput = form.querySelector('input[type="file"]');
    const fileName = form.querySelector("[data-file-name]");
    fileInput?.addEventListener("change", () => {
      if (fileName) fileName.textContent = fileInput.files?.[0]?.name || "No file selected";
      if (fileInput.files?.[0]) {
        const idInput = form.querySelector('input[name="pdb_id"]');
        if (idInput) idInput.value = "";
      }
    });
    form.querySelector('input[name="pdb_id"]')?.addEventListener("input", () => {
      if (form.querySelector('input[name="pdb_id"]')?.value.trim() && fileInput) {
        fileInput.value = "";
        if (fileName) fileName.textContent = "No file selected";
      }
    });
    form.addEventListener("submit", async event => {
      event.preventDefault();
      const id = form.querySelector('input[name="pdb_id"]')?.value.trim();
      const file = fileInput?.files?.[0];
      if (Boolean(id) === Boolean(file)) {
        setMessage("Enter one PDB ID or choose one PDB/mmCIF file, not both.", "error");
        return;
      }
      const payload = new FormData();
      if (id) payload.append("pdb_id", id);
      else payload.append("structure", file);
      setMessage("Submitting the structure for analysis…", "info");
      const progressPanel = byId("analysis-progress") || form.parentElement?.querySelector("[data-progress-panel]");
      show(progressPanel);
      form.querySelectorAll("button[type=submit]").forEach(button => { button.disabled = true; button.dataset.oldLabel = button.textContent; button.textContent = "Analysing…"; });
      try {
        const response = await fetch("/api/predict", { method: "POST", body: payload });
        const result = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(result.error || result.detail || `Request failed (${response.status})`);
        validateResult(result);
        currentResult = result;
        currentStructure = id ? { kind: "pdb", pdbId: result.pdb_id || id } : { kind: "upload", file, name: file.name };
        renderResult(result);
        saveHistory(result, currentStructure);
        setMessage(result.message || "Structure analysis completed.", "success");
        const chosenMode = form.querySelector('[name="analysis_mode"]')?.value || form.dataset.afterAnalysis || "beginner";
        navigate(chosenMode === "specialised" ? "specialised" : "beginner", chosenMode === "specialised" ? "specialised-overview" : "beginner-analyse");
      } catch (error) {
        setMessage(error.message || "The structure could not be analysed.", "error");
      } finally {
        show(progressPanel, false);
        form.querySelectorAll("button[type=submit]").forEach(button => { button.disabled = false; if (button.dataset.oldLabel) button.textContent = button.dataset.oldLabel; });
      }
    });
  });

  function validateResult(result) {
    if (!result || !result.prediction || ![0, 1].includes(Number(result.prediction.label))) throw new Error("The API response does not contain a valid binary prediction.");
    if (!Array.isArray(result.matrix) || result.matrix.length !== 20 || result.matrix.some(row => !Array.isArray(row) || row.length !== 20)) throw new Error("The API response did not contain a valid 20×20 matrix.");
  }

  function renderResult(result) {
    const meta = result.feature_metadata || {};
    const prediction = result.prediction || {};
    const positive = Number(prediction.label) === 1;
    const className = prediction.class_name || (positive ? "Antibody–Antigen Interaction" : "General Protein–Protein Interaction");
    text("beginner-result-title", className);
    text("beginner-result-text", prediction.interpretation || (positive ? "The CNN classified this structure as an antibody–antigen interaction." : "The CNN classified this structure as a general protein–protein interaction."));
    const resultMeta = byId("beginner-result-meta");
    if (resultMeta) resultMeta.textContent = `${result.pdb_id || "Uploaded structure"} · Partner A: ${chainsText(prediction.partner_a_chains)} · Partner B: ${chainsText(prediction.partner_b_chains)}`;
    show(byId("beginner-result-explainer"));
    show(byId("beginner-latest-empty"), false);
    show(byId("beginner-latest-result"));
    text("beginner-latest-title", result.pdb_id || (currentStructure?.name ?? "Uploaded structure"));
    text("beginner-latest-description", className);
    text("summary-pdb-id", result.pdb_id || currentStructure?.name || "Uploaded structure");
    text("summary-class-chip", `Class ${Number(prediction.label)} · ${className}`);
    text("summary-partner-a", chainsText(prediction.partner_a_chains));
    text("summary-partner-b", chainsText(prediction.partner_b_chains));
    text("summary-interface-a", numberText(meta.partner_a_interface_residues));
    text("summary-interface-b", numberText(meta.partner_b_interface_residues));
    show(byId("specialised-summary"));
    text("prediction-class", className);
    text("prediction-label", `Predicted class ${Number(prediction.label)} · ${positive ? "Antibody–Antigen" : "General PPI"}`);
    text("prediction-mark", String(Number(prediction.label)));
    const probability = Number(prediction.probability_antibody_antigen);
    text("prediction-probability", Number.isFinite(probability) ? `${(probability * 100).toFixed(1)}%` : "Not returned");
    text("prediction-interpretation", prediction.interpretation || "This model score is a classifier output, not an experimental confirmation or physical binding probability.");
    text("heatmap-title", `${result.pdb_id || "Uploaded structure"} · normalized interface feature`);
    text("chain-table-title", result.pdb_id || currentStructure?.name || "Uploaded structure");
    const ranges = meta.partner_a_residue_ranges || meta.partner_a_interface_residue_ranges;
    const rangesB = meta.partner_b_residue_ranges || meta.partner_b_interface_residue_ranges;
    const table = byId("chain-table-body");
    if (table) table.innerHTML = [
      ["Partner A", chainsText(prediction.partner_a_chains), meta.partner_a_interface_residues, meta.partner_a_observed_sidechain_contacts, ranges],
      ["Partner B", chainsText(prediction.partner_b_chains), meta.partner_b_interface_residues, meta.partner_b_observed_sidechain_contacts, rangesB]
    ].map(row => `<tr><td>${esc(row[0])}</td><td>${esc(row[1])}</td><td>${esc(numberText(row[2]))}</td><td>${esc(numberText(row[3]))}</td><td>${esc(formatRanges(row[4]))}</td></tr>`).join("");
    drawHeatmap();
    renderCdrAnalysis(result.cdr_analysis);
    updateViewerLabels(prediction);
    byId("download-result")?.removeAttribute("disabled");
    byId("download-matrix")?.removeAttribute("disabled");
    setupViewer();
  }

  function formatRanges(value) {
    if (value == null) return "Not returned";
    if (typeof value === "string") return value;
    try { return JSON.stringify(value); } catch { return "Not returned"; }
  }

  function renderCdrAnalysis(data) {
    const status = byId("cdr-analysis-status");
    const tag = byId("cdr-status-tag");
    const results = byId("cdr-analysis-results");
    const loopCards = byId("cdr-loop-cards");
    const table = byId("cdr-table-body");
    const sabdabDownload = byId("sabdab-structure-download");
    if (!status || !tag || !results || !loopCards || !table) return;
    const domains = Array.isArray(data?.antibody_domains) ? data.antibody_domains : [];
    status.textContent = data?.message || "No CDR analysis was included in the API response.";
    tag.textContent = data?.status ? data.status.replaceAll("_", " ").toUpperCase() : "NOT RETURNED";
    tag.classList.toggle("status-good", data?.status === "available");
    tag.classList.toggle("status-warning", data?.status !== "available");
    if (sabdabDownload) {
      sabdabDownload.href = data?.structure_download_url || "#";
      show(sabdabDownload, Boolean(data?.structure_download_url));
    }
    show(results, data?.status === "available" && domains.length > 0);
    if (!domains.length) {
      loopCards.innerHTML = "";
      table.innerHTML = "";
      return;
    }
    loopCards.innerHTML = domains.map(domain => {
      const chainName = domain.chain_type === "heavy" ? "Heavy chain CDR loops" : "Light chain CDR loops";
      const loopCardsMarkup = (domain.cdr_regions || []).map(region => `<article class="cdr-loop-card"><div class="cdr-loop-top"><span>${esc(region.name.replace("CDR-", ""))}</span><span class="cdr-residue-count"><strong>${numberText(region.residue_count)}</strong><small>residues</small></span></div><div class="cdr-loop-sequence">${esc(region.sequence || "No numbered residues")}</div><div class="cdr-interface-count">${numberText(region.interface_residue_count)} at interface</div></article>`).join("");
      return `<section class="cdr-loop-group"><div class="cdr-loop-heading"><h3>${chainName}</h3><span>Chain ${esc(domain.chain_id)} · ${esc(domain.partner)}${domain.species_hint ? ` · ${esc(domain.species_hint)}` : ""}</span></div><div class="cdr-loop-grid">${loopCardsMarkup}</div></section>`;
    }).join("");
    table.innerHTML = domains.flatMap(domain => (domain.cdr_regions || []).map(region => {
      const residues = Array.isArray(region.residues) ? region.residues : [];
      const structureIds = residues.map(residue => `${residue.pdb_residue_number}${residue.pdb_insertion_code || ""}`);
      const imgtIds = residues.map(residue => residue.imgt_position);
      return `<tr><td>${esc(domain.chain_id)}</td><td>${esc(domain.partner)}</td><td>${esc(region.name)}</td><td class="cdr-sequence">${esc(region.sequence || "—")}</td><td>${numberText(region.residue_count)}</td><td>${numberText(region.interface_residue_count)}</td><td>${structureIds.length ? `${esc(structureIds.join(", "))}<small>IMGT ${esc(imgtIds.join(", "))}</small>` : "—"}</td></tr>`;
    })).join("");
  }

  function drawHeatmap() {
    const canvas = byId("interaction-heatmap");
    const result = currentResult;
    if (!canvas || !result?.matrix) return;
    const ctx = canvas.getContext("2d");
    const matrix = result.matrix;
    const aa = Array.isArray(result.amino_acid_order) && result.amino_acid_order.length === 20 ? result.amino_acid_order : aaFallback;
    const size = 20;
    const cell = canvas.width / size;
    const vals = matrix.flat().map(Number).filter(Number.isFinite);
    const max = Math.max(...vals, 0.000001);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let row = 0; row < size; row++) for (let col = 0; col < size; col++) {
      const value = Number(matrix[row][col]) || 0;
      const intensity = Math.sqrt(Math.max(0, value) / max);
      const partnerA = row > col;
      const base = partnerA ? [118, 84, 232] : row < col ? [34, 183, 202] : [38, 49, 69];
      const alpha = row === col ? 0.72 : 0.1 + intensity * 0.88;
      ctx.fillStyle = `rgba(${base.join(",")},${alpha})`;
      ctx.fillRect(col * cell, row * cell, cell, cell);
      ctx.strokeStyle = "rgba(8, 13, 25, .82)";
      ctx.lineWidth = 1.25;
      ctx.strokeRect(col * cell, row * cell, cell, cell);
      if (value > max * 0.23 && cell > 25) {
        ctx.fillStyle = intensity > 0.67 ? "#fff" : "#f5f7ff";
        ctx.font = "500 10px 'DM Mono', monospace";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(value.toFixed(2), col * cell + cell / 2, row * cell + cell / 2);
      }
    }
    canvas.dataset.aaOrder = JSON.stringify(aa);
  }

  byId("interaction-heatmap")?.addEventListener("click", event => {
    const canvas = event.currentTarget;
    if (!currentResult?.matrix) return;
    const rect = canvas.getBoundingClientRect();
    const col = Math.max(0, Math.min(19, Math.floor((event.clientX - rect.left) / rect.width * 20)));
    const row = Math.max(0, Math.min(19, Math.floor((event.clientY - rect.top) / rect.height * 20)));
    const aa = currentResult.amino_acid_order || aaFallback;
    const value = Number(currentResult.matrix[row][col]);
    const inspector = byId("cell-inspector");
    if (inspector) inspector.innerHTML = `<span class="card-index">CELL INSPECTOR</span><h3>${esc(aa[row])} × ${esc(aa[col])}</h3><p>Row ${row + 1} · Column ${col + 1}</p><div class="cell-detail"><span>Actual matrix value</span><strong>${Number.isFinite(value) ? value.toFixed(6) : "Not returned"}</strong><p>${row === col ? "Diagonal is defined as zero." : row > col ? "Lower triangle: Partner A intrapartner interface contacts." : "Upper triangle: Partner B intrapartner interface contacts."}</p></div>`;
  });

  function updateViewerLabels(prediction = {}) {
    text("viewer-partner-a", chainsText(prediction.partner_a_chains));
    text("viewer-partner-b", chainsText(prediction.partner_b_chains));
  }

  function setupViewer() {
    const mount = byId("structure-viewer");
    if (!mount) return;
    if (!window.$3Dmol) {
      show(byId("viewer-message"));
      text("viewer-note", "The optional 3D viewer library did not load. Prediction and matrix results are still available.");
      return;
    }
    try {
      viewer = $3Dmol.createViewer(mount, { backgroundColor: "#080809", antialias: true });
      viewer.clear();
      viewerModel = null;
      show(byId("viewer-message"));
      const structure = currentStructure;
      if (!structure) return;
      if (structure.kind === "upload") {
        structure.file.text().then(contents => addStructure(contents, structure.file.name.toLowerCase().endsWith(".cif") ? "cif" : "pdb"));
      } else if (structure.pdbId) {
        fetch(`https://files.rcsb.org/download/${encodeURIComponent(structure.pdbId)}.cif`)
          .then(response => { if (!response.ok) throw new Error(`RCSB returned ${response.status}`); return response.text(); })
          .then(contents => addStructure(contents, "cif"))
          .catch(error => { show(byId("viewer-message")); text("viewer-note", `Could not load the coordinates from RCSB: ${error.message}. The prediction result remains available.`); });
      }
    } catch (error) {
      show(byId("viewer-message"));
      text("viewer-note", `The 3D viewer could not be initialized: ${error.message}`);
    }
  }

  function addStructure(contents, format) {
    if (!viewer) return;
    viewer.clear();
    viewerModel = viewer.addModel(contents, format);
    viewer.setStyle({}, { cartoon: { color: "#63738a", opacity: 0.84 } });
    applyViewerHighlight();
    viewer.zoomTo(); viewer.resize(); viewer.render();
    show(byId("viewer-message"), false);
  }

  function applyViewerHighlight() {
    if (!viewer || !viewerModel || !currentResult) return;
    const p = currentResult.prediction || {};
    const a = p.partner_a_chains || [];
    const b = p.partner_b_chains || [];
    const selected = byId("viewer-highlight")?.value || "both";
    viewer.setStyle({}, { cartoon: { color: "#63738a", opacity: 0.58 } });
    if (selected !== "b") a.forEach(chain => viewer.setStyle({ chain }, { cartoon: { color: "#a78bfa", opacity: 1 } }));
    if (selected !== "a") b.forEach(chain => viewer.setStyle({ chain }, { cartoon: { color: "#55d4e4", opacity: 1 } }));
    viewer.render();
  }
  byId("viewer-highlight")?.addEventListener("change", applyViewerHighlight);
  byId("viewer-reset")?.addEventListener("click", () => { viewer?.zoomTo(); viewer?.render(); });
  byId("viewer-fullscreen")?.addEventListener("click", async () => {
    const stage = byId("viewer-stage");
    if (!stage) return;
    try { if (!document.fullscreenElement) await stage.requestFullscreen(); else await document.exitFullscreen(); } catch (error) { setMessage(`Fullscreen is unavailable: ${error.message}`, "error"); }
    requestAnimationFrame(() => viewer?.resize());
  });
  document.addEventListener("fullscreenchange", () => requestAnimationFrame(() => viewer?.resize()));

  function getHistory() {
    try { const data = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); return Array.isArray(data) ? data : []; }
    catch { return []; }
  }
  function saveHistory(result, structure) {
    const history = getHistory();
    history.unshift({ id: `${Date.now()}-${Math.random().toString(16).slice(2)}`, timestamp: new Date().toISOString(), source: result.pdb_id || structure.name || "Uploaded structure", result, fileName: structure.name || null });
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 30))); } catch { setMessage("Analysis completed, but browser storage is full; the result is still visible until you leave this page.", "info"); }
    renderHistory();
  }
  function renderHistory() {
    const list = byId("history-list");
    if (!list) return;
    const history = getHistory();
    show(byId("history-empty"), history.length === 0);
    list.innerHTML = history.map(item => {
      const pred = item.result?.prediction || {};
      const label = pred.class_name || (Number(pred.label) === 1 ? "Antibody–Antigen Interaction" : "General Protein–Protein Interaction");
      const date = new Date(item.timestamp).toLocaleString();
      return `<article class="history-card content-card"><div><span class="card-index">${esc(date)}</span><h2>${esc(item.source)}</h2><p>${esc(label)} · Class ${esc(pred.label)}</p></div><div class="history-actions"><button class="button button-secondary" type="button" data-open-history="${esc(item.id)}">Open result</button><button class="text-button danger-text" type="button" data-delete-history="${esc(item.id)}" aria-label="Delete ${esc(item.source)}">Delete</button></div></article>`;
    }).join("");
  }
  byId("history-list")?.addEventListener("click", event => {
    const open = event.target.closest("[data-open-history]");
    const remove = event.target.closest("[data-delete-history]");
    const history = getHistory();
    if (open) {
      const item = history.find(entry => entry.id === open.dataset.openHistory);
      if (!item) return;
      currentResult = item.result;
      currentStructure = { kind: "history", pdbId: item.fileName ? null : item.result.pdb_id, name: item.fileName };
      renderResult(item.result);
      show(byId("viewer-message"));
      text("viewer-note", item.fileName ? "This uploaded structure is not saved in history. Upload it again to reopen its 3D coordinates." : "Coordinates can be reloaded from RCSB for this PDB ID.");
      navigate("specialised", "specialised-overview");
    } else if (remove) {
      try { localStorage.setItem(HISTORY_KEY, JSON.stringify(history.filter(entry => entry.id !== remove.dataset.deleteHistory))); } catch {}
      renderHistory();
    }
  });

  function download(filename, content, type) {
    const url = URL.createObjectURL(new Blob([content], { type }));
    const a = document.createElement("a"); a.href = url; a.download = filename; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  byId("download-result")?.addEventListener("click", () => { if (currentResult) download(`${currentResult.pdb_id || "structure"}-prediction.json`, JSON.stringify(currentResult, null, 2), "application/json"); });
  byId("download-matrix")?.addEventListener("click", () => {
    if (!currentResult?.matrix) return;
    const aa = currentResult.amino_acid_order || aaFallback;
    const csv = [",${aa.join(",")}", ...currentResult.matrix.map((row, i) => `${aa[i]},${row.join(",")}`)].join("\n");
    download(`${currentResult.pdb_id || "structure"}-matrix.csv`, csv, "text/csv");
  });

  const topics = {
    antibody: ["Antibody", "An antibody is an immune protein. It has variable regions that can recognize molecular surfaces. This classifier uses structure-derived features and does not itself determine therapeutic suitability."],
    antigen: ["Antigen", "An antigen is the molecule or molecular part recognized by an antibody. Here, the submitted structure is treated as a candidate interacting pair; chain labels do not by themselves prove biological identity."],
    interface: ["Interaction interface", "An interface is the set of residues near a partner across a structure-based distance cutoff. The backend detects cross-partner heavy-atom contacts to choose interface residues."],
    chains: ["Protein chains", "A structure may contain several chains. Partner A and Partner B are the chain groups selected by the backend. They are labels for the detected pair, not guaranteed antibody/antigen labels."],
    cdr: ["CDR regions", "CDRs are antibody variable-loop regions involved in antigen recognition. The current API does not return CDR residue annotations, and the model feature uses all selected interface residues."],
    pdb: ["PDB and mmCIF", "PDB and mmCIF files describe three-dimensional macromolecular structures. You can enter an ID for RCSB retrieval or upload a local coordinate file."],
    matrix: ["20×20 interaction matrix", "The matrix summarizes normalized intrapartner side-chain contact counts among interface residues. Partner A is in the lower triangle; Partner B is in the upper triangle; the diagonal is zero. It is a CNN input feature, not a physical binding-energy map."],
    cnn: ["CNN prediction", "A convolutional neural network receives the 20×20 matrix and outputs one of two dataset-defined classes. A class score is not an experimental validation or a probability that the proteins bind in reality."]
  };
  document.querySelectorAll("[data-topic]").forEach(button => button.addEventListener("click", () => {
    document.querySelectorAll("[data-topic]").forEach(item => item.classList.toggle("active", item === button));
    const [title, description] = topics[button.dataset.topic] || ["", ""];
    const article = byId("topic-content");
    if (article) article.innerHTML = `<span class="card-index">LEARN</span><h2>${esc(title)}</h2><p>${esc(description)}</p>`;
  }));

  document.querySelector('[data-topic="antibody"]')?.click();

  renderHistory();
  navigate("home");
})();
