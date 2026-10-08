/**
 * Antibody–Antigen Interaction Analyzer V2
 * Plain Vanilla JavaScript implementation (Zero Framework Dependencies).
 * Precision Structural Biology Research Interface.
 */

(function () {
  "use strict";

  // Configuration: easily configurable API base URL
  const API_BASE = window.location.protocol === "file:" ? "http://127.0.0.1:8000" : "";

  // 20 Standard Amino Acids in exact project order (Zhang et al. 2024)
  const AMINO_ACIDS = [
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"
  ];

  const AA_NAMES = {
    "A": "ALA (Alanine)",
    "R": "ARG (Arginine)",
    "N": "ASN (Asparagine)",
    "D": "ASP (Aspartate)",
    "C": "CYS (Cysteine)",
    "Q": "GLN (Glutamine)",
    "E": "GLU (Glutamate)",
    "G": "GLY (Glycine)",
    "H": "HIS (Histidine)",
    "I": "ILE (Isoleucine)",
    "L": "LEU (Leucine)",
    "K": "LYS (Lysine)",
    "M": "MET (Methionine)",
    "F": "PHE (Phenylalanine)",
    "P": "PRO (Proline)",
    "S": "SER (Serine)",
    "T": "THR (Threonine)",
    "W": "TRP (Tryptophan)",
    "Y": "TYR (Tyrosine)",
    "V": "VAL (Valine)"
  };

  // DOM Elements
  const pdbInput = document.getElementById("pdb-input");
  const analyzeBtn = document.getElementById("analyze-btn");
  const inputError = document.getElementById("input-error");
  const preAnalysisPanel = document.getElementById("pre-analysis-panel");
  const loadingPanel = document.getElementById("loading-panel");
  const loadingPdbTarget = document.getElementById("loading-pdb-target");
  const errorCard = document.getElementById("error-card");
  const errorTitle = document.getElementById("error-title");
  const errorType = document.getElementById("error-type");
  const errorDetail = document.getElementById("error-detail");
  const errorDismissBtn = document.getElementById("error-dismiss-btn");
  const resultsContainer = document.getElementById("results-container");
  const backendStatus = document.getElementById("backend-status");
  const backendStatusText = document.getElementById("backend-status-text");

  // Canvas & Matrix Elements
  const canvas = document.getElementById("matrix-canvas");
  const tooltip = document.getElementById("matrix-tooltip");
  const copyHashBtn = document.getElementById("copy-hash-btn");
  const copyHashLabel = document.getElementById("copy-hash-label");

  let currentMatrix = null;
  let hoveredCell = null;
  let stageTimer = null;

  /**
   * Health Check: queries GET /api/health to confirm backend availability.
   */
  async function checkBackendHealth() {
    try {
      const response = await fetch(`${API_BASE}/api/health`, {
        method: "GET",
        headers: { "Accept": "application/json" },
      });
      if (response.ok) {
        const data = await response.json();
        if (data.model_loaded) {
          setBackendStatus("online", "Model Online");
        } else {
          setBackendStatus("offline", "Model Not Loaded");
        }
      } else {
        setBackendStatus("offline", `Backend HTTP ${response.status}`);
      }
    } catch (err) {
      setBackendStatus("offline", "Backend Offline");
    }
  }

  function setBackendStatus(state, message) {
    if (!backendStatus || !backendStatusText) return;
    backendStatus.className = "status-pill";
    if (state === "online") {
      backendStatus.classList.add("status-online");
    } else {
      backendStatus.classList.add("status-offline");
    }
    backendStatusText.textContent = message;
  }

  /**
   * Validate PDB identifier format (4 alphanumeric characters).
   */
  function validatePdbInput(raw) {
    const clean = (raw || "").trim().toUpperCase();
    if (!clean) {
      return { valid: false, message: "Please enter a 4-character PDB accession ID." };
    }
    if (!/^[0-9A-Z]{4}$/.test(clean)) {
      return {
        valid: false,
        message: `Invalid format '${clean}'. PDB accession must be exactly 4 alphanumeric characters (e.g. 1EJO).`,
      };
    }
    return { valid: true, pdb: clean };
  }

  function showInputError(message) {
    if (!inputError) return;
    inputError.textContent = message;
    inputError.style.display = "block";
    if (pdbInput) pdbInput.focus();
  }

  function clearInputError() {
    if (!inputError) return;
    inputError.textContent = "";
    inputError.style.display = "none";
  }

  function showError(type, detail) {
    if (!errorCard) return;
    errorCard.style.display = "block";
    if (errorType) errorType.textContent = `Error: ${type || "InferenceError"}`;

    let friendlyDetail = detail;
    if (type === "InvalidPdbId") {
      friendlyDetail = "Invalid PDB identifier format. Expected exactly 4 alphanumeric characters (e.g. 1EJO).";
    } else if (type === "StructureNotFound") {
      friendlyDetail = "Structure could not be located in cache or downloaded from RCSB PDB coordinate archive. Please verify the accession code.";
    } else if (type === "InvalidComplex") {
      friendlyDetail = "The specified structure is not a recognized antibody–antigen complex (requires both antibody heavy/light and protein antigen chains).";
    } else if (type === "NoValidContacts") {
      friendlyDetail = "No intermolecular contacts <= 5.0 Å were detected between the antibody and antigen interface residues.";
    } else if (type === "ModelLoadError") {
      friendlyDetail = "The CNN model artifact is unavailable or incompatible. Please ensure the model is trained and serialized.";
    } else if (!detail) {
      friendlyDetail = "Analysis could not be completed. Please check that the backend is running and try again.";
    }

    if (errorDetail) errorDetail.textContent = friendlyDetail;
    if (resultsContainer) resultsContainer.style.display = "none";
    if (preAnalysisPanel) preAnalysisPanel.style.display = "none";
    if (loadingPanel) loadingPanel.style.display = "none";

    errorCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function clearError() {
    if (errorCard) errorCard.style.display = "none";
  }

  /**
   * Staged Loading Animation: simulates progressive pipeline checkpoints
   * without claiming individual backend stages are independently verified.
   */
  function startStagedLoading(cleanPdb) {
    if (loadingPdbTarget) loadingPdbTarget.textContent = cleanPdb;

    const stages = [
      { id: "stage-step-1", delay: 0 },
      { id: "stage-step-2", delay: 280 },
      { id: "stage-step-3", delay: 560 },
      { id: "stage-step-4", delay: 840 },
      { id: "stage-step-5", delay: 1120 },
      { id: "stage-step-6", delay: 1400 },
    ];

    // Reset all steps to initial state
    stages.forEach((s, idx) => {
      const el = document.getElementById(s.id);
      if (el) {
        el.className = "stage-step";
        const icon = el.querySelector(".stage-icon");
        if (icon) icon.textContent = idx === 0 ? "→" : "○";
      }
    });

    let currentStep = 0;
    if (stageTimer) clearInterval(stageTimer);

    stageTimer = setInterval(() => {
      currentStep++;
      if (currentStep > stages.length) {
        clearInterval(stageTimer);
        return;
      }

      for (let i = 0; i < stages.length; i++) {
        const el = document.getElementById(stages[i].id);
        if (!el) continue;
        const icon = el.querySelector(".stage-icon");
        if (i < currentStep - 1) {
          el.className = "stage-step done";
          if (icon) icon.textContent = "✓";
        } else if (i === currentStep - 1) {
          el.className = "stage-step active";
          if (icon) icon.textContent = "→";
        } else {
          el.className = "stage-step";
          if (icon) icon.textContent = "○";
        }
      }
    }, 280);
  }

  function stopStagedLoading() {
    if (stageTimer) {
      clearInterval(stageTimer);
      stageTimer = null;
    }
  }

  /**
   * Primary Analysis Pipeline: calls POST /api/predict
   */
  async function runAnalysis(pdbCode) {
    clearInputError();
    clearError();

    const validation = validatePdbInput(pdbCode);
    if (!validation.valid) {
      showInputError(validation.message);
      return;
    }

    const cleanPdb = validation.pdb;
    if (pdbInput) pdbInput.value = cleanPdb;

    // Enter loading state
    setLoadingState(true, cleanPdb);

    try {
      const response = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
        },
        body: JSON.stringify({ pdb_id: cleanPdb }),
      });

      const data = await response.json();

      if (!response.ok) {
        const errDetail = data.detail || {};
        const errType = typeof errDetail === "object" ? errDetail.error_type : "HttpError";
        const errMsg = typeof errDetail === "object" ? errDetail.detail : (data.detail || "Server error");
        showError(errType, errMsg);
        return;
      }

      // Success: render dynamic data
      renderResults(data);

    } catch (networkErr) {
      console.error("Network or inference request failed:", networkErr);
      showError(
        "NetworkError",
        "Could not connect to the analysis backend. Please ensure the server is running at " +
        (API_BASE || window.location.origin) + " and try again."
      );
    } finally {
      setLoadingState(false, cleanPdb);
    }
  }

  function setLoadingState(isLoading, cleanPdb) {
    if (isLoading) {
      if (analyzeBtn) {
        analyzeBtn.disabled = true;
        analyzeBtn.classList.add("loading");
      }
      if (loadingPanel) loadingPanel.style.display = "block";
      if (resultsContainer) resultsContainer.style.display = "none";
      if (preAnalysisPanel) preAnalysisPanel.style.display = "none";
      startStagedLoading(cleanPdb);
    } else {
      stopStagedLoading();
      if (analyzeBtn) {
        analyzeBtn.disabled = false;
        analyzeBtn.classList.remove("loading");
      }
      if (loadingPanel) loadingPanel.style.display = "none";
    }
  }

  /**
   * Render complete API response dynamically into the DOM.
   */
  function renderResults(data) {
    clearError();

    // 1. Result Summary Banner & Classification
    const resPdb = document.getElementById("res-pdb-id");
    if (resPdb) resPdb.textContent = data.pdb_id;

    const prob = data.prediction.probability;
    const predClass = data.prediction.predicted_class;
    const probPercent = (prob * 100).toFixed(2) + "%";
    const probDecimal = prob.toFixed(4);

    const resProb = document.getElementById("res-probability");
    if (resProb) resProb.textContent = `${probPercent} (${probDecimal})`;

    // Fill the horizontal probability gauge
    const probFill = document.getElementById("res-probability-fill");
    if (probFill) {
      const clampPct = Math.max(0, Math.min(100, prob * 100));
      probFill.style.width = `${clampPct}%`;
      if (predClass === 1) {
        probFill.style.background = "linear-gradient(90deg, #059669, #10b981)";
      } else {
        probFill.style.background = "linear-gradient(90deg, #d97706, #f59e0b)";
      }
    }

    const predBadge = document.getElementById("res-prediction-badge");
    if (predBadge) {
      predBadge.className = "prediction-classification-badge";
      if (predClass === 1) {
        predBadge.textContent = "COGNATE PAIR (Class 1)";
        predBadge.classList.add("badge-cognate");
      } else {
        predBadge.textContent = "MISMATCHED PAIR (Class 0)";
        predBadge.classList.add("badge-mismatched");
      }
    }

    const resInterp = document.getElementById("res-interpretation-text");
    if (resInterp) {
      resInterp.textContent = data.prediction.interpretation || "No interpretation provided.";
    }

    // 2. Macromolecular Entities
    const heavy = data.structure.antibody_heavy_chains || [];
    const light = data.structure.antibody_light_chains || [];
    const antigen = data.structure.antigen_chains || [];

    const heavyEl = document.getElementById("res-heavy-chains");
    if (heavyEl) heavyEl.textContent = heavy.length ? heavy.join(", ") : "None";

    const lightEl = document.getElementById("res-light-chains");
    if (lightEl) lightEl.textContent = light.length ? light.join(", ") : "None";

    const agEl = document.getElementById("res-antigen-chains");
    if (agEl) agEl.textContent = antigen.length ? antigen.join(", ") : "None";

    // CDR counts
    const cdrs = data.cdrs || {};
    const cdrMap = {
      "res-cdr-h1": cdrs.H1 || 0,
      "res-cdr-h2": cdrs.H2 || 0,
      "res-cdr-h3": cdrs.H3 || 0,
      "res-cdr-l1": cdrs.L1 || 0,
      "res-cdr-l2": cdrs.L2 || 0,
      "res-cdr-l3": cdrs.L3 || 0,
    };
    for (const [id, count] of Object.entries(cdrMap)) {
      const el = document.getElementById(id);
      if (el) el.textContent = count;
    }

    // 3. Contact & Interface Statistics
    const contacts = data.contacts || {};
    const interContacts = document.getElementById("res-inter-contacts");
    if (interContacts) interContacts.textContent = (contacts.intermolecular || 0).toLocaleString();

    const abInt = document.getElementById("res-ab-interface");
    if (abInt) abInt.textContent = (contacts.antibody_interface_residues || 0).toLocaleString();

    const cdrInt = document.getElementById("res-cdr-interface");
    if (cdrInt) cdrInt.textContent = (contacts.cdr_interface_residues || 0).toLocaleString();

    const agInt = document.getElementById("res-ag-interface");
    if (agInt) agInt.textContent = (contacts.antigen_interface_residues || 0).toLocaleString();

    // 4. Interaction Representation Meta
    const rep = data.representation || {};
    const canonShape = document.getElementById("res-canonical-shape");
    if (canonShape) canonShape.textContent = `Shape: [${(rep.canonical_shape || [20, 20]).join(", ")}]`;

    const cnnShape = document.getElementById("res-cnn-shape");
    if (cnnShape) cnnShape.textContent = `CNN Tensor: [${(rep.cnn_shape || [20, 21, 1]).join(", ")}]`;

    const nonzeroCount = document.getElementById("res-nonzero-count");
    if (nonzeroCount) nonzeroCount.textContent = `Nonzero Cells: ${rep.nonzero_count !== undefined ? rep.nonzero_count : "--"}`;

    const sha256El = document.getElementById("res-sha256");
    if (sha256El) sha256El.textContent = rep.sha256 || "None";

    // 5. Render 20x20 Heatmap
    if (rep.matrix_20x20 && Array.isArray(rep.matrix_20x20)) {
      currentMatrix = rep.matrix_20x20;
      drawHeatmap(currentMatrix);
    }

    // Reveal results container and hide intro panel
    if (preAnalysisPanel) preAnalysisPanel.style.display = "none";
    if (resultsContainer) {
      resultsContainer.style.display = "block";
      resultsContainer.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  /**
   * Draw the 20x20 contact matrix on HTML5 Canvas with axis labels.
   * Uses high-DPI devicePixelRatio for crystal-clear typography.
   */
  function drawHeatmap(matrix) {
    if (!canvas || !matrix) return;
    const ctx = canvas.getContext("2d");

    const displaySize = 520;
    const dpr = window.devicePixelRatio || 1;

    canvas.width = displaySize * dpr;
    canvas.height = displaySize * dpr;
    canvas.style.width = `${displaySize}px`;
    canvas.style.height = `${displaySize}px`;

    ctx.scale(dpr, dpr);

    const width = displaySize;
    const height = displaySize;

    ctx.clearRect(0, 0, width, height);

    const margin = 36; // pixels for row/col amino acid labels
    const gridSize = displaySize - margin;
    const cellSize = gridSize / 20;

    // Dark canvas background
    ctx.fillStyle = "#030813";
    ctx.fillRect(0, 0, width, height);

    // Axis Labels: Top Columns (Antibody) & Left Rows (Antigen)
    ctx.fillStyle = "#94a3b8";
    ctx.font = "600 11px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    for (let i = 0; i < 20; i++) {
      const char = AMINO_ACIDS[i];
      // Column label (Antibody)
      ctx.fillText(char, margin + (i + 0.5) * cellSize, margin / 2);
      // Row label (Antigen)
      ctx.fillText(char, margin / 2, margin + (i + 0.5) * cellSize);
    }

    // Draw cells
    for (let r = 0; r < 20; r++) {
      for (let c = 0; c < 20; c++) {
        const val = matrix[r][c] || 0.0;
        const x = margin + c * cellSize;
        const y = margin + r * cellSize;

        // Fill cell with sequential colormap
        ctx.fillStyle = getHeatmapColor(val);
        ctx.fillRect(x, y, cellSize, cellSize);

        // Subtle cell grid border
        ctx.strokeStyle = "rgba(26, 42, 66, 0.65)";
        ctx.lineWidth = 0.5;
        ctx.strokeRect(x, y, cellSize, cellSize);
      }
    }

    // Highlight hovered cell if active
    if (hoveredCell && hoveredCell.r >= 0 && hoveredCell.r < 20 && hoveredCell.c >= 0 && hoveredCell.c < 20) {
      const hx = margin + hoveredCell.c * cellSize;
      const hy = margin + hoveredCell.r * cellSize;

      ctx.strokeStyle = "#22d3ee";
      ctx.lineWidth = 2;
      ctx.strokeRect(hx + 1, hy + 1, cellSize - 2, cellSize - 2);
    }
  }

  /**
   * Precision Sequential Scientific Colormap:
   * 0.0: Dark obsidian (#07111f)
   * (0, 0.25]: Deep teal-navy (#083344)
   * (0.25, 0.65]: Vivid cyan (#06b6d4)
   * (0.65, 0.90]: Bright cyan-ice (#a5f3fc)
   * (0.90, 1.0]: Pure highlight (#ffffff)
   */
  function getHeatmapColor(val) {
    if (val <= 0.0) return "#07111f";
    const v = Math.min(1.0, Math.max(0.0, val));

    if (v < 0.25) {
      const t = v / 0.25;
      const r = Math.round(7 + (8 - 7) * t);
      const g = Math.round(17 + (51 - 17) * t);
      const b = Math.round(31 + (68 - 31) * t);
      return `rgb(${r}, ${g}, ${b})`;
    } else if (v < 0.65) {
      const t = (v - 0.25) / 0.4;
      const r = Math.round(8 + (6 - 8) * t);
      const g = Math.round(51 + (182 - 51) * t);
      const b = Math.round(68 + (212 - 68) * t);
      return `rgb(${r}, ${g}, ${b})`;
    } else if (v < 0.90) {
      const t = (v - 0.65) / 0.25;
      const r = Math.round(6 + (165 - 6) * t);
      const g = Math.round(182 + (243 - 182) * t);
      const b = Math.round(212 + (252 - 212) * t);
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      const t = (v - 0.90) / 0.10;
      const r = Math.round(165 + (255 - 165) * t);
      const g = Math.round(243 + (255 - 243) * t);
      const b = Math.round(252 + (255 - 252) * t);
      return `rgb(${r}, ${g}, ${b})`;
    }
  }

  /**
   * Canvas Tooltip and Hover tracking.
   */
  if (canvas) {
    canvas.addEventListener("mousemove", function (e) {
      if (!currentMatrix) return;
      const rect = canvas.getBoundingClientRect();
      const clientX = e.clientX - rect.left;
      const clientY = e.clientY - rect.top;

      const displaySize = 520;
      const scaleX = displaySize / rect.width;
      const scaleY = displaySize / rect.height;

      const mouseX = clientX * scaleX;
      const mouseY = clientY * scaleY;

      const margin = 36;
      const gridSize = displaySize - margin;
      const cellSize = gridSize / 20;

      if (mouseX >= margin && mouseY >= margin && mouseX <= margin + gridSize && mouseY <= margin + gridSize) {
        const c = Math.floor((mouseX - margin) / cellSize);
        const r = Math.floor((mouseY - margin) / cellSize);

        if (r >= 0 && r < 20 && c >= 0 && c < 20) {
          hoveredCell = { r, c };
          drawHeatmap(currentMatrix);

          const val = currentMatrix[r][c] || 0.0;
          const abAA = AMINO_ACIDS[c];
          const agAA = AMINO_ACIDS[r];
          const abName = AA_NAMES[abAA] || abAA;
          const agName = AA_NAMES[agAA] || agAA;

          if (tooltip) {
            tooltip.innerHTML = `
              <div style="font-weight: 700; color: #38bdf8; margin-bottom: 2px;">
                Antibody: ${abName} &times; Antigen: ${agName}
              </div>
              <div style="color: #cbd5e1;">Normalized Frequency: <strong>${val.toFixed(4)}</strong></div>
              <div style="font-size: 0.7rem; color: #94a3b8; margin-top: 2px;">
                Coordinates: [Row: ${r + 1} (${agAA}), Col: ${c + 1} (${abAA})]
              </div>
            `;
            tooltip.style.display = "block";
            tooltip.style.left = `${Math.min(rect.width - 240, Math.max(10, clientX + 15))}px`;
            tooltip.style.top = `${Math.max(10, clientY - 45)}px`;
          }
          return;
        }
      }

      hoveredCell = null;
      drawHeatmap(currentMatrix);
      if (tooltip) tooltip.style.display = "none";
    });

    canvas.addEventListener("mouseleave", function () {
      hoveredCell = null;
      if (currentMatrix) drawHeatmap(currentMatrix);
      if (tooltip) tooltip.style.display = "none";
    });
  }

  // Copy SHA-256 Hash action
  if (copyHashBtn) {
    copyHashBtn.addEventListener("click", function () {
      const hashEl = document.getElementById("res-sha256");
      if (!hashEl) return;
      const text = hashEl.textContent.trim();
      if (!text || text.startsWith("---")) return;

      navigator.clipboard.writeText(text).then(function () {
        if (copyHashLabel) {
          const original = copyHashLabel.textContent;
          copyHashLabel.textContent = "Copied!";
          setTimeout(function () {
            copyHashLabel.textContent = original;
          }, 2000);
        }
      }).catch(function (err) {
        console.error("Clipboard copy failed:", err);
      });
    });
  }

  // Event Listeners for Analysis Submission
  if (analyzeBtn) {
    analyzeBtn.addEventListener("click", function () {
      if (pdbInput) runAnalysis(pdbInput.value);
    });
  }

  if (pdbInput) {
    pdbInput.addEventListener("keydown", function (e) {
      if (e.key === "Enter") {
        e.preventDefault();
        runAnalysis(pdbInput.value);
      }
    });

    pdbInput.addEventListener("input", function () {
      clearInputError();
    });
  }

  // Example Chips Click Handling
  document.querySelectorAll(".benchmark-chip").forEach(function (btn) {
    btn.addEventListener("click", function () {
      const pdb = btn.getAttribute("data-pdb");
      if (pdb) {
        if (pdbInput) pdbInput.value = pdb;
        runAnalysis(pdb);
      }
    });
  });

  // Error Card Dismiss / Reset
  if (errorDismissBtn) {
    errorDismissBtn.addEventListener("click", function () {
      clearError();
      if (resultsContainer) resultsContainer.style.display = "none";
      if (preAnalysisPanel) preAnalysisPanel.style.display = "block";
      if (pdbInput) {
        pdbInput.value = "";
        pdbInput.focus();
      }
    });
  }

  // Initial Startup Check
  checkBackendHealth();

})();
