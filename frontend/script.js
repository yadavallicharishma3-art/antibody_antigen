/* Zipped_ABAG frontend. All prediction values come from the existing API. */
(() => {
  "use strict";

  const HISTORY_KEY = "zipped-abag-analysis-history-v1";
  const panels = [...document.querySelectorAll("[data-view-panel]")];
  const subviews = [...document.querySelectorAll("[data-subview-panel]")];
  const aaFallback = "ACDEFGHIKLMNPQRSTVWY".split("");
  const aminoAcids = {
    A: ["ALA", "Alanine"], C: ["CYS", "Cysteine"], D: ["ASP", "Aspartic acid"],
    E: ["GLU", "Glutamic acid"], F: ["PHE", "Phenylalanine"], G: ["GLY", "Glycine"],
    H: ["HIS", "Histidine"], I: ["ILE", "Isoleucine"], K: ["LYS", "Lysine"],
    L: ["LEU", "Leucine"], M: ["MET", "Methionine"], N: ["ASN", "Asparagine"],
    P: ["PRO", "Proline"], Q: ["GLN", "Glutamine"], R: ["ARG", "Arginine"],
    S: ["SER", "Serine"], T: ["THR", "Threonine"], V: ["VAL", "Valine"],
    W: ["TRP", "Tryptophan"], Y: ["TYR", "Tyrosine"]
  };
  function aminoAcidLabel(value) {
    const code = String(value || "").trim().toUpperCase();
    const match = Object.entries(aminoAcids).find(([one, names]) => one === code || names[0] === code);
    return match ? { short: match[1][0], full: match[1][1] } : { short: code || "UNK", full: code || "Unknown amino acid" };
  }
  let currentView = "home";
  let currentSubview = "";
  let currentResult = null;
  let currentStructure = null;
  let viewer = null;
  let viewerModel = null;
  let viewerLoadGeneration = 0;
  let viewerInitialView = null;
  let viewerIdleTimer = null;
  let viewerEventsBound = false;

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

  let learningContentLoaded = false;

  function formatLearningAnswer(answer) {
    let markup = "";
    let paragraph = [];
    let listType = "";

    function finishParagraph() {
      if (!paragraph.length) return;
      markup += "<p>" + esc(paragraph.join(" ")) + "</p>";
      paragraph = [];
    }
    function finishList() {
      if (!listType) return;
      markup += "</" + listType + ">";
      listType = "";
    }

    answer.split(/\r?\n/).forEach(line => {
      const cleanLine = line.trim();
      if (!cleanLine) {
        finishParagraph();
        finishList();
        return;
      }
      const item = cleanLine.match(/^(- |\d+\. )(.*)$/);
      if (item) {
        finishParagraph();
        const nextType = /^\d/.test(item[1]) ? "ol" : "ul";
        if (listType !== nextType) {
          finishList();
          listType = nextType;
          markup += "<" + listType + ">";
        }
        markup += "<li>" + esc(item[2]) + "</li>";
      } else {
        finishList();
        paragraph.push(cleanLine);
      }
    });
    finishParagraph();
    finishList();
    return markup;
  }

  async function loadLearningContent() {
    const container = byId("learning-chapters");
    if (!container || learningContentLoaded) return;
    container.textContent = "Loading lessons…";

    try {
      const response = await fetch("/static/antibody_antigen_learning_content.txt");
      if (!response.ok) throw new Error("Learning content file could not be loaded.");
      const content = await response.text();
      const chapterMatches = [...content.matchAll(/^CHAPTER \d+: (.+)$/gm)];
      const questionMatches = [...content.matchAll(/^(\d+)\. (What|How) ([^\r\n]+)$/gm)];
      if (chapterMatches.length !== 3 || questionMatches.length !== 8) {
        throw new Error("The learning content file does not contain the expected three chapters and eight questions.");
      }

      const sourceTitles = {
        "https://www.ncbi.nlm.nih.gov/books/NBK10757/": "NCBI — Immunobiology",
        "https://www.ncbi.nlm.nih.gov/books/NBK26884/": "NCBI — B Cells and Antibodies",
        "https://www.ncbi.nlm.nih.gov/books/NBK27124/": "NCBI — The Recognition and Effector Mechanisms of Adaptive Immunity",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC3792396/": "Sela-Culang et al. (2013) — The Structural Basis of Antibody-Antigen Recognition",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC3443979/": "Ramaraj et al. (2012) — Antigen-antibody interface properties",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC9307644/": "Peng et al. (2022) — Antibody CDR amino acids underlying recognition of diverse protein antigens"
      };

      const questionData = questionMatches.map((match, index) => {
        const bodyStart = match.index + match[0].length;
        const nextQuestion = questionMatches[index + 1]?.index ?? content.length;
        const nextChapter = chapterMatches.find(chapter => chapter.index > match.index)?.index ?? content.length;
        const libraryStart = content.indexOf("SOURCE LIBRARY");
        const fullBody = content.slice(bodyStart, Math.min(nextQuestion, nextChapter, libraryStart < 0 ? content.length : libraryStart)).trim();
        const sourceStartMatch = fullBody.match(/^Sources?:\s*$|^Source\s+[–-]/m);
        const sourceBlock = sourceStartMatch ? fullBody.slice(sourceStartMatch.index) : "";
        const answer = sourceStartMatch ? fullBody.slice(0, sourceStartMatch.index).trim() : fullBody;
        const urls = [...sourceBlock.matchAll(/https?:\/\/[^\s]+/g)].map(item => item[0]);
        const owningChapter = [...chapterMatches].reverse().find(chapter => chapter.index < match.index);

        return {
          title: match[2] + " " + match[3],
          chapterTitle: owningChapter?.[1] || "",
          answer,
          urls
        };
      });

      const chapters = chapterMatches.map((chapter, index) => {
        const chapterQuestions = questionData.filter(question => question.chapterTitle === chapter[1]);
        const questionsHtml = chapterQuestions.map(question => {
          const links = question.urls.map(url =>
            '<a href="' + esc(url) + '" target="_blank" rel="noopener noreferrer">' +
            esc(sourceTitles[url] || url) + '</a>'
          ).join("; ");
          const sources = links ? '<p class="learning-sources"><strong>Research sources:</strong> ' + links + '</p>' : "";
          return '<details class="learning-question"><summary>' + esc(question.title) +
            '</summary><div class="learning-answer">' + formatLearningAnswer(question.answer) +
            sources + '</div></details>';
        }).join("");
        return '<section class="learning-chapter content-card" aria-labelledby="learning-chapter-' +
          (index + 1) + '"><span class="card-index">CHAPTER 0' + (index + 1) +
          '</span><h2 id="learning-chapter-' + (index + 1) + '">' + esc(chapter[1]) +
          '</h2>' + questionsHtml + '</section>';
      });

      container.innerHTML = chapters.join("");
      container.querySelectorAll(".learning-question").forEach(question => {
        question.addEventListener("toggle", () => {
          if (!question.open) return;
          container.querySelectorAll(".learning-question[open]").forEach(other => {
            if (other !== question) other.open = false;
          });
        });
      });
      learningContentLoaded = true;
    } catch (error) {
      container.innerHTML = "";
      const message = document.createElement("p");
      message.setAttribute("role", "alert");
      message.textContent = error.message || "Learning content could not be loaded.";
      container.appendChild(message);
    }
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
    if (name === "beginner-learn") loadLearningContent();
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
      setMessage("Analysis is running…", "info");
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
        setMessage("Analysis completed successfully.", "success");
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
    text("beginner-result-text", prediction.interpretation || (positive ? "The structure was classified as an antibody–antigen interaction." : "The structure was classified as a general protein–protein interaction."));
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
    text("prediction-interpretation", prediction.interpretation || "This classification is not experimental confirmation of physical binding.");
    text("heatmap-title", `${result.pdb_id || "Uploaded structure"} · normalized interface feature`);
    text("chain-table-title", result.pdb_id || currentStructure?.name || "Uploaded structure");
    const table = byId("chain-table-body");
    if (table) table.innerHTML = [
      ["Partner A", chainsText(prediction.partner_a_chains), meta.partner_a_interface_residues, meta.partner_a_observed_sidechain_contacts],
      ["Partner B", chainsText(prediction.partner_b_chains), meta.partner_b_interface_residues, meta.partner_b_observed_sidechain_contacts]
    ].map(row => `<tr><td>${esc(row[0])}</td><td>${esc(row[1])}</td><td>${esc(numberText(row[2]))}</td><td>${esc(numberText(row[3]))}</td></tr>`).join("");
    renderInterfaceResidues(meta, prediction, result.cdr_analysis);
    drawHeatmap();
    renderCdrAnalysis(result.cdr_analysis);
    updateViewerLabels(prediction);
    byId("download-result")?.removeAttribute("disabled");
    byId("download-matrix")?.removeAttribute("disabled");
    setupViewer();
  }

  function structureResidueKey(chainId, number, insertionCode = "") {
    return `${String(chainId || "").trim().toUpperCase()}|${String(number ?? "").trim()}|${String(insertionCode || "").trim().toUpperCase()}`;
  }

  function renderInterfaceResidues(meta, prediction, cdrAnalysis) {
    const heavyCell = byId("heavy-interface-residues");
    const lightCell = byId("light-interface-residues");
    const antigenCell = byId("antigen-interface-residues");
    if (!heavyCell || !lightCell || !antigenCell) return;

    const domains = Array.isArray(cdrAnalysis?.antibody_domains) ? cdrAnalysis.antibody_domains : [];
    const heavyChains = new Set(domains.filter(domain => domain.chain_type === "heavy").map(domain => String(domain.chain_id).trim().toUpperCase()));
    const lightChains = new Set(domains.filter(domain => domain.chain_type === "light").map(domain => String(domain.chain_id).trim().toUpperCase()));
    const antibodyChains = new Set([...heavyChains, ...lightChains]);
    const partnerChains = new Set([
      ...(Array.isArray(prediction.partner_a_chains) ? prediction.partner_a_chains : []),
      ...(Array.isArray(prediction.partner_b_chains) ? prediction.partner_b_chains : [])
    ].map(chain => String(chain).trim().toUpperCase()));
    const antigenChains = new Set([...partnerChains].filter(chain => !antibodyChains.has(chain)));

    const cdrByResidue = new Map();
    domains.forEach(domain => (domain.cdr_regions || []).forEach(region => {
      (region.residues || []).forEach(residue => {
        cdrByResidue.set(
          structureResidueKey(domain.chain_id, residue.pdb_residue_number, residue.pdb_insertion_code),
          region.name
        );
      });
    }));

    const groups = { heavy: new Map(), light: new Map(), antigen: new Map() };
    const interfaceDetails = [
      ...(Array.isArray(meta.partner_a_interface_residue_details) ? meta.partner_a_interface_residue_details : []),
      ...(Array.isArray(meta.partner_b_interface_residue_details) ? meta.partner_b_interface_residue_details : [])
    ];
    const seen = new Set();
    interfaceDetails.forEach(residue => {
      const chainId = String(residue.chain_id || "").trim();
      const normalizedChain = chainId.toUpperCase();
      const key = structureResidueKey(chainId, residue.residue_number, residue.insertion_code);
      if (!chainId || seen.has(key)) return;
      seen.add(key);

      const role = heavyChains.has(normalizedChain) ? "heavy"
        : lightChains.has(normalizedChain) ? "light"
          : antigenChains.has(normalizedChain) ? "antigen" : null;
      if (!role) return;
      if (!groups[role].has(chainId)) groups[role].set(chainId, []);
      groups[role].get(chainId).push({ ...residue, cdrRegion: cdrByResidue.get(key) || "" });
    });

    const renderGroup = (cell, role, label) => {
      const chainGroups = groups[role];
      if (!domains.length) {
        cell.textContent = "Chain roles unavailable because no CDR annotations were returned.";
        return;
      }
      if (!chainGroups.size) {
        cell.textContent = `No ${label.toLowerCase()} interface residues identified.`;
        return;
      }
      cell.innerHTML = [...chainGroups.entries()].map(([chainId, residues]) => {
        residues.sort((a, b) => Number(a.residue_number) - Number(b.residue_number)
          || String(a.insertion_code || "").localeCompare(String(b.insertion_code || "")));
        const residueLabels = residues.map(residue => {
          const aminoAcid = aminoAcidLabel(residue.residue_name || residue.amino_acid).short;
          const structureId = `${residue.residue_number}${residue.insertion_code || ""}`;
          const cdr = residue.cdrRegion ? ` <small>${esc(residue.cdrRegion)}</small>` : "";
          return `<span class="interface-residue-item">${esc(aminoAcid)} ${esc(structureId)}${cdr}</span>`;
        }).join("");
        return `<div class="interface-residue-group"><strong>Chain ${esc(chainId)}</strong><div class="interface-residue-items">${residueLabels}</div></div>`;
      }).join("");
    };

    renderGroup(heavyCell, "heavy", "Heavy-chain");
    renderGroup(lightCell, "light", "Light-chain");
    renderGroup(antigenCell, "antigen", "Antigen-chain");
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
    const rowAminoAcid = aminoAcidLabel(aa[row]);
    const columnAminoAcid = aminoAcidLabel(aa[col]);
    const value = Number(currentResult.matrix[row][col]);
    const inspector = byId("cell-inspector");
    if (inspector) inspector.innerHTML = `<span class="card-index">CELL INSPECTOR</span><h3>${esc(rowAminoAcid.short)} × ${esc(columnAminoAcid.short)}</h3><p>${esc(rowAminoAcid.full)} × ${esc(columnAminoAcid.full)}</p><p>Row ${row + 1} · Column ${col + 1}</p><div class="cell-detail"><span>Actual matrix value</span><strong>${Number.isFinite(value) ? value.toFixed(6) : "Not returned"}</strong><p>${row === col ? "Diagonal is defined as zero." : row > col ? "Lower triangle: Partner A intrapartner interface contacts." : "Upper triangle: Partner B intrapartner interface contacts."}</p></div>`;
  });

  function updateViewerLabels(prediction = {}) {
    const roles = getViewerChainRoles(currentResult);
    text("viewer-heavy-chains", chainsText(roles.heavy));
    text("viewer-light-chains", chainsText(roles.light));
    text("viewer-antigen-chains", chainsText(roles.antigen));
  }

  function getViewerChainRoles(result) {
    const prediction = result?.prediction || {};
    const partnerA = [...new Set((prediction.partner_a_chains || []).map(String))];
    const partnerB = [...new Set((prediction.partner_b_chains || []).map(String))];
    const participating = [...new Set([...partnerA, ...partnerB])];
    const domains = Array.isArray(result?.cdr_analysis?.antibody_domains) ? result.cdr_analysis.antibody_domains : [];
    const chainIds = type => [...new Set(domains
      .filter(domain => domain.chain_type === type && domain.chain_id != null)
      .map(domain => String(domain.chain_id))
      .filter(chain => participating.includes(chain)))];
    const heavy = chainIds("heavy");
    const light = chainIds("light");
    const antibody = new Set([...heavy, ...light]);
    const antibodyPartners = new Set(domains.map(domain => String(domain.partner || "").toLowerCase()).filter(value => value === "partner a" || value === "partner b"));
    let antigen = [];
    if (antibodyPartners.has("partner a") && !antibodyPartners.has("partner b")) antigen = partnerB;
    else if (antibodyPartners.has("partner b") && !antibodyPartners.has("partner a")) antigen = partnerA;
    else if (domains.length) antigen = participating.filter(chain => !antibody.has(chain));
    return {
      heavy,
      light,
      antigen,
      partnerA,
      partnerB,
      domains
    };
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
      if (!viewer) viewer = $3Dmol.createViewer(mount, { backgroundColor: "#080809", antialias: true });
      bindViewerControls();
      const loadGeneration = ++viewerLoadGeneration;
      clearTimeout(viewerIdleTimer);
      viewerIdleTimer = null;
      viewer.spin(false);
      viewer.clear();
      viewerModel = null;
      viewerInitialView = null;
      show(byId("viewer-message"));
      resetViewerControls();
      const structure = currentStructure;
      if (!structure) return;
      if (structure.kind === "upload") {
        structure.file.text().then(contents => addStructure(contents, structure.file.name.toLowerCase().endsWith(".cif") ? "cif" : "pdb", loadGeneration));
      } else if (structure.pdbId) {
        fetch(`https://files.rcsb.org/download/${encodeURIComponent(structure.pdbId)}.cif`)
          .then(response => { if (!response.ok) throw new Error(`RCSB returned ${response.status}`); return response.text(); })
          .then(contents => addStructure(contents, "cif", loadGeneration))
          .catch(error => { if (loadGeneration !== viewerLoadGeneration) return; show(byId("viewer-message")); text("viewer-note", `Could not load the coordinates from RCSB: ${error.message}. The prediction result remains available.`); });
      } else {
        text("viewer-note", "This saved upload has no coordinate file in browser history. Upload it again to view its structure.");
      }
    } catch (error) {
      show(byId("viewer-message"));
      text("viewer-note", `The 3D viewer could not be initialized: ${error.message}`);
    }
  }

  function addStructure(contents, format, loadGeneration) {
    if (!viewer || loadGeneration !== viewerLoadGeneration) return;
    viewer.clear();
    viewerModel = viewer.addModel(contents, format);
    applyViewerHighlight();
    viewer.zoomTo();
    viewer.resize();
    viewer.render();
    viewerInitialView = viewer.getView();
    show(byId("viewer-message"), false);
    updateCdrViewerAvailability();
    scheduleViewerAutoRotation();
  }

  function applyViewerHighlight() {
    if (!viewer || !viewerModel || !currentResult) return;
    const roles = getViewerChainRoles(currentResult);
    const selected = byId("viewer-highlight")?.value || "both";
    const visible = {
      heavy: byId("viewer-show-heavy")?.checked !== false,
      light: byId("viewer-show-light")?.checked !== false,
      antigen: byId("viewer-show-antigen")?.checked !== false
    };
    const colorByRole = { heavy: "#3987ff", light: "#36c275", antigen: "#ec64a9" };
    const roleByChain = new Map();
    roles.heavy.forEach(chain => roleByChain.set(chain, "heavy"));
    roles.light.forEach(chain => roleByChain.set(chain, "light"));
    roles.antigen.forEach(chain => roleByChain.set(chain, "antigen"));

    // Reset every atom to a neutral cartoon before applying the current roles,
    // visibility, and CDR styles so a newly loaded structure cannot inherit old styling.
    viewer.setStyle({}, { cartoon: { color: "#718096", opacity: 0.9 } });
    for (const chain of [...new Set([...roles.partnerA, ...roles.partnerB])]) {
      const role = roleByChain.get(chain);
      if (!role) continue;
      const isVisible = visible[role];
      const partner = roles.partnerA.includes(chain) ? "a" : "b";
      const opacity = selected !== "both" && selected !== partner ? 0.22 : 1;
      viewer.setStyle({ chain }, isVisible
        ? { cartoon: { color: colorByRole[role], opacity } }
        : {});
    }

    const cdrEnabled = byId("viewer-show-cdr")?.checked !== false;
    if (cdrEnabled) {
      const cdrColors = { 1: "#ff9f43", 2: "#ffd43b", 3: "#f04452" };
      for (const domain of roles.domains) {
        const chain = String(domain.chain_id ?? "");
        const role = roleByChain.get(chain);
        if (!role || !visible[role]) continue;
        for (const region of domain.cdr_regions || []) {
          const match = String(region.name || "").match(/([123])$/);
          const color = match ? cdrColors[match[1]] : null;
          if (!color) continue;
          for (const residue of region.residues || []) {
            const residueNumber = Number(residue.pdb_residue_number);
            if (!Number.isFinite(residueNumber)) continue;
            const selector = { chain, resi: residueNumber };
            const insertionCode = String(residue.pdb_insertion_code || "").trim();
            if (insertionCode) selector.icode = insertionCode;
            const partner = roles.partnerA.includes(chain) ? "a" : "b";
            const opacity = selected !== "both" && selected !== partner ? 0.22 : 1;
            viewer.setStyle(selector, { cartoon: { color, opacity } });
          }
        }
      }
    }
    viewer.render();
  }

  function resetViewerControls() {
    for (const id of ["viewer-show-heavy", "viewer-show-light", "viewer-show-antigen", "viewer-show-cdr", "viewer-auto-rotate"]) {
      const control = byId(id);
      if (control) control.checked = true;
    }
    const highlight = byId("viewer-highlight");
    if (highlight) highlight.value = "both";
    updateCdrViewerAvailability();
  }

  function updateCdrViewerAvailability() {
    const control = byId("viewer-show-cdr");
    const roles = getViewerChainRoles(currentResult);
    const hasCdrResidues = roles.domains.some(domain => (domain.cdr_regions || []).some(region => (region.residues || []).length));
    if (control) {
      control.disabled = !hasCdrResidues;
      if (!hasCdrResidues) control.checked = false;
    }
    for (const [id, chainList] of [["viewer-show-heavy", roles.heavy], ["viewer-show-light", roles.light], ["viewer-show-antigen", roles.antigen]]) {
      const chainControl = byId(id);
      if (chainControl) chainControl.disabled = chainList.length === 0;
    }
    const hasRoles = roles.heavy.length || roles.light.length || roles.antigen.length;
    text("viewer-note", hasCdrResidues
      ? "Drag to rotate and scroll to zoom. Heavy, light, antigen, and exact annotated CDR residues are colored separately."
      : hasRoles
        ? "Drag to rotate and scroll to zoom. Chain colors use returned chain metadata; CDR highlighting is unavailable for this structure."
        : "Drag to rotate and scroll to zoom. Chain roles or CDR annotations were not returned for this structure.");
  }

  function pauseViewerAutoRotation() {
    if (!viewer || !viewerModel) return;
    viewer.spin(false);
    clearTimeout(viewerIdleTimer);
    viewerIdleTimer = null;
  }

  function scheduleViewerAutoRotation() {
    clearTimeout(viewerIdleTimer);
    viewerIdleTimer = null;
    if (!viewer || !viewerModel || byId("viewer-auto-rotate")?.checked === false) return;
    viewerIdleTimer = setTimeout(() => {
      viewerIdleTimer = null;
      if (viewer && viewerModel && byId("viewer-auto-rotate")?.checked !== false) viewer.spin("y", 0.15, true);
    }, 2200);
  }

  function bindViewerControls() {
    if (viewerEventsBound) return;
    viewerEventsBound = true;
    ["viewer-highlight", "viewer-show-heavy", "viewer-show-light", "viewer-show-antigen", "viewer-show-cdr"]
      .forEach(id => byId(id)?.addEventListener("change", applyViewerHighlight));
    byId("viewer-auto-rotate")?.addEventListener("change", () => {
      pauseViewerAutoRotation();
      if (byId("viewer-auto-rotate")?.checked) scheduleViewerAutoRotation();
    });
    const stage = byId("viewer-stage");
    ["pointerdown", "pointermove", "wheel", "touchstart", "touchmove"].forEach(eventName => stage?.addEventListener(eventName, () => {
      pauseViewerAutoRotation();
      scheduleViewerAutoRotation();
    }, { passive: true }));
    ["pointerup", "pointercancel", "touchend"].forEach(eventName => stage?.addEventListener(eventName, scheduleViewerAutoRotation, { passive: true }));
    byId("viewer-reset")?.addEventListener("click", () => {
      pauseViewerAutoRotation();
      if (viewerInitialView) viewer?.setView(viewerInitialView);
      else viewer?.zoomTo();
      viewer?.render();
      scheduleViewerAutoRotation();
    });
  }

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
    const aa = (currentResult.amino_acid_order || aaFallback).map(code => aminoAcidLabel(code).short);
    const csv = [`,${aa.join(",")}`, ...currentResult.matrix.map((row, i) => `${aa[i]},${row.join(",")}`)].join("\n");
    download(`${currentResult.pdb_id || "structure"}-matrix.csv`, csv, "text/csv");
  });

  document.querySelectorAll(".learning-question").forEach(question => question.addEventListener("toggle", () => { if(question.open) document.querySelectorAll(".learning-question[open]").forEach(other => { if(other!==question) other.open=false; }); }));

  renderHistory();
  navigate("home");
})();
