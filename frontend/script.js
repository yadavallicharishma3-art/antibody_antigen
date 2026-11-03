const form = document.querySelector("#prediction-form");
const pdbInput = document.querySelector("#pdb-id");
const fileInput = document.querySelector("#structure-file");
const statusBox = document.querySelector("#result-status");
const resultSection = document.querySelector("#result-section");
const predictionBox = document.querySelector("#prediction-result");
const matrixPanel = document.querySelector("#matrix-panel");
const AA_ORDER = "ACDEFGHIKLMNPQRSTVWY";

pdbInput.addEventListener("input", () => {
  if (pdbInput.value.trim()) fileInput.value = "";
  document.querySelector("#file-name").textContent = "Choose a PDB or mmCIF file";
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) pdbInput.value = "";
  document.querySelector("#file-name").textContent = fileInput.files[0]?.name || "Choose a PDB or mmCIF file";
});
document.querySelectorAll(".examples button").forEach((button) => {
  button.addEventListener("click", () => {
    pdbInput.value = button.dataset.pdb;
    fileInput.value = "";
    document.querySelector("#file-name").textContent = "Choose a PDB or mmCIF file";
    pdbInput.focus();
  });
});
function showStatus(message, kind) {
  statusBox.textContent = message;
  statusBox.className = `status-message ${kind}`;
}
function drawHeatmap(matrix) {
  const canvas = document.querySelector("#heatmap");
  const context = canvas.getContext("2d");
  const size = 700, left = 46, top = 42, cell = 31;
  canvas.width = size; canvas.height = size;
  context.clearRect(0, 0, size, size);
  context.font = "11px Segoe UI, Arial";
  context.textAlign = "center"; context.textBaseline = "middle";
  for (let row = 0; row < 20; row += 1) {
    for (let column = 0; column < 20; column += 1) {
      const value = matrix[row][column];
      const strength = Math.sqrt(Math.max(0, Math.min(1, value)));
      context.fillStyle = row === column ? "#1e293b" : row > column
        ? `rgb(${Math.round(7 + 12 * strength)},${Math.round(39 + 130 * strength)},${Math.round(58 + 170 * strength)})`
        : `rgb(${Math.round(55 + 180 * strength)},${Math.round(39 + 150 * strength)},${Math.round(12 + 70 * strength)})`;
      context.fillRect(left + column * cell, top + row * cell, cell - 1, cell - 1);
      if (value > 0) {
        context.fillStyle = strength > .62 ? "#07111f" : "#e2e8f0";
        context.font = "8px Segoe UI, Arial";
        context.fillText(value.toFixed(2), left + column * cell + cell / 2, top + row * cell + cell / 2);
        context.font = "11px Segoe UI, Arial";
      }
    }
  }
  context.fillStyle = "#94a3b8";
  for (let index = 0; index < 20; index += 1) {
    context.fillText(AA_ORDER[index], left + index * cell + cell / 2, 22);
    context.fillText(AA_ORDER[index], 23, top + index * cell + cell / 2);
  }
}
function displayPrediction(result) {
  const prediction = result.prediction;
  predictionBox.replaceChildren();
  predictionBox.className = "result-card prediction-card";
  const header = document.createElement("div");
  header.className = "prediction-header";
  const left = document.createElement("div");
  left.innerHTML = '<span class="card-eyebrow">STRUCTURE PREDICTION</span>';
  const title = document.createElement("h2");
  title.textContent = prediction.class_name;
  left.append(title);
  const badge = document.createElement("span");
  badge.className = `prediction-class ${prediction.label ? "" : "negative"}`;
  badge.textContent = `Class ${prediction.label} · ${prediction.label ? "ANTIBODY–ANTIGEN" : "GENERAL PROTEIN–PROTEIN"}`;
  header.append(left, badge);
  const details = document.createElement("div");
  details.className = "prediction-meta";
  for (const text of [
    `Structure: ${result.pdb_id}`,
    `Partner A: ${prediction.partner_a_chains.join(", ")}`,
    `Partner B: ${prediction.partner_b_chains.join(", ")}`,
  ]) {
    const item = document.createElement("span"); item.textContent = text; details.append(item);
  }
  predictionBox.append(header, details);
  document.querySelector("#result-pdb-id").textContent = result.pdb_id;
}
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const hasPdbId = Boolean(pdbInput.value.trim());
  const hasFile = Boolean(fileInput.files[0]);
  if (hasPdbId === hasFile) { showStatus("Enter a PDB ID or upload one structure file.", "error"); return; }
  resultSection.classList.add("hidden");
  const button = document.querySelector("#submit-button");
  button.disabled = true;
  button.innerHTML = "<span>Analyzing structure…</span><b>◌</b>";
  showStatus(hasPdbId ? `Retrieving ${pdbInput.value.trim()} from RCSB and analyzing…` : "Validating the uploaded structure and detecting contacts…", "working");
  try {
    const response = await fetch("/api/predict", { method: "POST", body: new FormData(form) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Structure analysis failed.");
    displayPrediction(result);
    drawHeatmap(result.matrix);
    predictionBox.classList.remove("hidden");
    matrixPanel.classList.remove("hidden");
    resultSection.classList.remove("hidden");
    showStatus("Structure validated. Prediction and interaction matrix are ready.", "success");
    resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showStatus(error.message, "error");
  } finally {
    button.disabled = false;
    button.innerHTML = "<span>Analyze structure</span><b>→</b>";
  }
});




