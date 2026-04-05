import { fetchSlip } from "./api.js";
import { compositeProb, formatMarketLabel, pct, showToast } from "./utils.js";

let recommendedSlip = [];
let currentLeague = "PL";

export function renderSlipBuilderPage() {
  return `
    <div class="page-shell page-enter">
      <div class="slip-toolbar">
        <div class="slip-toolbar-filters">
          <select class="slip-select" id="slip-league">
            <option value="PL">Premier League</option>
            <option value="BL1">Bundesliga</option>
            <option value="FL1">Ligue 1</option>
            <option value="SA">Serie A</option>
            <option value="PD">La Liga</option>
          </select>
          <button class="btn-primary" type="button" id="load-slip-btn">Load Recommended Slip</button>
        </div>
        <div class="slip-toolbar-actions">
          <button class="btn-ghost" type="button" id="export-slip-btn">Export Slip</button>
          <button class="btn-ghost" type="button" id="clear-manual-slip-btn">Clear Manual</button>
        </div>
      </div>

      <div class="slip-layout">
        <div id="slip-content"></div>
        <div id="slip-summary"></div>
      </div>
    </div>
  `;
}

export function initSlipBuilderPage() {
  const select = document.getElementById("slip-league");
  const loadButton = document.getElementById("load-slip-btn");
  const exportButton = document.getElementById("export-slip-btn");
  const clearButton = document.getElementById("clear-manual-slip-btn");

  if (!select || !loadButton || !exportButton || !clearButton) {
    return;
  }

  select.value = currentLeague;
  loadButton.addEventListener("click", () => {
    currentLeague = select.value || "PL";
    loadRecommendedSlip();
  });

  exportButton.addEventListener("click", () => {
    const slip = activeSlip();
    if (!slip.length) {
      showToast("No slip to export");
      return;
    }

    exportSlip(slip);
  });

  clearButton.addEventListener("click", () => {
    window.__pitchsenseStore.manualSlip = [];
    renderSlip();
    showToast("Manual slip cleared");
  });

  if (window.__pitchsenseStore.manualSlip.length) {
    renderSlip();
  } else {
    loadRecommendedSlip();
  }
}

async function loadRecommendedSlip() {
  const content = document.getElementById("slip-content");
  const summary = document.getElementById("slip-summary");
  if (content) {
    content.innerHTML = '<div class="slip-card"><div class="term-line term-muted">Loading slip...</div></div>';
  }
  if (summary) {
    summary.innerHTML = "";
  }

  try {
    const response = await fetchSlip({ league: currentLeague });
    recommendedSlip = Array.isArray(response.slip)
      ? response.slip.map((selection) => ({
          id: selection.match_id || selection.match,
          match: selection.match,
          selection: formatMarketLabel(selection.market),
          market: selection.market,
          prob: Number(selection.probability ?? selection.confidence ?? 0),
          league: selection.league,
        }))
      : [];
    renderSlip(response.message);
  } catch (error) {
    renderEmptySlip(error.message);
  }
}

function activeSlip() {
  const manualSlip = window.__pitchsenseStore.manualSlip || [];
  return manualSlip.length ? manualSlip : recommendedSlip;
}

function renderSlip(message = "") {
  const slip = activeSlip();
  const content = document.getElementById("slip-content");
  const summary = document.getElementById("slip-summary");

  if (!content || !summary) {
    return;
  }

  if (!slip.length) {
    renderEmptySlip(message || "No slip loaded yet.");
    return;
  }

  const label = window.__pitchsenseStore.manualSlip.length ? "Manual Slip" : "Recommended Slip";
  const composite = compositeProb(slip);

  content.innerHTML = `
    <div class="slip-card">
      <div class="slip-header">
        <div>
          <div class="slip-title">${label}</div>
          <div class="slip-subtitle">${message || "Selections arranged for the current session."}</div>
        </div>
        <div class="slip-confidence">
          <div class="slip-confidence-label">Total Slip Confidence</div>
          <div class="slip-confidence-value">${pct(composite)}</div>
        </div>
      </div>
      <div class="slip-list">
        ${slip.map((selection) => renderSlipEntry(selection)).join("")}
      </div>
    </div>
  `;

  summary.innerHTML = `
    <div class="status-card">
      <h3>Slip Summary</h3>
      <div class="summary-list">
        <div class="summary-item">
          <span class="summary-key">Selections</span>
          <span class="summary-value">${slip.length}</span>
        </div>
        <div class="summary-item">
          <span class="summary-key">Composite</span>
          <span class="summary-value">${pct(composite)}</span>
        </div>
        <div class="summary-item">
          <span class="summary-key">Source</span>
          <span class="summary-value">${window.__pitchsenseStore.manualSlip.length ? "Manual" : "Backend"}</span>
        </div>
      </div>
    </div>
  `;

  content.querySelectorAll("[data-remove-slip]").forEach((button) => {
    button.addEventListener("click", () => removeManualSelection(button.dataset.removeSlip));
  });
}

function renderSlipEntry(selection) {
  const pctValue = Math.round((selection.prob ?? 0) * 100);
  const removeButton = window.__pitchsenseStore.manualSlip.length
    ? `<button type="button" class="slip-entry-remove" data-remove-slip="${selection.id}">Remove</button>`
    : "";

  return `
    <div class="slip-entry">
      <div class="slip-entry-head">
        <div class="slip-entry-title">${selection.match}</div>
        ${removeButton}
      </div>
      <div class="slip-entry-meta">
        <span class="slip-selection">${selection.selection}</span>
        <span class="mono">${pct(selection.prob)}</span>
      </div>
      <div class="prob-row">
        <span class="prob-label-text">Pick</span>
        <div class="prob-bar-track">
          <div class="prob-bar-fill prob-green" style="width: ${pctValue}%"></div>
        </div>
        <span class="prob-pct">${pctValue}%</span>
      </div>
    </div>
  `;
}

function renderEmptySlip(message) {
  const content = document.getElementById("slip-content");
  const summary = document.getElementById("slip-summary");
  if (content) {
    content.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">&#9917;</div>
        <h3>No slip loaded yet</h3>
        <p>${message}</p>
      </div>
    `;
  }
  if (summary) {
    summary.innerHTML = "";
  }
}

function removeManualSelection(selectionId) {
  window.__pitchsenseStore.manualSlip = window.__pitchsenseStore.manualSlip.filter((entry) => entry.id !== selectionId);
  renderSlip();
}

function exportSlip(slip) {
  const lines = slip.map((selection) => `${selection.match}  |  ${selection.selection}  |  ${pct(selection.prob)}`);
  const text = [
    "PITCHSENSE SLIP",
    new Date().toLocaleString(),
    "",
    ...lines,
    "",
    `Composite: ${pct(compositeProb(slip))}`,
  ].join("\n");

  navigator.clipboard.writeText(text);
  showToast("Slip copied to clipboard");
}
