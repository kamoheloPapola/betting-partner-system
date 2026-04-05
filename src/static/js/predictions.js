import { fetchPredictions } from "./api.js";
import {
  formatDate,
  formatDateInput,
  formatTime,
  leagueFlag,
  pct,
  probabilityForMarket,
  selectionFromMatch,
  showToast,
} from "./utils.js";

let appStore = null;
let cachedPredictions = [];

export function renderPredictionsPage() {
  return `
    <div class="page-shell page-enter">
      <div class="filter-bar">
        <select class="filter-select" id="filter-league">
          <option value="">All Leagues</option>
          <option value="PL">Premier League</option>
          <option value="BL1">Bundesliga</option>
          <option value="FL1">Ligue 1</option>
          <option value="SA">Serie A</option>
          <option value="PD">La Liga</option>
        </select>
        <input type="date" class="filter-input" id="filter-date-from" placeholder="From">
        <input type="date" class="filter-input" id="filter-date-to" placeholder="To">
        <select class="filter-select" id="filter-market">
          <option value="">All Markets</option>
          <option value="1X2">1X2 (Result)</option>
          <option value="btts">Both Teams to Score</option>
          <option value="over25">Over 2.5 Goals</option>
          <option value="corners">Corners</option>
          <option value="cards">Cards</option>
        </select>
        <div class="filter-slider-group">
          <label for="filter-confidence">Min Confidence</label>
          <input type="range" id="filter-confidence" min="0" max="100" value="0">
          <span id="confidence-label">0%</span>
        </div>
        <button class="btn-primary" id="apply-filters-btn">Apply</button>
        <button class="btn-ghost" id="clear-filters-btn">Clear</button>
      </div>

      <div
        id="slow-load-msg"
        style="display: none; text-align: center; padding: 12px; color: var(--amber); font-size: 13px; font-family: var(--font-mono);"
      >
        Computing predictions - first load takes about 10s. Subsequent loads are served from cache.
      </div>

      <div id="predictions-root"></div>
    </div>
  `;
}

export function initPredictionsPage(store) {
  appStore = store;
  cachedPredictions = Array.isArray(store.predictions) ? [...store.predictions] : [];

  const slider = document.getElementById("filter-confidence");
  const applyButton = document.getElementById("apply-filters-btn");
  const clearButton = document.getElementById("clear-filters-btn");

  slider?.addEventListener("input", () => {
    const label = document.getElementById("confidence-label");
    if (label) {
      label.textContent = `${slider.value}%`;
    }
  });

  applyButton?.addEventListener("click", () => applyFilters());
  clearButton?.addEventListener("click", () => clearFilters());

  window.applyFilters = applyFilters;
  window.clearFilters = clearFilters;
  window.addToSlip = addToSlip;

  if (cachedPredictions.length) {
    renderPredictionResults(cachedPredictions);
    return;
  }

  loadPredictions();
}

async function loadPredictions() {
  renderSkeletons();
  const slowLoadMessage = document.getElementById("slow-load-msg");
  const slowTimer = window.setTimeout(() => {
    if (slowLoadMessage) {
      slowLoadMessage.style.display = "block";
    }
  }, 3000);

  try {
    const response = await fetchPredictions();
    cachedPredictions = Array.isArray(response.predictions) ? response.predictions : [];
    appStore.predictions = cachedPredictions;
    window.clearTimeout(slowTimer);
    if (slowLoadMessage) {
      slowLoadMessage.style.display = "none";
    }
    renderPredictionResults(cachedPredictions);
  } catch (error) {
    window.clearTimeout(slowTimer);
    if (slowLoadMessage) {
      slowLoadMessage.style.display = "none";
    }
    renderError(error.message);
  }
}

function renderSkeletons() {
  const root = document.getElementById("predictions-root");
  if (!root) {
    return;
  }

  root.innerHTML = `
    <div class="skeleton-grid">
      <div class="skeleton skeleton-card"></div>
      <div class="skeleton skeleton-card"></div>
      <div class="skeleton skeleton-card"></div>
    </div>
  `;
}

function renderPredictionResults(predictions) {
  const root = document.getElementById("predictions-root");
  if (!root) {
    return;
  }

  if (!predictions.length) {
    root.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">&#9917;</div>
        <h3>Predictions are not ready yet</h3>
        <p>We are still syncing the latest model data for this fixture window. Please refresh in a moment or widen the date range.</p>
        <div class="empty-state-actions">
          <button type="button" class="btn-primary" onclick="applyFilters()">Refresh</button>
          <button type="button" class="btn-ghost" onclick="clearFilters()">Reset Filters</button>
        </div>
      </div>
    `;
    return;
  }

  root.innerHTML = `
    <div class="card-grid">
      ${predictions.map((match) => renderMatchCard(match)).join("")}
    </div>
  `;
}

function renderError(message) {
  const root = document.getElementById("predictions-root");
  if (!root) {
    return;
  }

  root.innerHTML = `
    <div class="empty-state">
      <div class="empty-icon">&#9888;</div>
      <h3>Could not load predictions</h3>
      <p>${message}</p>
      <button type="button" onclick="applyFilters()">Try again</button>
    </div>
  `;
}

function renderMatchCard(match) {
  return `
    <div class="match-card" data-match-id="${match.id}">
      <div class="card-header">
        <div class="league-badge">
          <span class="league-flag">${leagueFlag(match.league)}</span>
          <span class="league-name">${match.league_name}</span>
        </div>
        <div class="match-meta">
          <span class="match-date">${formatDate(match.date)}</span>
          <span class="match-time">${formatTime(match.kickoff)}</span>
        </div>
      </div>

      <div class="teams-row">
        <div class="team home">
          <div class="team-name">${match.home_team}</div>
          <div class="form-strip">${renderFormStrip(match.home_form)}</div>
        </div>
        <div class="vs-badge">VS</div>
        <div class="team away">
          <div class="team-name">${match.away_team}</div>
          <div class="form-strip">${renderFormStrip(match.away_form)}</div>
        </div>
      </div>

      <div class="probability-section">
        <div class="prob-label">RESULT PROBABILITIES</div>
        ${renderProbBar("Home", match.home_prob, "green")}
        ${renderProbBar("Draw", match.draw_prob, "amber")}
        ${renderProbBar("Away", match.away_prob, "blue")}
      </div>

      <div class="stats-row">
        <div class="stat">
          <span class="stat-label">xG</span>
          <span class="stat-value">${match.expected_goals?.toFixed(1) ?? "\u2014"}</span>
        </div>
        <div class="stat">
          <span class="stat-label">BTTS</span>
          <span class="stat-value">${match.btts_prob != null ? pct(match.btts_prob) : "\u2014"}</span>
        </div>
        <div class="stat">
          <span class="stat-label">O2.5</span>
          <span class="stat-value">${match.over25_prob != null ? pct(match.over25_prob) : "\u2014"}</span>
        </div>
        <div class="stat">
          <span class="stat-label">Corners</span>
          <span class="stat-value">${match.corners_mean?.toFixed(1) ?? "\u2014"}</span>
        </div>
      </div>

      <div class="h2h-row">
        <span class="h2h-label">H2H</span>
        <span class="h2h-home">${match.home_team_short} W${match.h2h?.home_wins ?? 0}</span>
        <span class="h2h-draw">D${match.h2h?.draws ?? 0}</span>
        <span class="h2h-away">${match.away_team_short} W${match.h2h?.away_wins ?? 0}</span>
        <span class="h2h-span">(last 5)</span>
      </div>

      <div class="card-footer">
        <button class="btn-add-slip" type="button" onclick="addToSlip('${match.id}')">+ Add to Slip</button>
        <div class="confidence-badge confidence-${match.confidence_tier}">
          <span class="confidence-dot"></span>
          ${match.confidence_tier?.toUpperCase() ?? "LOW"}
        </div>
      </div>
    </div>
  `;
}

function renderFormStrip(form) {
  if (!form || !form.length) {
    return '<div class="form-strip-empty">No data</div>';
  }

  return form.slice(-5).map((result) => {
    const colorMap = { W: "form-w", D: "form-d", L: "form-l" };
    return `<span class="form-dot ${colorMap[result] || "form-d"}" title="${result}"></span>`;
  }).join("");
}

function renderProbBar(label, prob, color) {
  const pctValue = Math.round((Number(prob ?? 0)) * 100);
  return `
    <div class="prob-row">
      <span class="prob-label-text">${label}</span>
      <div class="prob-bar-track">
        <div class="prob-bar-fill prob-${color}" style="width: ${pctValue}%"></div>
      </div>
      <span class="prob-pct">${pctValue}%</span>
    </div>
  `;
}

async function applyFilters() {
  const league = document.getElementById("filter-league")?.value || "";
  const dateFrom = document.getElementById("filter-date-from")?.value || "";
  const dateTo = document.getElementById("filter-date-to")?.value || "";
  const market = document.getElementById("filter-market")?.value || "";
  const minConfidence = Number(document.getElementById("filter-confidence")?.value || 0);

  if (!cachedPredictions.length) {
    await loadPredictions();
    return;
  }

  const filtered = cachedPredictions.filter((match) => {
    if (league && match.league !== league) {
      return false;
    }

    const matchDate = formatDateInput(match.kickoff || match.date);
    if (dateFrom && matchDate < dateFrom) {
      return false;
    }
    if (dateTo && matchDate > dateTo) {
      return false;
    }

    if (market && probabilityForMarket(match, market) <= 0) {
      return false;
    }

    if (minConfidence > 0) {
      const confidence = probabilityForMarket(match, market);
      if ((confidence * 100) < minConfidence) {
        return false;
      }
    }

    return true;
  });

  renderPredictionResults(filtered);
}

function clearFilters() {
  const ids = ["filter-league", "filter-date-from", "filter-date-to", "filter-market"];
  ids.forEach((id) => {
    const element = document.getElementById(id);
    if (element) {
      element.value = "";
    }
  });

  const slider = document.getElementById("filter-confidence");
  const label = document.getElementById("confidence-label");
  if (slider) {
    slider.value = "0";
  }
  if (label) {
    label.textContent = "0%";
  }

  renderPredictionResults(cachedPredictions);
}

function addToSlip(matchId) {
  const match = cachedPredictions.find((entry) => entry.id === matchId);
  if (!match) {
    return;
  }

  const pick = selectionFromMatch(match);
  const store = window.__pitchsenseStore;
  const entry = {
    id: match.id,
    match: `${match.home_team} vs ${match.away_team}`,
    selection: pick.label,
    market: pick.market,
    prob: pick.prob,
    league: match.league,
  };

  const existingIndex = store.manualSlip.findIndex((item) => item.id === entry.id);
  if (existingIndex >= 0) {
    store.manualSlip.splice(existingIndex, 1, entry);
  } else {
    store.manualSlip.push(entry);
  }

  showToast("Added to slip");
  window.navigateTo("slip-builder");
}
