import { fetchApiHealth, fetchDriftStatus, fetchModelHealth } from "./api.js";
import { renderPredictionsPage, initPredictionsPage } from "./predictions.js";
import { renderSlipBuilderPage, initSlipBuilderPage } from "./slipBuilder.js";
import { renderTerminalPage, initTerminalPage } from "./terminal.js";

window.__pitchsenseStore = {
  predictions: [],
  manualSlip: [],
};

const routes = {
  predictions: {
    title: "Match Predictions",
    subtitle: "Probability estimates for upcoming fixtures",
    render: renderPredictionsPage,
    init: () => initPredictionsPage(window.__pitchsenseStore),
  },
  "slip-builder": {
    title: "Slip Builder",
    subtitle: "Build confidence-weighted selections and export them instantly",
    render: renderSlipBuilderPage,
    init: () => initSlipBuilderPage(),
  },
  "model-health": {
    title: "Model Health",
    subtitle: "Registry coverage, calibration quality, and productive model status",
    render: renderModelHealthPage,
    init: initModelHealthPage,
  },
  "drift-monitor": {
    title: "Drift Monitor",
    subtitle: "Live guardrail posture and recent monitoring alerts",
    render: renderDriftMonitorPage,
    init: initDriftMonitorPage,
  },
  terminal: {
    title: "Terminal",
    subtitle: "Execute backend commands from the browser",
    render: renderTerminalPage,
    init: initTerminalPage,
  },
};

function resolvePage() {
  const rawHash = window.location.hash.replace(/^#/, "");
  return routes[rawHash] ? rawHash : "predictions";
}

function setTopbar(page) {
  const config = routes[page];
  const title = document.getElementById("page-title");
  const subtitle = document.getElementById("page-subtitle");
  if (title) {
    title.textContent = config.title;
  }
  if (subtitle) {
    subtitle.textContent = config.subtitle;
  }

  const updatedAt = document.getElementById("updated-at");
  if (updatedAt) {
    updatedAt.textContent = `Updated ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
  }
}

function setActiveNav(page) {
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.classList.toggle("active", item.dataset.page === page);
  });
}

function renderPage(page) {
  const config = routes[page];
  const container = document.getElementById("page-container");
  if (!container || !config) {
    return;
  }

  setTopbar(page);
  setActiveNav(page);
  container.innerHTML = config.render();
  config.init();
}

window.navigateTo = function navigateTo(page) {
  window.location.hash = page;
};

async function updateApiStatus() {
  const pill = document.getElementById("api-pill");
  const sidebar = document.getElementById("sidebar-api-status");
  const label = document.getElementById("api-pill-text");
  const sidebarLabel = document.getElementById("sidebar-api-text");

  try {
    await fetchApiHealth();
    pill?.classList.add("online");
    pill?.classList.remove("offline");
    sidebar?.classList.add("online");
    sidebar?.classList.remove("offline");
    if (label) {
      label.textContent = "API Online";
    }
    if (sidebarLabel) {
      sidebarLabel.textContent = "API Online";
    }
  } catch (_error) {
    pill?.classList.add("offline");
    pill?.classList.remove("online");
    sidebar?.classList.add("offline");
    sidebar?.classList.remove("online");
    if (label) {
      label.textContent = "API Offline";
    }
    if (sidebarLabel) {
      sidebarLabel.textContent = "API Offline";
    }
  }
}

function bindNav() {
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.addEventListener("click", (event) => {
      event.preventDefault();
      window.navigateTo(item.dataset.page);
    });
  });
}

function renderModelHealthPage() {
  return `
    <div class="page-shell page-enter">
      <div class="status-card-grid">
        <div class="status-card full-width" id="model-health-summary">
          <h3>Loading model health...</h3>
        </div>
        <div class="status-card full-width" id="model-health-markets"></div>
      </div>
    </div>
  `;
}

async function initModelHealthPage() {
  const summary = document.getElementById("model-health-summary");
  const marketsNode = document.getElementById("model-health-markets");
  if (!summary || !marketsNode) {
    return;
  }

  try {
    const payload = await fetchModelHealth();
    const markets = Object.entries(payload.markets || {});
    summary.innerHTML = `
      <h3>Global Status</h3>
      <div class="kv-grid">
        <div class="kv-row"><span class="kv-key">Generated</span><span class="kv-value">${payload.generated_at || "-"}</span></div>
        <div class="kv-row"><span class="kv-key">Global Drift</span><span class="kv-value">${payload.global_drift_status || "-"}</span></div>
        <div class="kv-row"><span class="kv-key">Markets</span><span class="kv-value">${markets.length}</span></div>
      </div>
    `;

    marketsNode.innerHTML = markets.map(([market, entries]) => `
      <div class="status-card" style="margin-bottom: 16px;">
        <h3>${market}</h3>
        <div class="table-wrap">
          <table class="data-table">
            <thead>
              <tr>
                <th>Serving League</th>
                <th>Model League</th>
                <th>Version</th>
                <th>Brier</th>
                <th>Drift</th>
              </tr>
            </thead>
            <tbody>
              ${entries.map((entry) => `
                <tr>
                  <td>${entry.league}</td>
                  <td>${entry.model_league}</td>
                  <td>${entry.version}</td>
                  <td>${entry.brier_score ?? "-"}</td>
                  <td>${entry.drift_status ?? "-"}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      </div>
    `).join("");
  } catch (error) {
    summary.innerHTML = `<h3>Model health unavailable</h3><p class="muted">${error.message}</p>`;
    marketsNode.innerHTML = "";
  }
}

function renderDriftMonitorPage() {
  return `
    <div class="page-shell page-enter">
      <div class="status-card" id="drift-monitor-root">
        <h3>Loading drift state...</h3>
      </div>
    </div>
  `;
}

async function initDriftMonitorPage() {
  const root = document.getElementById("drift-monitor-root");
  if (!root) {
    return;
  }

  try {
    const payload = await fetchDriftStatus();
    const alerts = payload.recent_alerts || payload.alerts || [];
    root.innerHTML = `
      <h3>Drift Monitor</h3>
      <div class="kv-grid">
        <div class="kv-row"><span class="kv-key">Status</span><span class="kv-value">${payload.status || "-"}</span></div>
        <div class="kv-row"><span class="kv-key">Total Alerts</span><span class="kv-value">${payload.total_alerts ?? alerts.length}</span></div>
      </div>
      ${alerts.length ? `
        <div class="table-wrap" style="margin-top: 18px;">
          <table class="data-table">
            <thead>
              <tr>
                <th>Alert</th>
                <th>Value</th>
                <th>When</th>
              </tr>
            </thead>
            <tbody>
              ${alerts.map((alert) => `
                <tr>
                  <td>${alert.alert || alert.type || JSON.stringify(alert).slice(0, 32)}</td>
                  <td>${alert.value ?? alert.status ?? "-"}</td>
                  <td>${alert.timestamp || alert.date || "-"}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      ` : `<p class="muted" style="margin-top: 18px;">No drift alerts recorded.</p>`}
    `;
  } catch (error) {
    root.innerHTML = `<h3>Drift data unavailable</h3><p class="muted">${error.message}</p>`;
  }
}

window.addEventListener("hashchange", () => renderPage(resolvePage()));

document.addEventListener("DOMContentLoaded", () => {
  bindNav();
  updateApiStatus();
  renderPage(resolvePage());
  setInterval(updateApiStatus, 30000);
});
