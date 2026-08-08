export const API_BASE = window.location.origin;

async function fetchJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const text = await response.text();
  let data = {};

  if (text) {
    try {
      data = JSON.parse(text);
    } catch (_error) {
      data = { detail: text };
    }
  }

  if (!response.ok) {
    throw new Error(data.detail || data.message || `API error: ${response.status}`);
  }

  return data;
}

export async function fetchPredictions({ league, dateFrom, dateTo, market, minConfidence } = {}) {
  const params = new URLSearchParams();
  if (league) {
    params.set("league", league);
  }
  if (dateFrom) {
    params.set("date_from", dateFrom);
  }
  if (dateTo) {
    params.set("date_to", dateTo);
  }
  if (market) {
    params.set("market", market);
  }
  if (minConfidence) {
    params.set("min_confidence", minConfidence);
  }

  const query = params.toString();
  return fetchJson(`/predictions${query ? `?${query}` : ""}`);
}

export async function fetchModelHealth() {
  return fetchJson("/model-health");
}

export async function fetchDriftStatus() {
  return fetchJson("/drift-status");
}

export async function fetchApiHealth() {
  return fetchJson("/health");
}

export async function callCli(endpoint, { method = "GET", args = {} } = {}) {
  let path = endpoint;
  const options = {
    method,
    headers: { "Content-Type": "application/json" },
  };

  if (method === "GET" && Object.keys(args).length) {
    path += `?${new URLSearchParams(args).toString()}`;
  } else if (method !== "GET") {
    options.body = JSON.stringify(args);
  }

  return fetchJson(path, options);
}
