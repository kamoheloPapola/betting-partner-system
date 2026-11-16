"use strict";
const $ = id => document.getElementById(id);
const pct = value => Number.isFinite(value) ? (value * 100).toFixed(1) + "%" : "—";
const stamp = value => value ? value.replace("T", " ").replace(/\.\d+Z$/, " UTC") : "—";
function cell(row, text, className) {
  const td = document.createElement("td");
  td.textContent = text;
  if (className) td.className = className;
  row.appendChild(td);
}
async function get(path) {
  const response = await fetch(path, {cache: "no-store", signal: AbortSignal.timeout(10000)});
  if (!response.ok) throw new Error("Request failed (" + response.status + ")");
  return response.json();
}
async function refresh() {
  $("refresh").disabled = true;
  try {
    const query = $("league").value ? "?league=" + encodeURIComponent($("league").value) : "";
    const responses = await Promise.all([get("/api/health"), get("/api/forecasts" + query)]);
    const health = responses[0], data = responses[1];
    $("health").textContent = health.state;
    $("health").className = health.state === "GREEN" ? "good" : health.state === "YELLOW" ? "warn" : "bad";
    $("health-detail").textContent = (health.reasons || []).concat(health.warnings || []).join(" · ").replaceAll("_", " ") || "The latest run completed successfully.";
    $("last-success").textContent = stamp(health.last_success_utc);
    $("forecast-count").textContent = String(health.forecast_count || 0);
    $("notice").textContent = health.halt ? "New forecasts are halted for review. Existing records remain available." : "";
    $("notice").className = health.halt ? "notice bad" : "";
    $("rows").replaceChildren();
    const forecasts = data.forecasts || [];
    for (const f of forecasts) {
      const row = document.createElement("tr");
      cell(row, stamp(f.kickoff_utc));
      cell(row, f.home_name + " — " + f.away_name);
      cell(row, f.home_xg.toFixed(2) + " — " + f.away_xg.toFixed(2));
      for (const name of ["home_win", "draw", "away_win", "over_2_5", "btts_yes"]) cell(row, pct(f.markets[name]), "numeric");
      const result = f.settlement === "SETTLED" ? f.home_goals + "–" + f.away_goals + " · settled" : f.settlement.replaceAll("_", " ").toLowerCase();
      cell(row, result);
      $("rows").appendChild(row);
    }
    $("empty").hidden = forecasts.length > 0;
  } catch (error) {
    $("health").textContent = "UNAVAILABLE";
    $("health").className = "bad";
    $("notice").textContent = "The local record could not be read. Check the last job result; forecasts have not been regenerated.";
    $("notice").className = "notice bad";
  } finally { $("refresh").disabled = false; }
}
$("refresh").addEventListener("click", refresh);
$("league").addEventListener("change", refresh);
refresh();
setInterval(refresh, 30000);

