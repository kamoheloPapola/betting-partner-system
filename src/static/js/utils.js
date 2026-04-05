const EMPTY_VALUE = "\u2014";

export function pct(prob) {
  if (prob == null || Number.isNaN(Number(prob))) {
    return EMPTY_VALUE;
  }

  return `${Math.round(Number(prob) * 100)}%`;
}

export function formatDate(iso) {
  if (!iso) {
    return EMPTY_VALUE;
  }

  return new Date(iso).toLocaleDateString("en-GB", {
    weekday: "short",
    day: "numeric",
    month: "short",
  });
}

export function formatTime(iso) {
  if (!iso) {
    return EMPTY_VALUE;
  }

  return new Date(iso).toLocaleTimeString("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function leagueFlag(code) {
  const flags = {
    PL: "\u{1F3F4}\u{E0067}\u{E0062}\u{E0065}\u{E006E}\u{E0067}\u{E007F}",
    BL1: "\u{1F1E9}\u{1F1EA}",
    FL1: "\u{1F1EB}\u{1F1F7}",
    SA: "\u{1F1EE}\u{1F1F9}",
    PD: "\u{1F1EA}\u{1F1F8}",
  };

  return flags[code] || "\u26BD";
}

export function showToast(msg) {
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 2500);
}

export function compositeProb(slip) {
  if (!slip.length) {
    return 0;
  }

  return slip.reduce((acc, selection) => acc * (selection.prob ?? 0.5), 1) ** (1 / slip.length);
}

export function leagueName(code) {
  const names = {
    PL: "Premier League",
    BL1: "Bundesliga",
    FL1: "Ligue 1",
    SA: "Serie A",
    PD: "La Liga",
  };

  return names[code] || code || "Unknown League";
}

export function shortTeamName(name) {
  if (!name) {
    return "";
  }

  const parts = String(name).trim().split(/\s+/);
  if (parts.length <= 2) {
    return parts[0];
  }

  return parts.slice(0, 2).join(" ");
}

export function confidenceTier(prob) {
  const value = Number(prob ?? 0);
  if (value >= 0.65) {
    return "HIGH";
  }

  if (value >= 0.5) {
    return "MEDIUM";
  }

  return "LOW";
}

export function formFromRating(rating) {
  const value = Number(rating);
  if (Number.isNaN(value)) {
    return [];
  }

  const clamped = Math.max(0, Math.min(1, value));
  const wins = Math.round(clamped * 5);
  const draws = clamped > 0.35 && clamped < 0.65 ? 1 : 0;
  const losses = Math.max(0, 5 - wins - draws);
  const form = [];

  for (let index = 0; index < wins; index += 1) {
    form.push("W");
  }

  for (let index = 0; index < draws; index += 1) {
    form.push("D");
  }

  for (let index = 0; index < losses; index += 1) {
    form.push("L");
  }

  return form.slice(0, 5);
}

export function formatMarketLabel(market) {
  const labels = {
    HOME: "Home Win",
    DRAW: "Draw",
    AWAY: "Away Win",
    BTTS: "BTTS",
    "O2.5": "Over 2.5",
    "Cards O2.5": "Cards O2.5",
    "Cards U5.5": "Cards U5.5",
    "Corners O7.5": "Corners O7.5",
    "Corners U11.5": "Corners U11.5",
    HOME_WIN: "Home Win",
    AWAY_WIN: "Away Win",
    "GOALS_O2.5": "Over 2.5",
    BTTS_YES: "BTTS",
    "CARDS_O2.5": "Cards O2.5",
    "CARDS_U5.5": "Cards U5.5",
    "CORNERS_O7.5": "Corners O7.5",
    "CORNERS_U11.5": "Corners U11.5",
    DOUBLE_CHANCE: "Double Chance",
  };

  return labels[market] || market || "Selection";
}

export function selectionFromMatch(match) {
  const options = [
    { label: "Home Win", market: "HOME", prob: Number(match.home_prob ?? 0) },
    { label: "Draw", market: "DRAW", prob: Number(match.draw_prob ?? 0) },
    { label: "Away Win", market: "AWAY", prob: Number(match.away_prob ?? 0) },
    { label: "BTTS", market: "BTTS", prob: Number(match.btts_prob ?? 0) },
    { label: "Over 2.5", market: "O2.5", prob: Number(match.over25_prob ?? 0) },
    { label: "Corners", market: "CORNERS", prob: Number(match.corners_prob ?? 0) },
    { label: "Cards", market: "CARDS", prob: Number(match.cards_prob ?? 0) },
  ];

  return options.sort((left, right) => right.prob - left.prob)[0];
}

export function probabilityForMarket(match, market) {
  switch (market) {
    case "1X2":
      return Math.max(Number(match.home_prob ?? 0), Number(match.draw_prob ?? 0), Number(match.away_prob ?? 0));
    case "btts":
      return Number(match.btts_prob ?? 0);
    case "over25":
      return Number(match.over25_prob ?? 0);
    case "corners":
      return Number(match.corners_prob ?? 0);
    case "cards":
      return Number(match.cards_prob ?? 0);
    default:
      return Number(match.confidence ?? 0);
  }
}

export function formatDateInput(value) {
  if (!value) {
    return "";
  }

  const date = new Date(value);
  const month = `${date.getMonth() + 1}`.padStart(2, "0");
  const day = `${date.getDate()}`.padStart(2, "0");
  return `${date.getFullYear()}-${month}-${day}`;
}
