import { callCli, fetchApiHealth } from "./api.js";
import { pct } from "./utils.js";

let commandHistory = [];
let historyIndex = -1;

const COMMAND_MAP = {
  help: { endpoint: "/cli/help", method: "GET" },
  status: { endpoint: "/cli/status", method: "GET" },
  "model-health": { endpoint: "/cli/model-health", method: "GET" },
  "inspect-drift": { endpoint: "/cli/inspect-drift", method: "GET" },
  "reset-drift": { endpoint: "/cli/reset-drift", method: "POST" },
  "invalidate-cache": { endpoint: "/cli/invalidate-cache", method: "POST" },
  "cache-stats": { endpoint: "/cli/cache-stats", method: "GET" },
  predict: { endpoint: "/cli/predict", method: "POST" },
  train: { endpoint: "/cli/train", method: "POST" },
  "list-models": { endpoint: "/cli/list-models", method: "GET" },
  clear: null,
};

export function renderTerminalPage() {
  return `
    <div class="page-shell page-enter">
      <div class="terminal-page">
        <div class="terminal-header">
          <div class="terminal-title">
            <span class="terminal-icon">&#9633;</span>
            Terminal
          </div>
          <div class="terminal-status" id="term-status">
            <span class="status-dot" id="term-status-dot"></span>
            <span id="status-text">Connecting...</span>
          </div>
        </div>

        <div class="quick-commands">
          <span class="qc-label">Quick:</span>
          <button class="qc-btn" data-command="help">help</button>
          <button class="qc-btn" data-command="predict --league PL">predict PL</button>
          <button class="qc-btn" data-command="train --leagues PL BL1 FL1 SA PD">train all</button>
          <button class="qc-btn" data-command="model-health">model health</button>
          <button class="qc-btn" data-command="cache-stats">cache stats</button>
          <button class="qc-btn" data-command="reset-drift">reset drift</button>
          <button class="qc-btn" data-command="inspect-drift">inspect drift</button>
          <button class="qc-btn" data-command="status">status</button>
        </div>

        <div class="terminal-output" id="terminal-output">
          <div class="term-line term-info">Welcome to Pitchsense Terminal</div>
          <div class="term-line term-muted">Type 'help' or use Quick commands above.</div>
        </div>

        <div class="terminal-input-area">
          <span class="term-prompt">$</span>
          <input
            type="text"
            id="term-input"
            class="term-input"
            placeholder="Enter command..."
            autocomplete="off"
            autocorrect="off"
            spellcheck="false"
          >
          <button class="btn-run" id="term-run-btn">Run</button>
          <button class="btn-clear" id="term-clear-btn">Clear</button>
        </div>
      </div>
    </div>
  `;
}

export function initTerminalPage() {
  const input = document.getElementById("term-input");
  const runButton = document.getElementById("term-run-btn");
  const clearButton = document.getElementById("term-clear-btn");

  if (!input || !runButton || !clearButton) {
    return;
  }

  document.querySelectorAll(".qc-btn").forEach((button) => {
    button.addEventListener("click", () => runCommand(button.dataset.command || ""));
  });

  runButton.addEventListener("click", () => submitCommand());
  clearButton.addEventListener("click", () => clearTerminal());

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      submitCommand();
      return;
    }

    if (event.key === "ArrowUp") {
      event.preventDefault();
      if (historyIndex < commandHistory.length - 1) {
        historyIndex += 1;
        input.value = commandHistory[historyIndex];
      }
    }

    if (event.key === "ArrowDown") {
      event.preventDefault();
      if (historyIndex > 0) {
        historyIndex -= 1;
        input.value = commandHistory[historyIndex];
      } else {
        historyIndex = -1;
        input.value = "";
      }
    }
  });

  bindTerminalGlobals();
  checkConnection();
  input.focus();
}

function parseCommand(raw) {
  const parts = raw.trim().split(/\s+/);
  const cmd = (parts[0] || "").toLowerCase();
  const args = {};

  for (let index = 1; index < parts.length; index += 1) {
    if (!parts[index].startsWith("--")) {
      continue;
    }

    const key = parts[index].slice(2);
    const values = [];

    while (index + 1 < parts.length && !parts[index + 1].startsWith("--")) {
      values.push(parts[index + 1]);
      index += 1;
    }

    if (!values.length) {
      args[key] = true;
    } else if (values.length === 1) {
      args[key] = values[0];
    } else {
      args[key] = values.join(" ");
    }
  }

  return { cmd, args };
}

function appendLine(text, type = "term-muted") {
  const output = document.getElementById("terminal-output");
  if (!output) {
    return;
  }

  const line = document.createElement("div");
  line.className = `term-line ${type}`;
  line.textContent = text;
  output.appendChild(line);
  output.scrollTop = output.scrollHeight;
}

function appendPromptLine(text) {
  appendLine(text, "term-prompt-line");
}

function removeLastLine() {
  const output = document.getElementById("terminal-output");
  if (output?.lastChild) {
    output.removeChild(output.lastChild);
  }
}

async function runCommand(raw) {
  const inputValue = raw || document.getElementById("term-input")?.value.trim();
  if (!inputValue) {
    return;
  }

  appendPromptLine(inputValue);
  commandHistory.unshift(inputValue);
  historyIndex = -1;

  const input = document.getElementById("term-input");
  if (input) {
    input.value = "";
  }

  const { cmd, args } = parseCommand(inputValue);
  if (cmd === "clear") {
    clearTerminal();
    return;
  }

  const mapping = COMMAND_MAP[cmd];
  if (!mapping) {
    appendLine(`Unknown command: '${cmd}'. Type 'help' for commands.`, "term-error");
    return;
  }

  const runButton = document.getElementById("term-run-btn");
  if (runButton) {
    runButton.disabled = true;
  }
  appendLine("Running...", "term-muted");

  try {
    const data = await callCli(mapping.endpoint, { method: mapping.method, args });
    removeLastLine();
    renderCommandOutput(cmd, data);
  } catch (error) {
    removeLastLine();
    appendLine(`Connection error: ${error.message}`, "term-error");
  } finally {
    if (runButton) {
      runButton.disabled = false;
    }
    input?.focus();
  }
}

function renderCommandOutput(cmd, data) {
  if (cmd === "help") {
    appendLine("Available commands:", "term-ok");
    (data.commands || []).forEach((command) => {
      appendLine(`  ${command.name.padEnd(16)} ${command.description}`, "term-data");
    });
    return;
  }

  if (cmd === "status") {
    appendLine("System Status", "term-ok");
    Object.entries(data).forEach(([key, value]) => {
      appendLine(`  ${key.padEnd(20)} ${value}`, "term-data");
    });
    return;
  }

  if (cmd === "predict") {
    appendLine(`Generated ${data.count ?? 0} predictions.`, "term-ok");
    if (Array.isArray(data.predictions)) {
      data.predictions.slice(0, 5).forEach((prediction) => {
        appendLine(
          `  ${prediction.home_team} vs ${prediction.away_team}  ->  H:${pct(prediction.home_prob)} D:${pct(prediction.draw_prob)} A:${pct(prediction.away_prob)}`,
          "term-data",
        );
      });
      if (data.predictions.length > 5) {
        appendLine(`  ... and ${data.predictions.length - 5} more`, "term-muted");
      }
    }
    return;
  }

  if (cmd === "train") {
    (data.log || []).forEach((line) => {
      const trimmed = String(line).replace(/^\[(INFO|OK|ERR)\]\s*/i, "");
      const type = /^\[OK\]/i.test(line)
        ? "term-ok"
        : /^\[ERR\]/i.test(line)
          ? "term-error"
          : "term-info";
      appendLine(trimmed, type);
    });
    if (data.summary) {
      appendLine(data.summary, "term-ok");
    }
    return;
  }

  if (typeof data === "object" && data !== null) {
    Object.entries(data).forEach(([key, value]) => {
      appendLine(`  ${key}: ${JSON.stringify(value)}`, "term-data");
    });
    return;
  }

  appendLine(String(data), "term-ok");
}

function clearTerminal() {
  const output = document.getElementById("terminal-output");
  if (output) {
    output.innerHTML = '<div class="term-line term-muted">Terminal cleared.</div>';
  }
}

async function submitCommand() {
  const input = document.getElementById("term-input");
  const value = input?.value.trim();
  if (value) {
    await runCommand(value);
  }
}

async function checkConnection() {
  const dot = document.getElementById("term-status-dot");
  const text = document.getElementById("status-text");

  try {
    await fetchApiHealth();
    dot?.classList.remove("error");
    dot?.classList.add("connected");
    if (text) {
      text.textContent = "Connected to localhost:8000";
    }
  } catch (_error) {
    dot?.classList.remove("connected");
    dot?.classList.add("error");
    if (text) {
      text.textContent = "Cannot reach API";
    }
  }
}

function bindTerminalGlobals() {
  window.runCommand = runCommand;
  window.clearTerminal = clearTerminal;
  window.submitCommand = submitCommand;
}
