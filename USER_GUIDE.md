# Betting Partner System: User Guide

Welcome to the **Betting Partner System**, a development-stage, AI-driven football forecasting engine. It generates statistical match-outcome probabilities without using bookmaker prices.

---

## 🚀 1. Getting Started

### Prerequisites
- **Python 3.10+**
- **Git**
- **Virtual Environment** tool (venv or conda)

### Installation

1.  **Clone/Open the Repository**:
    Ensure you are in the project root: `c:\Users\papol\my_systems\betting_partner_system`

2.  **Set up Virtual Environment**:
    ```powershell
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```powershell
    pip install -r requirements.txt
    ```

4.  **Environment Configuration**:
    Create a `.env` file in the root directory. You can copy `.env.example` as a template.
    **Required Keys:**
    - `API_FOOTBALL_KEY` (for direct API-Sports/API-Football access)
    - `ODDS_API_KEY` (for The Odds API)
    - `FOOTBALL_DATA_API_KEY` (for football-data.org)

---

## 🔄 2. Daily Workflow

Follow this "Golden Path" every day to ensure predictions are fresh, safe, and accurate.

### Phase 1: Morning Data Refresh ☀️

**Step 0: Pre-flight Check**
Before running any commands, ensure your environment is healthy:
1. **Activate Virtual Environment**:
   ```powershell
   .\venv\Scripts\activate
   ```
2. **Verify `.env` Existence**:
   ```powershell
   Test-Path .env
   ```
   *If `False`, copy `.env.example` to `.env` and fill in your keys.*


Update the system with the latest results and upcoming fixtures.

1.  **Fetch Upcoming Fixtures** (for major leagues):
    ```powershell
    python src/commands/cli.py fetch-upcoming --league PL
    python src/commands/cli.py fetch-upcoming --league PD
    python src/commands/cli.py fetch-upcoming --league SA
    python src/commands/cli.py fetch-upcoming --league BL1
    ```
    *Leagues: PL (Premier League), PD (La Liga), SA (Serie A), BL1 (Bundesliga), FL1 (Ligue 1)*

2.  **Ingest Recent Results** (to update team form):
    ```powershell
    python src/commands/cli.py fetch-latest-season
    python src/commands/cli.py ingest-results
    ```
    > **Tip**: Regular ingestion keeps the models aligned with the current "state of play".


### Phase 2: Safety Checks 🛡️
Before generating any advice, verify the system's statistical integrity.

1.  **Refresh Drift Status**:
    ```powershell
    python src/commands/cli.py refresh-drift
    ```
    > **Note**: If this fails, the system detects "Drift" (abnormal unpredictability) and may block high-risk slips.

### Phase 3: Generate Predictions 🍎
Generate the "Forbidden Fruit" certified slip and archetypal accumulators.

1.  **Generate Slip & Accumulators**:
    ```powershell
    # For today's matches
    python src/commands/cli.py forbidden-fruit --date today

    # For the full weekend
    python src/commands/cli.py forbidden-fruit --date weekend
    ```
    *Output will display the single-pick slip followed by archetypal accumulators (SAFE, BALANCED, AGGRESSIVE).*

2.  **View Saved Accumulators**:
    ```powershell
    python src/commands/cli.py show-accumulators --date YYYYMMDD
    ```

### Phase 4: Exploration 🔭
View the entire universe of predictions, including those that didn't make the cut.

1.  **Show All Predictions**:
    ```powershell
    python src/commands/cli.py show-predictions --date today --all
    ```

### Phase 5: Closing Loop (Post-Match) 🏁
Verify how the system performed.

1.  **Reconcile Results**:
    ```powershell
    python src/commands/cli.py reconcile-results --date 2023-10-27
    ```
    *(Replace date with the relevant match day)*

2.  **Audit Accumulators**:
    ```powershell
    python src/commands/cli.py accumulator-audit
    ```
    *Analyzes historical accumulator outcomes and updates correlation penalties.*

---

## 📚 3. Core Concepts

### 🍎 Forbidden Fruit
The **Forbidden Fruit** is the system's "Flagship Product". It selects a localized, high-confidence slip (usually 4 legs) based on:
- **Structural Integrity**: Only teams with stable form.
- **Value**: High statistical probability independent of public odds.
- **Safety**: Automatically rejects markets if "Drift" is high.

### 🛡️ Drift Guard
The system monitors "concept drift" – when football reality changes (e.g., end-of-season weirdness).
- **Green**: Forecast generation is permitted by the current drift checks.
- **Red**: High variance detected. The system minimizes risk or halts.

### (@) Accumulator Engine (Meta-Layer)
A secondary intelligence layer that composes multi-leg slips from Forbidden Fruit candidates.
- **Strict Isolation**: Does NOT affect base model probabilities.
- **Diversity Enforcement**: Blocks same-match stacking and limits same-league/same-market concentrations.
- **Structural Learning**: Learns from joint outcomes to adjust "Correlation Penalties".

### 🧠 The Models
The system uses an ensemble of:
- **Poisson Distribution**: For goal expectancy.
- **XGBoost**: For complex pattern recognition.
- **Logistic Regression**: For binary outcomes.
*Note: The user does not need to manually retrain models daily; they adapt via the result ingestion pipeline.*

---

## 🛠️ 4. Advanced Commands Reference

| Command | Argument | Description |
| :--- | :--- | :--- |
| `fetch-history` | `--league [CODE] --season [YEAR]` | Download full historical season data. |
| `train` | `poisson` / `xgb` | Manually Retrain models (Advanced users only). |
| `backtest` | `--league [CODE] --test-season [YEAR]` | Run walk-forward validation to test model accuracy. |
| `enrich-data` | `--league [CODE]` | Add advanced stats (xG, shot maps) from FBref to existing data. |
| `show-accumulators` | `--date [YYYYMMDD]` | Display generated accumulators for a specific date. |
| `accumulator-audit` | | Perform structural learning on historical accumulator outcomes. |

---

## ❓ 5. Troubleshooting

**Issue: "Missing Odds" or API Errors**
- Check your internet connection.
- Verify API usage limits in your `.env` keys.
- **Encoding Check**: Ensure your `.env` is saved as **UTF-8 without BOM**.
- Run `fetch-upcoming` again.


**Issue: "Drift Verification Failed"**
- The system is protecting you. This happens when recent results have been highly unpredictable.
- **Action**: Treat forecasts as unavailable until the drift condition has been reviewed. Do not force predictions.

**Issue: System returns empty slip**
- No matches met the strict >= 60% probability criteria or the "Value" threshold.
- **Action**: Check `show-predictions --all` to see the "near misses".

---

*Verified by Antigravity System | v3.0 | 2025*
