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
    python -m src.cli fetch-upcoming --league PL
    python -m src.cli fetch-upcoming --league PD
    python -m src.cli fetch-upcoming --league SA
    python -m src.cli fetch-upcoming --league BL1
    ```
    *Leagues: PL (Premier League), PD (La Liga), SA (Serie A), BL1 (Bundesliga), FL1 (Ligue 1)*

2.  **Ingest Recent Results** (to update team form):
    ```powershell
    python -m src.cli fetch-latest-season
    python -m src.cli ingest-results
    ```
    > **Tip**: Regular ingestion keeps the models aligned with the current "state of play".


### Phase 2: Safety Checks 🛡️
Before generating forecasts, verify the system's statistical integrity.

1.  **Refresh Drift Status**:
    ```powershell
    python -m src.cli refresh-drift
    ```
    > **Note**: If this fails, the system has detected drift and may make forecasts unavailable pending review.

### Phase 3: Generate Match Probabilities
Generate match-probability forecasts for the requested date range.

1.  **Generate Forecasts**:
    ```powershell
    # For today's matches
    python -m src.cli show-predictions --date today

    # For the full weekend
    python -m src.cli show-predictions --date weekend
    ```
    *Output displays model probabilities and forecast context for the available fixtures.*

2.  **Review Model Health**:
    ```powershell
    python -m src.cli audit-coverage
    ```

### Phase 4: Exploration 🔭
View the entire universe of predictions, including those that didn't make the cut.

1.  **Show All Predictions**:
    ```powershell
    python -m src.cli show-predictions --all
    ```

### Post-Match Evaluation 🏁
Verify how the system performed.

1.  **Reconcile Results**:
    ```powershell
    python -m src.cli resolve-predictions
    ```
    *Maps pending forecasts to finalized match results.*

2.  **Check Drift by League**:
    ```powershell
    python -m src.cli check-drift --league PL
    ```
    *Reports recent calibration and forecast-health signals.*

---

## 📚 3. Core Concepts

### Match-Probability Forecasts
The system's supported product surface reports statistical probabilities for football match outcomes:
- **Inputs**: Team form, historical match data, and engineered features.
- **Outputs**: Model probabilities and supporting forecast context.
- **Availability**: Drift and model-state checks can make forecasts unavailable.

### 🛡️ Drift Guard
The system monitors "concept drift" – when football reality changes (e.g., end-of-season weirdness).
- **Green**: Forecast generation is permitted by the current drift checks.
- **Red**: High variance detected. Forecast generation is unavailable pending review.

### Model Health
The monitoring surface reports whether model artifacts and recent forecast behavior are healthy.
- **Coverage Audit**: Verifies that required models are available for each league.
- **Drift Status**: Reports calibration and distribution changes from resolved forecasts.
- **Model State**: Distinguishes development and locked inference states.

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
| `fetch-data` | `--league [CODE] --season [YEAR]` | Download historical season data. |
| `train` | `poisson` / `nb` | Manually retrain models (advanced users only). |
| `backtest` | `--league [CODE] --test-season [YEAR]` | Run walk-forward validation to test model accuracy. |
| `show-predictions` | `--league [CODE] --date [FILTER]` | Display match-probability forecasts. |
| `check-drift` | `--league [CODE]` | Report recent calibration and drift status. |
| `audit-coverage` | `--save` | Verify model availability and health by league. |

---

## ❓ 5. Troubleshooting

**Issue: Missing Fixture Data or API Errors**
- Check your internet connection.
- Verify API usage limits in your `.env` keys.
- **Encoding Check**: Ensure your `.env` is saved as **UTF-8 without BOM**.
- Run `fetch-upcoming` again.


**Issue: "Drift Verification Failed"**
- The system is protecting you. This happens when recent results have been highly unpredictable.
- **Action**: Treat forecasts as unavailable until the drift condition has been reviewed. Do not force predictions.

**Issue: No forecasts are displayed**
- No fixtures were available, or model-state and drift checks made forecasts unavailable.
- **Action**: Check `show-predictions --all`, then review `check-drift` and `audit-coverage`.

---

*Verified by Antigravity System | v3.0 | 2025*
