
# API Reference

The system exposes its functionality via a Command Line Interface (CLI). All commands are accessed via `python -m src.cli [COMMAND]`.

## Primary Commands

### `show-predictions`
Generates predictions for upcoming matches.

**Usage:**
```bash
python -m src.cli show-predictions [OPTIONS]
```

**Options:**
- `--date [today|tomorrow|weekend|YYYY-MM-DD]`: Filter matches by date (default: 'today').
- `--league [PL|PD|SA|BL1|FL1]`: Filter by specific league code.
- `--all / -a`: Show all future predictions (overrides date filter).

**Example:**
`python -m src.cli show-predictions --date weekend --league PL`

---

### `train`
Retrains machine learning models.

**Usage:**
```bash
python -m src.cli train [OPTIONS]
```

**Options:**
- `--league [CODE]`: Target league (default: None = Global + Big 5).
- `--model-type [poisson_home|nb_home_corners|...]`: Specific model architecture (default: all).
- `--mode [production|debug]`: Save to registry (production) or dry-run (debug).
- `--force / -f`: Force retraining even if recent model exists.

**Example:**
`python -m src.cli train --league PL --mode production`

---

### `fetch-upcoming`
Retrieves live fixture data from API-Football.

**Usage:**
```bash
python -m src.cli fetch-upcoming [OPTIONS]
```

**Options:**
- `--league [CODE]`: Target league code.
- `--days [INT]`: (Deprecated) Days window lookahead.

---

### `fetch-latest-season`
Downloads historical results CSVs (Football-Data.co.uk).

**Usage:**
```bash
python -m src.cli fetch-latest-season [OPTIONS]
```

---

### `ingest-results`
Processes raw results into the master feature matrix.

**Usage:**
```bash
python -m src.cli ingest-results
```

---

## Maintenance Commands

### `refresh-drift`
Updates rolling drift metrics to detect market regime changes.

**Usage:**
```bash
python -m src.cli refresh-drift
```

### `audit-coverage`
Checks model registry for missing models, coverage gaps, or silent fallbacks.

**Usage:**
```bash
python -m src.cli audit-coverage [--save]
```
