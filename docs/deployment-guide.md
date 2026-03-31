
# Deployment Guide

## Prerequisites
- Python 3.10+
- Pip
- Valid API Keys (odds-api, football-data, etc.) set in environment variables.

## Installation

1. **Clone Repository**
   ```bash
   git clone <repo-url>
   cd betting_partner_system
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Installation**
   Run the test suite:
   ```bash
   pytest tests/
   ```

## Production Workflow

### 1. Initialize Data
Fetch the latest historical data for all supported leagues:
```bash
python -m src.cli fetch-latest-season --league PL
python -m src.cli fetch-latest-season --league SA
# ... repeat for others
python -m src.cli ingest-results
```

### 2. Train Base Models
Train the foundational models. This usually takes 5-10 minutes.
```bash
python -m src.cli train --mode production
```

### 3. Verify System Health
Check for drift and coverage issues before going live:
```bash
python -m src.cli refresh-drift
python -m src.cli audit-coverage
```

### 4. Schedule Predictions
Set up a cron job or scheduled task to run daily at 08:00 AM:
```bash
0 8 * * * cd /path/to/app && python -m src.cli show-predictions --date today >> logs/daily.log 2>&1
```

## Environment Variables
Create a `.env` file in the root:
```ini
API_KEY_FOOTBALL=your_api_key
LOG_LEVEL=INFO
```
