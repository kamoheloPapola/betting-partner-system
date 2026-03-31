
# System Architecture

## Overview
The Betting Partner System is a CLI-driven machine learning application designed to predict football match outcomes using statistical distributions (Poisson, Negative Binomial, Zero-Inflated Poisson). It integrates historical data, trains league-specific models, and provides actionable betting insights through a strict quality gating process.

## Data Flow Pipeline

### 1. Data Ingestion (`src.data`)
- **Sources**: Football-Data.co.uk (Primary CSVs), API-Football (Live Fixtures), FBref (Advanced Stats).
- **Process**: 
  1. `fetch_data`: Downloads raw CSVs to `data/raw/{league}_{season}.csv`.
  2. `pipeline.run`: Normalizes raw data into a standard schema.
  3. **Feature Engineering**: Calculates rolling averages (goals/corners/cards), form ratings, and rest days.
  4. **Persistence**: Saves processed features to `data/processed/master_features_{DATE}.csv`.

### 2. Model Training (`src.cli.commands.training`)
- **Strategy**: Dual-Layer modeling.
  - **Global Models**: Trained on all leagues combined (Fallback).
  - **League Models**: Specialized models for specific leagues (e.g., PL, SA).
- **Engines**:
  - `PoissonEngine`: Goals (Home/Away).
  - `NegativeBinomialEngine`: Corners (Over-dispersed data).
  - `ZeroInflatedEngine`: Cards (Sparse data).
- **Persistence**: Models serialized to `models/` with `manifest.json` tracking metadata/metrics.

### 3. Prediction (`src.cli.commands.prediction`)
- **Inference**:
  - Loads Best Available Model (Local > Global if Local Score < Global Score).
  - Computes probability matrices for Goals, Corners, Cards.
- **Gates**:
  - **Selection Gate**: Filters high-confidence bets based on value/edge.
  - **Forbidden Fruit**: Identifies specific high-tier accumulator candidates.
  - **Drift Guardrail**: Blocks predictions if market regime shift detected.

### 4. Presentation
- **CLI**: Rich-formatted tables with color-coded confidence levels.
- **Monitoring**: JSONL event logging to `data/events/` for observability.

## Key Components

| Component | Responsibility |
|-----------|----------------|
| `ServiceContainer` | Dependency Injection (Singleton) for Registry/Pipeline. |
| `ModelRegistry` | Manages model versions, metadata, and fallback logic. |
| `FeaturePipeline` | One-way data flow: Raw -> Features -> Model Input. |
| `SelectionGate` | Enforces minimum probability thresholds (e.g. 1X2 > 45%). |

## Directory Structure
```
src/
  cli/          # Command-line interface logic (Typer)
  core/         # Shared utilities, exceptions, DI container
  data/         # Data fetching and validation logic
  features/     # Feature engineering pipeline
  ml/           # Probabilistic engines and model registry
  strategies/   # Betting strategies (Forbidden Fruit, Gates)
  monitoring/   # Event logging and metrics
  config/       # Configuration constants
```
