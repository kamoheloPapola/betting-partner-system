# DRIFT POLICY (IMMUTABLE)

## 1. Purpose
The goal of the Drift Check is **Early Warning**, not retraining. It serves as a safety gate to answer: *"Is the model behaving today like it did historically?"* If the answer is "no", the system must pause to prevent blind losses.

## 2. Mandatory Execution
- Rolling day drift checks are **MANDATORY**.
- Drift checks must run on every prediction session before any selections are made.
- Drift checks may not be disabled, skipped, or overridden.

## 3. Enforcement
- Status = **STOP** blocks all decisions.
- The pipeline must terminate before reaching the Forbidden Fruit selection layer if drift status is "STOP".
- All drift states must be persisted to `data/drift/rolling_90d_status.json` for historical traceability.

## 4. Monitored Metrics (The Four Pillars)
Only these metrics are considered for the drift status:
- **A. High-Confidence Hit Rate Drift (PRIMARY)**: Strong predictions must win at expected rates.
- **B. Calibration Drift (ECE)**: Probabilities must remain an honest reflection of outcome frequency.
- **C. Confidence Distribution Drift**: Detect model becoming timid (deflation) or reckless (inflation).
- **D. Selection Rate Drift**: Ensure selectivity is maintained (no leakage).

## 5. Threshold Adjustments
Drift thresholds are defensive and may only be adjusted after:
- Extensive historical backtesting.
- Documented technical justification.

**STATUS: ACTIVE & ENFORCED**
