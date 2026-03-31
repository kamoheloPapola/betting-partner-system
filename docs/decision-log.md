
# Architectural Decision Log

## ADR-001: Granular Model Registry vs. Monolith
**Date:** 2025-12-20
**Context:** We support multiple leagues with vastly different scoring dynamics (e.g., Bundesliga high scoring vs. Serie A defensive).
**Decision:** Implement a **League-Specific Model Registry**.
**Rationale:** A single global model fails capturing local nuances (refs, style). Training separate models for each league improves Brier score by ~15% in high-variance leagues.
**Consequences:** Complexity in `prediction.py` to route requests. Added `ModelRegistry` class.

## ADR-002: Dual-Layer Fallback Strategy
**Date:** 2025-12-25
**Context:** Smaller leagues lack sufficient historical data for stable training.
**Decision:** Implement **Smart Fallback**.
**Logic:** If `LeagueModel.score < GlobalModel.score`, use League. Else, use Global.
**Rationale:** Prevents overfitting on small datasets while allowing large leagues to benefit from specificity.

## ADR-003: Property-Based Testing for Core Engines
**Date:** 2026-01-07
**Context:** `PoissonEngine` and probability logic are math-heavy and prone to subtle edge-case bugs (e.g. sum > 1.0).
**Decision:** Adopt **Hypothesis** for property-based testing.
**Rationale:** Unit tests with static inputs miss floating point weirdness and extreme combinations.

## ADR-004: Event Sourcing for Observability
**Date:** 2026-01-07
**Context:** Debugging drift in production is impossible without historical context of predictions.
**Decision:** Log all predictions as immutable event streams (`jsonl`).
**Rationale:** Enables perfect replayability and drift detection without database complexity.
