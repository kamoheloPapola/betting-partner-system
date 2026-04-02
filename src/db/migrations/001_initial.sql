CREATE TABLE IF NOT EXISTS resolved_predictions (
    prediction_id TEXT PRIMARY KEY,
    match_hash TEXT NOT NULL,
    league TEXT NOT NULL,
    kickoff_date TIMESTAMP NULL,
    market TEXT NULL,
    probability REAL NULL,
    outcome TEXT NULL,
    resolved_at TIMESTAMP NULL
);

CREATE INDEX IF NOT EXISTS idx_resolved_predictions_match_hash
ON resolved_predictions (match_hash);

CREATE INDEX IF NOT EXISTS idx_resolved_predictions_market
ON resolved_predictions (market);

CREATE INDEX IF NOT EXISTS idx_resolved_predictions_league_kickoff
ON resolved_predictions (league, kickoff_date);

CREATE INDEX IF NOT EXISTS idx_resolved_predictions_resolved_at
ON resolved_predictions (resolved_at);

CREATE TABLE IF NOT EXISTS model_manifest_entries (
    manifest_key TEXT PRIMARY KEY,
    model_name TEXT NULL,
    version TEXT NULL,
    league TEXT NULL,
    model_type TEXT NULL,
    target TEXT NULL,
    filename TEXT NULL,
    mode TEXT NULL,
    status TEXT NULL,
    sklearn_version TEXT NULL,
    train_size INTEGER NULL,
    test_size INTEGER NULL,
    registered_at TIMESTAMP NULL,
    features_json TEXT NULL,
    params_json TEXT NULL,
    metrics_json TEXT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_model_manifest_name_league
ON model_manifest_entries (model_name, league);

CREATE INDEX IF NOT EXISTS idx_model_manifest_status
ON model_manifest_entries (status);

CREATE INDEX IF NOT EXISTS idx_model_manifest_registered_at
ON model_manifest_entries (registered_at);

CREATE TABLE IF NOT EXISTS drift_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    league TEXT NULL,
    market TEXT NULL,
    severity TEXT NULL,
    metric TEXT NULL,
    value REAL NULL,
    threshold REAL NULL,
    detected_at TIMESTAMP NULL,
    payload_json TEXT NULL
);

CREATE INDEX IF NOT EXISTS idx_drift_events_detected_at
ON drift_events (detected_at);

CREATE INDEX IF NOT EXISTS idx_drift_events_type_league
ON drift_events (event_type, league);

CREATE INDEX IF NOT EXISTS idx_drift_events_severity
ON drift_events (severity);
