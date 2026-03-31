
from datetime import datetime
from src.monitoring.events import log_event, PredictionEvent, get_prediction_metrics

def verify():
    # 1. Log generic event
    print("Logging test event...")
    timestamp = datetime.now()
    evt = PredictionEvent(
        timestamp=timestamp,
        event_type="prediction_generated",
        league="TEST_LG",
        match_id="test_match_001",
        model_version="v1.0-test",
        confidence=0.85,
        metadata={"test": True}
    )
    log_event(evt)
    
    # 2. Fetch metrics
    date_str = timestamp.strftime('%Y-%m-%d')
    print(f"Fetching metrics for {date_str}...")
    metrics = get_prediction_metrics(date_str)
    
    print("Metrics:", metrics)
    
    # 3. Assert
    assert metrics['total_predictions'] >= 1, "Should have at least 1 prediction event"
    assert metrics['avg_confidence'] > 0, "Confidence should be positive"
    print("Verification PASSED")

if __name__ == "__main__":
    verify()
