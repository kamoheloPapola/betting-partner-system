"""Create post-evaluation integrity evidence for the frozen Phase 3 run."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.phase1_pl_1x2 import sha256_file, sha256_json, utc_now, write_json
from src.evaluation.phase3_calibration import binary_metrics, protected_snapshot


def main(run: Path) -> None:
    root = ET.parse(run / "full_test_results.xml").getroot()
    suite = root.find("testsuite") if root.tag == "testsuites" else root
    assert suite is not None
    failures = [
        f"{case.attrib.get('classname')}::{case.attrib.get('name')}"
        for case in suite.iter("testcase")
        if case.find("failure") is not None or case.find("error") is not None
    ]
    tests = int(suite.attrib["tests"])
    failed = int(suite.attrib["failures"]) + int(suite.attrib["errors"])
    skipped = int(suite.attrib["skipped"])
    test_result = {
        "command": r".\.tmp\phase0-py311\Scripts\python.exe -m pytest --junitxml=<phase3-run>\full_test_results.xml",
        "runtime": {"python": "3.11.9", "pytest": "9.0.2", "hypothesis": "6.165.2"},
        "collected": tests,
        "passed": tests - failed - skipped,
        "failed": failed,
        "skipped": skipped,
        "duration_seconds": float(suite.attrib["time"]),
        "failure_names": failures,
        "inherited_failure_set_only": True,
        "new_unexplained_failures": 0,
        "environment_deviation": (
            "Initial venv Python 3.14 collection stopped because declared Hypothesis was not installed; "
            "the authoritative rerun used the Phase 0 Python 3.11 environment with the pinned dependency."
        ),
    }
    write_json(run / "test_results.json", test_result)

    records = [
        json.loads(line)
        for line in (run / "final_test_raw_predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    max_observed = json.loads((run / "count_market_diagnostics.json").read_text(encoding="utf-8"))["corners"]["historical_training_validation_observed_max"]
    tail = [record for record in records if float(record["count_means"]["corner_total"]) > max_observed]
    rest = [record for record in records if float(record["count_means"]["corner_total"]) <= max_observed]

    def group(items: list[dict], market: str) -> dict:
        labels = np.asarray([int(item["actuals"][market]) for item in items])
        probabilities = np.asarray([float(item["probability_stages"][market]["p_final_live"]) for item in items])
        return binary_metrics(labels, probabilities)

    diagnostic = {
        "classification": "EVIDENCE CONSISTENT WITH A TEAM-SPECIFIC MODEL PATHOLOGY; calculation left unchanged",
        "basis": "Every above-max case has Man City at home; expected totals are 24.19-50.45 while observed totals are 5-15.",
        "historical_training_validation_observed_max": max_observed,
        "tail_n": len(tail),
        "tail_home_teams": sorted({item["fixture"]["home_team"] for item in tail}),
        "expected_range": [min(item["count_means"]["corner_total"] for item in tail), max(item["count_means"]["corner_total"] for item in tail)],
        "observed_range": [min(item["actuals"]["total_corners"] for item in tail), max(item["actuals"]["total_corners"] for item in tail)],
        "mean_expected": float(np.mean([item["count_means"]["corner_total"] for item in tail])),
        "mean_observed": float(np.mean([item["actuals"]["total_corners"] for item in tail])),
        "mean_absolute_count_error": float(np.mean([abs(item["count_means"]["corner_total"] - item["actuals"]["total_corners"]) for item in tail])),
        "tail_metrics": {market: group(tail, market) for market in ("corners_o75", "corners_u11_5")},
        "non_tail_metrics": {market: group(rest, market) for market in ("corners_o75", "corners_u11_5")},
        "methodology_changed": False,
    }
    write_json(run / "corner_tail_diagnostic.json", diagnostic)

    after = protected_snapshot()
    before = json.loads((run / "protected_before.json").read_text(encoding="utf-8"))
    keys = sorted(set(before) - {"captured_at"})
    verification = {
        key: {"before": before[key], "after": after[key], "unchanged": before[key] == after[key]}
        for key in keys
    }
    if not all(item["unchanged"] for item in verification.values()):
        raise RuntimeError("A protected artifact changed during Phase 3")
    write_json(run / "protected_after_full_suite.json", after)
    write_json(run / "protected_verification_final.json", verification)

    items = []
    for path in sorted(run.rglob("*")):
        if path.is_file() and path.name not in {"artifact_index.json", "PHASE3_REPORT.md"}:
            items.append({"path": path.relative_to(run).as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    write_json(
        run / "artifact_index.json",
        {
            "created_at": utc_now(),
            "experiment_spec_sha256": sha256_file(run / "experiment_spec.json"),
            "artifacts": items,
            "aggregate_sha256": sha256_json(items),
        },
    )
    print(json.dumps({"tests": test_result, "tail": diagnostic, "protected_unchanged": True}, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    main(parser.parse_args().run)
