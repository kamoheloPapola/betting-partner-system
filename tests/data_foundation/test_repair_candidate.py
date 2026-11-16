import hashlib
import pytest
from src.data_foundation.repair import deduplicate_identical, relabel_2025, fields, RepairError


def test_exact_byte_dedup_keeps_opaque_outcomes():
    row=b"2024-08-01,A,B,\xff,\xfe\n"
    out,proof=deduplicate_identical(b"date,home,away,score,odds\n"+row+row)
    assert out.endswith(row) and proof["removed_identical_rows"]==1
    assert not proof["outcome_cells_decoded"]


def test_different_bytes_not_silently_collapsed():
    out,proof=deduplicate_identical(b"h\nA,1\nA,2\n")
    assert proof["removed_identical_rows"]==0 and out==b"h\nA,1\nA,2\n"


def sample():
    raw=b"date,home_team,away_team,home_score,away_score\n2026-01-01,BAD,BAD,\xff,\xfe\n"
    plan={"line":2,"status":"MATCHED","fixture_id":7,"home_id":1,"away_id":2,
          "source_sha256":hashlib.sha256(raw).hexdigest()}
    provider={7:{"home_id":1,"away_id":2,"home_meta":{"name":"Correct, Home"},"away_meta":{"name":"Away"}}}
    return raw,plan,provider


def test_relabel_only_identity_tokens_and_append_provider_ids():
    raw,plan,provider=sample()
    out,excluded,proof=relabel_2025(raw,[plan],provider)
    assert fields(out.splitlines()[1])[3:5]==[b"\xff",b"\xfe"]
    assert b'"Correct, Home"' in out and out.endswith(b",1,2,7\n")
    assert proof["nonidentity_tokens_equal"] and proof["excluded_rows"]==0


def test_exclusion_retains_original_opaque_bytes():
    raw,plan,provider=sample();plan["status"]="EXCLUDED"
    out,excluded,proof=relabel_2025(raw,[plan],provider)
    assert excluded==raw and len(out.splitlines())==1 and proof["excluded_rows"]==1


def test_plan_must_pin_actual_source():
    raw,plan,provider=sample();plan["source_sha256"]="wrong"
    with pytest.raises(RepairError,match="identity mismatch"):relabel_2025(raw,[plan],provider)


def test_missing_plan_cannot_silently_drop_a_row():
    raw,_,provider=sample()
    with pytest.raises(RepairError,match="cover every"):relabel_2025(raw,[],provider)

