from pathlib import Path
import json
import pytest
from src.data_foundation.audit import project_metadata,load_metadata,audit_group,inventory,MetaFixture,SensitiveMetadata,run_audit,PacketLimit
from src.data_foundation.identity import read_aliases

def test_sealed_poison_score_bytes_never_decode(tmp_path):
    path=tmp_path/"PL_2024.csv"
    path.write_bytes(b"date,home_team,away_team,home_score,away_score\n2024-07-01,A,B,\xff,\xfe\n")
    rows,errors,pin=load_metadata(path,"canonical","PL")
    assert not errors and len(rows)==1 and rows[0].home=="A"
    assert pin["size_bytes"]==path.stat().st_size

def test_quoted_names_comma_and_exact_case_preserved(tmp_path):
    path=tmp_path/"PL_2024.csv"
    path.write_bytes(b'date,home_team,away_team,home_score\n2024-07-01,"A, Club","B ""FC""",999\n')
    rows,errors,_=load_metadata(path,"canonical","PL")
    assert not errors and rows[0].home=="A, Club" and rows[0].away=='B "FC"'

def test_historical_names_dates_only_and_file_season_label(tmp_path):
    path=tmp_path/"Premier_League_Season_2024_2025.csv"
    path.write_bytes(b"Date,HomeTeam,AwayTeam,FTHG\n01/07/2024,Raw A,Raw B,\xff\n")
    rows,errors,_=load_metadata(path,"historical","PL")
    assert not errors and rows[0].day=="2024-07-01" and rows[0].season_label=="2024_2025"

def test_metadata_invalid_date_error_without_raw_values(tmp_path):
    path=tmp_path/"PL_2024.csv"
    path.write_bytes(b"date,home_team,away_team\nnot-a-date,A,B\n")
    rows,errors,_=load_metadata(path,"canonical","PL")
    assert not rows and errors[0]["type"]=="INVALID_METADATA_ROW"
    assert "not-a-date" not in str(errors)

def test_metadata_sensitive_value_stops_without_output(tmp_path):
    path=tmp_path/"PL_2024.csv"
    path.write_bytes(b"date,home_team,away_team\n2024-07-01,token=synthetic,A\n")
    with pytest.raises(SensitiveMetadata):load_metadata(path,"canonical","PL")

def test_duplicate_self_same_day_and_ordered_pair_counts():
    def f(line,day,home,away):
        return MetaFixture("PL","canonical","2024","synthetic",line,day,home,away)
    rows=[f(2,"2024-08-01","A","B"),f(3,"2024-08-01","A","B"),
          f(4,"2024-08-01","A","C"),f(5,"2024-08-02","A","A"),
          f(6,"2024-08-03","A","B")]
    audit=audit_group("PL","2024",rows)
    assert len(audit["self_fixtures"])==1
    assert len(audit["duplicate_metadata_fixtures"])==1
    assert audit["duplicate_metadata_fixtures"][0]["count"]==2
    assert len(audit["multiple_distinct_fixtures_per_team_date"])==1
    assert len(audit["repeated_ordered_pairs_after_metadata_dedupe"])==1
    name=[x for x in inventory(rows) if x["source_name"]=="A"][0]
    assert name["raw_match_rows"]==5  # self fixture counts as one match, not two.

def test_alias_csv_schema(tmp_path):
    path=tmp_path/"alias.csv"
    path.write_text("league,source_name,provider_team_id,note\nPL,Old Club,1,reviewed synthetic\n")
    assert read_aliases(path)[0]["provider_team_id"]==1
    path.write_text("source_name,league,provider_team_id,note\nOld Club,PL,1,note\n")
    with pytest.raises(ValueError):read_aliases(path)

def test_outputs_are_packet_only_and_exclusive(tmp_path):
    root=tmp_path/"repo";(root/"data/processed/matches").mkdir(parents=True)
    (root/"data/processed/matches/PL_2024.csv").write_bytes(b"date,home_team,away_team\n2024-07-01,A,B\n")
    packet=tmp_path/"packet";packet.mkdir()
    result=run_audit(root,packet/"audit",packet)
    assert result["sealed_scores_decoded"]==0
    with pytest.raises(FileExistsError):run_audit(root,packet/"audit",packet)
    with pytest.raises(Exception):run_audit(root,tmp_path/"escape",packet)
