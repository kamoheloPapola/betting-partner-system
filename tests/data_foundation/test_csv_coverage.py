import pytest
from src.data_foundation.csv_coverage import require_disjoint_seasons, SourceCoverageError


def source(tmp_path, name, dates):
    p = tmp_path/name
    p.write_bytes(b"Date,HomeTeam,AwayTeam,FTHG\n"+b"".join(d.encode()+b",A,B,\xff\n" for d in dates))
    return p


def test_two_files_covering_same_season_refused_even_when_bytes_identical(tmp_path):
    a=source(tmp_path,"A.csv",["01/08/2024"]);b=source(tmp_path,"B.csv",["01/08/2024"])
    with pytest.raises(SourceCoverageError,match="Two files"):
        require_disjoint_seasons([a,b],"PL")


def test_nonidentical_same_season_also_refused(tmp_path):
    a=source(tmp_path,"A.csv",["01/08/2024"]);b=source(tmp_path,"B.csv",["02/08/2024"])
    with pytest.raises(SourceCoverageError):
        require_disjoint_seasons([a,b],"PL")


def test_distinct_seasons_and_cutoff_are_allowed_without_score_decode(tmp_path):
    a=source(tmp_path,"A.csv",["31/05/2024"]);b=source(tmp_path,"B.csv",["01/06/2024"])
    result=require_disjoint_seasons([a,b],"PL")
    assert result[str(a)]==[2023] and result[str(b)]==[2024]


@pytest.mark.parametrize("dates",[[],["not-a-date"]])
def test_unknown_coverage_refused(tmp_path,dates):
    with pytest.raises(SourceCoverageError):
        require_disjoint_seasons([source(tmp_path,"X.csv",dates)],"PL")

