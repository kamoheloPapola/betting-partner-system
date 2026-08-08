from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/slips/forbidden-fruit",
        "/api/v1/slips/PL",
    ],
)
def test_removed_slip_routes_return_404(path):
    response = client.get(path)

    assert response.status_code == 404


def test_served_frontend_has_no_slip_builder_surface():
    index_html = (PROJECT_ROOT / "src" / "static" / "index.html").read_text(encoding="utf-8")
    app_js = (PROJECT_ROOT / "src" / "static" / "js" / "app.js").read_text(encoding="utf-8")
    api_js = (PROJECT_ROOT / "src" / "static" / "js" / "api.js").read_text(encoding="utf-8")
    predictions_js = (
        PROJECT_ROOT / "src" / "static" / "js" / "predictions.js"
    ).read_text(encoding="utf-8")
    slip_builder_js = PROJECT_ROOT / "src" / "static" / "js" / "slipBuilder.js"

    assert "Slip Builder" not in index_html
    assert 'data-page="slip-builder"' not in index_html
    assert "slipBuilder.js" not in app_js
    assert '"slip-builder"' not in app_js
    assert "manualSlip" not in app_js
    assert "/api/v1/slips" not in api_js
    assert "Add to Slip" not in predictions_js
    assert "addToSlip" not in predictions_js
    assert "manualSlip" not in predictions_js
    assert not slip_builder_js.exists()
