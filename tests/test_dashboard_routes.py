from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
LATEST_MANIFEST = ROOT / "reports/corrected_v2/latest.json"
DASHBOARD_ROUTES = (
    "Live Predictions",
    "Historical Trends",
    "Congestion Heatmap",
    "Road Comparison",
    "Model Performance",
    "Feature Importance",
    "Forecast Visualization",
    "Prediction Confidence",
    "Weather vs Traffic",
    "Data Validation",
    "Model Training",
)


def test_dashboard_routes_render_against_latest_corrected_run():
    if not LATEST_MANIFEST.is_file():
        pytest.skip("Run the corrected pipeline before dashboard route tests.")

    app = AppTest.from_file(
        str(ROOT / "dashboard/app.py"),
        default_timeout=120,
    )
    app.run(timeout=120)
    assert not app.exception, [
        f"Live Predictions: {exception.message}" for exception in app.exception
    ]

    for route in DASHBOARD_ROUTES[1:]:
        app.sidebar.radio[0].set_value(route)
        app.run(timeout=120)
        assert not app.exception, [
            f"{route}: {exception.message}" for exception in app.exception
        ]
