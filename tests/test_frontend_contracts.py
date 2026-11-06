import re
from pathlib import Path
import pytest


@pytest.fixture
def index_html_content():
    path = Path(__file__).resolve().parent.parent / "index.html"
    return path.read_text(encoding="utf-8")


@pytest.fixture
def script_js_content():
    path = Path(__file__).resolve().parent.parent / "script.js"
    return path.read_text(encoding="utf-8")


def test_index_html_has_no_stray_characters(index_html_content):
    """Verify index.html has no stray characters after </head>."""
    match = re.search(r"</head>(\s*)<body>", index_html_content)
    assert match is not None, "Stray character detected after </head>"
    # Ensure no alphanumeric character exists between </head> and <body>
    between = match.group(1)
    assert between.strip() == "", f"Unexpected characters between head and body: {repr(between)}"


def test_index_html_contains_error_label_element(index_html_content):
    """Verify index.html includes the missing error-label element referenced by script.js."""
    assert 'id="error-label"' in index_html_content
    assert 'class="error-label"' in index_html_content
    # Ensure error-label appears inside state-error
    assert '<p class="error-label" id="error-label"></p>' in index_html_content


def test_index_html_age_field_bounds(index_html_content):
    """Verify age input in index.html is bounded to 18–24."""
    assert 'min="18"' in index_html_content
    assert 'max="24"' in index_html_content
    assert '18–24 (Initial deployment)' in index_html_content


def test_script_js_dynamic_api_base(script_js_content):
    """Verify script.js has dynamic and configurable API base resolution."""
    assert "window.MINDSCOPE_API_BASE" in script_js_content
    assert "isLocal" in script_js_content
    assert "8000" in script_js_content


def test_script_js_defensive_error_rendering(script_js_content):
    """Verify script.js checks for element existence before setting textContent."""
    assert "if (errorLabelEl) errorLabelEl.textContent = label;" in script_js_content
    assert "if (errorCopyEl) errorCopyEl.textContent = copy;" in script_js_content


def test_script_js_duplicate_submission_prevention(script_js_content):
    """Verify script.js implements concurrency lock to prevent duplicate submissions."""
    assert "let isSubmitting = false;" in script_js_content
    assert "if (isSubmitting) return;" in script_js_content
    assert "setSubmitting(true);" in script_js_content
    assert "setSubmitting(false);" in script_js_content


def test_script_js_non_json_error_handling(script_js_content):
    """Verify script.js safely checks content-type and handles non-JSON responses."""
    assert "res.headers.get(\"content-type\")" in script_js_content
    assert "contentType.includes(\"application/json\")" in script_js_content


def test_script_js_numerical_bounds_alignment(script_js_content):
    """Verify script.js client-side validation enforces 18-24 age and bounded inputs."""
    assert '["age", 18, 24]' in script_js_content
    assert '["avg_daily_usage_hours", 0, 16]' in script_js_content
    assert '["daily_unlocks", 0, 500]' in script_js_content
    assert '["study_hours", 0, 16]' in script_js_content
    assert '["physical_activity_hours", 0, 8]' in script_js_content
    assert '["sleep_hours_per_night", 2, 16]' in script_js_content


def test_field_id_alignment_between_html_and_js(index_html_content, script_js_content):
    """Verify every field ID validated in script.js exists in index.html."""
    required_ids = [
        "predict-form", "submit-btn", "reset-btn", "error-retry-btn",
        "state-idle", "state-loading", "state-result", "state-error",
        "score-number", "score-band", "score-context", "gauge-fill",
        "error-label", "error-copy", "age", "gender", "country",
        "academic_level", "most_used_platform", "purpose_of_use",
        "avg_daily_usage_hours", "daily_unlocks", "study_hours",
        "physical_activity_hours", "sleep_hours_per_night",
        "stress_level", "stress_level_group"
    ]
    for element_id in required_ids:
        assert f'id="{element_id}"' in index_html_content, f"Missing element with id='{element_id}' in index.html"

