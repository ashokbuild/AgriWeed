"""Tests for core yield and detection helper logic in app.py."""

import numpy as np
import pytest
from app import calculate_yield, annotate_image, ACRE_TO_HECTARE


def test_calculate_yield_hectares():
    """Verify calculation in hectares: 10 ha * 5 t/ha = 50 t baseline."""
    res = calculate_yield(
        field_area=10.0,
        area_unit="hectare",
        baseline_yield_t_ha=5.0,
        loss_pct=20.0,
        control_pct=80.0,
    )
    assert res["area_ha"] == 10.0
    assert res["baseline_prod"] == 50.0
    assert res["uncontrolled_prod"] == 40.0
    assert res["controlled_prod"] == 48.0
    assert res["potential_loss"] == 10.0
    assert res["potential_gain"] == 8.0


def test_calculate_yield_acres():
    """Verify acre conversion: 100 acres * 0.40468564224 ha/acre."""
    res = calculate_yield(
        field_area=100.0,
        area_unit="acre",
        baseline_yield_t_ha=10.0,
        loss_pct=25.0,
        control_pct=85.0,
    )
    expected_ha = 100.0 * ACRE_TO_HECTARE
    assert res["area_ha"] == pytest.approx(expected_ha, rel=1e-3)
    expected_baseline = expected_ha * 10.0
    assert res["baseline_prod"] == pytest.approx(expected_baseline, rel=1e-2)
    assert res["potential_loss"] == pytest.approx(expected_baseline * 0.25, rel=1e-2)
    assert res["potential_gain"] == pytest.approx(expected_baseline * 0.25 * 0.85, rel=1e-2)


def test_annotate_image_runs():
    """Verify annotate_image draws boxes without crashing."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    dets = [
        {"box": [10, 10, 50, 50], "conf": 0.9, "class_name": "Weed", "is_weed": True},
        {"box": [60, 60, 100, 100], "conf": 0.85, "class_name": "Maize", "is_weed": False},
    ]
    ann = annotate_image(img, dets)
    assert ann.shape == img.shape
    assert ann.dtype == np.uint8
