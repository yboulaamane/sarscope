import math

import pytest

from sarscope.units import to_pactivity


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [(100.0, "nM", 7.0), (1.0, "uM", 6.0), (1.0, "μM", 6.0), (1e-8, "M", 8.0), (1.0, "pM", 12.0)],
)
def test_conversion(value, unit, expected):
    assert to_pactivity(value, unit) == pytest.approx(expected)


def test_matches_chembl_pchembl_for_a_real_value():
    # BRAF record: standard_value "16000.0" nM would have pchembl 4.80.
    assert round(to_pactivity(16000.0, "nM"), 2) == 4.80


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan])
def test_non_positive_is_an_error(value):
    with pytest.raises(ValueError, match="positive"):
        to_pactivity(value, "nM")


def test_unknown_unit_is_an_error():
    with pytest.raises(ValueError, match="unsupported unit"):
        to_pactivity(1.0, "ug.mL-1")
