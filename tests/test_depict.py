import base64

import pytest

from sarscope.depict import to_data_uri, to_png, to_svg

VEMURAFENIB = "CCCS(=O)(=O)Nc1ccc(F)c(C(=O)c2c[nH]c3ncc(-c4ccc(Cl)cc4)cc23)c1F"
RGROUP_CORE = "O=C(Nc1ccc(Oc2ccnc(NC(=O)C3([*:1])CC3)c2)c([*:2])c1)N"


def test_svg():
    svg = to_svg(VEMURAFENIB)
    assert svg is not None
    assert "<svg" in svg
    assert "path" in svg  # bonds were actually drawn


def test_png_is_a_png():
    png = to_png(VEMURAFENIB)
    assert png is not None and png[:4] == b"\x89PNG"


def test_data_uri_decodes_to_the_png():
    uri = to_data_uri("CCO")
    assert uri is not None and uri.startswith("data:image/png;base64,")
    assert base64.b64decode(uri.split(",", 1)[1])[:4] == b"\x89PNG"


def test_size_is_honoured():
    small = to_svg("CCO", (120, 90))
    assert "120" in small and "90" in small


def test_attachment_points_survive():
    # The numbered dummy atoms are the point of an R-group core.
    svg = to_svg(RGROUP_CORE)
    assert svg is not None and "<svg" in svg


def test_highlight_does_not_break_drawing():
    svg = to_svg(VEMURAFENIB, highlight="c1ccccc1")
    assert svg is not None and "<svg" in svg


def test_unknown_highlight_is_ignored_not_fatal():
    assert to_svg("CCO", highlight="not a pattern") is not None


@pytest.mark.parametrize("bad", ["", "C1CC", "not a smiles", None])
def test_bad_input_returns_none_rather_than_raising(bad):
    assert to_svg(bad) is None
    assert to_png(bad) is None
    assert to_data_uri(bad) is None
