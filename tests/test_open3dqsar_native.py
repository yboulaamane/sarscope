"""Documented-command and field-order contracts; external binaries not required."""

from pathlib import Path

import numpy as np
import pytest

from benchmarks.open3dqsar_native import commands, read_ascii
from sarscope.analysis.fields import GridSpec


def test_native_commands_exclude_test_before_pretreatment():
    text = commands(Path("/tmp/example"), GridSpec((1.0, 2.0, 3.0), (3, 4, 5), 1.0), [2, 7])
    assert "object_list=3,8 attribute=testset" in text
    assert text.index("attribute=testset") < text.index("sdcut")
    assert text.index("format=ascii") < text.index("sdcut")
    assert "x_start=1 x_nodes=3" in text
    assert "y_start=2 y_nodes=4" in text
    assert "z_start=3 z_nodes=5" in text
    assert "pls pc=3" in text and "cv type=loo pc=3" in text
    assert "type=mm_ele diel_dep=const diel_const=1" in text


def test_native_ascii_axis_order(tmp_path):
    grid = GridSpec((0.0, 0.0, 0.0), (2, 3, 4), 1.0)
    original = np.arange(grid.n_points).reshape(grid.shape)
    exported = original.transpose(2, 1, 0).ravel()
    file = tmp_path / "field.agrd"
    header = ["X: 0,1", "Y: 0,2", "Z: 0,3", "step: 1,1,1", "nodes: 2,3,4", ""]
    file.write_text("\n".join(header) + "\n" + "\n".join(str(v) for v in exported) + "\n")
    assert np.array_equal(read_ascii(file, grid), original.ravel())
    with pytest.raises(ValueError, match="dimensions"):
        read_ascii(file, GridSpec((0.0, 0.0, 0.0), (2, 4, 3), 1.0))
