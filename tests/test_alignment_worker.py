"""Serial-worker reuse and timeout attribution; no binding-mode validation."""

import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rdkit import Chem
from test_field_numerics import methanol
from test_field_qsar import molecule

from sarscope.analysis import align3d
from sarscope.conformers import validate_conformer_stereo
from sarscope.params import FieldQsarParams


def fake_context(monkeypatch, *, polls, responses):
    connection, child, process = Mock(), Mock(), Mock()
    connection.poll.side_effect = polls
    connection.recv.side_effect = responses
    process.pid = None
    process.start.side_effect = lambda: setattr(process, "pid", 123)
    process.is_alive.side_effect = [True, False]
    context = Mock()
    context.Pipe.return_value = connection, child
    context.Process.return_value = process
    monkeypatch.setattr(align3d.mp, "get_context", lambda _: context)
    monkeypatch.setattr(align3d, "time", SimpleNamespace(monotonic=lambda: 0.0))
    return context, connection, process


def test_one_worker_handles_multiple_requests_and_is_cleaned_up(monkeypatch):
    block = Chem.MolToMolBlock(methanol())
    context, connection, process = fake_context(
        monkeypatch,
        polls=[True, True, True],
        responses=[("ready", None), (True, (block, {})), (True, (block, {}))],
    )
    with align3d._AlignmentGenerator(FieldQsarParams(), 180) as worker:
        for _ in range(2):
            mol, _ = worker.generate("CO", None)
            validate_conformer_stereo(mol, "CO")
    assert context.Process.call_count == process.start.call_count == 1
    assert connection.send.call_count == 2
    process.terminate.assert_called_once()
    process.join.assert_called_once()
    connection.close.assert_called_once()


@pytest.mark.parametrize(
    ("deadline", "stage", "polls", "responses", "message"),
    [
        (180, "startup", [False], [], "worker startup exceeded its 30s limit"),
        (180, "job", [True, False], [("ready", None)], "conformer/MCS job exceeded its 30s"),
        (4, "job", [True, False], [("ready", None)], "Whole-series alignment budget"),
        (0, "startup", [], [], "Whole-series alignment budget"),
    ],
)
def test_timeout_reports_correct_budget_and_always_cleans_up(
    monkeypatch, deadline, stage, polls, responses, message
):
    _, connection, process = fake_context(monkeypatch, polls=polls, responses=responses)
    with pytest.raises(ValueError, match=message):
        with align3d._AlignmentGenerator(FieldQsarParams(), deadline) as worker:
            assert stage == "job"
            worker.generate("CO", None)
    connection.close.assert_called_once()
    if deadline == 0:
        process.start.assert_not_called()
        process.terminate.assert_not_called()
    else:
        process.terminate.assert_called_once()


@pytest.mark.parametrize("failure", [EOFError(), (False, "Stereo constraint failed")])
def test_worker_crash_and_chemistry_failures_are_not_silently_skipped(monkeypatch, failure):
    fake_context(monkeypatch, polls=[True, True], responses=[("ready", None), failure])
    message = "stopped unexpectedly" if isinstance(failure, EOFError) else "Stereo constraint"
    with pytest.raises(ValueError, match=message):
        with align3d._AlignmentGenerator(FieldQsarParams(), 180) as worker:
            worker.generate("CO", None)


def test_real_spawned_worker_reuses_pid_and_preserves_coordinate_stereo():
    params = FieldQsarParams(alignment="rigid_mcs", alignment_timeout=60)
    with align3d._AlignmentGenerator(params, time.monotonic() + 60) as worker:
        pid = worker.process.pid
        for smiles in ("C[C@H](O)C(=O)O", "C[C@H](F)CC"):
            mol, diagnostics = worker.generate(smiles, None)
            validate_conformer_stereo(mol, smiles)
            assert diagnostics["stereo_verified_from_3d"]
            assert worker.process.pid == pid
    assert not worker.process.is_alive()


def test_generated_series_reuses_reference_and_keeps_core_warnings(monkeypatch):
    params = FieldQsarParams(alignment="rigid_mcs")
    monkeypatch.setattr(align3d, "preparation_available", lambda: True)
    reference = Chem.RemoveHs(molecule("c1ccccc1"))
    calls = []

    class Generator:
        def __init__(self, settings, deadline):
            calls.append("worker")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            calls.append("closed")

        def generate(self, smiles, ref):
            calls.append(smiles)
            return molecule(smiles), {}

    monkeypatch.setattr(align3d, "_AlignmentGenerator", Generator)
    other = Chem.RemoveHs(Chem.MolFromSmiles("CCCCCCCCCCc1ccccc1"))
    result = align3d.align_series([reference, other], ["ref", "other"], params)
    assert calls == ["worker", Chem.MolToSmiles(other), "closed"]
    assert len(result.molecules) == 2
    assert result.report.iloc[1].core_coverage < 0.5
    assert "covers too little" in result.report.iloc[1].warning


def test_series_failure_identifies_position_without_returning_partial_alignment(monkeypatch):
    class Generator:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def generate(self, *args):
            raise ValueError("Whole-series alignment budget exhausted")

    monkeypatch.setattr(align3d, "preparation_available", lambda: True)
    monkeypatch.setattr(align3d, "_AlignmentGenerator", Generator)
    mols = [Chem.RemoveHs(molecule("c1ccccc1")) for _ in range(2)]
    with pytest.raises(ValueError, match=r"bad \(compound 2/2, 1 completed\).*Whole-series"):
        align3d.align_series(mols, ["ref", "bad"], FieldQsarParams(alignment="rigid_mcs"))
