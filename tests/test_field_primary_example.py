"""Reviewed-state/partition contracts and the actual opt-in primary-data UI path."""

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from threadpoolctl import threadpool_limits

from sarscope.field_primary_example import (
    RECORDED_SCORES,
    TEST_IDS,
    TRAIN_IDS,
    fetch_primary_archive,
    primary_thrombin_dataset,
)
from sarscope.field_qsar import dataset_from_upload, prescribed_split_indices, subset_dataset


def posed(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    return mol


def uploaded(molecules, *, partition=None, preserve=True):
    raw = pd.DataFrame(
        {
            "molecule_id": [f"mol{i}" for i in range(len(molecules))],
            "smiles": [Chem.MolToSmiles(Chem.RemoveHs(m)) for m in molecules],
            "pactivity": np.arange(len(molecules)) + 6.0,
        }
    )
    if partition is not None:
        raw["partition"] = partition
    return dataset_from_upload(
        raw,
        molecules,
        id_col="molecule_id",
        smiles_col="smiles",
        value_col="pactivity",
        unit="p",
        endpoint="Ki",
        source_sha256="synthetic-test",
        preserve_input_states=preserve,
        split_col="partition" if partition is not None else None,
    )


def test_reviewed_states_keep_ions_and_coordinates_without_aggregation():
    mol = posed("CC[NH3+]")
    before = mol.GetConformer().GetPositions().copy()
    result = uploaded([mol])
    assert Chem.GetFormalCharge(result.molecules[0]) == 1
    np.testing.assert_array_equal(result.molecules[0].GetConformer().GetPositions(), before)
    assert result.table.n_measurements.tolist() == [1]
    assert result.source["preserve_input_states"] is True
    assert result.accounting.name.tolist() == ["reviewed_input_states"]
    with pytest.raises(ValueError, match="Standardisation changed a molecular graph"):
        uploaded([mol], preserve=False)


def test_reviewed_state_mode_does_not_allow_duplicates_salts_or_fake_3d():
    mol = posed("CCO")
    with pytest.raises(ValueError, match="unique structures"):
        uploaded([mol, mol])
    with pytest.raises(ValueError, match="parent identity needs adjudication"):
        uploaded([posed("CCN"), posed("CC[NH3+]")])
    with pytest.raises(ValueError, match="single-component"):
        uploaded([posed("CC[NH3+].[Cl-]")])
    with pytest.raises(ValueError, match="3D conformer"):
        uploaded([Chem.MolFromSmiles("CCO")])
    with pytest.raises(ValueError, match="radical"):
        uploaded([posed("[CH2]CO")])


def test_supplied_partition_preserves_order_and_refuses_ambiguity_or_hidden_subsetting():
    data = uploaded(
        [posed("CCO"), posed("CCN"), posed("CCC")], partition=["train", "held_out", "train"]
    )
    assert prescribed_split_indices(data) == ([0, 2], [1])
    changed = subset_dataset(data, ["mol0", "mol1"])
    with pytest.raises(ValueError, match="complete declared compound set"):
        prescribed_split_indices(changed)
    with pytest.raises(ValueError, match="both training and held-out"):
        uploaded([posed("CCO"), posed("CCC")], partition=["train", "train"])
    with pytest.raises(ValueError, match="explicitly training or held-out"):
        uploaded([posed("CCO"), posed("CCC")], partition=["train", None])
    data.source["prescribed_split"]["test_ids"].append("mol0")
    with pytest.raises(ValueError, match="non-overlapping"):
        prescribed_split_indices(data)


def test_primary_archive_mismatch_is_rejected_before_parsing():
    with pytest.raises(ValueError, match="hash/size differs"):
        primary_thrombin_dataset(b"not source data")


def test_primary_download_is_bounded_and_rejects_changed_content(monkeypatch):
    import sarscope.field_primary_example as module

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            yield b"x" * (module.MAX_DOWNLOAD_BYTES + 1)

    monkeypatch.setattr(module.httpx, "stream", lambda *a, **k: Response())
    with pytest.raises(ValueError, match="4 MiB"):
        fetch_primary_archive()
    monkeypatch.setattr(Response, "iter_bytes", lambda self: iter([b"changed"]))
    with pytest.raises(ValueError, match="hash differs"):
        fetch_primary_archive()


@pytest.fixture(scope="module")
def primary_payload():
    path = Path(
        os.environ.get(
            "SARSCOPE_PRIMARY_ARCHIVE",
            "runs/3dqsar-validation-20261018/original-sutherland/jm0497141_si_002.zip",
        )
    )
    if not path.is_file():
        pytest.skip("Supply separately licensed hash-matching data via SARSCOPE_PRIMARY_ARCHIVE")
    return path.read_bytes()


def test_primary_reader_retains_all_source_states_values_and_frozen_partition(primary_payload):
    dataset = primary_thrombin_dataset(primary_payload)
    assert len(dataset.table) == 88
    assert dataset.table.molecule_id.tolist() == TRAIN_IDS + TEST_IDS
    assert prescribed_split_indices(dataset) == (list(range(59)), list(range(59, 88)))
    for mol, row in zip(dataset.molecules, dataset.table.itertuples(), strict=True):
        assert row.pactivity == float(mol.GetProp("ACTIVITY"))
        assert all(a.GetNumRadicalElectrons() == 0 for a in mol.GetAtoms())
    assert sum(Chem.GetFormalCharge(m) > 0 for m in dataset.molecules) == 79
    assert dataset.source["license"].startswith("CC-BY-NC")
    assert dataset.source["design_comparability_review_required"] is True


def test_primary_ui_is_opt_in_and_actual_profile_reproduces_native_scores(
    primary_payload, monkeypatch, request
):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    import sarscope.field_qsar_ui as ui

    calls = []

    def fetch(source_hash):
        calls.append("fetch")
        return primary_payload

    monkeypatch.setattr(ui, "_fetch_primary_example", fetch)
    app = AppTest.from_string(
        "from sarscope.field_qsar_ui import show_field_qsar\nshow_field_qsar(key='primary3d')",
        default_timeout=180,
    ).run()
    app.selectbox(key="primary3d_example_name").select("primary_thrombin").run()
    assert not calls and not app.exception
    app.checkbox(key="primary3d_primary_terms").check().run()
    assert not calls
    app.button(key="primary3d_fetch_primary").click().run()
    assert calls == ["fetch"] and "primary3d_result" not in app.session_state
    app.checkbox(key="primary3d_use_primary").check().run()
    app.button(key="primary3d_primary_reset").click().run()
    assert not app.exception and not app.error
    app.button(key="primary3d_prepare").click().run()
    assert not app.exception and not app.error
    assert app.button(key="primary3d_run").disabled
    assert app.selectbox(key="primary3d_split").value == "supplied"
    assert app.checkbox(key="primary3d_larger_workspace").value is True
    app.checkbox(key="primary3d_reviewed").check().run()
    with threadpool_limits(limits=1):
        app.button(key="primary3d_run").click().run(timeout=180)
    assert not app.exception and not app.error
    result = app.session_state["primary3d_active_result"]
    stats = result.statistics.query("algorithm == 'field_pls'").iloc[0]
    for metric, value in RECORDED_SCORES.items():
        assert stats[metric] == pytest.approx(value, abs=1e-6)
        request.node.user_properties.append((metric, str(float(stats[metric]))))
    request.node.user_properties.extend(
        [
            ("archive_sha256", hashlib.sha256(primary_payload).hexdigest()),
            ("field_settings", json.dumps(asdict(result.prepared.params), sort_keys=True)),
            ("source", json.dumps(result.prepared.dataset.source, sort_keys=True)),
            ("split_indices", json.dumps(result.provenance["split_indices"], sort_keys=True)),
            (
                "module_sha256",
                json.dumps(
                    {
                        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                        for name in (
                            "src/sarscope/field_qsar.py",
                            "src/sarscope/field_qsar_ui.py",
                            "src/sarscope/field_primary_example.py",
                            "src/sarscope/pipeline.py",
                        )
                    },
                    sort_keys=True,
                ),
            ),
        ]
    )
    assert result.prepared.params.spacing == 1
    assert result.fields.grid.n_points == 14300
    assert len(result.permutations) == 99
    assert result.prepared.dataset.source["preserve_input_states"] is True
    assert "not validated" in result.status
    assert not result.checks.loc[result.checks.check.eq("Alignment thresholds met"), "passed"].iloc[
        0
    ]
    assert result.predictions.query("split == 'held_out'").molecule_id.tolist() == TEST_IDS
    # Removing one compound may not silently redefine the source partition.
    app.multiselect(key="primary3d_compounds").set_value(TRAIN_IDS + TEST_IDS[:-1]).run()
    app.button(key="primary3d_prepare").click().run()
    assert app.button(key="primary3d_run").disabled
    assert any("complete declared compound set" in warning.value for warning in app.warning)
    # Changing examples cannot retain an unavailable split/reference, and reset
    # must put ordinary browser budgets back rather than inherit the 512 MB opt-in.
    app.selectbox(key="primary3d_example_name").select("ccr5").run()
    app.checkbox(key="primary3d_example").check().run()
    app.button(key="primary3d_example_reset").click().run()
    assert not app.exception and not app.error
    app.button(key="primary3d_prepare").click().run()
    assert app.selectbox(key="primary3d_split").value == "random"
    assert app.checkbox(key="primary3d_larger_workspace").value is False
