"""RunResults -> a folder a person can open, email, or attach as supplementary data.

    <out>/report.html          self-contained: figures inlined as base64 PNG
    <out>/provenance.json      versions, ChEMBL release, every parameter
    <out>/tables/*.csv         curation_log, curated_dataset, rejected,
                               descriptor_profile, pca_loadings,
                               scaffold_diversity, scaffold_enrichment,
                               cliffs_<fp>, identical_pairs_<fp>,
                               cliff_generators_<fp>, consensus_cliffs,
                               model_scores, rgroups_<n>_substituents/members
    <out>/figures/*.png        descriptor_distributions, chemical_space, cliffs_<fp>,
                               cliff_generators_<fp>, applicability_domain

Every table is written as CSV under its own descriptive name, so a reader can
take one into a spreadsheet without unpicking the HTML. Figures use
matplotlib's Agg backend, so no display is needed and it works in CI and on
Colab.

The report opens with the curation log and the list of skipped steps, before
any result: a reader should know what the numbers are numbers *of* first.
"""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from sarscope import provenance  # noqa: E402
from sarscope.analysis.descriptors import CORE_DESCRIPTORS  # noqa: E402
from sarscope.analysis.landscape import cliff_generators  # noqa: E402
from sarscope.depict import to_data_uri  # noqa: E402
from sarscope.pipeline import RunResults  # noqa: E402

#: Categorical slots 1 and 2 of the reference palette, plus a recessive grey.
GROUP_COLORS = {1: "#2a78d6", 2: "#eb6834"}
CLIFF_COLOR = "#e34948"
MUTED = "#8a8a85"

STYLE = """
:root { --bg:#fcfcfb; --fg:#0b0b0b; --muted:#52514e; --line:#e2e2dd; --card:#ffffff; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#1a1a19; --fg:#ffffff; --muted:#c3c2b7; --line:#3a3a37; --card:#222220; }
}
* { box-sizing: border-box; }
body { margin:0; padding:32px 16px; background:var(--bg); color:var(--fg);
  font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }
main { max-width: 1000px; margin: 0 auto; }
h1 { font-size:1.9rem; margin:0 0 4px; letter-spacing:-0.02em; }
h2 { font-size:1.3rem; margin:40px 0 4px; padding-top:20px; border-top:1px solid var(--line); }
h3 { font-size:1rem; margin:24px 0 4px; }
p.sub { color:var(--muted); margin:0 0 24px; }
p.note { color:var(--muted); font-size:0.9rem; margin:4px 0 12px; }
table { border-collapse:collapse; width:100%; font-size:0.86rem; margin:12px 0; }
th,td { text-align:right; padding:6px 10px; border-bottom:1px solid var(--line); }
th:first-child,td:first-child { text-align:left; }
th { color:var(--muted); font-weight:600; white-space:nowrap; }
td.smiles { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:0.78rem;
  max-width:340px; overflow-wrap:anywhere; text-align:left; }
img { max-width:100%; height:auto; display:block; margin:12px 0; }
.cards { display:flex; flex-wrap:wrap; gap:12px; margin:16px 0; }
.card { flex:1 1 150px; background:var(--card); border:1px solid var(--line);
  border-radius:10px; padding:12px 14px; }
.card .k { color:var(--muted); font-size:0.78rem; }
.card .v { font-size:1.4rem; font-weight:600; letter-spacing:-0.02em; }
.warn { background:var(--card); border-left:3px solid #eda100; padding:10px 14px;
  border-radius:0 8px 8px 0; margin:12px 0; }
code { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:0.82em;
  overflow-wrap:anywhere; }
.structures { display:flex; flex-wrap:wrap; gap:14px; margin:14px 0; }
.structures figure { margin:0; background:#ffffff; border:1px solid var(--line);
  border-radius:10px; padding:8px; }
.structures img { margin:0; }
.structures figcaption { color:var(--muted); font-size:0.78rem; margin-top:6px;
  max-width:260px; }
footer { color:var(--muted); font-size:0.82rem; margin-top:48px;
  border-top:1px solid var(--line); padding-top:16px; }
"""


def write_report(results: RunResults, out_dir: Path) -> Path:
    """Write everything above into ``out_dir`` (created if needed); return report.html.

    Raises FileExistsError for a non-empty directory that lacks a
    provenance.json, so a typo in --out cannot scatter files over an unrelated
    folder; a previous report's folder is overwritten. provenance.json is
    ``results.provenance``, or a fresh record when that is empty (``analyse``
    called directly).
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not (out_dir / "provenance.json").exists():
        raise FileExistsError(
            f"{out_dir} is not empty and holds no provenance.json, so it is not a "
            "SARscope report folder; refusing to write into it"
        )
    tables = out_dir / "tables"
    figures = out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    record = results.provenance or provenance.collect(results.params, {"kind": "unknown"})
    (out_dir / "provenance.json").write_text(json.dumps(record, indent=2))

    written = _write_tables(results, tables)
    images = _write_figures(results, figures)
    path = out_dir / "report.html"
    path.write_text(_render(results, record, written, images), encoding="utf-8")
    return path


# -- tables ------------------------------------------------------------------


def _write_tables(results: RunResults, out: Path) -> dict[str, pd.DataFrame]:
    log = pd.DataFrame(
        [
            {
                "step": s.name,
                "records_in": s.records_in,
                "records_out": s.records_out,
                "removed": s.removed,
                "molecules_out": s.molecules_out,
                "detail": s.detail,
            }
            for s in results.curation.steps
        ]
    )
    loadings = results.pca.loadings.reset_index(names="property")
    explained = pd.DataFrame(
        [["explained variance", *results.pca.explained.to_numpy()]], columns=loadings.columns
    )
    cumulative = pd.DataFrame(
        [["cumulative variance", *results.pca.cumulative.to_numpy()]], columns=loadings.columns
    )

    written: dict[str, pd.DataFrame] = {
        "curation_log": log,
        "curated_dataset": results.table,
        "rejected": results.curation.rejected,
        "pca_loadings": pd.concat([loadings, explained, cumulative], ignore_index=True),
        "scaffold_diversity": results.diversity.reset_index(names="class"),
        "scaffold_enrichment": results.enrichment,
        "consensus_cliffs": results.consensus_cliffs,
    }
    if results.profile is not None:
        written["descriptor_profile"] = results.profile.stats.reset_index().merge(
            results.profile.p_values.rename("p_value").rename_axis("property").reset_index(),
            on="property",
            how="left",
        )
    for name, sas in results.landscapes.items():
        written[f"cliffs_{name}"] = sas.cliffs
        written[f"identical_pairs_{name}"] = sas.identical_pairs
        written[f"cliff_generators_{name}"] = cliff_generators(
            sas.cliffs, results.params.landscape.generator_sd
        )
    if results.models is not None:
        written["model_scores"] = results.models.scores
    for i, sar in enumerate(results.rgroups, start=1):
        written[f"rgroups_{i:02d}_substituents"] = sar.substituents
        written[f"rgroups_{i:02d}_members"] = sar.members

    for name, frame in written.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    return written


# -- figures -----------------------------------------------------------------


def _save(fig: Any, path: Path) -> str:
    """Write the PNG and return it base64-encoded for inlining."""
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=MUTED, alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)


def _write_figures(results: RunResults, out: Path) -> dict[str, str]:
    images: dict[str, str] = {}
    table = results.table

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    for ax, prop in zip(axes.ravel(), CORE_DESCRIPTORS, strict=True):
        for group, color in GROUP_COLORS.items():
            values = table.loc[table["group"] == group, prop]
            if len(values):
                ax.hist(values, bins=30, alpha=0.62, color=color, label=f"Group {group}")
        ax.set_title(prop, fontsize=10)
        _style(ax)
    axes.ravel()[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Physicochemical properties by activity group", fontsize=11)
    fig.tight_layout()
    images["descriptor_distributions"] = _save(fig, out / "descriptor_distributions.png")

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for group, color in GROUP_COLORS.items():
        rows = results.pca.scores[table["group"].to_numpy() == group]
        ax.scatter(
            rows["PC1"],
            rows["PC2"],
            s=11,
            c=color,
            alpha=0.55,
            linewidths=0,
            label=f"Group {group}",
        )
    var = results.pca.explained
    ax.set_xlabel(f"PC1 ({var.iloc[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var.iloc[1]:.1%})")
    ax.set_title("Chemical space (property PCA)", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    images["chemical_space"] = _save(fig, out / "chemical_space.png")

    for name, sas in results.landscapes.items():
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        pairs = sas.cliffs
        if len(pairs):
            ax.scatter(
                pairs["similarity"], pairs["delta"], s=9, c=CLIFF_COLOR, alpha=0.6, linewidths=0
            )
        ax.axvline(
            results.params.landscape.similarity_threshold, color=MUTED, lw=1, ls="--", alpha=0.7
        )
        ax.axhline(
            results.params.landscape.activity_threshold, color=MUTED, lw=1, ls="--", alpha=0.7
        )
        ax.set_xlabel("Tanimoto similarity")
        ax.set_ylabel("|delta potency| (log units)")
        ax.set_title(f"Activity cliffs ({name})", fontsize=11)
        _style(ax)
        images[f"cliffs_{name}"] = _save(fig, out / f"cliffs_{name}.png")

        gens = cliff_generators(sas.cliffs, results.params.landscape.generator_sd)
        if len(gens):
            fig, ax = plt.subplots(figsize=(7.2, 4.4))
            ax.bar(
                range(len(gens)),
                gens["n_cliffs"],
                color=[CLIFF_COLOR if g else GROUP_COLORS[1] for g in gens["is_generator"]],
                width=1.0,
            )
            threshold = gens.attrs.get("threshold", float("nan"))
            if threshold == threshold:  # not NaN
                ax.axhline(threshold, color=MUTED, lw=1.2, ls="--")
            ax.set_xlabel("Molecules in at least one cliff")
            ax.set_ylabel("Cliffs formed")
            ax.set_title(f"Activity cliff generators ({name})", fontsize=11)
            _style(ax)
            images[f"cliff_generators_{name}"] = _save(fig, out / f"cliff_generators_{name}.png")

    if results.domain is not None:
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        train = results.domain.train_scores
        query = results.domain.query_scores
        ax.scatter(
            train[:, 0],
            train[:, 1],
            s=10,
            c=GROUP_COLORS[1],
            alpha=0.45,
            linewidths=0,
            label="Train",
        )
        if query.shape[0]:
            ax.scatter(
                query[:, 0],
                query[:, 1],
                s=16,
                c=GROUP_COLORS[2],
                alpha=0.75,
                linewidths=0,
                label="Test",
            )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Applicability domain (PCA bounding box)", fontsize=11)
        ax.legend(frameon=False, fontsize=9)
        _style(ax)
        images["applicability_domain"] = _save(fig, out / "applicability_domain.png")
    return images


# -- html --------------------------------------------------------------------


def _table_html(frame: pd.DataFrame, limit: int = 25, smiles_cols: tuple[str, ...] = ()) -> str:
    shown = frame.head(limit)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in shown.columns)
    body = []
    for _, row in shown.iterrows():
        cells = []
        for col, value in row.items():
            klass = ' class="smiles"' if col in smiles_cols else ""
            text = f"{value:,.4g}" if isinstance(value, float) else str(value)
            cells.append(f"<td{klass}>{html.escape(text)}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    more = (
        f"<p class='note'>Showing {limit} of {len(frame):,} rows; "
        "the full table is in <code>tables/</code>.</p>"
        if len(frame) > limit
        else ""
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>{more}"


def _cards(pairs: list[tuple[str, str]]) -> str:
    cells = "".join(
        f"<div class='card'><div class='k'>{html.escape(k)}</div>"
        f"<div class='v'>{html.escape(v)}</div></div>"
        for k, v in pairs
    )
    return f"<div class='cards'>{cells}</div>"


def _structures(items: list[tuple[str, str]], size: tuple[int, int] = (260, 190)) -> str:
    """A row of drawn structures with captions. Skips anything RDKit cannot draw."""
    cells = []
    for smiles, caption in items:
        uri = to_data_uri(smiles, size)
        if uri is None:
            continue
        cells.append(
            f"<figure><img alt='{html.escape(caption)}' src='{uri}'>"
            f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        )
    return f"<div class='structures'>{''.join(cells)}</div>" if cells else ""


def _figure(images: dict[str, str], key: str) -> str:
    if key not in images:
        return ""
    return f"<img alt='{html.escape(key)}' src='data:image/png;base64,{images[key]}'>"


def _render(
    results: RunResults,
    record: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    images: dict[str, str],
) -> str:
    source = record.get("source", {})
    title = source.get("target_name") or source.get("path") or "SARscope report"
    table = results.table
    parts: list[str] = []

    parts.append(f"<h1>{html.escape(str(title))}</h1>")
    subtitle = ", ".join(
        str(x)
        for x in [
            source.get("target_id"),
            source.get("organism"),
            source.get("release"),
            f"SARscope {record.get('sarscope', '')}",
            record.get("created_utc", ""),
        ]
        if x
    )
    parts.append(f"<p class='sub'>{html.escape(subtitle)}</p>")

    parts.append("<h2>1. Curation</h2>")
    parts.append(
        "<p class='note'>Every record that left the dataset, and the step that removed it. "
        "These choices change the numbers in every later section, so they are recorded "
        "rather than assumed.</p>"
    )
    parts.append(
        _cards(
            [
                ("Raw records", f"{source.get('raw_records', len(table)):,}"),
                ("Molecules", f"{len(table):,}"),
                ("Group 1", f"{int((table['group'] == 1).sum()):,}"),
                ("Group 2", f"{int((table['group'] == 2).sum()):,}"),
            ]
        )
    )
    parts.append(_table_html(tables["curation_log"], limit=40))
    for step, reason in results.skipped.items():
        parts.append(
            f"<div class='warn'><b>{html.escape(step)} skipped.</b> {html.escape(reason)}</div>"
        )

    parts.append("<h2>2. Physicochemical properties</h2>")
    parts.append(_figure(images, "descriptor_distributions"))
    if "descriptor_profile" in tables:
        parts.append("<h3>Descriptor statistics by group</h3>")
        parts.append(
            "<p class='note'>Kurtosis is Fisher excess kurtosis (normal = 0). p-values are "
            "two-sided Mann-Whitney U, Group 1 against Group 2.</p>"
        )
        parts.append(_table_html(tables["descriptor_profile"], limit=14))
    parts.append("<h3>PCA loadings</h3>")
    parts.append(
        "<p class='note'>Properties are standardised before the PCA, or molecular weight "
        "would dominate every component. Component signs are fixed so runs are comparable.</p>"
    )
    parts.append(_table_html(tables["pca_loadings"]))
    parts.append(_figure(images, "chemical_space"))

    parts.append("<h2>3. Scaffolds</h2>")
    parts.append("<h3>Murcko scaffold diversity</h3>")
    parts.append(
        "<p class='note'>Cyclic skeletons (Ncsk) use RDKit's generic scaffold. "
        '"Cyclic skeleton" has no single '
        "definition, so skeleton counts from different tools are not comparable.</p>"
    )
    parts.append(_table_html(tables["scaffold_diversity"]))
    parts.append("<h3>Scaffold enrichment</h3>")
    parts.append(
        "<p class='note'>EF is the Group 1 fraction within a scaffold over the Group 1 fraction "
        "of the whole dataset. A singleton scores the maximum EF on one molecule, so the table "
        "is sorted by <b>ef_lower</b>, the Wilson lower bound, which requires evidence.</p>"
    )
    ranked = results.enrichment[results.enrichment["n"] >= 5]
    if len(ranked):
        parts.append(
            _structures(
                [
                    (
                        str(r.scaffold),
                        f"{r.n} molecules, {r.frac_group1:.0%} Group 1, EF {r.ef:.2f}",
                    )
                    for r in ranked.head(8).itertuples()
                ]
            )
        )
    parts.append(_table_html(tables["scaffold_enrichment"], smiles_cols=("scaffold",)))

    if results.rgroups:
        parts.append("<h2>4. R-group SAR</h2>")
        parts.append(
            "<p class='note'>Per-position substituent statistics for the most populated series: "
            "the input to a medicinal-chemistry SAR read, not a substitute for one. "
            "<b>delta</b> compares a substituent's median potency against the molecules of the "
            "same series that differ at that position. Those molecules may differ elsewhere too, "
            "so read delta together with <b>n</b> and <b>p_value</b>.</p>"
        )
        for i, sar in enumerate(results.rgroups[:5], start=1):
            parts.append(
                f"<h3>Series {i} - {sar.n_molecules} molecules, "
                f"{len(sar.positions)} varying positions</h3>"
            )
            parts.append(f"<p class='note'><code>{html.escape(sar.scaffold)}</code></p>")
            parts.append(
                _structures(
                    [(sar.scaffold, "Shared scaffold"), (sar.core, "Labelled core")],
                    size=(300, 220),
                )
            )
            order = sar.substituents["delta"].abs().sort_values(ascending=False).index
            ranked_subs = sar.substituents.reindex(order)
            parts.append(
                _structures(
                    [
                        (
                            str(r.substituent),
                            f"{r.position}: n={r.n}, delta {r.delta:+.2f}, p={r.p_value:.1e}",
                        )
                        for r in ranked_subs.head(6).itertuples()
                    ],
                    size=(190, 145),
                )
            )
            parts.append(_table_html(ranked_subs, limit=12, smiles_cols=("substituent",)))

    parts.append("<h2>5. Activity landscape</h2>")
    parts.append(
        "<p class='note'>Every pair of molecules, placed by structural similarity and potency "
        "difference. Cliffs are similar pairs with very different potency; pairs whose "
        "fingerprints are identical are listed separately, since their SALI is undefined.</p>"
    )
    for name, sas in results.landscapes.items():
        parts.append(f"<h3>{name}</h3>")
        parts.append(
            _cards(
                [(k.replace("_", " ").title(), f"{v:,}") for k, v in sas.region_counts.items()]
                + [("Identical pairs", f"{len(sas.identical_pairs):,}")]
            )
        )
        parts.append(_figure(images, f"cliffs_{name}"))
        if len(sas.cliffs):
            smiles = results.table.set_index("molecule_id")["smiles"]
            potency = results.table.set_index("molecule_id")["pactivity"]
            top = sas.cliffs.iloc[0]
            pair = [
                (str(smiles[m]), f"{m}, potency {potency[m]:.2f}")
                for m in (top["id_a"], top["id_b"])
                if m in smiles.index
            ]
            if pair:
                parts.append(
                    "<p class='note'>Steepest cliff on this fingerprint: similarity "
                    f"{top['similarity']:.2f}, {top['delta']:.2f} log units apart. "
                    "What differs between these two structures is the SAR.</p>"
                )
                parts.append(_structures(pair, size=(300, 220)))
        parts.append(_figure(images, f"cliff_generators_{name}"))
    if results.consensus_generators:
        parts.append(
            "<p class='note'>Cliff generators found under every fingerprint: "
            f"{html.escape(', '.join(results.consensus_generators))}</p>"
        )

    if results.models is not None:
        parts.append("<h2>6. QSAR models</h2>")
        parts.append(
            "<p class='note'><b>leak_free</b> selects features and resamples inside training "
            "folds only, after the split. <b>naive</b> is the common ordering - select and "
            "oversample on everything, then split - which puts copies of training molecules in "
            "the test set and chooses features using the held-out rows. The gap between the two "
            "is how much that ordering would have flattered these models. The winner is chosen "
            "on cross-validated MCC, never on the test set.</p>"
        )
        parts.append(f"<p>Best model: <b>{html.escape(results.models.best_algorithm)}</b></p>")
        parts.append(_table_html(tables["model_scores"], limit=30))
        if results.domain is not None:
            parts.append("<h3>Applicability domain</h3>")
            parts.append(
                f"<p class='note'>{results.domain.coverage:.1%} of test compounds fall inside "
                "the training set's PCA bounding box. A box in two components is a generous "
                "criterion: a molecule can sit inside it and still be far from every training "
                "compound.</p>"
            )
            parts.append(_figure(images, "applicability_domain"))

    packages = record.get("packages", {})
    versions = ", ".join(f"{k} {v}" for k, v in packages.items() if v)
    parts.append(
        f"<footer>Generated by SARscope {html.escape(str(record.get('sarscope', '')))}. "
        f"{html.escape(versions)}.<br>Tables in <code>tables/</code>, figures in "
        "<code>figures/</code>, full settings in <code>provenance.json</code>.</footer>"
    )

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(str(title))} - SARscope</title>"
        f"<style>{STYLE}</style></head><body><main>{''.join(parts)}</main></body></html>"
    )
