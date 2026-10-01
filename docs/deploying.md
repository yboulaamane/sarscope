# Deploying the Streamlit app

The app is a thin layer over the library, with optional visualization and docking
dependencies. Three things about Streamlit Community Cloud are worth knowing, each
learned from a failed deploy.

## The three files

| File | Read by | Purpose |
|---|---|---|
| `requirements.txt` | pip | Python dependencies |
| `packages.txt` | apt-get | system libraries |
| `.streamlit/config.toml` | Streamlit | theme, telemetry off |

**`packages.txt` must contain nothing but package names, one per line.** Its
lines are passed to `apt-get install` verbatim, so a comment line is treated as
a package name and fails the whole install step. Do not add comments to it.

**`requirements.txt` must not list `.` or `.[app]`.** A self-referencing
requirements file makes pip resolve the project as a dependency of itself. The
first deploy did this and spent its time backtracking through every matplotlib
release before trying to build one from source.

**Streamlit Cloud does not read `runtime.txt`.** The Python version is chosen
in **Advanced settings** when the app is created, and cannot be changed
afterwards — you have to delete the app and recreate it. Python 3.12 is the
safe choice: Sorbent requires >= 3.12, and every dependency publishes wheels
for it. Newer versions may lack a wheel for some package and fall back to
building from source, which is slow and often fails.

## Why packages.txt exists

RDKit's drawing extension is a compiled library that links against the system
X11 libraries:

```console
$ ldd .../rdkit/Chem/Draw/rdMolDraw2D*.so | grep X
	libXrender.so.1 => /lib/x86_64-linux-gnu/libXrender.so.1
	libX11.so.6     => /lib/x86_64-linux-gnu/libX11.so.6
	libXext.so.6    => /lib/x86_64-linux-gnu/libXext.so.6
```

The RDKit wheel bundles cairo and freetype but not these, so on a minimal
container `import rdkit.Chem.Draw` raises ImportError.

**The app does not require them.** `sarscope.depict` imports the extension
defensively: if it is missing, structures fall back to SMILES text and the app
says why. Everything else — curation, descriptors, scaffolds, cliffs, models,
the downloadable report — works unchanged. So if `packages.txt` ever causes
trouble, deleting it can disable the drawings and the optional Vina docking engine;
the remaining analyses still work.

## Setting the app up

1. Push to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), **Create app** from
   the repository.
3. Main file path: `streamlit_app.py`.
4. **Advanced settings → Python 3.12.**
5. Deploy.

The first page render performs no ChEMBL request. Target lookup, curation,
medicinal-chemistry analyses, ML, explanation and report ZIP creation each run
only from their corresponding button. This is intentional: a collapsed
Streamlit expander still executes its body, so expensive work must be gated by
buttons rather than merely hidden in an expander.

The deployment requirements include `shap` and `umap-learn`. Streamlit supports
both: TreeSHAP is opt-in for the descriptor Random Forest (100 held-out
compounds maximum), and ECFP4/Jaccard UMAP is a separate button with a 2,000
compound sample cap. Neither package is imported on the landing page. The
first UMAP run after a restart may take longer due to Numba kernel compilation.
Dependency-file changes trigger a cloud rebuild; these optional analyses do
not eliminate Community Cloud's cold-start delays.

The cliff-pair docking step adds `autodock-vina` to `packages.txt` and Meeko,
ProLIF, Gemmi, py3Dmol and filelock to `requirements.txt`. These Python packages
are not imported on the landing page. Vina runs as a standalone executable,
avoiding dependence on Python-specific Vina wheels. ProLIF network HTML is
generated without Jupyter/IPython and displayed in isolated Streamlit component
iframes so the two diagrams cannot conflict. No paid interaction-diagram service
is needed. Browser diagram/viewer JavaScript can load from public CDNs; molecular
files are not submitted to an external structural-analysis service.

Only an explicit pair-docking button starts work. Jobs use one CPU, permit one
active job per host, and terminate after the user-selected total timeout (maximum
300 seconds). Uploaded receptors and temporary worker files are deleted when
the job ends; results and downloadable inputs remain in that user's session
until it is reset or ends. These bounds reduce, but do not guarantee avoidance
of, shared-host resource limits. Rebuilds can take longer after adding these
dependencies. See [cliff-pair docking](cliff-docking.md) for supported chemistry.

Native descriptor selection, fingerprint–descriptor hybrids and regression
diagnostics add no deployment dependencies. Molfeat stays an optional extra
because its core requires PyTorch; it is not in requirements.txt. Browser
explanations cap descriptors at 50. Metric bootstraps are an explicit,
prediction-only 500-repeat step, not repeated model training.

For a large target, the app shows download-page and curation progress plus
separate phase timings. This is mostly a data-processing cost, not a Streamlit
rendering limit. The release-keyed raw download cache lives on local storage;
the bounded standardized-structure cache lives in the Python process. Both
can be lost when Community Cloud restarts the app, because its local file
storage [is not guaranteed to persist](https://docs.streamlit.io/develop/concepts/connections/connecting-to-data).

## When a deploy fails

Click **Manage app** in the lower right to open the build log. The first line
that says `ERROR` names the step that failed:

- *during apt* — a bad line in `packages.txt` (a comment, or a package name
  that does not exist on the container's Debian release).
- *during pip, with backtracking* — a dependency has no wheel for the selected
  Python version and pip is searching older releases. Recreate the app on
  Python 3.12.
- *at app start, ImportError* — a Python package imported fine but its
  compiled part could not load. That means a missing system library, so it
  belongs in `packages.txt`.
