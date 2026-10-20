# Getting started

Use the [hosted app](https://sarscope.streamlit.app/) without installing anything,
or run the current source locally with Python 3.12 or newer.

## Local browser app

From a fresh environment:

```bash
git clone https://github.com/yboulaamane/sarscope.git
cd sarscope
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[app]"
python -m streamlit run streamlit_app.py
```

On Windows, activate with `.\.venv\Scripts\Activate.ps1` instead.
If your Linux distribution lacks `venv`, install its matching Python venv package.
The scientific stack and app are installed together; optional docking and
specialized representation libraries are separate extras.

## First analysis

1. Find a target by ChEMBL ID, gene/protein or disease; confirm its identity.
2. Curate the measurements and inspect the retained assay evidence.
3. Choose the analyses you need: chemical space, scaffolds, SAR or activity cliffs.
4. Fit and evaluate a 2D model, or open the 3D-QSAR route with reviewed coordinates.
5. Export the results, settings and source records.

Each expensive step is user-triggered. Opening the app does not download a dataset
or run docking. For your own 3D series, choose **3D QSAR · own compounds**;
SMILES alone do not supply an alignment.

## CLI

```bash
sarscope target CHEMBL5145
sarscope run CHEMBL5145 --out report/
sarscope run --input compounds.csv --out report/
```

Example upload:

```csv
molecule_id,smiles,pactivity
compound_1,CCOc1ccccc1,6.8
```

Declare a consistent measured endpoint. Qualitative PubChem Active/Inactive calls
use the separate screening route and are not converted into potency values.

See the [workflow guide](user-guide.md) for time/scaffold splits, descriptors,
prediction exports, optional extras and detailed commands.
