# Primary-source thrombin: reproducible worked example

This is a source-audited retrospective example, not a fully qualified predictive
model, independently reviewed binding alignment or validated lead-optimisation
service. No new source data are bundled with SARscope. The publisher labels the
supporting archive CC BY-NC 4.0; review its terms for your intended use.

## Run it in Streamlit

1. Open the **3D QSAR · uncertainty-aware 3D SAR** upload workflow (choose
   **Upload my compounds** when a ChEMBL assay is also available), then expand
   **Try a measured 3D-QSAR example**.
2. Choose **Thrombin · 88 primary-source compounds · fixed-partition worked
   example**. Review the linked publications and separate data licence.
3. Acknowledge the terms and click **Fetch original thrombin example**. This
   explicitly downloads the original 3.3 MB supporting archive. Page loading
   alone does not fetch it or fit a model. Changed hashes cause refusal.
4. Tick **Use original thrombin example**, then click **Apply recorded thrombin
   profile (offline recommended)**. This changes visible settings only. It does
   not approve the overlay, prepare a model or declare the model valid.
5. Click **Prepare and review alignment**. Inspect all supplied poses and source
   assumptions. The recorded profile retains explicit hydrogens and skips the
   common-core metric rather than pretending it passed. These are deposited
   prealigned conformations, not verified crystal poses. Only tick the overlay
   review declaration after your own inspection.
6. Check **Supplied partition · random nested training CV**: 59 training and 29
   held-out compounds. The fraction slider is disabled; its value does not
   select another partition. Then explicitly click **Run experimental 3D QSAR**.
7. Read the current run's validation checks, 2D RF/constant baselines and
   prediction errors before inspecting contours. Sensitivity and design remain
   separate actions. Historical assay comparability requires a separate review
   before matched-pair design; it is not inferred from the benchmark name.

The recorded profile uses 1 Å spacing, 4 Å padding, five Gaussian fields,
Gasteiger charges, training-derived per-field screening/block scaling, capped
component selection, three-fold nested CV, seed 42 and 99 training permutations.
It explicitly opts into a bounded **20,000-point / 512 MB** estimated field
workspace; ordinary browser defaults remain **12,000 / 256 MB**. This is not a
cap on total process memory. Prefer local/offline execution on constrained or
busy shared hosts. A resource refusal is not permission to remove inconvenient
compounds or silently change the frozen profile.

## What should reproduce

The 18 October source-adjudicated native run gave pooled nested q² **0.607052**,
held-out predictive R² **0.507182**, RMSE **0.771571** and three components.
The same-split ECFP4 RF RMSE was 0.950122. The app's full native path has been
executed through Streamlit AppTest with the actual 99-permutation model, not a
stub returning stored predictions. The prior recorded-metric caption is clearly
labelled as such; the live model table is produced only by Run.

The strict held-out R² > 0.6 and native-release checks remain failed. Common-core
alignment qualification is unassessed; one recorded training-only pose variant
failed its stability flag. No threshold or seed was changed to hide these facts.
The [validation report](3d-qsar-validation-20261018.md) explains the retrospective
scope, conditional uncertainty and historical test-label/export diagnostic.

## Use your own supplied states and partition

For a coordinate-bearing SDF, explicitly choose **Keep my reviewed SDF protonation
and tautomer states**. This requires one finite 3D pose per unique, single-component
closed-shell structure. The source states are retained without neutralisation,
tautomerisation or aggregation. Ordinary curation is reused as a *shadow identity
audit* so that alternate states/replicates of one parent cannot become independent
observations; its changed structures and aggregated activities are not used.
The ordinary standardising upload route remains the default.

You can then select an optional supplied train/test column and explicitly choose
its training label. Every row must have one of exactly two labels, both sides
must be present, and IDs must be unique/non-overlapping. No activity-based split
is inferred. If you remove compounds, the source partition refuses scoring until
the complete declared set is restored or you explicitly choose a different split.
Supplying a partition does not establish that it is independent or prospective.

Use the SDF, not the CSV, to reproduce the example's deposited frame. CSV/SMILES
has no coordinates; generated alignment is a different experiment. Every original
activity is retained unchanged; no pIC50 values were invented or converted from
qualitative screening calls.

## Evidence limits

AppTest verifies Python-side UI flow, settings, source state/partition retention
and native metric reproduction. It does not render the browser's WebGL overlay
or prove that a human reviewed the poses. An independent chemical review and an
untouched final-workflow evaluation remain required for broader claims.
