# Preserved first attempt

The run stopped during alignment QC before computing any QSAR scores:
`androstanediol: Too many symmetry mappings; use pre-aligned input or a tighter reference.`

The old MCS setting required matching chiral-tag presence. For aldosterone /
androstanediol it reduced the core to an isolated six-membered ring, with
24 × 36 = 864 candidate mappings, exceeding the default limit of 256.

The correction constrains aromaticity/hybridisation **during** MCS search and
does not require equal chiral-tag presence at substituted sp3 centres. Each
input molecule still must pass assigned stereochemistry validation from its
actual 3D geometry. Uploaded heavy-atom coordinates are not altered.

No model scores were produced or discarded by this attempt. The source audit
and primary-reference tables are retained. The final run is in a separate
versioned directory.
