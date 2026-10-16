# Preserved second preparation attempt

The aromaticity/hybridisation-aware MCS search resolved the first pair's
overbroad symmetry (2 × 2 candidate mappings rather than 24 × 36), but estrone
still exceeded the browser's default 256-map limit. No QSAR scores were produced.

The benchmark is now explicitly configured for a bounded 4096-map limit in
`protocol.json`, with a 300-second total geometry-QC budget. This is a recorded
offline benchmark resource allowance, not an activity-dependent change or a
change to any deposited coordinate or stereochemistry. Browser defaults are
unchanged. Final outputs use a new directory.
