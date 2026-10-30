# Calibration record provenance

The JSON files in this directory are the compact, tracked records required by
the calibration tests and runtime adapters.  They intentionally contain no
publisher PDF, solver fields, incoming handoff scripts, or bulk mesh output.

## `pla_notched_tension.json`

- 54 tension rows selected verbatim from the 108-row PLA notched-test
  transcription used by the project.
- Source DOI: [10.1111/ffe.12746](https://doi.org/10.1111/ffe.12746).
- Source-file SHA-256 before selection:
  `69661b9582e7420890c0a051f92649dbb87b9e8b0acc3105c98729ec9f1f08a7`.
- The rows retain table/page and measured geometry/load fields so the frozen
  split and numerical calibration remain auditable.

## `abs_nominal_tension.json`

- 54 nominal-tension rows selected verbatim from the 108-row ABS experimental
  record used by the Ng--Susmel campaign adapter.
- Source-record SHA-256 before selection:
  `b96f0dffbf7410065fa333579255fdb9eefd4ef540faa8794b7aa5b62f291229`.
- Selection is exactly the six declared families `UB`, `UI`, `US`, `OB`, `OI`,
  and `OS`; bending records are excluded by the stated campaign contract.

## `pla_weibull_frozen.json`

- Compact tracked copy of the already-used frozen parameter metadata, retained
  only so the mesh adapter does not depend on an ignored private handoff.
- It records the original source path and hash; it is not claimed to be a new
  numerical solve or source dataset.

## `qualified_weibull_reductions.json`

- Minimal `m_grid`/`logJ` reductions and mesh hashes for the two PLA raster-0
  calibration families (`UB`, `US`) from qualified h=.18 NGSolve evidence.
- This is a compact derived fixture from the project's numerical result, not a
  replacement for a fresh external solve.  The full campaign still fails
  loudly when any non-fixture mesh evidence is unavailable.

These small numeric transcriptions are included solely to make the repository
self-contained and reproducible.  Reusers must consult the cited source and
applicable terms before using the data beyond this research-validation scope.
