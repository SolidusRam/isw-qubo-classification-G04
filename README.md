# README information

## Preprocessing note

The current preprocessing implementation works correctly on numeric columns, but it does not yet handle non-numeric feature columns robustly. If a CSV contains text or categorical values, the preprocessing step may fail during z-score computation unless those columns are excluded or converted first.
