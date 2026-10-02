# Slice 02 — Historical Aggregation

## Module
`aggregation`

## What to build
Aggregate bureau, bureau-balance, previous-application, POS/CASH, installment, and credit-card histories to applicant level.

## Acceptance criteria
- [ ] Output has one row per `SK_ID_CURR`.
- [ ] Historical tables are reduced before joining.
- [ ] Large tables are processed in chunks where necessary.
- [ ] `TARGET` is not used during aggregation.
- [ ] A local processed feature table is produced.

## Out of scope
Advanced modeling.
