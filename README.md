# Shipment Record Anomaly Detection

An unsupervised anomaly detector for shipment/port records, built as an
extension of an existing shipment-data validation pipeline (Excel VBA +
Google Apps Script automation, plus a Python port-distance calculator).

## The problem

The existing validation pipeline catches known error patterns with explicit
rules (date format checks, port-code allowlists, dropdown validation). One
recurring edge case that rules struggle with: a port code that's actually a
country code (e.g. `PAR` instead of the real port code for a French port),
caught previously only by manual review. More generally: **combinations** of
mildly-off fields (distance slightly too long, weight slightly too high,
date near a year boundary) that are jointly unusual but don't individually
cross any one hardcoded threshold.

## Why unsupervised learning, specifically

There is no labeled dataset of "this shipment record was wrong" to train a
classifier on. An Isolation Forest is trained purely on the structure of
normal records and flags records that are structurally unusual relative to
the rest of the batch — no labels required. This is a deliberate and
correct modeling choice for this problem, not a limitation to hide.

## Evaluation methodology

Since no real labeled error data can be used, five known anomaly types are
injected into synthetic shipment data at an 8% rate, with labels kept
strictly separate from training and used only afterward for scoring — the
same approach used to evaluate fraud/anomaly detectors when true labels
don't exist.

Anomaly types injected: port code swapped for a country code, distance
wildly inconsistent with the real route, physically implausible container
weight, invalid ship date, and a "combo_subtle" case where several fields
are only mildly off individually but jointly unusual.

## Results (this run — see `evaluation_results.csv` for full per-record output)

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Isolation Forest | 0.979 | 0.979 | 0.979 |
| Rule-based baseline | 1.000 | 0.771 | 0.871 |

Isolation Forest ROC-AUC: 1.000

**Recall by anomaly type — this is the actual finding:**

| Anomaly type | ML recall | Rule recall |
|---|---|---|
| bad_date | 0.96 | 1.00 |
| combo_subtle | 0.95 | **0.00** |
| impossible_weight | 1.00 | 1.00 |
| distance_mismatch | 1.00 | 1.00 |
| port_code_is_country_code | 1.00 | 1.00 |

The rule-based baseline matches or ties the ML model on every anomaly type
that's a clear single-field violation — rules are perfectly adequate there,
and there's no need to claim otherwise. The gap is entirely on
`combo_subtle`: cases where no single field crosses a hardcoded threshold,
but several fields are jointly unusual. That is the actual justification
for using ML here, evidenced by a number, not an assumption.

## Honest limitations

- **All data here is synthetic.** The near-perfect scores reflect a clean,
  controlled evaluation set, not real-world performance. Real shipment data
  will be noisier, anomalies will be less cleanly separable, and precision/
  recall will almost certainly be lower.
- The `contamination` parameter (expected anomaly rate) is set to match the
  known injection rate. In production this would need to be tuned against a
  small hand-reviewed validation sample or a business tolerance for false
  positives, not assumed.
- This has not been run against real historical shipment records. Before
  relying on it operationally, it needs validation against actual flagged
  cases from the existing manual-review process.

## Files

- `generate_data.py` — synthetic shipment data + labeled anomaly injection
- `features.py` — feature engineering (continuous signals, not pre-solved
  binary flags, so the model does real work rather than relaying a rule)
- `detect.py` — trains the Isolation Forest, runs the rule-based baseline,
  evaluates both against held-out labels
- `evaluation_results.csv` — full per-record output from the last run

## Run it

```bash
pip install -r requirements.txt
python detect.py
```
