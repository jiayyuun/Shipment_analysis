"""

Method: for each flagged record, compute how many standard deviations each
feature is from the dataset's mean (a z-score), and report the 3 features
that deviate the most.
"""

import numpy as np
import pandas as pd

from generate_data import generate_dataset
from features import build_features
from detect import rule_based_baseline
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURE_LABELS = {
    "distance_km": "declared distance",
    "weight_kg": "declared weight",
    "teu_count": "container count",
    "weight_per_teu": "weight per container",
    "distance_ratio": "distance vs. expected route length",
    "origin_code_edit_dist": "origin port code (doesn't match a known port)",
    "dest_code_edit_dist": "destination port code (doesn't match a known port)",
    "ship_year": "ship year",
    "ship_day_of_year": "ship day-of-year",
}


def explain_record(feats: pd.DataFrame, feature_means: pd.Series, feature_stds: pd.Series,
                    idx: int, top_n: int = 3) -> str:
    row = feats.loc[idx]
    z_scores = ((row - feature_means) / feature_stds.replace(0, 1)).abs()
    top_features = z_scores.sort_values(ascending=False).head(top_n)

    reasons = []
    for feat_name, z in top_features.items():
        if z < 1.5:  # not actually unusual -- don't manufacture a reason
            continue
        label = FEATURE_LABELS.get(feat_name, feat_name)
        reasons.append(f"{label} is {z:.1f} standard deviations from typical")

    if not reasons:
        return "No individual field stands out strongly -- flagged on a subtle combination."
    return "; ".join(reasons)


def run():
    df = generate_dataset()
    feats = build_features(df)

    X_scaler = StandardScaler()
    X = X_scaler.fit_transform(feats.values)

    model = IsolationForest(n_estimators=300, contamination=0.08, random_state=42)
    model.fit(X)
    df["pred_ml"] = model.predict(X) == -1
    df["pred_rule"] = rule_based_baseline(feats)

    feature_means = feats.mean()
    feature_stds = feats.std()

    print("=" * 78)
    print("SAMPLE EXPLANATIONS -- what you'd actually show a reviewer or client")
    print("=" * 78)

    # Show one example per anomaly type that the ML model actually flagged
    for atype in ["port_code_is_country_code", "bad_date", "impossible_weight",
                  "distance_mismatch", "combo_subtle"]:
        matches = df[(df["anomaly_type"] == atype) & (df["pred_ml"])]
        if matches.empty:
            continue
        idx = matches.index[0]
        row = df.loc[idx]
        explanation = explain_record(feats, feature_means, feature_stds, idx)

        print(f"\nShipment {row['shipment_id']}  (true cause: {atype})")
        print(f"  Route: {row['origin_code']} -> {row['destination_code']}, "
              f"{row['distance_km']} km, {row['weight_kg']} kg, ship day {row['ship_day_of_year']}/{row['ship_year']}")
        print(f"  Rule-based flag: {'YES' if row['pred_rule'] else 'no'}")
        print(f"  ML flag: YES")
        print(f"  --> Reviewer message: \"Flagged for review -- {explanation}.\"")

    print("\n" + "-" * 78)
    print("Honest takeaway: for clear single-field problems (bad port code, bad date,")
    print("impossible weight), the explanation is crisp and specific -- basically as")
    print("good as a rule's message. For 'combo_subtle' cases, the explanation is")
    print("necessarily softer ('a few things are somewhat off together') because")
    print("that IS the actual situation -- there is no single field to point at.")
    print("That's a real limitation to say out loud, not something to paper over.")


if __name__ == "__main__":
    run()
