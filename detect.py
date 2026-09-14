"""
Trains an unsupervised Isolation Forest on shipment features and evaluates it
against the injected ground-truth labels (used ONLY for evaluation, never for
training -- this mirrors the real situation where you don't have labeled
fraud/error data to train on).

Also runs a simple rule-based baseline for comparison, and reports where each
approach wins -- the honest framing is "ML wins on multivariate combinations
no single threshold catches", not "ML is unconditionally better".
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from generate_data import generate_dataset
from features import build_features


def rule_based_baseline(feats: pd.DataFrame) -> np.ndarray:
    """
    A reasonable hand-written validation-rule baseline, in the same spirit as
    the existing shipment-validation pipeline: flag a record if any single
    field crosses an obviously-wrong threshold.
    """
    # this is where we change the model parameters
    flags = (
        (feats["distance_ratio"] > 2.5) #this parameter stands for the the ratio between the row's distance and great circle distance provided by the training dataset 
        | (feats["distance_ratio"] < 0.8)
        | (feats["origin_code_edit_dist"] > 0.4) #this parameter stands for the difference in the number of the characters between the row's port name and the actual name of the port provided by the training dataset
        | (feats["dest_code_edit_dist"] > 0.4) #this parameter stands for the difference in the number of the characters between the row's port name and the actual name of the port provided by the training dataset
        | (feats["weight_per_teu"] > 30480 * 1.5) # ratio of the weight of the containers to the number of containers - why *1.5 
        | (feats["ship_year"] < 2020)
        | (feats["ship_year"] > 2030)
    )
    return flags.values


def run():
    df = generate_dataset()
    feats = build_features(df)
    y_true = df["is_anomaly"].values

    X = StandardScaler().fit_transform(feats.values)

    model = IsolationForest(
        n_estimators=300,
        contamination=0.08,  # matches the injection rate; in practice you'd
                              # tune this against a small labeled validation
                              # slice or business tolerance for false alarms
        random_state=42,
    )
    model.fit(X)

    # decision_function: higher = more normal. Flip sign so higher = more anomalous.
    anomaly_score = -model.decision_function(X)
    y_pred_ml = model.predict(X) == -1  # sklearn: -1 = outlier, 1 = inlier

    y_pred_rule = rule_based_baseline(feats)

    print("=" * 70)
    print("EVALUATION (against held-out injected labels, not used in training)")
    print("=" * 70)

    for name, y_pred in [("Isolation Forest", y_pred_ml), ("Rule-based baseline", y_pred_rule)]:
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"\n{name}:")
        print(f"  Precision: {prec:.3f}   Recall: {rec:.3f}   F1: {f1:.3f}")

    auc = roc_auc_score(y_true, anomaly_score)
    print(f"\nIsolation Forest ROC-AUC (rank quality, threshold-independent): {auc:.3f}")

    # Break down recall BY anomaly type -- this is the honest part: show
    # exactly which anomaly types each method catches and misses, rather
    # than only reporting an aggregate number.
    print("\n" + "-" * 70)
    print("Recall by anomaly type (does each method catch this type?)")
    print("-" * 70)
    df_eval = df.copy()
    df_eval["pred_ml"] = y_pred_ml
    df_eval["pred_rule"] = y_pred_rule
    for atype in df_eval.loc[df_eval["is_anomaly"], "anomaly_type"].unique():
        subset = df_eval[df_eval["anomaly_type"] == atype]
        ml_recall = subset["pred_ml"].mean()
        rule_recall = subset["pred_rule"].mean()
        print(f"  {atype:28s}  ML recall: {ml_recall:.2f}   Rule recall: {rule_recall:.2f}  (n={len(subset)})")

    df_eval.to_csv("evaluation_results.csv", index=False)
    print("\nFull per-record results saved to evaluation_results.csv")

    return df_eval


if __name__ == "__main__":
    run()
