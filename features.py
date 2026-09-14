"""
Feature engineering for the shipment anomaly detector.

Deliberately uses continuous/raw-ish signals rather than pre-computed clean
binary "is_valid" flags -- if we handed the model a perfect is_valid flag,
the model would just be relaying a rule we already wrote, not learning
anything. Instead we give it noisier signals (edit distance to nearest known
code, a distance ratio, weight-per-teu) and let it find what combinations of
those are jointly unusual.
"""

import difflib
import numpy as np
import pandas as pd

from generate_data import PORTS, _haversine_km


def _code_edit_distance_to_nearest_port(code: str) -> float:
    """1 - similarity ratio to the closest known port code. 0 = exact match."""
    best = max(difflib.SequenceMatcher(None, code, known).ratio() for known in PORTS)
    return 1.0 - best


def _gc_distance_or_nan(origin_code: str, dest_code: str) -> float:
    if origin_code not in PORTS or dest_code not in PORTS:
        return np.nan
    o_lat, o_lon, *_ = PORTS[origin_code]
    d_lat, d_lon, *_ = PORTS[dest_code]
    return _haversine_km(o_lat, o_lon, d_lat, d_lon)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)

    feats["distance_km"] = df["distance_km"]
    feats["weight_kg"] = df["weight_kg"]
    feats["teu_count"] = df["teu_count"]
    feats["weight_per_teu"] = df["weight_kg"] / df["teu_count"].clip(lower=1)

    gc_km = df.apply(lambda r: _gc_distance_or_nan(r["origin_code"], r["destination_code"]), axis=1)
    # when a code doesn't resolve to a known port at all, distance_ratio can't
    # be computed -- fill with a large sentinel so it reads as "unverifiable /
    # suspicious" rather than silently imputing a plausible-looking value
    feats["distance_ratio"] = np.where(gc_km.notna() & (gc_km > 0), df["distance_km"] / gc_km, 9.99)

    feats["origin_code_edit_dist"] = df["origin_code"].apply(_code_edit_distance_to_nearest_port)
    feats["dest_code_edit_dist"] = df["destination_code"].apply(_code_edit_distance_to_nearest_port)

    feats["ship_year"] = df["ship_year"]
    feats["ship_day_of_year"] = df["ship_day_of_year"]

    return feats
