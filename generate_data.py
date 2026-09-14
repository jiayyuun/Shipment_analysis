"""
Synthetic shipment dataset generator with injected, labeled anomalies.

The labels are NEVER passed to the anomaly detector -- they exist purely so
we can measure precision/recall afterwards, the same way you'd inject known
fraud patterns into a test set when you don't have real labeled fraud data.
"""

import numpy as np
import pandas as pd

# A small reference table of real ports: code -> (lat, lon, country, currency)
# Mirrors the structure of the port/airport lookups already used in the
# shipment-validation pipeline (5-char-ish codes + an allowlist).
PORTS = {
    "SGSIN": (1.29, 103.85, "SG", "SGD"),   # Singapore
    "CNSHA": (31.23, 121.47, "CN", "CNY"),  # Shanghai
    "NLRTM": (51.92, 4.48, "NL", "EUR"),    # Rotterdam
    "USLAX": (33.74, -118.27, "US", "USD"), # Los Angeles
    "AEJEA": (25.02, 55.06, "AE", "AED"),   # Jebel Ali
    "DEHAM": (53.55, 9.99, "DE", "EUR"),    # Hamburg
    "KRPUS": (35.10, 129.04, "KR", "KRW"),  # Busan
    "GBFXT": (51.45, 0.72, "GB", "GBP"),    # Felixstowe
    "JPYOK": (35.44, 139.64, "JP", "JPY"),  # Yokohama
    "INNSA": (18.95, 72.95, "IN", "INR"),   # Nhava Sheva
    "MYPKG": (3.00, 101.39, "MY", "MYR"),   # Port Klang
    "TWKHH": (22.61, 120.27, "TW", "TWD"),  # Kaohsiung
}

# Country codes that a careless data-entry step might paste in instead of a
# real port code (the exact confusion described in the shipment-validation
# work: "PAR" instead of Paris's actual port code).
COUNTRY_CODE_CONFUSABLES = ["PAR", "USA", "CHN", "DEU", "GBR", "JPN", "SGP", "KOR"]

CONTAINER_TYPES = {
    "20ft": (18000, 28000),   # plausible laden weight range in kg
    "40ft": (26000, 30480),
    "40hc": (26000, 30480),
}


def _haversine_km(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, atan2
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def generate_dataset(n=1200, anomaly_rate=0.08, seed=13):
    rng = np.random.default_rng(seed)
    port_codes = list(PORTS.keys())
    rows = []

    n_anomalies = int(n * anomaly_rate)
    anomaly_flags = np.array([True] * n_anomalies + [False] * (n - n_anomalies))
    rng.shuffle(anomaly_flags)

    for i in range(n):
        is_anom = anomaly_flags[i]
        anomaly_type = "none"

        origin, dest = rng.choice(port_codes, size=2, replace=False)
        o_lat, o_lon, _, o_cur = PORTS[origin]
        d_lat, d_lon, _, d_cur = PORTS[dest]
        gc_km = _haversine_km(o_lat, o_lon, d_lat, d_lon)

        # realistic sea route is always somewhat longer than great-circle
        # (coastlines, straits, canals) -- typically 1.05x-1.4x
        route_factor = rng.uniform(1.05, 1.4)
        distance_km = gc_km * route_factor

        container_type = rng.choice(list(CONTAINER_TYPES.keys()))
        wmin, wmax = CONTAINER_TYPES[container_type]
        weight_kg = rng.uniform(wmin, wmax)
        teu_count = 1 if container_type == "20ft" else rng.choice([1, 2])
        currency = o_cur
        ship_year = 2025
        ship_day_of_year = int(rng.uniform(1, 365))

        if is_anom:
            anomaly_type = rng.choice([
                "port_code_is_country_code",
                "distance_mismatch",
                "impossible_weight",
                "bad_date",
                "combo_subtle",
            ])

            if anomaly_type == "port_code_is_country_code":
                if rng.random() < 0.5:
                    origin = rng.choice(COUNTRY_CODE_CONFUSABLES)
                else:
                    dest = rng.choice(COUNTRY_CODE_CONFUSABLES)

            elif anomaly_type == "distance_mismatch":
                # declared distance wildly inconsistent with the real route
                distance_km = gc_km * rng.uniform(3.0, 8.0)

            elif anomaly_type == "impossible_weight":
                # e.g. a 20ft container "weighing" as much as a 40ft x5
                weight_kg = wmax * rng.uniform(3.0, 6.0)

            elif anomaly_type == "bad_date":
                ship_year = int(rng.choice([1999, 2041]))
                ship_day_of_year = int(rng.uniform(1, 365))

            elif anomaly_type == "combo_subtle":
                # each factor only mildly off on its own -- this is the case
                # a single hardcoded threshold on any one field would miss,
                # but which is jointly unusual across several fields at once
                distance_km = gc_km * rng.uniform(1.6, 2.0)       # mildly high
                weight_kg = wmax * rng.uniform(1.15, 1.35)        # mildly high
                ship_day_of_year = int(rng.uniform(360, 365))     # year-end edge

        rows.append({
            "shipment_id": f"SHP{i:05d}",
            "origin_code": origin,
            "destination_code": dest,
            "distance_km": round(distance_km, 1),
            "container_type": container_type,
            "weight_kg": round(weight_kg, 1),
            "teu_count": teu_count,
            "currency": currency,
            "ship_year": ship_year,
            "ship_day_of_year": ship_day_of_year,
            "is_anomaly": is_anom,          # held out from the model
            "anomaly_type": anomaly_type,   # held out from the model
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("shipments.csv", index=False)
    print(df["anomaly_type"].value_counts())
    print(f"\nTotal: {len(df)} shipments, {df['is_anomaly'].sum()} labeled anomalies "
          f"({df['is_anomaly'].mean():.1%})")
