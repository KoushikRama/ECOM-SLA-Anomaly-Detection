import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.common.load_main_config import load_data_config, get_data_filepath
from src.data.generate_data import add_noise, apply_load_dynamics, apply_hourly_rules, random_in_range, compute_hour_factor


# =========================================
# ANOMALY FUNCTION
# =========================================
def inject_anomaly(values, config):

    anomaly_cfg = config["anomaly_injection"]

    # ✅ probability check INSIDE function
    if np.random.rand() > anomaly_cfg["probability"]:
        return values, False, None

    rules = anomaly_cfg["rules"]

    anomaly_type = np.random.choice(list(rules.keys()))
    rule = rules[anomaly_type]

    targets = rule["targets"]
    factor_min, factor_max = rule["factor_range"]

    # ✅ realistic intensity variation
    if np.random.rand() < 0.7:
        factor = np.random.uniform(factor_min, (factor_min + factor_max) / 2)
    else:
        factor = np.random.uniform(factor_min, factor_max)

    # ----------------------------
    # APPLY ANOMALY
    # ----------------------------
    for t in targets:

        if t not in values:
            continue

        # latency → only increase
        if t in ["success_rt_avg", "fail_rt_avg"]:
            values[t] *= factor

        # success volume (can go up or down depending on factor)
        elif t == "success_vol":
            values[t] *= factor

        # failure volume → increase
        elif t == "fail_vol":
            values[t] *= factor

    return values, True, anomaly_type


# =========================================
# MAIN GENERATOR
# =========================================
def generate_test_data(start_date, hours, config, seed=None):

    if seed is not None:
        np.random.seed(seed)

    RANDOMNESS = config["noise"]
    operations_config = config["operations"]
    hourly_rules = config["hourly_rules"]

    data = []
    current = start_date

    for _ in range(hours):
        hour = current.hour
        hour_factor = compute_hour_factor(hour, config)

        for op, cfg in operations_config.items():

            base_vol = cfg["success_vol"]

            # ----------------------------
            # TRAFFIC
            # ----------------------------
            success_vol = base_vol * hour_factor

            # ----------------------------
            # NOISE
            # ----------------------------
            values = {
                "success_vol": add_noise(success_vol, RANDOMNESS["success_vol_std_pct"]),
                "fail_vol": add_noise(cfg["fail_vol"], RANDOMNESS["fail_vol_std_pct"]),
                "success_rt_avg": random_in_range(*cfg["success_rt_avg"], RANDOMNESS["latency_std_pct"]),
                "fail_rt_avg": random_in_range(*cfg["fail_rt_avg"], RANDOMNESS["latency_std_pct"])
            }

            # ----------------------------
            # SYSTEM DYNAMICS
            # ----------------------------
            values = apply_load_dynamics(op, values, cfg, config)
            values = apply_hourly_rules(op, hour, values, hourly_rules)

            # ----------------------------
            # ANOMALY INJECTION
            # ----------------------------
            values, is_anomaly, anomaly_type = inject_anomaly(values, config)

            # ----------------------------
            # FINAL OUTPUT
            # ----------------------------
            data.append({
                "timestamp": current,
                "operation": op,
                "success_vol": int(values["success_vol"]),
                "success_rt_avg": round(values["success_rt_avg"], 3),
                "fail_vol": int(values["fail_vol"]),
                "fail_rt_avg": round(values["fail_rt_avg"], 3),
                "is_anomaly": is_anomaly,
                "anomaly_type": anomaly_type
            })

        current += timedelta(hours=1)

    return pd.DataFrame(data)


# =========================================
# SAVE
# =========================================
def save_test_data(df):
    filepath = get_data_filepath().replace(".csv", "_test.csv")
    df.to_csv(filepath, index=False)
    print(f"Test Data saved to: {filepath}")


# =========================================
# MAIN
# =========================================
def main():

    config = load_data_config()

    df = generate_test_data(
        start_date=datetime(2025, 4, 1),
        hours=24 * 7,
        config=config
    )

    save_test_data(df)

    print(df.head())
    print("\nAnomaly Rate:", round(df["is_anomaly"].mean(), 3))


# =========================================
# RUN
# =========================================
if __name__ == "__main__":
    main()