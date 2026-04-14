import requests
import pandas as pd


# =========================================
# CONFIG
# =========================================
API_URL = "https://wlrih2v7gb.execute-api.us-east-1.amazonaws.com/prod/predict"


# =========================================
# SAMPLE PAYLOAD (🔥 THIS IS WHAT YOU WANT)
# =========================================
def get_sample_payload():
    return [
        {
            "timestamp": "2025-04-01 10:00:00",
            "operation": "browse_products",
            "success_vol": 8000,
            "success_rt_avg": 150.0,
            "fail_vol": 300,
            "fail_rt_avg": 120.0,
            "is_anomaly": False,
            "anomaly_type": None
        },
        {
            "timestamp": "2025-04-01 11:00:00",
            "operation": "checkout",
            "success_vol": 5000,
            "success_rt_avg": 180.0,
            "fail_vol": 600,
            "fail_rt_avg": 200.0,
            "is_anomaly": True,
            "anomaly_type": "traffic_spike"
        }
    ]


# =========================================
# API TEST FUNCTION
# =========================================
def test_api():

    payload = get_sample_payload()

    print("\n📤 Sending Payload:")
    print(payload)

    try:
        response = requests.post(API_URL, json=payload)
    except Exception as e:
        print("❌ Request failed:", e)
        return

    print("\n📥 Status Code:", response.status_code)

    if response.status_code != 200:
        print("❌ API Error:")
        print(response.text)
        return

    data = response.json()

    print("\n📥 Raw Response:")
    print(data)

    # 🔥 SAFE PARSING
    if isinstance(data, dict):
        data = [data]

    df = pd.DataFrame(data)

    print("\n📊 Parsed DataFrame:")
    print(df.head())

    print("\n📊 Columns:", df.columns.tolist())

    # ----------------------------
    # ANOMALY CHECK
    # ----------------------------
    if "Status" in df.columns:
        anomalies = df[df["Status"] != "Normal ✅"]
        print(f"\n🚨 Detected {len(anomalies)} anomalies")

    print("\n✅ API Test Completed")


# =========================================
# RUN
# =========================================
if __name__ == "__main__":
    test_api()