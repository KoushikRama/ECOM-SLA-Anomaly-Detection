import requests

url = "http://localhost:8080/predict"

payload = [{
    "timestamp": "2025-01-01 10:00:00",
    "operation": "browse_products",
    "success_vol": 8000,
    "fail_vol": 300,
    "success_rt_avg": 150,
    "fail_rt_avg": 120
}]

res = requests.post(url, json=payload)

print(res.json())