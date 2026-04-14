import boto3
import json

runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

ENDPOINT_NAME = "ecom-sla-anomaly-endpoint-CLI"

payload = [
    {
        "timestamp": "2025-01-01 10:00:00",
        "operation": "browse_products",
        "success_vol": 8000,
        "fail_vol": 300,
        "success_rt_avg": 150,
        "fail_rt_avg": 120
    }
]

response = runtime.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType="application/json",
    Body=json.dumps(payload)
)

result = response["Body"].read().decode()

print("Response:")
print(result)