import os
import boto3

BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "eu.amazon.nova-micro-v1:0")
AWS_REGION = os.environ.get("AWS_REGION", "eu-north-1")

client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

def generate(prompt):
    response = client.converse(
        modelId=BEDROCK_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
    )
    return response["output"]["message"]["content"][0]["text"]