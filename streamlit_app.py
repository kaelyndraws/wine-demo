"""
Streamlit UI for the Wine Quality classifier hosted on SageMaker.

Reads endpoint name and region from environment variables.
boto3 picks up AWS credentials from:
  - the EC2 instance profile (when running on EC2 with LabInstanceProfile), OR
  - ~/.aws/credentials (when running locally)
"""

import json
import os

import boto3
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError


ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "wine-xgb-endpoint")
REGION = os.environ.get("AWS_REGION", "us-east-1")


@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)


def invoke_endpoint(features: list[float]) -> dict:
    runtime = get_runtime_client()
    payload = {"instances": [features]}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


st.set_page_config(page_title="Wine Quality Classifier")
st.title("Wine Quality Classifier")

st.markdown(
    "Predicts wine quality as **low**, **medium**, or **high** from "
    "11 physico-chemical features."
)

col1, col2 = st.columns(2)
with col1:
    fixed_acidity = st.number_input("Fixed acidity", value=7.4, step=0.1)
    volatile_acidity = st.number_input("Volatile acidity", value=0.7, step=0.01)
    citric_acid = st.number_input("Citric acid", value=0.0, step=0.01)
    residual_sugar = st.number_input("Residual sugar", value=1.9, step=0.1)
    chlorides = st.number_input("Chlorides", value=0.076, step=0.001, format="%.3f")
    free_so2 = st.number_input("Free sulfur dioxide", value=11.0, step=1.0)
with col2:
    total_so2 = st.number_input("Total sulfur dioxide", value=34.0, step=1.0)
    density = st.number_input("Density", value=0.9978, step=0.0001, format="%.4f")
    pH = st.number_input("pH", value=3.51, step=0.01)
    sulphates = st.number_input("Sulphates", value=0.56, step=0.01)
    alcohol = st.number_input("Alcohol", value=9.4, step=0.1)

if st.button("Predict", type="primary"):
    features = [
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_so2, total_so2, density, pH, sulphates, alcohol,
    ]
    try:
        result = invoke_endpoint(features)
    except NoCredentialsError:
        st.error(
            "No AWS credentials found. If running on EC2, attach LabInstanceProfile. "
            "If running locally, configure ~/.aws/credentials."
        )
    except ClientError as e:
        st.error(f"AWS error: {e.response['Error'].get('Message', str(e))}")
    else:
        label = result["labels"][0]
        probs = result["probabilities"][0]

        st.success(f"Predicted quality: **{label}**")
        st.write("Class probabilities:")
        st.bar_chart({"probability": probs})
