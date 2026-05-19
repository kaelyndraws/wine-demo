"""SageMaker inference entry point.

Generic sklearn Pipeline loader. Works for any pipeline saved as model.joblib,
regardless of which classifier is inside.

Four functions form the SageMaker contract:
    model_fn   - load model from disk (called once per container)
    input_fn   - parse request body (called per request)
    predict_fn - run inference (called per request)
    output_fn  - serialize response (called per request)
"""

import json
import os

import joblib
import numpy as np
import pandas as pd


JSON_CONTENT_TYPE = "application/json"
CSV_CONTENT_TYPE = "text/csv"

CLASS_NAMES = ["low", "medium", "high"]

FEATURE_NAMES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def model_fn(model_dir: str):
    """Load the pickled sklearn Pipeline from model.joblib."""
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(request_body, request_content_type: str) -> pd.DataFrame:
    """Parse incoming request body into a DataFrame.

    Accepts JSON: {"instances": [[...11 floats...], ...]}
    Or CSV: one row per instance, 11 comma-separated values.
    """
    if request_content_type == JSON_CONTENT_TYPE:
        payload = json.loads(request_body)
        instances = payload["instances"]
        return pd.DataFrame(instances, columns=FEATURE_NAMES)

    if request_content_type == CSV_CONTENT_TYPE:
        if isinstance(request_body, (bytes, bytearray)):
            request_body = request_body.decode("utf-8")
        rows = [
            [float(x) for x in line.split(",")]
            for line in request_body.strip().splitlines()
            if line.strip()
        ]
        return pd.DataFrame(rows, columns=FEATURE_NAMES)

    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data: pd.DataFrame, pipeline) -> dict:
    """Run inference. Returns probabilities, predicted class IDs, and labels."""
    probs = pipeline.predict_proba(input_data)
    class_ids = np.argmax(probs, axis=1)
    labels = [CLASS_NAMES[int(i)] for i in class_ids]
    return {
        "probabilities": probs.tolist(),
        "predictions": class_ids.tolist(),
        "labels": labels,
    }


def output_fn(prediction: dict, accept_content_type: str):
    """Serialize the prediction dict for the response body."""
    if accept_content_type == JSON_CONTENT_TYPE:
        return json.dumps(prediction), JSON_CONTENT_TYPE
    raise ValueError(f"Unsupported accept type: {accept_content_type}")
