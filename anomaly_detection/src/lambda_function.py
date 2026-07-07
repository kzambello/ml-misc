import json
import joblib
import numpy as np
import pandas as pd
import onnxruntime as ort

SCALER_PATH = "scaler.joblib"
ONNX_PATH = "autoencoder.onnx"
BEST_THRESHOLD_PATH = "best_threshold"

scaler = joblib.load(SCALER_PATH)

ort_session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

best_threshold = float(np.loadtxt(BEST_THRESHOLD_PATH))

features = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
            "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
            "V21","V22","V23","V24","V25","V26","V27","V28","Amount"]

def lambda_handler(event, context):

    # Expect:
    #
    # {
    #   "transactions": [
    #      {"Time": ..., "V1": ..., ..., "V28": ..., "Amount": ...},
    #      ...
    #   ]
    # }

    if 'body' in event:
        body = json.loads(event['body'])
        txs = body["transactions"]
    else:
        txs = event["transactions"]

    X = []
    for t in txs:
        row = dict(t)

        row["Time"] = int(row["Time"] / 3600) % 24
        x = [row[f] for f in features]
        X.append(x)

    X = np.asarray(X, dtype=np.float32)
    X = pd.DataFrame(X, columns=features)

    onnx_input = scaler.transform(X).astype(np.float32)
    onnx_output = ort_session.run(None, {input_name: onnx_input})[0]

    mse = np.mean(np.power(onnx_input - onnx_output, 2), axis=1)

    pred = (mse > best_threshold).astype(int)


    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "predictions": pred.tolist(),
            "mse": mse.tolist(),
            "threshold": best_threshold
        })
    }

