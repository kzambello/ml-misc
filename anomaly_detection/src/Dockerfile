FROM public.ecr.aws/lambda/python:3.14

RUN dnf -y update && dnf -y install gcc gcc-c++ make && dnf clean all

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -t .

COPY lambda_function.py .
COPY model/scaler.joblib ./scaler.joblib
COPY model/autoencoder.onnx ./autoencoder.onnx
COPY model/best_threshold ./best_threshold

CMD ["lambda_function.lambda_handler"]
