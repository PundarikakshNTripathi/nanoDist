from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from distributed_trainer.serving.tensor_serializer import TensorSerializer

app = FastAPI(title="nanoDist Inference Engine")

class InferenceRequest(BaseModel):
    features: list[float]

@app.get("/health")
def health_check():
    return {"status": "ready"}

@app.post("/predict")
def predict(request: InferenceRequest):
    # Core inference stub. In production, load the trained parameters here.
    
    # y_pred, _ = mlp_forward(x, params)
    dummy_pred = np.zeros((1, 10), dtype=np.float32)
    
    return {"prediction": TensorSerializer.serialize(dummy_pred)}
