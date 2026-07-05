from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="nanoDist Serving API")

class InferenceRequest(BaseModel):
    features: list[float]

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict(request: InferenceRequest):
    # Stub for model inference. In a real scenario, load the trained parameters
    # and pass request.features through mlp_forward.
    
    x = np.array([request.features], dtype=np.float32)
    # y_pred, _ = mlp_forward(x, loaded_params)
    
    return {"prediction": [0.0] * 10} # Dummy response
