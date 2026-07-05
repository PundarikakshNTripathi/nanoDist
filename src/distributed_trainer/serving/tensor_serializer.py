import base64
import numpy as np

class TensorSerializer:
    @staticmethod
    def serialize(arr: np.ndarray) -> dict:
        """Serialize a numpy array into a JSON-friendly dict."""
        return {
            "dtype": str(arr.dtype),
            "shape": arr.shape,
            "data": base64.b64encode(arr.tobytes()).decode('utf-8')
        }

    @staticmethod
    def deserialize(data: dict) -> np.ndarray:
        """Deserialize a dict back into a numpy array."""
        arr_bytes = base64.b64decode(data["data"])
        arr = np.frombuffer(arr_bytes, dtype=np.dtype(data["dtype"]))
        return arr.reshape(data["shape"])
