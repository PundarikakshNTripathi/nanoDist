import numpy as np
from distributed_trainer.core.memory_manager import compute_param_memory_bytes

def test_compute_param_memory_bytes():
    params = {
        'W1': np.zeros((10, 10), dtype=np.float32),
        'b1': np.zeros(10, dtype=np.float32)
    }
    # 10x10x4 = 400 bytes, 10x4 = 40 bytes. Total 440 bytes.
    total = compute_param_memory_bytes(params)
    assert total == 440
