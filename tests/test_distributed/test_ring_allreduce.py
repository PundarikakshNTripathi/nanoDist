import numpy as np
from distributed_trainer.distributed.ring_allreduce import ring_all_reduce_mean

def test_ring_all_reduce_mean():
    # 3 workers, array of size 6
    w1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    w2 = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    w3 = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    
    out = ring_all_reduce_mean([w1, w2, w3])
    
    # Expected mean is 2.0 for all elements
    assert np.all(out == 2.0)
