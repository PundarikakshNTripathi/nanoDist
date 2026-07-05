import numpy as np
from distributed_trainer.data.distributed_sharder import shard_dataset_across_workers

def test_shard_dataset():
    x = np.arange(10).reshape(5, 2)
    y = np.arange(5).reshape(5, 1)
    
    shards = shard_dataset_across_workers(x, y, num_workers=2)
    assert len(shards) == 2
    
    # np.array_split will give 3 to the first worker, 2 to the second
    assert shards[0][0].shape == (3, 2)
    assert shards[1][0].shape == (2, 2)
