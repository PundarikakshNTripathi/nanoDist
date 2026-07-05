from distributed_trainer.data.ingestion import make_synthetic_regression_batch
from distributed_trainer.core.layers import init_mlp_params

def test_make_synthetic_regression_batch():
    x, y = make_synthetic_regression_batch(batch_size=10, in_dim=5, out_dim=2)
    assert x.shape == (10, 5)
    assert y.shape == (10, 2)

def test_init_mlp_params():
    params = init_mlp_params(in_dim=10, hidden_dim=20, out_dim=2)
    assert 'W1' in params
    assert params['W1'].shape == (10, 20)
