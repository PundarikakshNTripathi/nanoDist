# nanoDist: Distributed Training Engine in NumPy

nanoDist is a production-grade, distributed machine learning training engine implemented entirely from scratch using pure NumPy. The project demonstrates advanced memory optimization and parallel computing primitives typically abstracted by frameworks like PyTorch or DeepSpeed, scaled down for analytical study and custom infrastructure deployments.

## Architecture and Core Features

The engine implements several state-of-the-art techniques for training massive neural networks under strict memory constraints:

1.  **Distributed Data Parallelism (DDP):** Implements horizontal scaling by sharding datasets across multiple simulated workers.
2.  **Ring All-Reduce Communication:** Custom peer-to-peer network synchronization algorithm implemented via NumPy matrix partitioning to aggregate gradients across distributed nodes without a centralized parameter server bottleneck.
3.  **ZeRO Optimizer (Stage 2 Partitioning):** Memory footprint reduction through distributed optimizer state tracking. First and second momentum tensors (Adam) are sharded across workers, minimizing redundant memory allocation.
4.  **Activation Checkpointing:** Algorithmic trade-off between computation and memory. Intermediate activation matrices are discarded during the forward pass and selectively recomputed during the backward pass to prevent Out-Of-Memory (OOM) errors during deep network backpropagation.
5.  **Mixed Precision Training:** Computational acceleration and memory preservation via strategic downcasting. Forward and backward passes operate in FP16, while optimizer steps safely execute in FP32 using loss scaling to prevent gradient underflow.
6.  **Gradient Accumulation:** Simulation of large global batch sizes across memory-constrained micro-batches.

## Infrastructure and MLOps

The repository adheres to production software engineering standards, integrating a modern MLOps stack:

*   **Dependency Management:** `uv`
*   **Configuration Orchestration:** `hydra-core` (hierarchical YAML configurations with CLI overrides)
*   **Telemetry and Logging:** `wandb` (Weights & Biases) for experiment tracking, `loguru` for structured terminal logging
*   **Hyperparameter Tuning:** `optuna` for automated parameter sweeps
*   **Containerization:** `docker` and `docker-compose` for local distributed node simulation
*   **Continuous Integration:** GitHub Actions triggering `pytest` unit test suites and `ruff` static analysis
*   **API Serving:** `fastapi` and `pydantic` for model serialization and inference serving interfaces

## Repository Structure

*   `src/distributed_trainer/core/`: Contains the fundamental ML math, including forward/backward passes, memory accounting, and the primary orchestration loop.
*   `src/distributed_trainer/distributed/`: Implements the dataset sharder and Ring All-Reduce algorithms.
*   `src/distributed_trainer/optim/`: Houses the Base, standard Adam, and ZeRO-sharded Adam optimizers.
*   `src/distributed_trainer/serving/`: FastAPI application endpoints and NumPy tensor serialization utilities.
*   `conf/`: Hydra configuration definitions.
*   `benchmarks/`: HPC simulation scripts and matplotlib-driven memory optimization analysis tools.
*   `tests/`: Comprehensive unit testing for algebraic integrity and distributed primitive validation.

## Execution

### Local Environment Setup
Install the `uv` package manager, sync dependencies, and install pre-commit hooks:
```bash
make setup
```

### Execution via CLI
Trigger the primary training orchestrator with dynamic Hydra configuration overriding:
```bash
uv run python main.py training.batch_size=128 training.num_workers=4
```

### Benchmarking and Testing
Execute the validation suite or memory profiling simulations:
```bash
make test
uv run python benchmarks/benchmark_memory.py
```
