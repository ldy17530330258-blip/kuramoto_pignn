# Kuramoto-PIGNN v1

Structured prototype for a Kuramoto dynamics physics-informed graph neural network.

## Directory overview
- `configs/`: experiment configuration
- `data_generation/`: graph generation, Kuramoto simulation, dataset building
- `models/`: GNN model definition
- `physics/`: Kuramoto dynamics and residual utilities
- `training/`: losses and training loop
- `scripts/`: runnable entry points
- `outputs/`: checkpoints, logs, figures

## Quick start
Build dataset:
```bash
python scripts/run_build_kuramoto_dataset.py
```

Train model:
```bash
python scripts/run_train_kuramoto_pignn.py
```
