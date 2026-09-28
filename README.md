# DAPS: Drift-Aware Prototype Stabilization for Federated Graph Learning Under Domain Shift

This repository contains only the code path required to run DAPS on the Twitch federated graph benchmark with the paper's alpha=5 client allocation. It excludes dataset files, unrelated dataset and baseline code, result files, logs, and checkpoints.

## Experiment configuration

- Dataset: Twitch, six language domains.
- Alpha=5 client allocation: 2 clients per domain for EN, ES, FR, PT, RU, and 10 clients for DE; 20 clients total.
- Method: DAPS, implemented in `models/fggp.py`.
- Backbone: PMLP-GCN, hidden dimension 128.
- Training: 200 communication rounds, 6 local epochs, full client participation, node-count-weighted FedAvg.
- Optimizer: SGD, momentum 0.9, weight decay 1e-5.
- DAPS settings: local learning rate 0.030, ALPR weight 0.0003, anchor momentum 0.60, warm-up 5, anchor temperature 0.30, confidence threshold 0.75, drift temperature 0.75.
- Reported score: mean accuracy over the final five communication rounds.

These paper settings are the CLI defaults in this release. The command below repeats them explicitly for clarity:

```bash
python main_graph.py --dataset fl_twitch --model fggp --backbone pmlp_gcn --communication_epoch 200 --local_epoch 6 --local_lr 0.030 --daps_lambda 0.0003 --daps_momentum 0.60 --daps_warmup 5 --daps_temp 0.30 --daps_conf 0.75 --daps_drift_temp 0.75 --seed 100 --device_id 0 --csv_log
```

`--device_id` selects the CUDA device when available; the program falls back to CPU otherwise. Change `--seed` to repeat the run. Logs, processed data, and checkpoints are local outputs ignored by Git.

## Data requirements

The Twitch dataset is not included. Place the six domain files at `datasets/data/Twitch/<DOMAIN>/raw/<DOMAIN>.npz`, where `<DOMAIN>` is `DE`, `EN`, `ES`, `FR`, `PT`, or `RU`. Each file must contain `features`, `target`, and `edges` arrays as read by `datasets/twitch.py`. Obtain the data from its original source and follow its terms.

## Environment

Use Python 3.10 or 3.11 with a PyTorch, torchvision, and PyTorch Geometric combination matching the local CPU/CUDA runtime. The compiled PyG extensions `torch-scatter` and `torch-sparse` must match that combination. Install the listed dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Code map

- `main_graph.py`: Twitch-only experiment entry point.
- `datasets/twitch.py` and `datasets/utils/splitter.py`: Twitch loading and client partitioning.
- `models/fggp.py`: DAPS model.
- `backbone/gnn/pmlp.py`: PMLP-GCN backbone and graph augmentation.
- `utils/`: training, FINCH clustering, model utilities, and local CSV logging.

The training code has no application authentication and requires no API key or token. GitHub repository access is managed outside the experiment runtime by the hosting integration.

