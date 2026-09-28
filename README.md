# DAPS on Twitch (alpha = 5)

This repository contains the experiment code needed to run the DAPS method on the Twitch federated graph benchmark under the paper's alpha=5 client protocol. It does not contain the Twitch data, generated experiment outputs, logs, checkpoints, or result tables.

## Scope

- Method: DAPS implementation in `models/fggp.py` (RDPA anchor aggregation and ALPR local regularization).
- Dataset loader: Twitch only.
- Client protocol: the paper's alpha=5 setting; 2 clients for each of EN, ES, FR, PT, RU and 10 clients for DE, for 20 clients total.
- Backbone: PMLP-GCN, hidden dimension 128.
- Training: 200 communication rounds, 6 local epochs per round, all clients participating, node-count-weighted model aggregation.
- Optimizer: SGD, momentum 0.9, weight decay 1e-5.
- Reported score: mean over the final five communication rounds.

The command below explicitly supplies the Twitch alpha=5 hyperparameters reported in `论文9.24/supplementary_material.tex`: local learning rate 0.030, ALPR weight 0.0003, anchor momentum 0.60, warm-up 5 rounds, anchor temperature 0.30, confidence threshold 0.75, and drift temperature 0.75. The source parser defaults differ, so keep these overrides when reproducing the paper setting.

## Environment

Use Python 3.10 or 3.11 and install a PyTorch / torchvision / PyTorch Geometric combination compatible with your CPU or CUDA runtime. PyTorch Geometric's compiled extensions (`torch-scatter` and `torch-sparse`) must match the selected PyTorch and CUDA versions. Then install the remaining packages:

```bash
python -m pip install -r requirements.txt
```

## Data (not included)

Place the six source Twitch domain files at:

```text
datasets/data/Twitch/<DOMAIN>/raw/<DOMAIN>.npz
```

where `<DOMAIN>` is `DE`, `EN`, `ES`, `FR`, `PT`, or `RU`. Each NPZ is expected to contain `features`, `target`, and `edges` arrays, as consumed by `datasets/twitch.py`. The loader creates processed files locally. Obtain the dataset from its original source and follow its terms; this repository does not redistribute it.

## Run the alpha=5 experiment

From the repository root:

```bash
python main_graph.py --dataset fl_twitch --model fggp --backbone pmlp_gcn --communication_epoch 200 --local_epoch 6 --local_lr 0.030 --daps_lambda 0.0003 --daps_momentum 0.60 --daps_warmup 5 --daps_temp 0.30 --daps_conf 0.75 --daps_drift_temp 0.75 --seed 100 --device_id 0 --csv_log
```

`--device_id` selects the CUDA device when CUDA is available; the code falls back to CPU otherwise. Change `--seed` to repeat the run with another seed. CSV logs and checkpoints are written locally to `data/` and `checkpoint/`; both locations are ignored by Git.

## Code map

- `main_graph.py`: experiment entry point and CLI options.
- `datasets/twitch.py`: Twitch loading, split setup, and alpha=5 domain-to-client counts.
- `models/fggp.py`: FGGP base implementation with DAPS components.
- `backbone/`: graph backbones and model support code.
- `utils/`: training, metrics, logging, and FINCH clustering helpers.
- `configs/twitch_alpha5.json`: machine-readable record of the paper configuration.

## Authentication

The experiment has no application-level authentication and requires no API key or access token. GitHub authentication is handled by the GitHub hosting integration and is not part of the training code or runtime configuration.

