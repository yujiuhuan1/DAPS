# DAPS: Drift-Aware Prototype Stabilization for Federated Graph Learning Under Domain Shift

Code for the DAPS Twitch federated graph experiment with the paper's α=5 client allocation.

- Dataset: Twitch, 20 clients (DE: 10; EN, ES, FR, PT, RU: 2 each).
- Method/backbone: DAPS with PMLP-GCN.
- Training: 200 communication rounds, 6 local epochs; SGD (momentum 0.9, weight decay 1e-5).
- DAPS: learning rate 0.030, λ=0.0003, anchor momentum 0.60, warm-up 5, temperatures 0.30/0.75, confidence threshold 0.75.

## Run

```bash
python -m pip install -r requirements.txt
python main_graph.py --dataset fl_twitch --model fggp --backbone pmlp_gcn --communication_epoch 200 --local_epoch 6 --local_lr 0.030 --daps_lambda 0.0003 --daps_momentum 0.60 --daps_warmup 5 --daps_temp 0.30 --daps_conf 0.75 --daps_drift_temp 0.75 --seed 100 --device_id 0 --csv_log
```

Put the six Twitch domain files under `datasets/data/Twitch/<DOMAIN>/raw/<DOMAIN>.npz` (`DE`, `EN`, `ES`, `FR`, `PT`, `RU`). Each file must contain `features`, `target`, and `edges`. Dataset files and generated outputs are not included.
