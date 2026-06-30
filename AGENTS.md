# XLinear — Lightweight MLP Time-Series Forecasting (AAAI 2026)

PyTorch implementation of XLinear, a lightweight MLP-based model for long-term time series forecasting with exogenous inputs.

## Project

- **Paper**: AAAI 2026 — XLinear: A Lightweight and Accurate MLP-Based Model for Long-Term Time Series Forecasting with Exogenous Inputs
- **Stack**: Python + PyTorch 2.6.0
- **Entry point**: `run_longExp.py` — argparse-driven experiment runner
- **Models**: XLinear (base, `models/XLinear.py`), XLinear-ES (`models/XLinear_ES.py`), XLinear-GT (`models/XLinear_GT.py`), XLinear_FFT (`models/XLinear_FFT.py` — complex-valued frequency domain variant), TimesNet (`models/TimesNet.py`)

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run an experiment via shell script
bash script/multi_forcasting/etth1.sh

# Run directly
python -u run_longExp.py \
    --is_training 1 \
    --model XLinear \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --enc_in 7 \
    --d_model 128 \
    --t_ff 512 \
    --c_ff 7 \
    --des 'Exp'

# Test-only (load checkpoint)
python -u run_longExp.py --is_training 0 --model XLinear --data ETTh1 ...
```

Available script categories under `script/`:
- `multi_forcasting/` — multivariate forecasting (ETTh1, ETTh2, ETTm1, ETTm2, weather, electricity, traffic)
- `forcast_exogenous/` — univariate with exogenous inputs (crop, DO, GTD, electricity_ms, etc.)
- `ablation/` — channel/temporal gating ablations
- `512_multi/` — multivariate with 512 lookback window
- `model_input_vary/` — varying input lengths (48–336)
- `pems/` — PeMS traffic datasets

## Architecture

| Module | Path | Role |
|---|---|---|
| `run_longExp.py` | root | Argparse entry point, experiment loop (train → test → predict) |
| `exp/exp_main.py` | `exp/` | `Exp_Main` — build model, train/val/test/predict loops, metric logging |
| `exp/exp_basic.py` | `exp/` | Base experiment class (sets device, model, data) |
| `models/XLinear.py` | `models/` | `Model` class with `Forcast_multi` / `Forcast_with_exogenous` backbones; dual gating (temporal + variable) with sigmoid |
| `models/XLinear_FFT.py` | `models/` | FFT-based variant using `ComplexLinear` + frequency-domain gating |
| `models/XLinear_ES.py` | `models/` | Exogenous skip connection variant |
| `models/XLinear_GT.py` | `models/` | Ground-truth-informed variant |
| `layers/Conv_Blocks.py` | `layers/` | `Inception_Block_V1` — multi-scale 2D convolutions averaged |
| `layers/Embed.py` | `layers/` | `DataEmbedding` — value + positional + temporal embedding |
| `data_provider/data_factory.py` | `data_provider/` | Dataset/loader factory |
| `data_provider/data_loader.py` | `data_provider/` | `Dataset_Custom` / `Dataset_Pred` loaders (CSV-based) |
| `utils/tools.py` | `utils/` | EarlyStopping, LR scheduler, visual (matplotlib), FLOP counting |
| `utils/metrics.py` | `utils/` | MSE, MAE, NSE, KGE, MAPE |
| `utils/timefeatures.py` | `utils/` | Time feature encoding |

## Conventions

- **Model interface**: Every model in `models/*.py` exports a class `Model(nn.Module)` with `__init__(self, configs)` (argparse.Namespace) and `forward(self, x_enc)` → `[B, pred_len, enc_in]`.
- **Feature modes**: `--features M` (multivariate → multivariate), `S` (uni → uni), `MS` (multi → uni, last channel is target).
- **Normalization**: RevIN-style instance norm inside the model (subtract mean/std, restore at output), controlled by `--usenorm`.
- **Experiment structure**: Training runs `args.itr` times (default 2), each with a unique setting string. Validation checks every epoch with early stopping (`--patience`).
- **Optimizer**: Adam (`lr=0.0001` default), ReduceLROnPlateau scheduler (`--lradj type3`). Loss: MSE.
- **Scripts**: Shell scripts under `script/` organize hyperparameters per dataset + prediction length.
- **Naming**: Snake_case for files/variables. Model classes named `Model`. Experiment setting format: `{model_id}_{model}_{data}_ft{features}_sl{seq_len}_ll{label_len}_pl{pred_len}_...`.

## Notes

<!-- Quick-add notes below this line -->
