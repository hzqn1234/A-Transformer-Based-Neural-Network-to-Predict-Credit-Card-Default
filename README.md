# A Transformer-Based Neural Network to Predict Credit Card Default

This repository contains the public code for the paper experiments on credit card default prediction. It separates the two dataset workflows used in the paper:

- `scripts/amex/`: American Express default prediction workflow.
- `scripts/taiwan/`: Taiwan credit card default workflow.
- `src/credit_default_prediction/`: shared preprocessing, model, metric, and training utilities.

The repository is intended for reproducibility and inspection. It does not include raw datasets, processed data, checkpoints, logs, submissions, or credentials.

## Paper

The corresponding paper is available from MDPI Electronics:
https://www.mdpi.com/2079-9292/15/12/2656

## Data

Use a local data directory outside the Git-tracked code repository. A convenient local layout is:

```text
credit-default-prediction/
  code/   # this repository
  data/   # raw data, processed data, checkpoints, logs, and local environments
```

From `credit-default-prediction/code`, for example:

```bash
export CREDIT_DEFAULT_DATA_DIR=../data
```

Recommended layout:

```text
$CREDIT_DEFAULT_DATA_DIR/
  amex/
    raw/
      train_data.csv
      train_labels.csv
      test_data.csv
      sample_submission.csv
    prepared/
  taiwan/
    prepared_seed0/
```

### AMEX

The AMEX data are from the Kaggle American Express Default Prediction competition. The raw data must be obtained from Kaggle under its terms and placed locally. The official test labels are not publicly available, so full local test-set evaluation requires submitting predictions to Kaggle.

If encoded `nn_series.feather` plus `train_df_count_df.feather` already exist in the AMEX raw directory, `scripts/amex/prepare_data.py` uses them automatically for full preprocessing runs. Otherwise, if denoised `train.feather` and `test.feather` intermediates exist, it uses those before falling back to the raw CSV files. When `--nrows` is set, the script reads from the CSV files for small checks.

### Taiwan

The Taiwan workflow can prepare the public UCI Default of Credit Card Clients dataset through `ucimlrepo`, or from a local CSV copy. The paper uses a 70/30 random split and AUC as the evaluation metric.

## Environment

Create a clean Python 3.10+ environment, then install the repository in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install torch  # choose the wheel matching your CUDA driver/runtime
pip install -e .
```

For GPU training, install a PyTorch build that matches your CUDA driver/runtime before running the training scripts. For example, CUDA 12.8 systems should use a CUDA 12.8-compatible PyTorch wheel rather than the newest default wheel if it targets a newer CUDA runtime.

## Presets

The command-line examples below correspond to the release presets in `configs/`:

- `configs/taiwan_lgbm.yaml`
- `configs/taiwan_transformer.yaml`
- `configs/amex_lgbm_manual.yaml`
- `configs/amex_lgbm_series_oof.yaml`
- `configs/amex_transformer_no_fe.yaml`
- `configs/amex_transformer_with_fe.yaml`

The YAML files document the recommended release settings. The scripts are command-line entry points and do not require the YAML files at runtime.
For release reproduction, follow the README commands or the matching YAML files; script defaults are convenience defaults for command-line use.

## AMEX Workflow

Prepare the AMEX sequence tensors:

```bash
python scripts/amex/prepare_data.py \
  --raw-dir "$CREDIT_DEFAULT_DATA_DIR/amex/raw" \
  --output-dir "$CREDIT_DEFAULT_DATA_DIR/amex/prepared"
```

The paper-compatible AMEX workflow trains four model variants:

1. manual-feature LightGBM;
2. manual plus row-level series-OOF LightGBM;
3. Transformer/GRU on sequence features;
4. Transformer/GRU on sequence plus engineered feature-table features.

Build the tabular feature tables with the row-level LightGBM intermediate:

```bash
python scripts/amex/train_series_lgbm.py \
  --raw-dir "$CREDIT_DEFAULT_DATA_DIR/amex/raw" \
  --output-dir "$CREDIT_DEFAULT_DATA_DIR/runs/amex_series_lgbm"

python scripts/amex/prepare_features.py manual-blocks \
  --raw-dir "$CREDIT_DEFAULT_DATA_DIR/amex/raw" \
  --output-dir "$CREDIT_DEFAULT_DATA_DIR/amex/features"

python scripts/amex/prepare_features.py model-tables \
  --feature-dir "$CREDIT_DEFAULT_DATA_DIR/amex/features" \
  --output-dir "$CREDIT_DEFAULT_DATA_DIR/amex/prepared" \
  --train-labels-file "$CREDIT_DEFAULT_DATA_DIR/amex/raw/train_labels.csv" \
  --row-oof-file "$CREDIT_DEFAULT_DATA_DIR/runs/amex_series_lgbm/oof.csv" \
  --row-submission-file "$CREDIT_DEFAULT_DATA_DIR/runs/amex_series_lgbm/submission.csv"
```

Train the two AMEX LightGBM variants:

```bash
python scripts/amex/train_lgbm.py \
  --data-dir "$CREDIT_DEFAULT_DATA_DIR/amex/prepared" \
  --output-dir "$CREDIT_DEFAULT_DATA_DIR/runs/amex_lgbm_manual" \
  --feature-set manual

python scripts/amex/train_lgbm.py \
  --data-dir "$CREDIT_DEFAULT_DATA_DIR/amex/prepared" \
  --output-dir "$CREDIT_DEFAULT_DATA_DIR/runs/amex_lgbm_series_oof" \
  --feature-set series_oof
```

Train the two AMEX Transformer/GRU variants:

```bash
python scripts/amex/train_transformer.py train \
  --data-dir "$CREDIT_DEFAULT_DATA_DIR/amex/prepared" \
  --output-dir "$CREDIT_DEFAULT_DATA_DIR/runs/amex_transformer_no_fe" \
  --device cuda \
  --epochs 12 \
  --batch-size 256 \
  --learning-rate 0.0000825 \
  --attention-heads 16 \
  --output-dropout 0.05 \
  --positional-encoding none \
  --no-feature-complement \
  --fixed-sequence-length 13

python scripts/amex/train_transformer.py train \
  --data-dir "$CREDIT_DEFAULT_DATA_DIR/amex/prepared" \
  --output-dir "$CREDIT_DEFAULT_DATA_DIR/runs/amex_transformer_with_fe" \
  --device cuda \
  --epochs 12 \
  --batch-size 256 \
  --learning-rate 0.015 \
  --attention-heads 32 \
  --output-dropout 0.1 \
  --positional-encoding sinusoidal \
  --fixed-sequence-length 13 \
  --use-features

python scripts/amex/train_transformer.py predict \
  --data-dir "$CREDIT_DEFAULT_DATA_DIR/amex/prepared" \
  --checkpoint-dir "$CREDIT_DEFAULT_DATA_DIR/runs/amex_transformer_no_fe" \
  --output-file "$CREDIT_DEFAULT_DATA_DIR/runs/amex_transformer_no_fe/submission.csv"

python scripts/amex/train_transformer.py predict \
  --data-dir "$CREDIT_DEFAULT_DATA_DIR/amex/prepared" \
  --checkpoint-dir "$CREDIT_DEFAULT_DATA_DIR/runs/amex_transformer_with_fe" \
  --output-file "$CREDIT_DEFAULT_DATA_DIR/runs/amex_transformer_with_fe/submission.csv"
```

Optionally average the four saved prediction files with:

```bash
python scripts/amex/ensemble.py \
  --prediction "$CREDIT_DEFAULT_DATA_DIR/runs/amex_lgbm_manual/submission.csv" --weight 0.30 \
  --prediction "$CREDIT_DEFAULT_DATA_DIR/runs/amex_lgbm_series_oof/submission.csv" --weight 0.35 \
  --prediction "$CREDIT_DEFAULT_DATA_DIR/runs/amex_transformer_no_fe/submission.csv" --weight 0.15 \
  --prediction "$CREDIT_DEFAULT_DATA_DIR/runs/amex_transformer_with_fe/submission.csv" --weight 0.10 \
  --output-file "$CREDIT_DEFAULT_DATA_DIR/runs/amex_average/submission.csv"
```

## Taiwan Workflow

Prepare Taiwan data:

```bash
python scripts/taiwan/prepare_data.py \
  --output-dir "$CREDIT_DEFAULT_DATA_DIR/taiwan/prepared_seed0" \
  --seed 0
```

Train the transformer model:

```bash
python scripts/taiwan/train_transformer.py \
  --data-dir "$CREDIT_DEFAULT_DATA_DIR/taiwan/prepared_seed0" \
  --output-dir "$CREDIT_DEFAULT_DATA_DIR/runs/taiwan_transformer" \
  --device cuda \
  --use-features \
  --epochs 12 \
  --learning-rate 0.01 \
  --feature-hidden-layers 3 \
  --positional-encoding none \
  --gru-pooling last_output \
  --feature-complement \
  --fixed-sequence-length 13 \
  --reload-best-for-oof \
  --optimizer-weight-decay 0.00000001 \
  --clip-grad-norm 0
```

These Taiwan transformer options reproduce the paper-compatible feature-augmented neural network protocol.

Generate Taiwan transformer test predictions:

```bash
python scripts/taiwan/predict_transformer.py \
  --data-dir "$CREDIT_DEFAULT_DATA_DIR/taiwan/prepared_seed0" \
  --checkpoint-dir "$CREDIT_DEFAULT_DATA_DIR/runs/taiwan_transformer" \
  --output-file "$CREDIT_DEFAULT_DATA_DIR/runs/taiwan_transformer/submission.csv"
```

Train the LightGBM baseline:

```bash
python scripts/taiwan/train_lgbm.py \
  --data-dir "$CREDIT_DEFAULT_DATA_DIR/taiwan/prepared_seed0" \
  --output-dir "$CREDIT_DEFAULT_DATA_DIR/runs/taiwan_lgbm"
```

The Taiwan LightGBM entry point uses the raw tabular Taiwan features and defaults to the DART LightGBM protocol used for the paper's Taiwan baseline.

Evaluate or ensemble saved predictions:

```bash
python scripts/taiwan/evaluate.py \
  --labels "$CREDIT_DEFAULT_DATA_DIR/taiwan/prepared_seed0/test_labels.csv" \
  --predictions "$CREDIT_DEFAULT_DATA_DIR/runs/taiwan_transformer/submission.csv"
```

## Reproducibility Notes

- AMEX test-set evaluation requires Kaggle submission because test labels are withheld.
- The training code writes checkpoints, OOF predictions, summaries, and submissions to `$CREDIT_DEFAULT_DATA_DIR/runs/` or another user-specified output directory. These generated files are ignored by Git.
- Small checks can use AMEX row limits such as `--nrows`, `--max-train-customers`, and `--max-test-customers` where available; Transformer smoke runs can additionally use `--epochs 1` and a small batch size.
- This repository provides paper-compatible workflows and release presets rather than a byte-for-byte copy of the original experimental environment. Small differences can occur because of hardware, CUDA/PyTorch/LightGBM/scikit-learn versions, random seeds, GPU nondeterminism, and Kaggle evaluation. The release presets were chosen to keep the code maintainable while reproducing the paper's main behavior and conclusions.

## Acknowledgements

The AMEX workflow was informed by public Kaggle community solutions for the American Express Default Prediction competition. The code in this repository has been organized around the model and reproducibility workflow used for the paper.
