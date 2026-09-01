# ChildLang Mandarin Tone Recognition

This project fine-tunes a **wav2vec 2.0** model using Fairseq to evaluate Mandarin speech, with particular focus on **tone recognition in child speech**.

## Setup

The project assumes the following environment:

* Python 3.9
* Fairseq
* PyTorch
* `pypinyin`
* ChildLang dataset
* Pretrained wav2vec 2.0 model

The project uses the existing ChildLang Python environment:

```bash
source /usr/local/python/venv/childlang/bin/activate
```

The main project directory is:

```text
/projects/assigned/ChildLang
```

## Directory Structure

```text
finetune_fairseq/
├── configs/       # Experiment configuration files
├── manifest/      # Generated speech manifests
├── scripts/       # Manifest, training, and testing scripts
├── outputs/       # Model checkpoints
├── results/       # Evaluation results
└── logs/          # Experiment logs
```

The project also uses:

```text
models/            # Pretrained models
fairseq/           # Fairseq source
```

## Experiments

Experiments are controlled through `.cfg` files. The pipeline supports:

1. No additional training
2. Regular fine-tuning
3. Multi-stage curriculum learning

Curriculum experiments train sequentially through stage directories, using the best checkpoint from each stage as the starting model for the next stage.

The primary evaluation metric is **UER (Unit Error Rate)**--in this case, since the units are tone, this actually represents Tone Error Rate.

## Running

An experiment is run with:

```bash
./run_experiments.sh configs/<experiment>.cfg
```

This generates the required manifest, trains the model when requested, and evaluates the resulting checkpoint.

Individual stages can also be evaluated using `test_manifest.sh`.

## Data

Mandarin characters are converted to Pinyin and then to tone labels (`T1`–`T5`) using `pypinyin`. Fairseq manifests provide the audio paths and corresponding labels used for training and evaluation.

The test set is kept separate from the development data and is used for final evaluation only.
