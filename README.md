# SPO-DDI

Official source code for **SPO-DDI: Pair-Adaptive Molecular Fusion and
Ordinal-Aware Learning for Drug-Drug Interaction Severity Prediction**.

The released model contains three main additions to the MSFCL backbone:

1. frozen MoLFormer representations as a complementary molecular view;
2. PairGate for drug-pair-conditioned residual molecular fusion; and
3. joint nominal and ordinal learning for three-level severity prediction.

This repository contains only the final-model implementation and the code
needed to prepare its molecular features. It does not include experimental
outputs, comparison implementations, or alternative model variants.

## Layout

```text
SPO-DDI/
├── src/
│   ├── main_molformer_pairgate_ordinal.py       # final-model entry point
│   ├── layer_molformer_pairgate_ordinal.py      # network architecture
│   ├── train_molformer_pairgate_ordinal.py      # training and evaluation
│   ├── molecular_fusion.py                      # TrimNet–MoLFormer fusion
│   ├── pair_conditioned_gate_ordinal.py         # PairGate and ordinal head
│   ├── data_preprocess.py                       # DDInter data loader
│   ├── extract_molformer_embeddings.py          # frozen feature extraction
│   ├── run_pairgate_ordinal_5fold.sh            # final five-fold launcher
│   └── trimnet/                                 # structural feature code
├── tests/test_core_components.py
├── environment.yml
└── requirements.txt
```

## Environment

The reference environment uses Linux, Python 3.8, PyTorch 2.1.1, CUDA 12.1,
and PyTorch Geometric 2.4.0.

```bash
conda env create -f environment.yml
conda activate spo-ddi
```

Install CUDA-matched PyTorch Geometric extension wheels if they are not
resolved automatically on your system.

## Data preparation

The code expects processed DDInter files under
`src/trimnet/data/ddinter/`. Required filenames and array schemas are described
in [`docs/DATA.md`](docs/DATA.md). Dataset files, pretrained weights,
checkpoints, caches, and generated outputs are intentionally not tracked.

The MoLFormer feature archive must contain two arrays:

- `drug_ids`: canonical drug identifiers;
- `embeddings`: one frozen MoLFormer vector per drug.

Both TrimNet and MoLFormer features are aligned by drug ID before training.

## Train one fold

Run from `src/`:

```bash
python main_molformer_pairgate_ordinal.py \
  --dataset ddinter \
  --zhongzi 0 \
  --molformer_feature_path trimnet/data/ddinter/molformer_ibm_molformer_xl_both_10pct.npz
```

## Train all five folds

```bash
cd src
bash run_pairgate_ordinal_5fold.sh
```

The launcher accepts `GPU`, `PYTHON`, `EPOCHS`, `BATCH_SIZE`, `RUN_ROOT`, and
`MOLFORMER_FEATURE_PATH` as environment variables.

## Validate the released components

```bash
python -m compileall -q src tests
python tests/test_core_components.py
```

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Update the
author list and paper DOI when the bibliographic record is finalized.

## License

See [`LICENSE`](LICENSE) and [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md).

