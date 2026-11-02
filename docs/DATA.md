# Data and artifact layout

## DDInter

The repository does not redistribute the processed DDInter records. Obtain the
source data under the terms of the DDInter project, construct the predefined
five folds, and place the resulting files under `src/trimnet/data/ddinter/`.

Each split CSV must contain the columns `Drug1_ID`, `Drug2_ID`, `Y`, and
`class`. `Y` is the zero-based severity target used by the classifier. The
`class` field is retained for compatibility with the original MSFCL loader.

## Required assets

| Path | Purpose |
|---|---|
| `ddinter_smiles.csv` | Canonical drug ID and SMILES ordering |
| `drug_ids.npy` | Drug ID order for TrimNet embeddings |
| `drug_emb_trimnet{fold}.npy` | Fold-specific 128-D structural embeddings |
| `drug_similarity.pkl` | Drug IDs and molecular similarity matrix |
| `molformer_ibm_molformer_xl_both_10pct.npz` | Frozen MoLFormer IDs and embeddings |
| `{fold}/ddi_training1.csv` | Training pairs for one fold |
| `{fold}/ddi_validation1.csv` | Validation pairs for one fold |
| `{fold}/ddi_test1.csv` | Test pairs for one fold |

The loaders validate drug membership and reorder features into the canonical
`ddinter_smiles.csv` order. Do not remove these checks.

## Excluded artifacts

Checkpoints, optimizer states, full prediction tables, logits, embeddings,
token caches, metrics, and logs are intentionally excluded from Git.

