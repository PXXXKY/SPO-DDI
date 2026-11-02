from pathlib import Path
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)

root = Path(__file__).resolve().parent / "ddinter"
input_file = root / "ddinter_triplets.csv"

data = pd.read_csv(input_file)

required = {"Drug1_ID", "Drug2_ID", "Y", "class"}
missing = required - set(data.columns)
if missing:
    raise RuntimeError(f"Missing columns: {missing}")

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

for fold, (train_val_idx, test_idx) in enumerate(
    skf.split(data, data["Y"])
):
    fold_dir = root / str(fold)
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_val = data.iloc[train_val_idx].reset_index(drop=True)
    test = data.iloc[test_idx].reset_index(drop=True)

    train, val = train_test_split(
        train_val,
        test_size=0.125,
        random_state=42 + fold,
        stratify=train_val["Y"],
    )

    train.to_csv(
        fold_dir / "ddi_training1.csv",
        index=False,
    )
    val.to_csv(
        fold_dir / "ddi_validation1.csv",
        index=False,
    )
    test.to_csv(
        fold_dir / "ddi_test1.csv",
        index=False,
    )

    print(f"\nFold {fold}")
    for name, frame in [
        ("train", train),
        ("val", val),
        ("test", test),
    ]:
        print(
            name,
            len(frame),
            frame["Y"].value_counts(
                normalize=True
            ).sort_index().round(4).to_dict(),
        )
