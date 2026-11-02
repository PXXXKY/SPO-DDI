import os
import pandas as pd
from sklearn.model_selection import KFold, train_test_split


def five_fold_split(input_file, output_dir):
    data = pd.read_csv(input_file)
    print(f"Loaded dataset with {len(data)} samples.")


    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_idx = 0
    for train_val_idx, test_idx in kf.split(data): 
        fold_dir = os.path.join(output_dir, f"{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        
        train_val_data = data.iloc[train_val_idx]
        test_data = data.iloc[test_idx]

        
        test_file = os.path.join(fold_dir, f"ddi_test1.csv")
        test_data.to_csv(test_file, index=False)

        
        train_data, val_data = train_test_split(
            train_val_data, test_size=1 / 8, random_state=42
        )

        
        train_file = os.path.join(fold_dir, f"ddi_training1.csv")
        val_file = os.path.join(fold_dir, f"ddi_validation1.csv")
        train_data.to_csv(train_file, index=False)
        val_data.to_csv(val_file, index=False)

        print(f"Fold {fold_idx}:")
        print(f"  Training set saved to: {train_file} ({len(train_data)} samples)")
        print(f"  Validation set saved to: {val_file} ({len(val_data)} samples)")
        print(f"  Test set saved to: {test_file} ({len(test_data)} samples)")
        fold_idx += 1


input_file = "ddinter/ddinter_triplets.csv"
output_dir = "ddinter"
five_fold_split(input_file, output_dir)
