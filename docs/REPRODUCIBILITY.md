# Reproducibility protocol

The final SPO-DDI implementation uses five predefined DDInter folds. Select a
fold with `--zhongzi 0` through `--zhongzi 4`, or use the supplied five-fold
launcher.

The default configuration uses:

- seed 2024;
- 100 epochs;
- batch size 256;
- Adam with learning rate `1e-3` and weight decay `5e-4`;
- nominal label smoothing 0.1;
- ordinal loss weight 0.2;
- nominal/ordinal inference blend 0.85/0.15;
- validation Macro-F1 for checkpoint selection.

Each fold must use an isolated checkpoint directory. Generated checkpoints,
logs, predictions, metrics, and feature caches are ignored by Git and are not
part of this source release.

The data loader aligns TrimNet, similarity, and MoLFormer arrays by drug ID.
Validation and test pairs are not inserted into the training message-passing
graph.

