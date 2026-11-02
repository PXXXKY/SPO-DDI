# SPO-DDI

本仓库提供论文 **SPO-DDI: Pair-Adaptive Molecular Fusion and Ordinal-Aware
Learning for Drug-Drug Interaction Severity Prediction** 的最终模型源码。

最终模型在 MSFCL 骨干网络上加入三项设计：

1. 冻结的 MoLFormer 分子语义特征；
2. 面向具体药物对的 PairGate 残差融合；
3. 三分类与有序学习的联合训练和推断。

公开内容仅包括最终模型及其必要的分子特征处理代码，不包含实验输出、对比模型实现
或其他模型变体。

## 环境安装

```bash
conda env create -f environment.yml
conda activate spo-ddi
```

## 数据准备

按照 [`docs/DATA.md`](docs/DATA.md) 将处理后的 DDInter 文件放入
`src/trimnet/data/ddinter/`。数据集、预训练权重、检查点、缓存和运行输出不会被
Git 跟踪。

## 运行最终模型

单折训练：

```bash
cd src
python main_molformer_pairgate_ordinal.py --dataset ddinter --zhongzi 0
```

五折训练：

```bash
cd src
bash run_pairgate_ordinal_5fold.sh
```

## 代码检查

```bash
python -m compileall -q src tests
python tests/test_core_components.py
```

引用信息见 [`CITATION.cff`](CITATION.cff)，第三方来源和许可说明见
[`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md)。

