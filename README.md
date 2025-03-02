# Entropy-Guided Meta-Initialization Regularization for Few-Shot Text Classification (EGMI)

This code is based on the implementations of [AMGS](https://github.com/Tianyi-Lei/Adaptive-Meta-learner-via-Gradient-Similarity-for-Few-shot-Text-Classification).

## Requirements


```setup
pip install -r requirements.txt
```

## Datasets

We provide the huffpost dataset in ./dataset by default. You can download other datasets through this [link](https://github.com/hccngu/MLADA)

## Training & Test

To train and test the Bert model with **EGMI** in the paper, run this command:

```
bash scripts/egmi.sh
```
If you want to train standard **MAML**, run this command:

```
bash scripts/maml.sh
```