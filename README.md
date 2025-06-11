
# Entropy-Guided Meta-Initialization Regularization for Few-Shot Text Classification (EGMI)
 
🎉 Our paper "Entropy-Guided Meta-Initialization Regularization for Few-Shot Text Classification (EGMI)" has been accepted for publication in Knowledge-Based Systems (Elsevier, SCI Q1, IF 8.8).


## Requirements

Before running the code, ensure that you have all the necessary Python dependencies installed. The required libraries are listed in the `requirements.txt` file.

To install the dependencies, run:

```bash
pip install -r requirements.txt
```

This will install all the packages and libraries required to run the code.

## Datasets

### Provided Dataset:
- The **HuffPost** dataset is provided in the `./dataset` folder by default.

### Additional Datasets:
- You can also download other datasets for few-shot text classification via the following [link](https://github.com/hccngu/MLADA).

Ensure you place the datasets in the appropriate folder as required by the scripts for proper loading during training.

## Training & Testing

To train and test a BERT model using **EGMI**, follow these steps:

### Training & Testing EGMI:

1. To train and test the model with **EGMI**, run the following command:

```bash
bash scripts/egmi.sh
```

This script will execute the training and evaluation for the **EGMI** method.

### Training Standard MAML:

2. If you want to train the model using the **MAML** method (Meta-learning for few-shot classification), run:

```bash
bash scripts/maml.sh
```

This script will execute the training and evaluation using the standard **MAML** algorithm.

## Summary:

- Install dependencies via `pip install -r requirements.txt`.
- Datasets are available in the `./dataset` folder and additional ones can be downloaded from the provided link.
- For training with **EGMI**, use `bash scripts/egmi.sh`.
- For training with **MAML**, use `bash scripts/maml.sh`.


