# Decoding VOC Oxidation Kinetics via Interpretable Graph Neural Networks

This repository contains the implementation code for the paper "Decoding VOC Oxidation Kinetics via Interpretable Graph Neural Networks." 
## Directory Structure

```
src/
├── train.py                    # Main training script with k-fold cross-validation
├── gnn_explainer.py            # Interpretability analysis and visualization
├── model.py                    # GNN model architecture definition
├── dataset.py                  # Dataset processing and graph data conversion
├── trainer.py                  # Training, validation, and testing utilities
├── utils.py                    # Utility functions (plotting, scheduling, ensemble)
├── path.py                     # Path configurations
├── all-0.1-42-kf-stratify/     # Processed dataset files
│   ├── train_all/
│   └── test_all/
└── logs/                       # Training logs and model checkpoints
    ├── GNNResModel-all-all_data-7-42-0.1_all_2024-07-02_12-00-18/
    └── custom-single-explainer-group_all_2025-08-20_11-12-23/
```

## Installation

1. **Create and activate conda environment:**
   ```bash
   conda create -n py39 python=3.9
   conda activate py39
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Model Training

Train the GNN model with k-fold cross-validation:

```bash
python train.py --react all --device cuda:0 
```

#### Training Arguments:
- `--react`: Reactant type (`all`, `Cl`, `OH`, `O3`, `NO3`)
- `--device`: Device for training (`cuda:0`, `cpu`)
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 500)
- `--lr_start`: Initial learning rate (default: 0.001)
- `--warmup_rate`: Warmup rate for scheduler (default: 0.1)
- `--seed`: Random seed for reproducibility (default: 3407)
- `--kfold`: Number of folds for cross-validation (default: 10)
- `--num_layers`: Number of ResNet layers (default: 7)
- `--heads`: Number of attention heads (default: 8)
- `--hidden_size`: Hidden layer size (default: 256)
- `--test_rate`: Test set ratio (default: 0.1)
- `--prefix`: Log directory prefix

### 2. Interpretability Analysis

Analyze model predictions and generate functional group contributions:

```bash
python gnn_explainer.py --ckpt GNNResModel-all-all_data-7-42-0.1_all_2024-07-02_12-00-18 --react all --device cuda:0 --batch_size 1 --seed 42
```


## Example Workflow

1. **Prepare Data**: Ensure CSV data is available at the path specified in `path.py`
2. **Train Model**: Run `train.py` with desired parameters
3. **Analyze Results**: Use `gnn_explainer.py` with the trained model checkpoint
4. **Interpret Outputs**: Review contribution files and visualizations


## Citation

If you use this code in your research, please cite our paper:
```
[Paper citation details to be added]
```
