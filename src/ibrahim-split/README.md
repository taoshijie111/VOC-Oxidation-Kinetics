# Reference-partition benchmark (partition of the baseline study, ref 11)

This folder holds the model that was **retrained on the train/test partition published with the
reference kinetic dataset** used for the two baselines of the manuscript (Al Ibrahim & Farooq,
referred to as Ref 1 and Ref 2 in Fig. 3a, ref 11 in the text), together with its predictions,
metrics and figure.

## Why this experiment exists

The two baselines in Fig. 3a are **literature values**, obtained by the authors of the reference
study on **their own molecule-level partition** of the underlying database, whereas the GNN results
reported in the manuscript were obtained with the oxidant-stratified split of the individual
measurements (`src/all-0.1-42-kf-stratify/`). The original comparison therefore mixes two
different data partitions.

To make the comparison like-for-like, the unchanged model, hyperparameters, training schedule and
ensembling procedure were retrained **on the exact partition of the reference study**. Only the
partition differs.

## The partition

Two published SMILES lists define the split at the **molecule** level — no molecule, and therefore
no temperature measurement of a molecule, appears on both sides:

| file | content |
|---|---|
| `ibrahim_train_smiles.csv` | 1,270 molecules (training side) |
| `ibrahim_test_smiles.csv` | 317 molecules (test side) |

The two lists are disjoint and together cover all 1,587 molecules of this dataset, which yields
5,027 measurements:

| side | molecules | measurements | share |
|---|---|---|---|
| train | 1,270 | 3,965 | 78.9% |
| test | 317 | 1,062 | 21.1% |

Test side by oxidant: OH 753, Cl 166, NO&#8323; 77, O&#8323; 66; temperature range 50–1367 K.

`src/ibrahim-42-kf-group/` contains the processed `rateDataset` files built from this partition, and
`src/logs/GNNResModel-all-all_data-IBRAHIM-7-42-0.1_all_2026-10-09_09-09-22/split_info.csv` records
the side assigned to every measurement.

## Training configuration

Identical to the main run (`src/logs/GNNResModel-all-all_data-7-42-0.1_all_2024-07-02_12-00-18/`,
prefix `GNNResModel-all-all_data`): reactant `all`, 7 residual layers, 8 attention heads, hidden
size 256, batch size 32, AdamW with `lr = 1e-3` and 10% warm-up, 500 epochs, seed 42, 10-fold
cross-validation with a 10-model ensemble.

Two things change with respect to `src/train.py`, both concerning the split only:

```python
# 1. outer train/test split: published molecule lists instead of a stratified 90:10 split
train_val_df = df_all[df_all['smiles'].isin(train_smiles)].reset_index(drop=True)   # train_smiles from
test_df      = df_all[df_all['smiles'].isin(test_smiles)].reset_index(drop=True)    # ibrahim_*.csv

# 2. inner cross-validation grouped by molecule instead of StratifiedKFold
kf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=args.seed)
```

`model.py`, `trainer.py`, `dataset.py` and `utils.py` are used unmodified.

```
cd src
TQDM_DISABLE=1 python train.py --react all --device cuda:0 --epochs 500 --kfold 10 --seed 42 \
    --prefix GNNResModel-all-all_data-IBRAHIM
```

## Results — 10-model ensemble, 1,062 test measurements of 317 unseen molecules

| metric | value |
|---|---|
| RMSE / MAE | **0.426 / 0.275** |
| R² | **0.948** |
| per-fold test RMSE | 0.424 – 0.908 (mean 0.531 ± 0.132) |
| per-molecule RMSE, mean / median | 0.322 / 0.225 |
| share of measurements with abs. error > 1 log unit | 3.3% |

### Per oxidant, on the same partition

| oxidant | n | R² | this work (ensemble) | this work (best-validation fold) | FFNN (ref 11) | FFNN + QM9 (ref 11) |
|---|---|---|---|---|---|---|
| OH | 753 | 0.86 | **0.383** | 0.369 | 0.43 | 0.38 |
| Cl | 166 | 0.91 | **0.512** | 0.521 | 1.25 | 0.45 |
| NO&#8323; | 77 | 0.90 | **0.546** | 0.562 | 0.93 | 0.70 |
| O&#8323; | 66 | 0.88 | **0.483** | 0.541 | 0.82 | 0.67 |

With a common partition the model is more accurate than the standard FFNN for **all four**
oxidants, and more accurate than the QM9-enhanced FFNN for NO&#8323; (0.546 vs 0.70) and O&#8323;
(0.483 vs 0.67), essentially equivalent for OH (0.383 vs 0.38) and slightly less accurate for Cl
(0.512 vs 0.45). The single model selected by validation RMSE within each split gives the same
picture (column above).

Per-fold test RMSE: 0.455, 0.483, 0.521, 0.481, 0.470, 0.492, 0.581, 0.491, 0.908, 0.424.

## Files

```
ibrahim-split/
├── README.md
├── ibrahim_train_smiles.csv        # 1,270 training molecules (published with ref 11)
├── ibrahim_test_smiles.csv         #   317 test molecules      (published with ref 11)
├── results/
│   ├── r15_metrics.json            # all metrics above (overall / per oxidant / per fold)
│   ├── r15_test_preds.csv          # per-measurement predictions: ensemble + each of the 10 folds
│   └── r15_per_molecule.csv        # per-molecule RMSE (317 molecules)
└── figures/
    ├── R1-5_fig1_parity_rmse.png   # parity plot + RMSE per oxidant vs the two baselines
    └── R1-5_fig1_parity_rmse.pdf
```

Model checkpoints (10 folds), training logs and the assignment of every measurement to a side:

```
src/ibrahim-42-kf-group/{train,test}_all/processed/                    # cached graphs of the partition
src/logs/GNNResModel-all-all_data-IBRAHIM-7-42-0.1_all_2026-10-09_09-09-22/
    args.json  best_model.txt  split_info.csv
    best_model_all_{1..10}.pth                                         # the 10 fold checkpoints
    loss_data_all_{1..10}.dat  test_pred_all_{1..10}.dat  test_pred_all.dat
```

`best_model.txt` lists the validation RMSE of each fold; `test_pred_all.dat` is the ensemble
prediction of the held-out set as written by the training script.

## Figure

**Figure R1-5-1.** Performance of the model after being retrained on the train/test partition of the
reference study used for the baselines. **(a)** Predicted versus experimental log&#8321;&#8320;k for
the 1,062 test measurements, which cover the 317 molecules that appear nowhere in the training data
(10-model ensemble; overall RMSE = 0.426 log units, R² = 0.948); symbols are coloured by oxidant and
the legend gives the coefficient of determination of each oxidant; the dashed line is 1:1 and the
shaded band spans ±1 log unit. **(b)** RMSE per oxidant for this model and for the two baseline
values reported by the reference study on the same partition; the colour encodes the oxidant and the
fill pattern the method.

## Note on the baseline values

The FFNN and FFNN + QM9 numbers are the values reported by the reference study, as also shown in
Fig. 3a of the manuscript. They are quoted as reported; if that study's tables correspond to a
different subset (for example only records carrying an Arrhenius parameter), the comparison should
carry a footnote.
