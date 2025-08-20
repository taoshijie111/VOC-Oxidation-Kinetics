import os
import json
import shutil
import glob
import torch
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from dataset import rateDataset
from model import GNNModel
import torch_geometric.data as gdata
import math
from trainer import Trainer
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from utils import CosineWarmupScheduler, add_time_to_log_dir, ensure_dir, seed_all, EnsembleModel # written by tsj
from path import csv_path, fuhao

parser = argparse.ArgumentParser(description='')
parser.add_argument('--react', default='all', type=str, help='reactant', choices=['all', 'Cl', 'OH', 'O3', 'NO3'])
parser.add_argument('--device', default='cuda:0', type=str, help='device')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--epochs', default=500, type=int, help='epochs')
parser.add_argument('--lr_start', default=0.001, type=float, help='learning rate start')
parser.add_argument('--warmup_rate', default=0.1, type=float, help='warmup rate')
parser.add_argument('--seed', default=3407, type=int, help='random seed')
parser.add_argument('--nmr', default='yes', type=str, help='whether to use nmr data', choices=['yes', 'no'])
parser.add_argument('--kfold', default=10, type=int, help='kfold')
parser.add_argument('--num_layers', default=7, type=int, help='num layers of resnet')
parser.add_argument('--heads', default=8, type=int)
parser.add_argument('--hidden_size', default=256, type=int)
parser.add_argument('--test_rate', default=0.1, type=float)
parser.add_argument('--prefix', default='GNNResModel-all-all_data', type=str, help='prefix of log directory')
args = parser.parse_args()

name = add_time_to_log_dir(args.react, args.prefix+'-'+str(args.num_layers)+'-'+str(args.seed)+'-'+str(args.test_rate))
log_dir = f'logs/' + name
ensure_dir(log_dir)

with open(os.path.join(log_dir, 'args.json'), 'w') as f:
    json.dump(vars(args), f, indent=4)

data_csv = os.path.join(csv_path, 'gnn_data_all-reacts_allT.csv')

batch_size = args.batch_size
epochs = args.epochs
lr_start = args.lr_start
react = args.react
nmr = args.nmr
k = args.kfold
num_layers= args.num_layers
heads = args.heads
hidden_size = args.hidden_size
device = args.device
test_rate = args.test_rate
seed_all(args.seed)

df_all = pd.read_csv(data_csv)

if react == "all":
    df_all = df_all[df_all['logK'].notnull()]
else:
    df_all = df_all[(df_all['react'] == react) & df_all['logK'].notnull()]

train_val_df, test_df = train_test_split(df_all, test_size=test_rate, random_state=args.seed, stratify=df_all['react'])
train_dataset = rateDataset("train", train_val_df, root=f'all-{test_rate}-{args.seed}-kf-stratify/{fuhao}train_{react}')
test_dataset = rateDataset("test", test_df, root=f'all-{test_rate}-{args.seed}-kf-stratify/{fuhao}test_{react}')

class_counts = train_val_df['react'].value_counts()
total_count = len(train_val_df)
class_weights = total_count / class_counts
weights = train_val_df['react'].map(class_weights).values

def custom_collate_fn(data_list):
    return gdata.Batch.from_data_list(data_list)


kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=args.seed)
fold_models = []
rmse = []
for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset, train_dataset._df['react'])):
    print(f'fold {fold + 1}')
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)


    sampler = WeightedRandomSampler(weights[train_idx], num_samples=len(weights[train_idx]), replacement=True)

    train_dataloader = DataLoader(train_subset, batch_size, collate_fn=custom_collate_fn, drop_last=True, sampler=sampler)
    val_dataloader = DataLoader(val_subset, batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    model = GNNModel(train_dataset.num_node_features,train_dataset.num_edge_features, train_dataset[0].g.size(-1), train_dataset[0].y.size(-1), num_layers=num_layers, heads=heads, hidden_channels=hidden_size)
    print("Total number of trainable parameters: {:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_start)


    scheduler = CosineWarmupScheduler(optimizer, args.warmup_rate, math.ceil(len(train_subset) / batch_size) * epochs)  # warmup 策略

    trainer = Trainer(model, optimizer, scheduler, device, train_dataloader, val_dataloader, val_dataloader, react, epochs, log_dir, fold+1)

    test_rmse = trainer.train()
    rmse.append(test_rmse)

    fold_models.append(model)

test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=custom_collate_fn)
ensemble_model = EnsembleModel(fold_models)
ensemble_model.eval()
tot_loss = 0
tot_mol = 0
test_labels = []
test_predictions = []
test_smiles = []
test_react = []

trainer = Trainer(ensemble_model, None, None, device, None, None, test_dataloader, react, epochs, log_dir)
results = trainer.test()
test_y = results[1]
pred_y = results[2]
rmse_test = results[3]
test_smiles = results[4]
test_reacts = results[5]

with open(os.path.join(log_dir, f'best_model.txt'), 'w') as f:
    for i in range(len(rmse)):
        f.write(f'{i+1} th model rmse: {rmse[i]} \n')
    f.write(f"\nTest RMSE using best model: {rmse.index(min(rmse)) + 1} with rmse: {min(rmse)}")
import numpy as np
test_y_str = [f'{y:.4f}' for y in test_y]
pred_y_str = [f'{y:.4f}' for y in pred_y]
np.savetxt(os.path.join(log_dir, f'test_pred_{react}.dat'),
           np.column_stack((test_smiles, test_reacts, test_y_str, pred_y_str)), header='test_smiles test_reacts test_y pred_y',
           comments='', fmt='%s')
