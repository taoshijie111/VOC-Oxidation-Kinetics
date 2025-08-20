import os
import json
import shutil
import glob
import torch
import numpy as np
import pandas as pd
import argparse
from dataset import rateDataset
from model import GNNModel
import torch_geometric.data as gdata
import math
from trainer import Trainer
from torch.utils.data import DataLoader
from utils import plotUtils, add_time_to_log_dir, ensure_dir, seed_all, EnsembleModel # written by tsj
from path import csv_path, fuhao
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser(description='')
parser.add_argument('--ckpt', default='GNNResModel-all-all_data-7-42-0.1_all_2024-07-02_12-00-18', type=str, help='checkpoint path')
parser.add_argument('--react', default='all', type=str, help='reactant', choices=['all', 'Cl', 'OH', 'O3', 'NO3'])
parser.add_argument('--device', default='cuda:0', type=str, help='device')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')  
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--prefix', default='custom-single-explainer-group', type=str, help='prefix of log directory')
args = parser.parse_args()

name = add_time_to_log_dir(args.react, args.prefix)
log_dir = f'logs/' + name
ensure_dir(log_dir)

with open(os.path.join(log_dir, 'args.json'), 'w') as f:
    json.dump(vars(args), f, indent=4)

ckpt = args.ckpt
batch_size = args.batch_size
react = args.react
device = args.device
epochs=1
seed = args.seed
seed_all(args.seed)

test_df = None

test_dataset = rateDataset("test", test_df, root=f'all-0.1-{args.seed}-kf-stratify/{fuhao}test_{react}')
train_dataset = rateDataset("train", test_df, root=f'all-0.1-{args.seed}-kf-stratify/{fuhao}train_{react}')

def custom_collate_fn(data_list):
    return gdata.Batch.from_data_list(data_list)


test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=custom_collate_fn)
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False, collate_fn=custom_collate_fn)

fold_models = []
ckpt_path = os.path.join('logs', ckpt)
with open(os.path.join(ckpt_path, 'args.json'), 'r') as f:
    ckpt_args = json.load(f)

# model_files = glob.glob(os.path.join(ckpt_path, '*.pth'))
# for model_file in model_files:
#     model = GNNModel(test_dataset.num_node_features,test_dataset.num_edge_features, test_dataset[0].g.size(-1), test_dataset[0].y.size(-1), num_layers=ckpt_args['num_layers'], heads=ckpt_args['heads'], hidden_channels=ckpt_args['hidden_size'])
#     model.load_state_dict(torch.load(model_file, map_location='cpu'))
#     fold_models.append(model.to(device))
#
# # rmse_files = glob.glob(os.path.join(ckpt_path, 'test_pred_*_?*.dat'))
# # rmse_weight = []
# # for file in rmse_files:
# #     rmse = pd.read_csv(file, sep=' ', header=0, names=['smiles', 'react', 'test_y', 'pred_y'])
# #     rmse_weight.append(np.sqrt(mean_squared_error(rmse['test_y'], rmse['pred_y'])))
# # rmse_weight_norm = [i / sum(rmse_weight) for i in rmse_weight]
#
# ensemble_model = EnsembleModel(fold_models, weights=None)
# ensemble_model.eval()

with open(os.path.join(ckpt_path, 'best_model.txt')) as f:
    lines = f.readlines()
    last_line = lines[-1]
    best_model_id = int(last_line.split(':')[1].split("with")[0])

model_file = os.path.join(ckpt_path, f'best_model_all_{best_model_id}.pth')
ensemble_model = GNNModel(test_dataset.num_node_features,test_dataset.num_edge_features, test_dataset[0].g.size(-1), test_dataset[0].y.size(-1), num_layers=ckpt_args['num_layers'], heads=ckpt_args['heads'], hidden_channels=ckpt_args['hidden_size'])
ensemble_model.load_state_dict(torch.load(model_file, map_location='cpu'))
ensemble_model.eval()

tot_loss = 0
tot_mol = 0
test_labels = []
test_predictions = []
test_smiles = []
test_react = []

# test
trainer = Trainer(ensemble_model, None, None, device, None, None, test_dataloader, react, epochs, log_dir)
results = trainer.group_mask_pred()
test_y = results[0]
pred_y = results[1]
rmse_test = results[2]
test_smiles = results[3]
test_reacts = results[4]
contributions = results[5]
test_y_str = [f'{y:.4f}' for y in test_y]
pred_y_str = [f'{y:.4f}' for y in pred_y]

np.savetxt(os.path.join(log_dir, f'test_pred_{args.react}.dat'),
           np.column_stack((test_smiles, test_reacts, test_y_str, pred_y_str)), header='test_smiles test_reacts test_y pred_y',
           comments='', fmt='%s')

contributions = pd.DataFrame(contributions)
contributions.to_csv(os.path.join(log_dir, f'test_contributions.csv'), index=False)
result = pd.read_csv(os.path.join(log_dir, f'test_pred_{args.react}.dat'), sep=' ')
test_y = result.test_y.values
pred_y = result.pred_y.values
react = result.test_reacts.values
fig, ax = plt.subplots()
ax.text(0.6,0.45,"R2 = " + str(round(r2_score(test_y, pred_y), 3)), fontsize=16, color='blue', transform=ax.transAxes)

colors = {'Cl': 'green', 'NO3': 'blue', 'O3': 'red', 'OH': 'black'}
rmse = {}
for label, group in result.groupby('test_reacts'):
    ax.scatter(group['test_y'], group['pred_y'], label=label, color=colors[label])
    rmse[label] = np.sqrt(mean_squared_error(group['test_y'], group['pred_y']))
ax.legend()
ax.set_xlabel("logK-EXP",fontsize=18)
ax.set_ylabel("logK-GNN",fontsize=18)
plt.savefig(os.path.join(log_dir, 'test_pred.png'), bbox_inches='tight')

with open(os.path.join(log_dir, 'test_rmse.txt'), 'w') as f:
    for label in rmse.keys():
        f.write(label + ', ')
    f.write('\n')
    for label in rmse.keys():
        f.write(str(rmse[label]) + ', ')

trainer = Trainer(ensemble_model, None, None, device, None, None, train_dataloader, react, epochs, log_dir)
results = trainer.group_mask_pred()
train_y = results[0]
pred_y = results[1]
rmse_test = results[2]
train_smiles = results[3]
train_reacts = results[4]
contributions = results[5]
train_y_str = [f'{y:.4f}' for y in train_y]
pred_y_str = [f'{y:.4f}' for y in pred_y]

np.savetxt(os.path.join(log_dir, f'train_pred_{args.react}.dat'),
           np.column_stack((train_smiles, train_reacts, train_y_str, pred_y_str)), header='train_smiles train_reacts train_y pred_y',
           comments='', fmt='%s')

contributions = pd.DataFrame(contributions)
contributions.to_csv(os.path.join(log_dir, f'train_contributions.csv'), index=False)
result = pd.read_csv(os.path.join(log_dir, f'train_pred_{args.react}.dat'), sep=' ')
train_y = result.train_y.values
pred_y = result.pred_y.values
react = result.train_reacts.values
fig, ax = plt.subplots()
ax.text(0.6,0.45,"R2 = " + str(round(r2_score(train_y, train_y), 3)), fontsize=16, color='blue', transform=ax.transAxes)

colors = {'Cl': 'green', 'NO3': 'blue', 'O3': 'red', 'OH': 'black'}
rmse = {}
for label, group in result.groupby('train_reacts'):
    ax.scatter(group['train_y'], group['pred_y'], label=label, color=colors[label])
    rmse[label] = np.sqrt(mean_squared_error(group['train_y'], group['pred_y']))

ax.legend()
ax.set_xlabel("logK-EXP",fontsize=18)
ax.set_ylabel("logK-GNN",fontsize=18)
plt.savefig(os.path.join(log_dir, 'train_pred.png'), bbox_inches='tight')

with open(os.path.join(log_dir, 'train_rmse.txt'), 'w') as f:
    for label in rmse.keys():
        f.write(label + ', ')
    f.write('\n')
    for label in rmse.keys():
        f.write(str(rmse[label]) + ', ')