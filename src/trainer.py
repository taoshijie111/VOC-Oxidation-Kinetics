import os

from tqdm import tqdm
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error
import math
from utils import plotUtils
from rdkit.Chem import RDConfig, AllChem
from rdkit import Chem


def load_functional_groups():
    fg_file = os.path.join(RDConfig.RDDataDir, 'FunctionalGroups.txt')
    functional_groups = {}
    with open(fg_file, 'r') as f:
        for line in f:
            if line.startswith('//') or line.strip() == '':
                continue
            parts = line.strip().split()
            fg_name = parts[0]
            fg_smarts = parts[1]
            functional_groups[fg_name] = fg_smarts
    return functional_groups


class Trainer:
    def __init__(self, model, optimizer, scheduler, device, train_dataloader, val_dataloader, test_dataloader, react,
                 epochs, log_dir, fold=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.react = react
        self.scheduler = scheduler
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.loss_fn = nn.MSELoss()
        self.best_val_loss = float('inf')  
        if fold is not None:
            self.best_model_path = os.path.join(log_dir, f"best_model_{self.react}_{fold}.pth")
        else:
            self.best_model_path = os.path.join(log_dir, f"best_model_{self.react}.pth") 
        self.epochs = epochs
        self.log_dir = log_dir
        self.fold = fold

    def train(self):
        best_model = 0
        train_rmse_list = []
        val_rmse_list = []
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            train_rmse_list.append(np.sqrt(train_loss))
            val_loss = self.val(epoch)
            val_rmse_list.append(np.sqrt(val_loss))
            print(f"\nEpoch {epoch}:")
            print(f"Train RMSE: {np.sqrt(train_loss)}")
            print(f"Validation RMSE: {np.sqrt(val_loss)}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_path)
                best_model = epoch

        self.model.load_state_dict(torch.load(self.best_model_path))
        test_results = self.test()
        test_loss = test_results[0]
        test_y = test_results[1]
        pred_y = test_results[2]
        rmse_test = test_results[3]
        test_smiles = test_results[4]
        test_reacts = test_results[5]
        print(f"\nTest RMSE using best model (epoch{best_model}) : {math.sqrt(test_loss)}")
        plotUtils.plot_test_pred(test_y, pred_y, rmse_test, self.react, self.log_dir, self.fold)
        plotUtils.plot_loss(self.epochs, train_rmse_list, val_rmse_list, self.react, self.log_dir, self.fold)
        test_y_str = [f'{y:.4f}' for y in test_y]
        pred_y_str = [f'{y:.4f}' for y in pred_y]

        if self.fold is not None:
            out_dir = f'test_pred_{self.react}_{self.fold}.dat'
        else:
            out_dir = f'test_pred_{self.react}.dat'

        np.savetxt(os.path.join(self.log_dir, out_dir),
                   np.column_stack((test_smiles, test_reacts, test_y_str, pred_y_str)),
                   header='test_smiles test_reacts test_y pred_y',
                   comments='', fmt='%s')

        return rmse_test

    def train_epoch(self, epoch):
        self.model.train()
        tot_loss = 0
        tot_mol = 0
        with tqdm(total=len(self.train_dataloader)) as pbar:
            pbar.set_description_str(f"train {epoch}")
            for n, batch_data in enumerate(self.train_dataloader):
                batch_data = batch_data.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(batch_data)
                loss = self.loss_fn(pred, batch_data.y.view(-1, 1))
                loss.backward()
                self.optimizer.step()
                self.scheduler.step() 
                num_mols = batch_data.num_graphs
                tot_mol += num_mols
                tot_loss += (loss.item() * num_mols)

                pbar.set_postfix_str(f"RMSE: {np.sqrt(tot_loss / tot_mol)}")
                pbar.update(1)
        return tot_loss / tot_mol

    def val(self, epoch):
        self.model.eval()
        tot_loss = 0
        tot_mol = 0
        with torch.no_grad():
            with tqdm(total=len(self.val_dataloader)) as pbar:
                pbar.set_description_str(f"validate {epoch}")
                for n, batch_data in enumerate(self.val_dataloader):
                    batch_data = batch_data.to(self.device)
                    pred = self.model(batch_data)
                    loss = self.loss_fn(pred, batch_data.y.view(-1, 1))

                    num_mols = batch_data.num_graphs
                    tot_mol += num_mols
                    tot_loss += (loss.item() * num_mols)

                    pbar.set_postfix_str(f"RMSE: {np.sqrt(tot_loss / tot_mol)}")
                    pbar.update(1)
        return tot_loss / tot_mol

    def test(self):
        self.model.eval()
        tot_loss = 0
        tot_mol = 0
        test_labels = []
        test_predictions = []
        test_smiles = []
        test_react = []
        with torch.no_grad():
            with tqdm(total=len(self.test_dataloader)) as pbar:
                pbar.set_description_str(f"test ")
                for n, batch_data in enumerate(self.test_dataloader):
                    batch_data = batch_data.to(self.device)
                    pred = self.model(batch_data)
                    loss = self.loss_fn(pred, batch_data.y.view(-1, 1))

                    num_mols = batch_data.num_graphs
                    tot_mol += num_mols
                    tot_loss += (loss.item() * num_mols)

                    test_labels.extend(batch_data.y.cpu().numpy().flatten().tolist())
                    test_predictions.extend(pred.cpu().numpy().flatten().tolist())
                    test_smiles.extend(batch_data.smiles)  
                    test_react.extend(batch_data.react) 

                    pbar.set_postfix_str(f"RMSE: {np.sqrt(tot_loss / tot_mol)}")
                    pbar.update(1)
                    rmse = np.sqrt(mean_squared_error(test_labels, test_predictions))

        return tot_loss / tot_mol, test_labels, test_predictions, rmse, test_smiles, test_react

    def pred(self):
        self.model.eval()

        test_labels = []
        test_predictions = []
        test_smiles = []
        test_react = []

        contributions = []

        with torch.no_grad():
            with tqdm(total=len(self.test_dataloader)) as pbar:
                pbar.set_description_str(f"test ")
                for n, batch_data in enumerate(self.test_dataloader):  # batch_data = 1
                    batch_data = batch_data.to(self.device)

                    single_contributions = {}

                    single_contributions['smiles'] = batch_data.smiles[0]
                    original_x = batch_data.x.clone()

                    original_output = self.model(batch_data)
                    single_contributions['react'] = batch_data.react[0]
                    single_contributions['T'] = batch_data.g[0][-1].cpu().detach().item()
                    single_contributions['original_y'] = batch_data.y.cpu().detach().item()
                    single_contributions['original_output'] = original_output.cpu().detach().item()

                    for i in range(batch_data.num_nodes):
                        batch_data.x = original_x
                        mask_x = batch_data.x.clone()
                        mask_x[i] = torch.zeros_like(mask_x[i])
                        batch_data.x = mask_x

                        masked_output = self.model(batch_data)

                        contribution = original_output - masked_output
                        single_contributions[f'atom_{i}'] = contribution.cpu().detach().item()

                    test_labels.extend(batch_data.y.cpu().numpy().flatten().tolist())
                    test_predictions.extend(original_output.cpu().numpy().flatten().tolist())
                    test_smiles.extend(batch_data.smiles)  
                    test_react.extend(batch_data.react)  
                    contributions.append(single_contributions)

                    pbar.update(1)
                    rmse = np.sqrt(mean_squared_error(test_labels, test_predictions))

        return test_labels, test_predictions, rmse, test_smiles, test_react, contributions

    def group_mask_pred(self):
        self.model.eval()

        test_labels = []
        test_predictions = []
        test_smiles = []
        test_react = []
        contributions = []

        with torch.no_grad():

            smarts_list = ['*-[O;D1]', '*-[O;D1]-*', '*-C=[O;D1]', '*-C(=O)-*', '*-O-C(=O)-*', '*-O-C(=O)-O-*',
                           '*-C(=O)-O', '*-OO', '*-C(=O)-OO', '*-C(=O)-C(=O)', '*-C(=O)-C(=O)-*', '*-O-[N+](=[O])-[O-]',
                           '*-O-O-[N+](=[O])-[O-]', '*-C(=O)-O-O-[N+](=[O])-[O-]', '*-[N+](=[O])-[O-]', '*-O-N=O',
                           '*-[F]', '*-[Cl]', '*-[Br]', '*-[I]', '*[Si](*)(*)(*)', '*[P](*)(*)', '*-[S](=O)(=O)-*',
                           '*-[S]-*', '*=S', '*-[S](=O)-*', '*-N(*)-*', '*=N-*', '*#N', '*-C#C-*', '*-C=C-*',
                           '*-c1c(*)c(*)c(*)c(*)c1-*']
            functional_groups = {}
            for smart in smarts_list:
                functional_groups[smart] = smart

            with tqdm(total=len(self.test_dataloader)) as pbar:
                pbar.set_description_str(f"pred")
                for n, batch_data in enumerate(self.test_dataloader): 
                    batch_data = batch_data.to(self.device)

                    single_contributions = {}

                    single_contributions['smiles'] = batch_data.smiles[0]
                    mol = AllChem.MolFromSmiles(single_contributions['smiles'])

                    original_x = batch_data.x.clone()
                    original_edge_feat = batch_data.edge_attr.clone()

                    original_output = self.model(batch_data)
                    single_contributions['react'] = batch_data.react[0]
                    single_contributions['T'] = batch_data.g[0][-1].cpu().detach().item()
                    single_contributions['original_y'] = batch_data.y.cpu().detach().item()
                    single_contributions['original_output'] = original_output.cpu().detach().item()

                    group_counts = {}

                    for fg_name, fg_smarts in functional_groups.items():

                        fg_pattern = Chem.MolFromSmarts(fg_smarts)
                        matches = mol.GetSubstructMatches(fg_pattern)

                        hit_bond = []  
                        hit_atom = []  

                        for match in matches:
                            match_bond = []
                            for bond in fg_pattern.GetBonds():
                                aid1 = match[bond.GetBeginAtomIdx()]
                                aid2 = match[bond.GetEndAtomIdx()]
                                match_bond.append(mol.GetBondBetweenAtoms(aid1, aid2).GetIdx())
                            hit_bond.append(match_bond)
                            hit_atom.append(match)
                        group_counts[fg_smarts] = {'bond': hit_bond, 'atom': hit_atom}

                    group_counts = {k: v for k, v in group_counts.items() if v['atom']} # 去掉空值
                    """
                    for example:
                    {'-C(=O)O': {'bond': [[3, 4, 5]], 'atom': [(3, 4, 5, 6)]},
                     '-O': {'bond': [[0], [5]], 'atom': [(1, 0), (4, 6)]},
                     '=O': {'bond': [[4]], 'atom': [(4, 5)]}}
                    """
                    single_contributions['group_idx'] = group_counts

                    delta = {}
                    masked_atom_idx = []
                    masked_edge_idx = []
                    for g_name in group_counts:
                        for g_idx in range(len(group_counts[g_name]['atom'])):
                            batch_data.x = original_x
                            batch_data.edge_attr = original_edge_feat
                            mask_x = batch_data.x.clone()
                            for i in group_counts[g_name]['atom'][g_idx]:
                                mask_x[i] = torch.zeros_like(mask_x[i])
                            mask_edge_feat = batch_data.edge_attr.clone()
                            for i in group_counts[g_name]['bond'][g_idx]:
                                mask_edge_feat[2*i] = torch.zeros_like(mask_edge_feat[2*i])
                                mask_edge_feat[2*i+1] = torch.zeros_like(mask_edge_feat[2*i+1])

                            masked_atom_idx.extend(group_counts[g_name]['atom'][g_idx])
                            masked_edge_idx.extend(group_counts[g_name]['bond'][g_idx])

                            batch_data.x = mask_x
                            batch_data.edge_attr = mask_edge_feat

                            masked_output = self.model(batch_data)

                            contribution = original_output - masked_output
                            delta[f'{g_name}_{g_idx}'] = contribution.cpu().detach().item()

                    masked_atom_idx = set(masked_atom_idx)
                    masked_edge_idx = set(masked_edge_idx)
                    for i in range(batch_data.num_nodes):   
                        if i not in masked_atom_idx:
                            batch_data.x = original_x
                            mask_x = batch_data.x.clone()
                            mask_x[i] = torch.zeros_like(mask_x[i])
                            batch_data.x = mask_x

                            masked_output = self.model(batch_data)

                            contribution = original_output - masked_output
                            delta[f'atom_{i}'] = contribution.cpu().detach().item()
                        else:
                            continue

                    for i in range(batch_data.num_edges // 2):   
                        if i not in masked_edge_idx:
                            batch_data.edge_attr = original_edge_feat
                            mask_edge_feat = batch_data.edge_attr.clone()
                            mask_edge_feat[2*i] = torch.zeros_like(mask_edge_feat[2*i])
                            mask_edge_feat[2*i+1] = torch.zeros_like(mask_edge_feat[2*i+1])

                            batch_data.edge_attr = mask_edge_feat

                            masked_output = self.model(batch_data)

                            contribution = original_output - masked_output
                            delta[f'edge_{i}'] = contribution.cpu().detach().item()
                        else:
                            continue

                    single_contributions['delta'] = delta

                    test_labels.extend(batch_data.y.cpu().numpy().flatten().tolist())
                    test_predictions.extend(original_output.cpu().numpy().flatten().tolist())
                    test_smiles.extend(batch_data.smiles)  
                    test_react.extend(batch_data.react)  
                    contributions.append(single_contributions)

                    pbar.update(1)
                    rmse = np.sqrt(mean_squared_error(test_labels, test_predictions))

        return test_labels, test_predictions, rmse, test_smiles, test_react, contributions