import torch
import torch_geometric.data as gdata
from torch_geometric.data import Data as GData
from rdkit.Chem.rdchem import Atom, Bond, Mol
from rdkit.Chem import AllChem
import numpy as np
from tqdm import tqdm
from rdkit.Chem import BondType

class GNNDataUtils:

    _element_symbols = ['C', 'H', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Si']
    _hybridizations = ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']

    @classmethod
    def _get_node_features_from_rdatom(cls, atom):
        atomic_symbol = atom.GetSymbol()
        atomic_num = list(cls._element_symbols).index(atomic_symbol)
        atomic_num1 = atom.GetAtomicNum()
        # One-hot encode atomic number
        atomic_num_one_hot = np.zeros(len(cls._element_symbols))
        atomic_num_one_hot[atomic_num] = 1

        # One-hot encode total number of hydrogen atoms
        total_num_hs = atom.GetTotalNumHs()
        total_num_hs_one_hot = np.zeros(5)
        if total_num_hs < 5:
            total_num_hs_one_hot[total_num_hs] = 1
        else:
            total_num_hs_one_hot[-1] = 1

        # One-hot encode formal charge
        formal_charge = atom.GetFormalCharge()
        formal_charge_one_hot = np.zeros(9)
        formal_charge_index = formal_charge + 4 
        formal_charge_one_hot[formal_charge_index] = 1

        # One-hot encode number of radical electrons
        num_radical_electrons = atom.GetNumRadicalElectrons()
        num_radical_electrons_one_hot = np.zeros(3)
        if num_radical_electrons < 3:
            num_radical_electrons_one_hot[num_radical_electrons] = 1
        else:
            num_radical_electrons_one_hot[-1] = 1

        # One-hot encode total valence
        total_valence = atom.GetTotalValence()
        total_valence_one_hot = np.zeros(9)
        if total_valence < 9:
            total_valence_one_hot[total_valence] = 1
        else:
            total_valence_one_hot[-1] = 1

        # One-hot encode hybridization
        hybridization = atom.GetHybridization()
        hybridization_one_hot = np.zeros(len(cls._hybridizations))
        hybridization_index = cls._hybridizations.index(str(hybridization))
        hybridization_one_hot[hybridization_index] = 1

        # One-hot encode whether the atom is in a ring or aromatic
        in_ring_one_hot = np.array([atom.IsInRing(), atom.GetIsAromatic()], dtype=int)

        return np.concatenate([atomic_num_one_hot,[total_num_hs],[formal_charge],[total_valence],hybridization_one_hot,in_ring_one_hot], axis=0)  #这个是按照之前的GNN改的

 
    @classmethod
    def _get_nodes_features_from_rdmol(cls, mol):
        return [cls._get_node_features_from_rdatom(mol.GetAtomWithIdx(i)) for i in range(mol.GetNumAtoms())]

    @classmethod
    def _get_edge_features_from_rdbond(cls, bond):
        res = [0]*5
        if bond.GetBondType() == BondType.SINGLE:
            idx = 0
        elif bond.GetBondType() == BondType.DOUBLE:
            idx = 1
        elif bond.GetBondType() == BondType.TRIPLE:
            idx= 2
        elif bond.GetBondType() == BondType.AROMATIC:
            idx = 3
        else:
            idx = 4
        res[idx] = 1
        return res

    @classmethod
    def _get_connection_from_rdbond(cls, bond, reverse= False) :
        if reverse:
            return [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        else:
            return [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]

    @classmethod
    def _get_edges_features_and_connections_from_rdmol(cls, mol):
        edges_features = []
        connections = []
        for bond in mol.GetBonds():
            edges_features.append(cls._get_edge_features_from_rdbond(bond))
            edges_features.append(cls._get_edge_features_from_rdbond(bond))
            connections.append(cls._get_connection_from_rdbond(bond, False))
            connections.append(cls._get_connection_from_rdbond(bond, True))
        return edges_features, connections

    @classmethod
    def get_gnn_data_from_rdmol(cls, mol: Mol) -> GData:
        edges_features, connections = cls._get_edges_features_and_connections_from_rdmol(mol)
        x = torch.tensor(cls._get_nodes_features_from_rdmol(mol), dtype=torch.float)
        edge_index = torch.tensor(connections, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edges_features, dtype=torch.float)
        return GData(x=x, edge_index=edge_index, edge_attr=edge_attr)
       
    @classmethod
    def get_gnn_data_from_smiles(cls, smiles: str) -> GData:
        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            print(smiles)
        return cls.get_gnn_data_from_rdmol(mol)

class rateDataset(gdata.InMemoryDataset):
    min_T = float('inf')
    max_T = float('-inf')

    def __init__(self, train_or_test, df, root, transform=None, pre_transform=None):
        self._train_or_test = train_or_test  
        self._df = df  
        super(rateDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return [f'gnn_rate_{self._train_or_test}']
 

    def process(self):
        data_list = []
        index = 0
        
        with tqdm(total=len(self._df)) as pbar:
            pbar.set_description_str(f"{self._train_or_test} process data")
            for _, row in self._df.iterrows():  
                             
                smiles = row.smiles 
                mol = AllChem.MolFromSmiles(smiles)
                atoms = mol.GetAtoms()
                molecule_elements = {atom.GetSymbol() for atom in atoms}

                if molecule_elements.issubset(GNNDataUtils._element_symbols):
                    data = GNNDataUtils.get_gnn_data_from_smiles(smiles)

                    reacts_list = ['OH','O3','NO3','Cl']
                    react_num = reacts_list.index(row['react'])
                    react_one_hot = np.zeros(len(reacts_list))
                    react_one_hot[react_num] = 1
                    react_one_hot = torch.tensor(react_one_hot, dtype=torch.float) 
                    T = torch.tensor([row['T']], dtype=torch.float) 
                    data.g = torch.cat([react_one_hot, T], dim=0).view(1, -1)
                    data.y = torch.tensor([row['logK']], dtype=torch.float)

                    data.smiles = smiles  
                    data.react = row['react'] 

                    data_list.append((data))
                pbar.update(1)


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == "__main__":
    print(GNNDataUtils.get_gnn_data_from_smiles("Cc1ccccc1").y)
