from operator import index
import torch
from torch_geometric.data import Data
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pandas as pd
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit import DataStructs
from tqdm import tqdm
import pickle
import os
import torch
import pickle

import torch.utils.data
import time
import os
import numpy as np

import csv

import dgl

from scipy import sparse as sp
import numpy as np
import networkx as nx
import hashlib

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom,
                explicit_H=True,
                use_chirality=False):

    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C','N','O', 'S','F','Si','P', 'Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B','V','K','Tl',
            'Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H', 'Li','Ge','Cu','Au','Ni','Cd','In',
            'Mn','Zr','Cr','Pt','Hg','Pb','Unknown'
        ]) + [atom.GetDegree()/10, atom.GetImplicitValence(),
                atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
                ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if explicit_H:
        results = results + [atom.GetTotalNumHs()]

    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)

    return torch.from_numpy(results)

def bond_features(bond):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, 0, 0]
    bond_feats = bond_feats + one_of_k_encoding_unk(bond.GetIsConjugated(), [0, 1, "nonbond"]) + \
                                 one_of_k_encoding_unk(bond.IsInRing(), [0, 1, "nonbond"])
    bond_feats = bond_feats + one_of_k_encoding_unk(
                        str(bond.GetStereo()),
                        ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE", "nonbond"])
    results = np.array(bond_feats).astype(np.float32)
    return torch.from_numpy(results)

def save_data(data, filename):
    dirname = f'{"data/ddinter"}'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = dirname + '/' + filename
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f'\nData saved as {filename}!')


def generate_drug_data_dgl(mol_graph, atom_symbols):
    edge_indices = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol_graph.GetBonds()]
    edge_feats = [bond_features(b) for b in mol_graph.GetBonds()]

    edge_list = torch.LongTensor(edge_indices)
    edge_feats = torch.stack(edge_feats) if edge_feats else torch.FloatTensor([])

    if len(edge_list) == 0:
        edge_list = torch.LongTensor([])
        edge_feats = torch.FloatTensor([])

    edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    edge_feats = torch.cat([edge_feats, edge_feats], dim=0) if len(edge_feats) else edge_feats

    features = [(atom.GetIdx(), atom_features(atom)) for atom in mol_graph.GetAtoms()]
    features.sort()
    _, features = zip(*features)
    features = torch.stack(features)

    line_graph_edge_index = torch.LongTensor([])
    if edge_list.nelement() != 0:
        conn = (edge_list[:, 1].unsqueeze(1) == edge_list[:, 0].unsqueeze(0)) & (
                    edge_list[:, 0].unsqueeze(1) != edge_list[:, 1].unsqueeze(0))
        line_graph_edge_index = conn.nonzero(as_tuple=False).T

    new_edge_index = edge_list.T
    node_feature = features.float()
    edge_feature = edge_feats.float()
    max_atoms = 9
    if node_feature.shape[0] < max_atoms:
        padding_atoms = max_atoms - node_feature.shape[0]
        padding_features = torch.zeros(padding_atoms, node_feature.shape[1])
        node_feature = torch.cat([node_feature, padding_features], dim=0)
    g = dgl.DGLGraph()
    g.add_nodes(node_feature.shape[0])
    g.ndata['feat'] = node_feature
    for src, dst in edge_list:
        g.add_edges(src.item(), dst.item())
    g.edata['feat'] = edge_feature
    return g

drug_smiles_path = pd.read_csv('data/ddinter/ddinter_smiles.csv')
drug_pairs_path = pd.read_csv('data/ddinter/ddinter_triplets.csv')
drug_smiles = [(h, t) for h,t in zip(drug_smiles_path['drug_id'],drug_smiles_path['smiles'])]
drug_dict = {}
drug_id_mol_tup = []
smiles_rdkit_list = []
symbols = list()
smiles_list = []
ID_list=[]
for i in drug_smiles:
    ID_list.append(i[0])
    smiles_list.append(i[1])
moleculse = [Chem.MolFromSmiles(smile) for smile in smiles_list]
fingerprints = [rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024) for mol in moleculse]
similarity_matrix = np.zeros((len(fingerprints),len(fingerprints)))
for i in range(len(fingerprints)):
    similarities = BulkTanimotoSimilarity(fingerprints[i],fingerprints)
    similarity_matrix[i] = similarities

drug_similarity=[ID_list, similarity_matrix]
save_data(drug_similarity, 'drug_similarity.pkl')


# for id in drug_smiles:
#     drug_dict[id[0]] = id[1]
#
# for id,smiles in drug_dict.items():
#     mol = Chem.MolFromSmiles(smiles.strip())
#     if mol is not None:
#         drug_id_mol_tup.append((id,mol))
#         symbols.extend(atom.GetSymbol() for atom in mol.GetAtoms())

# for m in drug_id_mol_tup:
#     smiles_rdkit_list.append(m[-1])
# symbols = list(set(symbols))
#drug_pos_data = {id: generate_drug_data(mol, symbols) for id, mol in tqdm(drug_id_mol_tup, desc='Processing drugs_lap')}
#drug_dgl_data = {id: generate_drug_data_dgl(mol, symbols) for id, mol in tqdm(drug_id_mol_tup, desc='Processing drugs_dgl')}
# drug_dgl_data_padding = {id: generate_drug_data_dgl(mol, symbols) for id, mol in tqdm(drug_id_mol_tup, desc='Processing drugs_dgl')}
# drug_pairs = [(h, t) for h, t in zip(drug_pairs_path['Drug1_ID'], drug_pairs_path['Drug2_ID'])]
# drug_pairs = [(h, t) for h, t in zip(drug_pairs_path['d1'], drug_pairs_path['d2'])]
# association_matrix = np.zeros((1706,1706),dtype=int)
# ID_index = {drug : idx for idx, drug in enumerate(ID_list)}
# for drug1,drug2 in drug_pairs:
#     idx1 = ID_index[drug1]
#     idx2 = ID_index[drug2]
#     association_matrix[idx1, idx2] = 1
#save_data(drug_pos_data,'drug_pos_data.pkl')
#save_data(drug_dgl_data,'drug_dgl_data.pkl')
# save_data(ID_index, 'ID_index.pkl')
# save_data(drug_dgl_data_padding,'drug_dgl_data_padding.pkl')
# association = [ID_index, association_matrix]
# save_data(association,'drugbank_association_matrix.pkl')