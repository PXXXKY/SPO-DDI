# -*- coding: utf-8 -*-
from pathlib import Path
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import itertools
from collections import defaultdict
from operator import neg
import random
import math
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, MolFromSmiles, AllChem
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

class MoleculeData(Data):
    """Molecular Data with explicit edge-index increments."""

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return int(self.x.size(0))
        return super().__inc__(
            key, value, *args, **kwargs
        )

from rdkit import Chem
import pandas as pd
import numpy as np


df_drugs_smiles = pd.read_csv('data/ddinter/ddinter_smiles.csv')
# df_drugs_smiles = pd.read_csv('data/drugbank_2/drugbank_2_smiles.csv')

DRUG_TO_INDX_DICT = {drug_id: indx for indx, drug_id in enumerate(df_drugs_smiles['drug_id'])}

drug_id_mol_graph_tup = [(id, Chem.MolFromSmiles(smiles.strip()), smiles.strip()) for id, smiles in zip(df_drugs_smiles['drug_id'], df_drugs_smiles['smiles'])]
drug_id_mol_graph_tup1 = [(id, smiles) for id, smiles in zip(df_drugs_smiles['drug_id'], df_drugs_smiles['smiles'])]


# Gettings information and features of atoms
ATOM_MAX_NUM = np.max([m[1].GetNumAtoms() for m in drug_id_mol_graph_tup])
AVAILABLE_ATOM_SYMBOLS = list({a.GetSymbol() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})
AVAILABLE_ATOM_DEGREES = list({a.GetDegree() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})
AVAILABLE_ATOM_TOTAL_HS = list({a.GetTotalNumHs() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})
max_valence = max(a.GetImplicitValence() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup))
max_valence = max(max_valence, 9)
AVAILABLE_ATOM_VALENCE = np.arange(max_valence + 1)

MAX_ATOM_FC = abs(np.max([a.GetFormalCharge() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)]))
MAX_ATOM_FC = MAX_ATOM_FC if MAX_ATOM_FC else 0
MAX_RADICAL_ELC = abs(np.max([a.GetNumRadicalElectrons() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)]))
MAX_RADICAL_ELC = MAX_RADICAL_ELC if MAX_RADICAL_ELC else 0


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


def get_atom_features(atom, mode='one_hot'):

    if mode == 'one_hot':
        atom_feature = torch.cat([
            one_of_k_encoding_unk(atom.GetSymbol(), AVAILABLE_ATOM_SYMBOLS),
            one_of_k_encoding_unk(atom.GetDegree(), AVAILABLE_ATOM_DEGREES),
            one_of_k_encoding_unk(atom.GetTotalNumHs(), AVAILABLE_ATOM_TOTAL_HS),
            one_of_k_encoding_unk(atom.GetImplicitValence(), AVAILABLE_ATOM_VALENCE),
            torch.tensor([atom.GetIsAromatic()], dtype=torch.float)
        ])
    else:
        atom_feature = torch.cat([
            one_of_k_encoding_unk(atom.GetSymbol(), AVAILABLE_ATOM_SYMBOLS),
            torch.tensor([atom.GetDegree()]).float(),
            torch.tensor([atom.GetTotalNumHs()]).float(),
            torch.tensor([atom.GetImplicitValence()]).float(),
            torch.tensor([atom.GetIsAromatic()]).float()
        ])

    return atom_feature
def onehot_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]
def bond_attr(mol, use_chirality=True):

    feat = []
    index = []
    n = mol.GetNumAtoms()
    for i in range(n):
        for j in range(n):
            if i != j:
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond is not None:
                    bt = bond.GetBondType()
                    bond_feats = [
                        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
                        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
                        bond.GetIsConjugated(),
                        bond.IsInRing()
                    ]
                    if use_chirality:
                        bond_feats = bond_feats + onehot_encoding_unk(
                            str(bond.GetStereo()),
                            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
                    feat.append(bond_feats)
                    index.append([i, j])

    return np.array(feat)
def get_mol_edge_list_and_feat_mtx(mol_graph,smiles):
    #print(111111111111111111111111)
    #print(mol_graph.shape)
    #print(smiles)
    features = [(atom.GetIdx(), atom_features(atom)) for atom in mol_graph.GetAtoms()]
    features.sort() # to make sure that the feature matrix is aligned according to the idx of the atom
    _, features = zip(*features)
    features = torch.stack(features)

    edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol_graph.GetBonds()])
    edge_attr=(bond_attr(MolFromSmiles(smiles)))
    edge_attr=torch.FloatTensor(edge_attr)
    undirected_edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    
    # assert TOTAL_ATOM_FEATS == features.shape[-1], "Expected atom n_features and retrived n_features not matching"
    # TrimNet当前使用10维键特征。
    # 普通化学键会被展开为两个方向，因此edge_index列数必须
    # 与edge_attr行数完全一致。
    bond_feature_dim = 10

    if undirected_edge_list.numel() == 0:
        # 合法的单原子、无化学键分子
        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )
        edge_attr = torch.empty(
            (0, bond_feature_dim),
            dtype=torch.float,
        )
    else:
        edge_index = (
            undirected_edge_list
            .reshape(-1, 2)
            .transpose(0, 1)
            .contiguous()
            .long()
        )

        # 正常情况下edge_attr已经是[num_edges, 10]。
        # 仅在它退化成一维时才重新整理形状。
        if edge_attr.ndim == 1:
            if edge_attr.numel() % bond_feature_dim != 0:
                raise RuntimeError(
                    "Cannot reshape bond features: "
                    f"shape={tuple(edge_attr.shape)}, "
                    f"numel={edge_attr.numel()}"
                )

            edge_attr = edge_attr.reshape(
                -1,
                bond_feature_dim,
            )

        elif edge_attr.ndim != 2:
            raise RuntimeError(
                "Unexpected edge_attr dimensions: "
                f"shape={tuple(edge_attr.shape)}"
            )

        edge_attr = edge_attr.contiguous().float()

        if edge_attr.size(1) != bond_feature_dim:
            raise RuntimeError(
                "Unexpected bond feature dimension: "
                f"shape={tuple(edge_attr.shape)}, "
                f"expected second dimension={bond_feature_dim}"
            )

        if edge_index.size(1) != edge_attr.size(0):
            raise RuntimeError(
                "Edge count and edge-attribute count differ: "
                f"edge_index={tuple(edge_index.shape)}, "
                f"edge_attr={tuple(edge_attr.shape)}"
            )

    return (
        edge_index,
        features.contiguous(),
        edge_attr.contiguous(),
    )


_GRAPH_CACHE_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "ddinter"
    / "mol_edge_list_feat_mtx.pt"
)

if _GRAPH_CACHE_PATH.exists():
    print(f"[GRAPH CACHE] Loading {_GRAPH_CACHE_PATH}")
    MOL_EDGE_LIST_FEAT_MTX = torch.load(
        _GRAPH_CACHE_PATH,
        map_location="cpu",
    )
    print(
        "[GRAPH CACHE] Loaded "
        f"{len(MOL_EDGE_LIST_FEAT_MTX)} molecular graphs"
    )
else:
    print("[GRAPH CACHE] Cache not found; building molecular graphs")
    MOL_EDGE_LIST_FEAT_MTX = {drug_id: get_mol_edge_list_and_feat_mtx(mol,smiles)
                                    for drug_id, mol,smiles in drug_id_mol_graph_tup}
    MOL_EDGE_LIST_FEAT_MTX = {drug_id: mol for drug_id, mol in MOL_EDGE_LIST_FEAT_MTX.items() if mol is not None}

    _GRAPH_CACHE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    torch.save(
        MOL_EDGE_LIST_FEAT_MTX,
        _GRAPH_CACHE_PATH,
    )
    print(
        "[GRAPH CACHE] Saved "
        f"{len(MOL_EDGE_LIST_FEAT_MTX)} molecular graphs "
        f"to {_GRAPH_CACHE_PATH}"
    )

# 每个药物的PyG Data对象只创建一次。
# 原代码在每个batch内重复构造Data，长期运行后可能触发
# PyG/PyTorch底层原生内存段错误。
MOL_GRAPH_DATA_CACHE = {
    drug_id: MoleculeData(
        x=graph_parts[1],
        edge_index=graph_parts[0],
        edge_attr=graph_parts[2],
        num_nodes=int(graph_parts[1].size(0)),
    )
    for drug_id, graph_parts in MOL_EDGE_LIST_FEAT_MTX.items()
}
#print(MOL_EDGE_LIST_FEAT_MTX.shape)
TOTAL_ATOM_FEATS = (next(iter(MOL_EDGE_LIST_FEAT_MTX.values()))[1].shape[-1])


##### DDI statistics and counting #######
df_all_pos_ddi = pd.read_csv('data/ddinter/ddinter_triplets.csv')
# df_all_pos_ddi = pd.read_csv('data/drugbank_2/drugbank_2_triplets.csv')
all_pos_tup = [(h, t, r) for h, t, r in zip(df_all_pos_ddi['Drug1_ID'], df_all_pos_ddi['Drug2_ID'], df_all_pos_ddi['Y'])]


ALL_DRUG_IDS, _, _ = zip(*drug_id_mol_graph_tup)

# 保持ddinter_smiles.csv中的药物顺序。
# 原代码使用set，可能导致不同进程或不同fold中的药物顺序变化，
# 进而使drug_ids.npy与drug_emb_trimnet*.npy错位。
ALL_DRUG_IDS = np.array(
    list(dict.fromkeys(ALL_DRUG_IDS)),
    dtype=object,
)
ALL_TRUE_H_WITH_TR = defaultdict(list)
ALL_TRUE_T_WITH_HR = defaultdict(list)
# I won't corrupt the relations
# self.true_r_with_ht = defaultdict(list)
FREQ_REL = defaultdict(int)
ALL_H_WITH_R = defaultdict(dict)
ALL_T_WITH_R = defaultdict(dict)
ALL_TAIL_PER_HEAD = {}
ALL_HEAD_PER_TAIL = {}

for h, t, r in all_pos_tup:
    ALL_TRUE_H_WITH_TR[(t, r)].append(h)
    ALL_TRUE_T_WITH_HR[(h, r)].append(t)
    FREQ_REL[r] += 1.0
    ALL_H_WITH_R[r][h] = 1
    ALL_T_WITH_R[r][t] = 1

for t, r in ALL_TRUE_H_WITH_TR:
    ALL_TRUE_H_WITH_TR[(t, r)] = np.array(list(set(ALL_TRUE_H_WITH_TR[(t, r)])))
for h, r in ALL_TRUE_T_WITH_HR:
    ALL_TRUE_T_WITH_HR[(h, r)] = np.array(list(set(ALL_TRUE_T_WITH_HR[(h, r)])))

for r in FREQ_REL:
    ALL_H_WITH_R[r] = np.array(list(ALL_H_WITH_R[r].keys()))
    ALL_T_WITH_R[r] = np.array(list(ALL_T_WITH_R[r].keys()))
    ALL_HEAD_PER_TAIL[r] = FREQ_REL[r] / len(ALL_T_WITH_R[r])
    ALL_TAIL_PER_HEAD[r] = FREQ_REL[r] / len(ALL_H_WITH_R[r])

#######    ****** ###############

class DrugDataset(Dataset):
    def __init__(self, tri_list, ratio=1.0,  neg_ent=1, disjoint_split=False, shuffle=True):
        ''''disjoint_split: Consider whether entities should appear in one and only one split of the dataset
        ''' 
        self.neg_ent = neg_ent
        self.tri_list = []
        self.ratio = ratio

        for h, t, r, *_ in tri_list:
            if ((h in MOL_EDGE_LIST_FEAT_MTX) and (t in MOL_EDGE_LIST_FEAT_MTX)):
                self.tri_list.append((h, t, r))

        if disjoint_split:
            d1, d2, *_ = zip(*self.tri_list)
            self.drug_ids = np.array(list(set(d1 + d2)))
        else:
            self.drug_ids = ALL_DRUG_IDS

        self.drug_ids = np.array([id for id in self.drug_ids if id in MOL_EDGE_LIST_FEAT_MTX])
        
        if shuffle:
            random.shuffle(self.tri_list)
        limit = math.ceil(len(self.tri_list) * ratio)
        self.tri_list = self.tri_list[:limit]

    def __len__(self):
        return len(self.tri_list)
    
    def __getitem__(self, index):
        return self.tri_list[index]

    def collate_fn(self, batch):
        pos_rels = []
        pos_h_samples = []
        pos_t_samples = []
        neg_rels = []
        neg_h_samples = []
        neg_t_samples = []

        hlist=[]
        for h, t, r in batch:
            hlist.append(h)
            pos_rels.append(r)
            h_data = self.__create_graph_data(h)
            t_data = self.__create_graph_data(t)
            pos_h_samples.append(h_data)
            pos_t_samples.append(t_data)


        pos_h_samples = Batch.from_data_list(pos_h_samples)
        pos_t_samples = Batch.from_data_list(pos_t_samples)
        pos_rels = torch.LongTensor(pos_rels)
        pos_tri = (pos_h_samples, pos_t_samples, pos_rels)



        return pos_tri, pos_tri,hlist
            
    def __create_graph_data(self, id):
        # Batch.from_data_list不会修改源Data对象，
        # 因此可以跨batch和epoch安全复用CPU缓存图。
        return MOL_GRAPH_DATA_CACHE[id]

    def __corrupt_ent(self, other_ent, r, other_ent_with_r_dict, max_num=1):
        corrupted_ents = []
        current_size = 0
        while current_size < max_num:
            candidates = np.random.choice(self.drug_ids, (max_num - current_size) * 2)
            mask = np.isin(candidates, other_ent_with_r_dict[(other_ent, r)], assume_unique=True, invert=True)
            corrupted_ents.append(candidates[mask])
            current_size += len(corrupted_ents[-1])
        
        if corrupted_ents != []:
            corrupted_ents = np.concatenate(corrupted_ents)

        return np.asarray(corrupted_ents[:max_num])
        
    def __corrupt_head(self, t, r, n=1):
        return self.__corrupt_ent(t, r, ALL_TRUE_H_WITH_TR, n)

    def __corrupt_tail(self, h, r, n=1):
        return self.__corrupt_ent(h, r, ALL_TRUE_T_WITH_HR, n)
    
    def __normal_batch(self, h, t, r, neg_size):
        neg_size_h = 0
        neg_size_t = 0
        # prob = self.tail_per_head[r] / (self.tail_per_head[r] + self.head_per_tail[r])
        prob = ALL_TAIL_PER_HEAD[r] / (ALL_TAIL_PER_HEAD[r] + ALL_HEAD_PER_TAIL[r])
        # prob = 2
        for i in range(neg_size):
            if random.random() < prob:
                neg_size_h += 1
            else:
                neg_size_t +=1
        
        return (self.__corrupt_head(t, r, neg_size_h),
                self.__corrupt_tail(h, r, neg_size_t))


class DrugDataset1(Dataset):
    def __init__(self, tri_list, ratio=1.0, neg_ent=1, disjoint_split=False, shuffle=True):
        ''''disjoint_split: Consider whether entities should appear in one and only one split of the dataset
        '''
        self.neg_ent = neg_ent
        self.tri_list = []
        self.ratio = ratio

        for h, t, r, *_ in tri_list:
            if ((h in MOL_EDGE_LIST_FEAT_MTX) and (t in MOL_EDGE_LIST_FEAT_MTX)):
                self.tri_list.append((h, t, r))

        if disjoint_split:
            d1, d2, *_ = zip(*self.tri_list)
            self.drug_ids = np.array(list(set(d1 + d2)))
        else:
            self.drug_ids = ALL_DRUG_IDS

        self.drug_ids = np.array([id for id in self.drug_ids if id in MOL_EDGE_LIST_FEAT_MTX])
        #self.drug_ids=np.load('drug_ids.npy')

        #print(self.drug_ids)
        np.save('data/ddinter/drug_ids.npy',self.drug_ids)
        # np.save('data/drugbank_2/drug_ids.npy', self.drug_ids)

        if shuffle:
            random.shuffle(self.tri_list)
        limit = math.ceil(len(self.tri_list) * ratio)
        self.tri_list = self.tri_list[:limit]

    def __len__(self):
        return len(self.tri_list)

    def __getitem__(self, index):
        return self.tri_list[index]

    def collate_fn(self, batch):
        drugs=[]
        for i in self.drug_ids:
            i_data = self.__create_graph_data(i)
            drugs.append(i_data)
        drugs = Batch.from_data_list(drugs)
        pos_rels=[]
        for h, t, r in batch:

            pos_rels.append(r)


        pos_rels = torch.LongTensor(pos_rels)

        pos_tri = (drugs, drugs, pos_rels)


        return pos_tri

    def __create_graph_data(self, id):
        # Batch.from_data_list不会修改源Data对象，
        # 因此可以跨batch和epoch安全复用CPU缓存图。
        return MOL_GRAPH_DATA_CACHE[id]

        #return Data(x=features, edge_index=edge_index)

    def __corrupt_ent(self, other_ent, r, other_ent_with_r_dict, max_num=1):
        corrupted_ents = []
        current_size = 0
        while current_size < max_num:
            candidates = np.random.choice(self.drug_ids, (max_num - current_size) * 2)
            mask = np.isin(candidates, other_ent_with_r_dict[(other_ent, r)], assume_unique=True, invert=True)
            corrupted_ents.append(candidates[mask])
            current_size += len(corrupted_ents[-1])

        if corrupted_ents != []:
            corrupted_ents = np.concatenate(corrupted_ents)

        return np.asarray(corrupted_ents[:max_num])

    def __corrupt_head(self, t, r, n=1):
        return self.__corrupt_ent(t, r, ALL_TRUE_H_WITH_TR, n)

    def __corrupt_tail(self, h, r, n=1):
        return self.__corrupt_ent(h, r, ALL_TRUE_T_WITH_HR, n)

    def __normal_batch(self, h, t, r, neg_size):
        neg_size_h = 0
        neg_size_t = 0
        # prob = self.tail_per_head[r] / (self.tail_per_head[r] + self.head_per_tail[r])
        prob = ALL_TAIL_PER_HEAD[r] / (ALL_TAIL_PER_HEAD[r] + ALL_HEAD_PER_TAIL[r])
        prob = ALL_TAIL_PER_HEAD[r] / (ALL_TAIL_PER_HEAD[r] + ALL_HEAD_PER_TAIL[r])
        # prob = 2
        for i in range(neg_size):
            if random.random() < prob:
                neg_size_h += 1
            else:
                neg_size_t += 1

        return (self.__corrupt_head(t, r, neg_size_h),
                self.__corrupt_tail(h, r, neg_size_t))

# Simple DrugDataLoaderWrapper to use the customized collate_fn
class DrugDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)

