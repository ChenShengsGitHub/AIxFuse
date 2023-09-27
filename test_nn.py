import joblib
from dgllife.model import model_zoo
from dgllife.utils import AttentiveFPAtomFeaturizer
from dgllife.utils import AttentiveFPBondFeaturizer
from torch.utils.data import DataLoader
from dgllife.data import MoleculeCSVDataset
from dgllife.utils import mol_to_bigraph
import dgl
from multiprocessing import Pool
from rdkit import Chem
import torch
import numpy as np
from functools import partial
import pandas as pd
from scipy.stats import pearsonr
import argparse
from glob import glob
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import sys
sys.path.append('utils')
import dual_MCTS
from iter_gen import task
from matplotlib import pyplot as plt

pool=Pool(64)

def collate_molgraphs(data):
    smiles_list, graph_list = map(list, zip(*data))
    
    bg = dgl.batch(graph_list)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    return smiles_list, bg

class GraphDataset(object):
    def __init__(self,smiles_list,smiles_to_graph):
        self.smiles=smiles_list
        if len(smiles_list) > 100:
            self.graphs = pool.map(smiles_to_graph,self.smiles)
        else:
            self.graphs = []
            for s in self.smiles:
                self.graphs.append(smiles_to_graph(s))
        

    def __getitem__(self, item):
        return self.smiles[item], self.graphs[item]

    def __len__(self):
        """Size for the dataset

        Returns
        -------
        int
            Size for the dataset
        """
        return len(self.smiles)

def eval(args):
    target1,target2=args.task.split('_')
    pdb_id1=task[args.task]['pdb_id1']
    pdb_id2=task[args.task]['pdb_id2']
    prec=task[args.task]['prec']
    generated_dir=os.path.join(args.generated_dir,f'{target1}_{target2}')
    model_dir=os.path.join(args.model_dir,f'{target1}_{target2}')
    final_train_csv=os.path.join(generated_dir,'util_iter_4.csv')
    test_csv=os.path.join(generated_dir,'gen_iter_5_scores.csv')
    train_df=pd.read_csv(final_train_csv)
    train_set=set(train_df['SMILES'])
    test_df=pd.read_csv(test_csv)
    mol_list=[]
    score1_list=[]
    score2_list=[]
    for smiles,score1,score2 in zip(test_df['SMILES'],test_df[f'{pdb_id1}_{prec}'],test_df[f'{pdb_id2}_{prec}']):
        if smiles in train_set or np.isnan(score1) or np.isnan(score2):
            continue
        mol=Chem.MolFromSmiles(smiles)
        mol_list.append(mol)
        score1_list.append(-score1)
        score2_list.append(-score2)
    print(len(mol_list))
    
    target1_mse_list=[]
    target2_mse_list=[]
    target1_r2_list=[]
    target2_r2_list=[]
    for iter in range(5):
        model_path = os.path.join(model_dir,f'gen_iter_{iter}.pt')
        mtatfp_model=dual_MCTS.MTATFP_model(model_path)
        scores1,scores2=mtatfp_model(mol_list)
        target1_mse=mean_squared_error(score1_list, scores1)
        target1_mse_list.append(target1_mse)
        target2_mse=mean_squared_error(score2_list, scores2)
        target2_mse_list.append(target2_mse)
        target1_r2=r2_score(score1_list, scores1)
        target1_r2_list.append(target1_r2)
        target2_r2=r2_score(score2_list, scores2)
        target2_r2_list.append(target2_r2)
        print(model_path,"target1|mean_squared_error:", target1_mse)
        print(model_path,"target1|r2 score:", target1_r2)
        print(model_path,"target2|mean_squared_error:", target2_mse)
        print(model_path,"target2|r2 score:", target2_r2)
    img_dir=os.path.join(args.img_dir,args.task)
    plt.scatter(-np.array(score1_list),-np.array(scores1),s=1)
    data=-np.concatenate([score1_list,scores1])
    min_data=np.min(data)
    max_data=np.max(data)
    plt.plot([min_data,max_data],[min_data,max_data],c='black')
    plt.xlim(min_data,max_data)
    plt.ylim(min_data,max_data)
    plt.savefig(os.path.join(img_dir,'plot1.png'))
    plt.clf()

    plt.scatter(-np.array(score2_list),-np.array(scores2),s=1)
    data=-np.concatenate([score2_list,scores2])
    min_data=np.min(data)
    max_data=np.max(data)
    plt.plot([min_data,max_data],[min_data,max_data],c='black')
    plt.xlim(min_data,max_data)
    plt.ylim(min_data,max_data)
    plt.savefig(os.path.join(img_dir,'plot2.png'))
    plt.clf()
    width =0.3
    plt.bar(np.array(range(len(target1_mse_list)))-0.25 , target1_mse_list,width=width)
    # plt.savefig(os.path.join(img_dir,'target1_mse.png'))
    # plt.clf()
    plt.bar(np.array(range(len(target1_r2_list)))+0.25, target1_r2_list,width=width)
    plt.savefig(os.path.join(img_dir,'target1_r2_mse.png'))
    plt.clf()

    plt.bar(range(len(target2_mse_list)), target2_mse_list)
    plt.savefig(os.path.join(img_dir,'target2_mse.png'))
    plt.clf()

    

    plt.bar(range(len(target2_r2_list)), target2_r2_list)
    plt.savefig(os.path.join(img_dir,'target2_r2.png'))
    plt.clf()
        
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',default='rorgt_dhodh')
    parser.add_argument('--generated_dir',default='data/outputs/generated/')
    parser.add_argument('--model_dir',default='data/models/dgl')
    parser.add_argument('--img_dir',default='data/outputs/images/nn_test')
    args = parser.parse_args()
    eval(args)


    