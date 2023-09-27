import os

import warnings
warnings.filterwarnings('ignore')


from rdkit import Chem
from rdkit.Chem import AllChem, rdShapeHelpers
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit import RDConfig
from rdkit import rdBase, RDLogger
import argparse
rdBase.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.*')  # https://github.com/rdkit/rdkit/issues/2683

# Set up features to use in FeatureMap
fdefName = os.path.join('/home/chensheng/anaconda3/envs/mtdd/share/RDKit/Data/', 'BaseFeatures.fdef')
fdef = AllChem.BuildFeatureFactory(fdefName)

fmParams = {}
for k in fdef.GetFeatureFamilies():
    fparams = FeatMaps.FeatMapParams()
    fmParams[k] = fparams

keep = ('Donor', 'Acceptor', 'NegIonizable', 'PosIonizable',
        'ZnBinder', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe')


def get_FeatureMapScore(query_mol, ref_mol):
    featLists = []
    for m in [query_mol, ref_mol]:
        rawFeats = fdef.GetFeaturesForMol(m)
        # filter that list down to only include the ones we're intereted in
        featLists.append([f for f in rawFeats if f.GetFamily() in keep])
    fms = [FeatMaps.FeatMap(feats=x, weights=[1] * len(x), params=fmParams) for x in featLists]
    fms[0].scoreMode = FeatMaps.FeatMapScoreMode.Best
    fm_score = fms[0].ScoreFeats(featLists[1]) / min(fms[0].GetNumFeatures(), len(featLists[1]))

    return fm_score


def calc_SC_RDKit_score(query_mol, ref_mol):
    # fm_score = get_FeatureMapScore(query_mol, ref_mol)

    # protrude_dist = rdShapeHelpers.ShapeProtrudeDist(query_mol, ref_mol,
    #                                                  allowReordering=False)
    # SC_RDKit_score = 0.5 * fm_score + 0.5 * (1 - protrude_dist)

    # return SC_RDKit_score
    protrude_dist = rdShapeHelpers.ShapeTanimotoDist(query_mol, ref_mol,allowReordering=False)
    return 1 - protrude_dist

def get_best(mol_list_gen):
    mol_list_gen_best={}
    for mol in mol_list_gen:
        title=mol.GetProp('_Name')
        score=float(mol.GetProp('r_i_docking_score'))
        if title not in mol_list_gen_best or score < float(mol_list_gen_best[title].GetProp('r_i_docking_score')):
            mol_list_gen_best[title]=mol
    return mol_list_gen_best


def scoring(mol_gen,mol_list_ref):
    best_score=0
    for title in mol_list_ref:
        mol_ref=mol_list_ref[title]
        score=calc_SC_RDKit_score(mol_gen, mol_ref)
        best_score = max(best_score,score)
    return best_score

from functools import partial
from multiprocessing import Pool
import random
import numpy as np
import pandas as pd
from iter_gen import task
pool=Pool(64)


def get_score(result_csv,mol_list_gen1,mol_list_gen2,mol_list_ref1,mol_list_ref2):
    mol1_list=[]
    mol2_list=[]
    title_list=[]
    score_list=[]
    score_list1=[]
    score_list2=[]
    smiles_list=[]
    le_list1=[]
    le_list2=[]
    for title in mol_list_gen1:
        mol_gen1=mol_list_gen1[title]
        smiles=Chem.MolToSmiles(mol_gen1,isomericSmiles=False)
        if title in mol_list_gen2:
            mol_gen2=mol_list_gen2[title]
            smiles_list.append(smiles)
            mol1_list.append(mol_gen1)
            mol2_list.append(mol_gen2)
            title_list.append(title)
            score_list.append(float(mol_gen1.GetProp('r_i_docking_score'))+float(mol_gen2.GetProp('r_i_docking_score')))
            score_list1.append(float(mol_gen1.GetProp('r_i_docking_score')))
            score_list2.append(float(mol_gen2.GetProp('r_i_docking_score')))
            le_list1.append(float(mol_gen1.GetProp('r_i_docking_score'))/len(mol_gen1.GetAtoms()))
            le_list2.append(float(mol_gen2.GetProp('r_i_docking_score'))/len(mol_gen2.GetAtoms()))
    print(len(score_list))
    ind_list=np.argsort(score_list)
    score_list=[score_list[int(i)] for i in ind_list]
    score_list1=[score_list1[int(i)] for i in ind_list]
    score_list2=[score_list2[int(i)] for i in ind_list]
    le_list1=[le_list1[int(i)] for i in ind_list]
    le_list2=[le_list2[int(i)] for i in ind_list]
    mol1_list=[mol1_list[int(i)] for i in ind_list]
    mol2_list=[mol2_list[int(i)] for i in ind_list]
    title_list=[title_list[int(i)] for i in ind_list]
    smiles_list=[smiles_list[int(i)] for i in ind_list]
    
    working_func=partial(scoring,mol_list_ref=mol_list_ref1)
    results1 = pool.map(working_func, mol1_list)
    working_func=partial(scoring,mol_list_ref=mol_list_ref2)
    results2 = pool.map(working_func, mol2_list)
    df=pd.DataFrame({'SMILES':smiles_list,'Title':title_list,'Docking Score 1':score_list1,'Docking Score 2':score_list2,'LE1':le_list1,'LE2':le_list2,'Docking Score':score_list,'Pose Sim 0':results1,'Pose Sim 1':results2})
    print(np.mean(results1),np.mean(results2))
    df.to_csv(result_csv,index=False)
from moses.metrics.utils import get_mol, mapper
def comp(csv,thrh0,thrh1):
    for method in csv:
        df=pd.read_csv(f'data/compare/rorgt_dhodh/{method}.csv')
        mols=mapper(64)(get_mol,df['SMILES'])
        for i,mol in enumerate(mols):
            try:
                df['SMILES'][i]=Chem.MolToSmiles(mol)
            except:
                continue
        sim_1='3D_sim_rorgt'if'3D_sim_rorgt' in df else 'best_3d_sim_rorgt'
        sim_2='3D_sim_dhodh'if'3D_sim_dhodh' in df else 'best_3d_sim_dhodh'
        print(method,len(set(df['SMILES'][(df[sim_1]>=thrh0)])),len(set(df['SMILES'][(df[sim_2]>=thrh1)])),len(set(df['SMILES'][(df[sim_1]>=thrh0)*(df[sim_2]>=thrh1)])))



if __name__=='__main__':
    jmc_14d_rorgt=0.5426944971537
    jmc_14d_dhodh=0.548154239766081
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',default='rorgt_dhodh')
    parser.add_argument('--method',default='AIxFuse')
    args = parser.parse_args()
    # target1,target2=args.task.split('_')
    # pdb_id1=task[args.task]['pdb_id1']
    # pdb_id2=task[args.task]['pdb_id2']
    # prec=task[args.task]['prec']
    # mol_list_gen1 = get_best(Chem.SDMolSupplier(f'utils/docking/{pdb_id1}_{prec}/{args.method}.sdf'))
    # mol_list_gen2 = get_best(Chem.SDMolSupplier(f'utils/docking/{pdb_id2}_{prec}/{args.method}.sdf'))
    # mol_list_ref1 = get_best(Chem.SDMolSupplier(f'utils/docking/{pdb_id1}_{prec}/{target1}_act.sdf'))
    # mol_list_ref2 = get_best(Chem.SDMolSupplier(f'utils/docking/{pdb_id2}_{prec}/{target2}_act.sdf'))
    result_csv=['ReINVENT','RationaleRL','MARS','AIxFuse_init','AIxFuse']
    
    # get_score(result_csv,mol_list_gen1,mol_list_gen2,mol_list_ref1,mol_list_ref2)
    comp(result_csv,jmc_14d_rorgt,jmc_14d_dhodh)

    
    
    


    
