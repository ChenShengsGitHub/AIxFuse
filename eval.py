from rdkit import Chem
import numpy as np
import joblib
import pandas as pd
import argparse
from moses.metrics import weight, logP, SA, QED,mol_passes_filters
from moses.metrics.utils import get_mol, mapper
from rdkit.Chem import AllChem, rdShapeHelpers
from rdkit.Chem import AllChem,Descriptors,Lipinski,DataStructs
import moses
import glob
import os
import warnings 
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')
# class rf_model:
#     def __init__(self,clf_path):
#         self.clf = joblib.load(clf_path)

#     def __call__(self, mol_list):
#         fps = []
#         ind_list=[]
#         for i,mol in enumerate(mol_list):
#             try:
#                 smiles=Chem.MolToSmiles(mol,isomericSmiles=False)
#                 norm_mol=Chem.MolFromSmiles(smiles)
#                 fp = AllChem.GetMorganFingerprintAsBitVect(norm_mol,2, nBits=2048)
#                 fps.append(fp)
#                 ind_list.append(i)
#             except:
#                 pass
#         scores=[0 for _ in mol_list]
#         # fps = np.concatenate(fps, axis=0)
#         if fps:
#             s = self.clf.predict_proba(fps)[:, 1]
#         for i,ind in enumerate(ind_list):
#             scores[ind]=s[i]
#         return np.float32(scores)
# gsk3b_model=rf_model('data/models/rf/gsk3.pkl')
# jnk3_model=rf_model('data/models/rf/jnk3.pkl')

task={'rorgt_dhodh':{
        'pdb_id1':'5NTP',
        'pdb_id2':'6QU7',
        'prec':'XP'},
    'gsk3b_jnk3':{
        'pdb_id1':'6Y9S',
        'pdb_id2':'4WHZ',
        'prec':'SP'}}

def calc_SC_RDKit_score(query_mol, ref_mol):
    protrude_dist = rdShapeHelpers.ShapeTanimotoDist(query_mol, ref_mol)
    return 1 - protrude_dist

def scoring(data,mol_list_ref):
    mol_gen,this_title=data
    best_score=-1
    best_title=None
    best_mol_ref=None
    score_dict={}
    for title in mol_list_ref:
        mol_ref=mol_list_ref[title]
        score=calc_SC_RDKit_score(mol_gen, mol_ref)
        score_dict[title]=score
        # if score > best_score:
        #     best_mol_ref=mol_ref
        #     best_score =score
        #     best_title=title
    # fps_ref=AllChem.GetMorganFingerprintAsBitVect(best_mol_ref,2,1024)
    # fps_gen=AllChem.GetMorganFingerprintAsBitVect(mol_gen,2,1024)
    # sim_2d=DataStructs.TanimotoSimilarity(fps_ref,fps_gen)
    return this_title,score_dict

from functools import partial
from multiprocessing import Pool
pool = Pool(64)

def get_mol_dict(mol_list_gen):
    mol_list_gen_best={}
    for mol in mol_list_gen:
        title=mol.GetProp('_Name').strip()
        score=float(mol.GetProp('r_i_docking_score'))
        if title not in mol_list_gen_best or score < float(mol_list_gen_best[title].GetProp('r_i_docking_score')):
            mol_list_gen_best[title]=mol
    return mol_list_gen_best

def get_properties(args):
    target1,target2=args.task.split('_')
    pdb_id1=task[args.task]['pdb_id1']
    pdb_id2=task[args.task]['pdb_id2']
    prec=task[args.task]['prec']
    compare_dir=os.path.join(args.compare_dir,f'{target1}_{target2}')
    docking_dir1=os.path.join(args.docking_dir,f'{pdb_id1}_{prec}')
    docking_dir2=os.path.join(args.docking_dir,f'{pdb_id2}_{prec}')
    ref_dir1=os.path.join(docking_dir1,f'{target1}_act.sdf')
    ref_dir2=os.path.join(docking_dir2,f'{target2}_act.sdf')
    ref_mols1=get_mol_dict(Chem.SDMolSupplier(ref_dir1))
    ref_mols2=get_mol_dict(Chem.SDMolSupplier(ref_dir2))
    ref_fps1=[]
    ref_title1=[]
    csv=os.path.join(compare_dir,f'{target1}.csv')
    df=pd.read_csv(csv)
    for title,smiles in zip(df['Title'],df['SMILES']):
        mol=Chem.MolFromSmiles(smiles)
        if mol:
            ref_fps1.append(AllChem.GetMorganFingerprintAsBitVect(mol,2,1024))
            ref_title1.append(title)
    
    csv=os.path.join(compare_dir,f'{target2}.csv')
    df=pd.read_csv(csv)
    ref_fps2=[]
    ref_title2=[]
    for title,smiles in zip(df['Title'],df['SMILES']):
        mol=Chem.MolFromSmiles(smiles)
        if mol:
            ref_fps2.append(AllChem.GetMorganFingerprintAsBitVect(mol,2,1024))
            ref_title2.append(title)
    
    name_list=[target1,target2,'JMC','ReINVENT','RationaleRL','MARS','AIxFuse_init','AIxFuse']
    # name_list=['ReINVENT','RationaleRL','MARS','AIxFuse_init']
    name_list=['RationaleRL']
    for name in name_list:
        csv=os.path.join(compare_dir,f'{name}.csv')
        if not os.path.exists(csv):
            print(csv,'not exist!')
            continue
        df=pd.read_csv(csv)
        mols=mapper(args.ncpu)(get_mol,df['SMILES'])
        tmp_list=[]
        ind_list=[]
        title_list=[]
        sim_2d_dict1={}
        sim_2d_dict2={}
        for i,(mol,title) in enumerate(zip(mols,df['Title'])):
            title=str(title)
            if mol is not None:
                tmp_list.append(mol)
                ind_list.append(i)
                title_list.append(title)
                fp=AllChem.GetMorganFingerprintAsBitVect(mol,2,1024)
                sim_2d_list1=DataStructs.BulkTanimotoSimilarity(fp,ref_fps1)
                sim_2d_list2=DataStructs.BulkTanimotoSimilarity(fp,ref_fps2)
                sim_2d_d={}
                for ref_title, sim2d in zip(ref_title1,sim_2d_list1):
                    sim_2d_d[ref_title]=sim2d
                sim_2d_dict1[title]=sim_2d_d
                sim_2d_d={}
                for ref_title, sim2d in zip(ref_title2,sim_2d_list2):
                    sim_2d_d[ref_title]=sim2d
                sim_2d_dict2[title]=sim_2d_d
        mols = tmp_list
        df['Weight']=[None for _ in range(len(df))]
        df['Weight'].iloc[ind_list]=mapper(args.ncpu)(weight,mols)
        df['LogP']=[None for _ in range(len(df))]
        df['LogP'].iloc[ind_list]=mapper(args.ncpu)(logP,mols)
        df['SA']=[None for _ in range(len(df))]
        df['SA'].iloc[ind_list]=mapper(args.ncpu)(SA,mols)
        df['QED']=[None for _ in range(len(df))]
        df['QED'].iloc[ind_list]=mapper(args.ncpu)(QED,mols)
        df['TPSA']=[None for _ in range(len(df))]
        df['TPSA'].iloc[ind_list]=mapper(1)(Descriptors.TPSA,mols)
        df['HBA']=[None for _ in range(len(df))]
        df['HBA'].iloc[ind_list]=mapper(1)(Lipinski.NumHAcceptors,mols)
        df['HBD']=[None for _ in range(len(df))]
        df['HBD'].iloc[ind_list]=mapper(1)(Lipinski.NumHDonors,mols)
        df['RotBond']=[None for _ in range(len(df))]
        df['RotBond'].iloc[ind_list]=mapper(1)(Lipinski.NumRotatableBonds,mols)
        df['Filtered']=[None for _ in range(len(df))]
        df['Filtered'].iloc[ind_list]=mapper(args.ncpu)(mol_passes_filters,mols)
        df[f'best_3d_sim_{target1}']=[None for _ in range(len(df))]
        df[f'best_2d_sim_{target1}']=[None for _ in range(len(df))]
        df[f'best_3d_2d_sim_{target1}']=[None for _ in range(len(df))]
        df[f'best_3d_match_{target1}']=[None for _ in range(len(df))]
        df[f'best_2d_match_{target1}']=[None for _ in range(len(df))]
        df[f'best_3d_2d_match_{target1}']=[None for _ in range(len(df))]
        df[f'best_3d_sim_{target2}']=[None for _ in range(len(df))]
        df[f'best_2d_sim_{target2}']=[None for _ in range(len(df))]
        df[f'best_3d_2d_sim_{target2}']=[None for _ in range(len(df))]
        df[f'best_3d_match_{target2}']=[None for _ in range(len(df))]
        df[f'best_2d_match_{target2}']=[None for _ in range(len(df))]
        df[f'best_3d_2d_match_{target2}']=[None for _ in range(len(df))]
        docking_sdf1=os.path.join(docking_dir1,f'{name}.sdf')
        docking_sdf1=docking_sdf1 if os.path.exists(docking_sdf1) else os.path.join(docking_dir1,f'{name}_act.sdf')
        docking_sdf2=os.path.join(docking_dir2,f'{name}.sdf')
        docking_sdf2=docking_sdf2 if os.path.exists(docking_sdf2) else os.path.join(docking_dir2,f'{name}_act.sdf')
        print(os.path.exists(docking_sdf1),os.path.exists(docking_sdf2))
        if os.path.exists(docking_sdf1):
            docking_mols1=Chem.SDMolSupplier(docking_sdf1)
            mols1=[]
            for mol in docking_mols1:
                mols1.append((mol,mol.GetProp('_Name').strip()))
            working_func=partial(scoring,mol_list_ref=ref_mols1)
            results1 = pool.map(working_func, mols1)
            result_dict1={}
            for this_title,score_dict in results1:
                best_3d_sim=0
                best_3d_match=None
                best_2d_sim=0
                best_2d_match=None
                best_3d_2d_sim=0
                best_3d_2d_match=None
                for ref_title in score_dict:
                    if this_title in sim_2d_dict1 and ref_title in sim_2d_dict1[this_title]:
                        sim_3d=score_dict[ref_title]
                        sim_2d=sim_2d_dict1[this_title][ref_title]
                        sim_3d_2d=sim_3d*sim_2d
                        if sim_3d>best_3d_sim:
                            best_3d_sim=sim_3d
                            best_3d_match=ref_title
                        if sim_2d>best_2d_sim:
                            best_2d_sim=sim_2d
                            best_2d_match=ref_title
                        if sim_3d_2d>best_3d_2d_sim:
                            best_3d_2d_sim=sim_3d_2d
                            best_3d_2d_match=ref_title
                result_dict1[this_title]=(best_3d_sim,best_2d_sim,best_3d_2d_sim,best_3d_match,best_2d_match,best_3d_2d_match)
            for title,ind in zip(title_list,ind_list):
                if title in result_dict1:
                    best_3d_sim,best_2d_sim,best_3d_2d_sim,best_3d_match,best_2d_match,best_3d_2d_match=result_dict1[title]
                    df[f'best_3d_sim_{target1}'][ind]=best_3d_sim
                    df[f'best_2d_sim_{target1}'][ind]=best_2d_sim
                    df[f'best_3d_2d_sim_{target1}'][ind]=best_3d_2d_sim
                    df[f'best_3d_match_{target1}'][ind]=best_3d_match
                    df[f'best_2d_match_{target1}'][ind]=best_2d_match
                    df[f'best_3d_2d_match_{target1}'][ind]=best_3d_2d_match
        if os.path.exists(docking_sdf2):
            docking_mols2=Chem.SDMolSupplier(docking_sdf2)
            mols2=[]
            for mol in docking_mols2:
                mols2.append((mol,mol.GetProp('_Name').strip()))
            working_func=partial(scoring,mol_list_ref=ref_mols2)
            results2 = pool.map(working_func, mols2)
            result_dict2={}
            for this_title,score_dict in results2:
                best_3d_sim=0
                best_3d_match=None
                best_2d_sim=0
                best_2d_match=None
                best_3d_2d_sim=0
                best_3d_2d_match=None
                for ref_title in score_dict:
                    if this_title in sim_2d_dict2 and ref_title in sim_2d_dict2[this_title]:
                        sim_3d=score_dict[ref_title]
                        sim_2d=sim_2d_dict2[this_title][ref_title]
                        sim_3d_2d=sim_3d*sim_2d
                        if sim_3d>best_3d_sim:
                            best_3d_sim=sim_3d
                            best_3d_match=ref_title
                        if sim_2d>best_2d_sim:
                            best_2d_sim=sim_2d
                            best_2d_match=ref_title
                        if sim_3d_2d>best_3d_2d_sim:
                            best_3d_2d_sim=sim_3d_2d
                            best_3d_2d_match=ref_title
                result_dict2[this_title]=(best_3d_sim,best_2d_sim,best_3d_2d_sim,best_3d_match,best_2d_match,best_3d_2d_match)
            for title,ind in zip(title_list,ind_list):
                if title in result_dict2:
                    best_3d_sim,best_2d_sim,best_3d_2d_sim,best_3d_match,best_2d_match,best_3d_2d_match=result_dict2[title]
                    df[f'best_3d_sim_{target2}'][ind]=best_3d_sim
                    df[f'best_2d_sim_{target2}'][ind]=best_2d_sim
                    df[f'best_3d_2d_sim_{target2}'][ind]=best_3d_2d_sim
                    df[f'best_3d_match_{target2}'][ind]=best_3d_match
                    df[f'best_2d_match_{target2}'][ind]=best_2d_match
                    df[f'best_3d_2d_match_{target2}'][ind]=best_3d_2d_match
        df.to_csv(csv,index=False)

def eval_moses(args):
    target1,target2=args.task.split('_')
    compare_dir=os.path.join(args.compare_dir,f'{target1}_{target2}')
    target1_smiles=pd.read_csv(os.path.join(compare_dir,f'{target1}.csv'))['SMILES'].to_list()
    target2_smiles=pd.read_csv(os.path.join(compare_dir,f'{target2}.csv'))['SMILES'].to_list()
    statics_dict={'Method':[]}
    for csv in glob.glob(f'{compare_dir}/*'):
        df=pd.read_csv(csv)
        name=csv.split('/')[-1].split('.')[0]
        gen_smiles=df['SMILES'].to_list()
        print(f'{name}_{target1}')
        statics_dict['Method'].append(f'{name}_{target1}')
        moses_result1=moses.get_all_metrics(gen=gen_smiles,test=target1_smiles,train=target1_smiles,n_jobs=args.ncpu)
        for key in moses_result1:
            if key not in statics_dict:
                statics_dict[key]=[]
            statics_dict[key].append(moses_result1[key])
        print(f'{name}_{target2}')
        statics_dict['Method'].append(f'{name}_{target2}')
        moses_result2=moses.get_all_metrics(gen=gen_smiles,test=target2_smiles,train=target2_smiles,n_jobs=args.ncpu)
        for key in moses_result2:
            statics_dict[key].append(moses_result2[key])
    result_csv=os.path.join(args.compare_dir,f'{target1}_{target2}.csv')
    statics_df=pd.DataFrame(statics_dict)
    statics_df.to_csv(result_csv,index=False)

def analyse_func(df,col_list,comp_list,thrh_list):
    succ_ind_list=[]
    single_succ_list=[]
    for col, comp, thrh in zip(col_list,comp_list,thrh_list):
        single_ind=comp(df[col],thrh)
        succ_ind_list.append(single_ind)
        single_succ=len(set(df['SMILES'][single_ind]))
        single_succ_list.append(single_succ)
    
    succ_ind=succ_ind_list[0]
    for i in range(1,len(succ_ind_list)):
        succ_ind*=succ_ind_list[i]
    print('%.2f'%(100*np.sum(succ_ind)/len(df)),end='\t')
    for single_succ in single_succ_list:
        print(single_succ,end='\t')
    print(len(set(df['SMILES'][succ_ind])))

def he(data,thrh):
    return data>=thrh

def le(data,thrh):
    return data<=thrh
    
def analyse(args):
    target1,target2=args.task.split('_')
    compare_dir=os.path.join(args.compare_dir,f'{target1}_{target2}')
    pdb_id1=task[args.task]['pdb_id1']
    pdb_id2=task[args.task]['pdb_id2']
    prec=task[args.task]['prec']
    target1_df=pd.read_csv(os.path.join(compare_dir,f'{target1}.csv'))
    target2_df=pd.read_csv(os.path.join(compare_dir,f'{target2}.csv'))

    # sa_med=np.median(target1_df['SA'].to_list()+target2_df['SA'].to_list())
    # qed_med=np.median(target1_df['QED'].to_list()+target2_df['QED'].to_list())
    # aff1_med=np.median(target1_df[f'{pdb_id1}_{prec}'].dropna())
    # aff2_med=np.median(target2_df[f'{pdb_id2}_{prec}'].dropna())

    sa_avg=np.mean(target1_df['SA'].to_list()+target2_df['SA'].to_list())
    qed_avg=np.mean(target1_df['QED'].to_list()+target2_df['QED'].to_list())
    aff1_avg=np.mean(target1_df[f'{pdb_id1}_{prec}'].dropna())
    aff2_avg=np.mean(target2_df[f'{pdb_id2}_{prec}'].dropna())
    
    # print(sa_med,qed_med,aff1_med,aff2_med)
    print(sa_avg,qed_avg,aff1_avg,aff2_avg)
    
    
    print('Method    ','SuccR','SA','QED','Aff1','Aff2','Succ',sep='\t')
    col_list=['SA','QED',f'{pdb_id1}_{prec}',f'{pdb_id2}_{prec}']
    comp_list=[le,he,le,le]
    thrh_list=[sa_avg,qed_avg,aff1_avg,aff2_avg]
    name_list=[target1,target2,'JMC','ReINVENT','RationaleRL','MARS','AIxFuse_init','AIxFuse']
    df_dict={}
    for name in name_list:
        csv=os.path.join(compare_dir,f'{name}.csv')
        if not os.path.exists(csv):
            continue
    # for csv in glob.glob(f'{compare_dir}/*'):
    #     name=csv.split('/')[-1].split('.')[0]
        df=pd.read_csv(csv)
        mols=mapper(args.ncpu)(get_mol,df['SMILES'])
        for i,mol in enumerate(mols):
            try:
                df['SMILES'][i]=Chem.MolToSmiles(mol)
            except:
                continue
        df_dict[name]=df
        if f'{pdb_id1}_{prec}' not in df:
            continue
        print(name+' '*(10-len(name)),end='\t')
        analyse_func(df,col_list,comp_list,thrh_list)
    print('************************************************************************')
    if target1=='rorgt':
        sa_14d,qed_14d,aff1_14d,aff2_14d=3.74619658218488,0.3683087692459719,-13.2639883105383,-10.2413786134841
        thrh_list=[sa_14d,qed_14d,aff1_14d,aff2_14d]
        for name in df_dict:
            df=df_dict[name]
            print(name+' '*(10-len(name)),end='\t')
            analyse_func(df,col_list,comp_list,thrh_list)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',default='rorgt_dhodh')
    parser.add_argument('--ncpu',type=int,default=64)
    parser.add_argument('--compare_dir',default='data/compare')
    parser.add_argument('--docking_dir',default='utils/docking')
    parser.add_argument("--get_properties", action="store_true")
    parser.add_argument("--eval_moses", action="store_true")
    parser.add_argument("--analyse", action="store_true")


    args = parser.parse_args()
    if args.get_properties:
        get_properties(args)
    if args.eval_moses:
        eval_moses(args)
    if args.analyse:
        analyse(args)
    