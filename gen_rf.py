import sys
import os
import pandas as pd
sys.path.append('utils')
sys.setrecursionlimit(3000)
import argparse
import warnings
import pickle
import numpy as np
import dual_MCTS
import time
warnings.filterwarnings('ignore')

task={'rorgt|dhodh':{'pdb_id1':'5NTP','pdb_id2':'6QU7','prec':'XP','device':'gpu'},'gsk3b|jnk3':{'pdb_id1':'6Y9S','pdb_id2':'4WHZ','prec':'SP','device':'cpu'}}

def iter_gen(args,iter):
    target1,target2=args.task.split('|')
    generated_dir=os.path.join(args.generated_dir,f'{target1}_{target2}')
    cache_dir=os.path.join(args.cache_dir,f'{target1}_{target2}')
    iter_csv=os.path.join(generated_dir,f'gen_iter_rf.csv')
    agent_dir=os.path.join(cache_dir,f'dual_MCTS_Agent_iter_0.pkl')
    model_ckpt='data/models/rf/gsk3.pkl,data/models/rf/jnk3.pkl'

    with open(agent_dir,'rb') as r:
        dual_MCTS_Agent=pickle.load(r)
    dual_MCTS_Agent.iter_explore(args.gen_num,model_path=model_ckpt,docking_result=None,gen_csv=iter_csv,final_gen=True,model_type='rf')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',default='rorgt|dhodh')
    parser.add_argument('--cache_dir',default='data/temp_data/')
    parser.add_argument('--generated_dir',default='data/outputs/generated/')
    parser.add_argument('--model_dir',default='data/models/dgl')
    parser.add_argument('--gen_num',type=int,default=10000)
    parser.add_argument('--iter',default='0,1,2,3,4')
    parser.add_argument('--ligpre_dir',default='/public/home/chensheng/project/aixfuse2/data/outputs/ligpre/')
    parser.add_argument('--nchir',default=32)
    parser.add_argument('--grid_dir',default='data/inputs/target_structures/grids/')
    parser.add_argument("--single_process", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--ligpre", action="store_true")
    parser.add_argument("--docking", action="store_true")
    parser.add_argument("--training", action="store_true")


    args = parser.parse_args()
    iter_gen(args,-1)