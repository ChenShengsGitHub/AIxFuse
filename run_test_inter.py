import sys
import os
import pandas as pd
import importlib
sys.path.append('utils')
import utils_chem
importlib.reload(utils_chem)
import utils_common
importlib.reload(utils_common)
from rdkit import Chem
import numpy as np
import utils_interaction
importlib.reload(utils_interaction)
import extr_core_info
importlib.reload(extr_core_info)
import random
import warnings
import pickle
import dual_MCTS
importlib.reload(dual_MCTS)
warnings.filterwarnings('ignore')

csv_dhodh='data/inputs/active_ligands/dhodh/dhodh.csv'
dir_dhodh='data/outputs/docking/dhodh/act_pdbs_comp'
conformer_dir_dhodh='data/outputs/docking/dhodh/act_pdbs_pose'
csv_rorgt='data/inputs/active_ligands/rorgt/rorgt.csv'
dir_rorgt='data/outputs/docking/rorgt/act_pdbs_comp'
conformer_dir_rorgt='data/outputs/docking/rorgt/act_pdbs_pose'
with open('temp.pkl','rb') as r:
    dual_MCTS_Agent=pickle.load(r)
dual_MCTS_Agent.iter_explore(10000)