import sys
sys.path.append('utils')
sys.setrecursionlimit(3000)
import warnings
import pickle
import argparse
import dual_MCTS
from ligpre import ligpre
warnings.filterwarnings('ignore')

def init_gen(target1_pkl,target2_pkl,gen_num,init_csv,init_mae):
    with open(target1_pkl,'rb') as r:
        core_info_rorgt=pickle.load(r)
    with open(target2_pkl,'rb') as r:
        core_info_dhodh=pickle.load(r)
    dual_MCTS_Agent=dual_MCTS.DualMCTS(core_info_rorgt,core_info_dhodh)
    print(dual_MCTS_Agent.sampler1.rationale_count,dual_MCTS_Agent.sampler2.rationale_count)
    df=dual_MCTS_Agent.init_explore(gen_num)
    df.to_csv(init_csv,index=False)
    ligpre(init_csv,init_mae)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target1_pkl',deafault='data/temp_data/core_info_rorgt.pkl')
    parser.add_argument('--target2_pkl',deafault='data/temp_data/core_info_dhodh.pkl')
    parser.add_argument('--gen_num',deafault=20000)
    parser.add_argument('--init_csv',default='/public/home/chensheng/project/aixfuse2/data/outputs/generated/init_gen_10w.csv')
    parser.add_argument('--init_mae',default='/public/home/chensheng/project/aixfuse2/data/outputs/ligpre/init_gen_10w.maegz')

    args = parser.parse_args()
    init_gen(args.target1_pkl,args.target2_pkl,args.gen_num,args.target2,args.prec)
