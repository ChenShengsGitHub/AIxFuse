
import dgl
from multiprocessing import Pool
from rdkit import Chem
import torch
import sys
sys.path.append('utils')
import dual_MCTS
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdmolfiles, rdmolops
import matplotlib.pyplot as plt

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

def eval():
    
    for smiles in ['CCn1c(CO)nn(-c2ccc3c(=O)n(-c4ccccc4CCC4CCN(C(=O)C(C)C)CC4)cc(C(C)=C)c3c2)c1=O']:
        mol_list=[]
        mol=Chem.MolFromSmiles(smiles)
        mol_list.append(mol)
        
        model_path = 'data/models/dgl/rorgt_dhodh/gen_iter_4.pt'
        mtatfp_model=dual_MCTS.MTATFP_model(model_path)
        (scores1,scores2),node_weight=mtatfp_model(mol_list,get_node_weight=True)
        smiles = Chem.MolToSmiles(mol,isomericSmiles=False)
        mol = Chem.MolFromSmiles(smiles)
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)

        # 2. 创建红绿渐变的颜色映射
        cmap = plt.cm.get_cmap('YlOrBr')  # 使用RdYlGn颜色映射，即红绿渐变
        print(node_weight)
        data=node_weight[0][:,0]-torch.min(node_weight[0][:,0])
        weights=(data/torch.max(data)).cpu().detach().numpy()
        # 3. 可视化高亮原子
        highlight_atoms = {}  # 存储需要高亮显示的原子索引及对应的颜色
        for i, atom in enumerate(mol.GetAtoms()):
            color = cmap(weights[i])[:3]  # 根据权重获取对应颜色，并反转权重以获得深浅变化
            highlight_atoms[i] = color
        d = rdMolDraw2D.MolDraw2DCairo(500, 500)
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=list(range(len(mol.GetAtoms()))),highlightAtomColors=highlight_atoms)
        d.FinishDrawing()
        d.WriteDrawingText(f'tmp/{smiles}.png')
        
        

if __name__ == "__main__":
    eval()


    