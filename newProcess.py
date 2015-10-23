import json
import pickle
import numpy as np
from rdkit import Chem


class process:

    def __init__(self):
        
        self.molmap=dict([unit[::-1] for unit in enumerate('0,C,H,O,N,P,S,B,F,Cl,Br,I,Se,s,o,n,c'.split(','))])
        
        #{'C': 195960, 'O': 115177, 'P': 11672, 'H': 287328, 
        # 'S': 2889, 'N': 38099, 'Cl': 406, 'Br': 111, 
        # 'I': 29, 'F': 222, 'Se': 10, 'Cr': 1, 
        # 'Hg': 2, 'B': 1, 'Sn': 1, 'Mo': 2} 
        #  less than five was moved
        
        self.seqmap=dict([unit[::-1] for unit in enumerate('0GAVLIMPSCNQTFYWDERKH')])
        
        #{'M': 172190, 'A': 617707, 'F': 286423, 'S': 422553, 
        # 'L': 660382, 'K': 399446, 'Q': 252083, 'I': 399685, 
        # 'P': 355917, 'T': 381837, 'H': 172184, 'R': 354485,
        # 'D': 403284, 'N': 289279, 'C': 94591, 'V': 510624, 
        # 'G': 545163, 'Y': 234358, 'E': 461396, 'W': 96201, 
        # 'X': 96, 'U': 4, 'B': 3, 'O': 2}
        # less than five was moved , X was moved
        
        self.bond_dict={'SINGLE': 1, 'DOUBLE': 2, 'AROMATIC': 3, 'TRIPLE': 4}
        self.data=[]
        self.n=0

        with open("bin\Kcat_combination_0918.json", "rt") as f:
            self.rawdata=json.load(f)

        for sample in self.rawdata:
            
            seq,smiles,value=sample["Sequence"],sample["Smiles"],np.array([sample["Value"]],float)

            if sample["ECNumber"]=='4.2.1.78':continue   
            #keep NCS enzyme from repeating

            if value<=0:continue  
            # this work https://doi.org/10.1101/2022.11.23.517595 prepose to remove data which is apporching to 0,and DLkcat remove all the values below 0
            
            if 'X' in seq or 'B' in seq or 'U' in seq or 'O' in seq:continue 
            #remove instandard Amino acid and rare amino
            
            mol=Chem.AddHs(Chem.MolFromSmiles(smiles))
            
            adjacency = np.array(Chem.GetAdjacencyMatrix(mol),float)
            
            if 0 in sum(adjacency):continue
            #remove unbonded atom
            
            atoms = [a.GetSymbol() for a in mol.GetAtoms()] 
            # a list contain symbol with H in the last

            for atom in mol.GetAromaticAtoms() : atoms[atom.GetIdx()] = atoms[atom.GetIdx()].lower()   
                     
            if len(seq)>self.n:self.n=len(seq)
            # recoed the longest length

            try:atom_encode = np.array([self.molmap[atom] for atom in atoms])  
            
            except:
                print('skip',smiles)
                continue
            
            DA=self.get_DA(adjacency)

            E=self.get_E(mol) # information of edges

            seq_encode = np.array([self.seqmap[aa] for aa in seq])
            self.data.append([seq_encode,atom_encode,DA,E,np.log2(value)])
            
            if round(len(self.data)/len(self.rawdata)*100,2)%10==0:print(round(len(self.data)/len(self.rawdata)*100,1),"%")
        
        with open("dataset.pkl", "wb") as f:pickle.dump(self.data, f)

        print('longest',self.n,'raw',len(self.rawdata),'contain',len(self.data),'molecules',len({str(i[1]) for i in self.data}),'seq',len({str(i[0]) for i in self.data}))
        print("finish")



    def get_E(self,mol):

        N=mol.GetNumAtoms()
        E=np.zeros((N,N))
        
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx() #get both edge atom in a bond
            bondtype = self.bond_dict[str(bond.GetBondType())]
            E[i, j] = E[j, i] = bondtype
        
        return self.DS(E)
        
        

    def DS(self,E):

        N=E.shape[0]
        
        e=np.array([E[i]*(1/np.sum(E[i])) for i in range(N)])

        ehat=np.zeros((N,N))
        S=np.sum(e,0)

        for i in range(N):
            for j in range(N):
                ehat[i,j]=np.sum([e[i]*e[j]/S])

        return ehat
        


    def get_DA(self,adjacency)->np.ndarray:
      
        A_hat = adjacency+np.eye(adjacency.shape[0])
        D_hat = np.diag(1/np.sum(A_hat, axis=1))
        
        DA=np.multiply(D_hat, A_hat)
        
        return DA
   
    

pr=process()