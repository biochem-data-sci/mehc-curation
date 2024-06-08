import pandas as pd
import numpy as np
from parallel_pandas import ParallelPandas
from rdkit import Chem

def remove_mixtures(smiles):
    # Check if a input is a mixture or not
    # If a input is a mixture --> Return 1
    if smiles.find('.') != -1:
        if smiles.find('.[') == -1:
            smiles = int(1)
        else:
            if smiles.count('.') != smiles.count('.['):
                smiles = int(1)
    return smiles

def remove_inorganic_compound(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # Check for the absence of carbon atoms
    has_carbon = any(atom.GetSymbol() == 'C' for atom in mol.GetAtoms())
    if has_carbon is not True:
        smiles = 1
    return smiles