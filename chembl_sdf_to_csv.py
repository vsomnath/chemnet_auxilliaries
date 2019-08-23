import deepchem as dc
import pandas as pd
import numpy as np
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Descriptors import *
import multiprocessing
import time


property_list = ["MolWt", "HeavyAtomMolWt", "MolLogP", "MolMR", "TPSA", "LabuteASA",
            "HeavyAtomCount", "NHOHCount", "NOCount", "NumHAcceptors", "NumHDonors", "NumHeteroatoms",
            "NumRotatableBonds", "NumRadicalElectrons", "NumValenceElectrons",
            "NumAromaticRings", "NumSaturatedRings", "NumAliphaticRings",
            "NumAromaticCarbocycles", "NumSaturatedCarbocycles", "NumAliphaticCarbocycles",
            "NumAromaticHeterocycles", "NumSaturatedHeterocycles", "NumAliphaticHeterocycles",
            "PEOE_VSA1", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5", "PEOE_VSA6", "PEOE_VSA7",
            "PEOE_VSA8", "PEOE_VSA9", "PEOE_VSA10", "PEOE_VSA11", "PEOE_VSA12", "PEOE_VSA13", "PEOE_VSA14",
            "SMR_VSA1", "SMR_VSA2", "SMR_VSA3", "SMR_VSA4", "SMR_VSA5",
            "SMR_VSA6", "SMR_VSA7", "SMR_VSA8", "SMR_VSA9", "SMR_VSA10",
            "SlogP_VSA1", "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6",
            "SlogP_VSA7", "SlogP_VSA8", "SlogP_VSA9", "SlogP_VSA10", "SlogP_VSA11", "SlogP_VSA12",
            "EState_VSA1", "EState_VSA2", "EState_VSA3", "EState_VSA4", "EState_VSA5", "EState_VSA6",
            "EState_VSA7", "EState_VSA8", "EState_VSA9", "EState_VSA10", "EState_VSA11",
            "VSA_EState1", "VSA_EState2", "VSA_EState3", "VSA_EState4", "VSA_EState5",
            "VSA_EState6", "VSA_EState7", "VSA_EState8", "VSA_EState9", "VSA_EState10",
            "BalabanJ", "BertzCT", "Ipc", "Kappa1", "Kappa2", "Kappa3", "HallKierAlpha",
            "Chi0", "Chi1", "Chi0n", "Chi1n", "Chi2n", "Chi3n", "Chi4n",
            "Chi0v", "Chi1v", "Chi2v", "Chi3v", "Chi4v"]

def get_properties_mol(smiles):
    """Computes the properties as in ChemNet paper for each molecule."""
    mol = MolFromSmiles(smiles)
    properties = dict()
    for prop_name in property_list:
        properties[prop_name] = globals()[prop_name](mol)
    return properties

start_time = time.time()
df = pd.read_csv("chembl_25_smiles.csv")

results = list()

cpu_count = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=cpu_count)

for i in range(len(df)):
    result = pool.apply_async(get_properties_mol, (df.iloc[i]))
    results.append(result)
pool.close()

properties_all = list()
props_intermediate = list()

for idx, result in enumerate(results):
    if (idx+1) % 10000 == 0:
        ## Save every 10000
        inter_df = pd.DataFrame(props_intermediate)
        inter_df.to_csv("chembl_25_props_{}.csv".format(idx), index=False, header=True)
        props_intermediate = list()

    props = result.get()
    properties_all.append(props)
    props_intermediate.append(props)

prop_df = pd.DataFrame(properties_all)
prop_df.to_csv("chembl_25_props.csv", index=False, header=True)

end_time = time.time()
print("Time taken to process molecules: ", end_time - start_time)
