# Chemnet Auxilliaries

Contains auxilliary training and preprocessing scripts used for replicating the results in [Goh et. al](https://arxiv.org/pdf/1712.02734.pdf). More precisely, the following scripts are present:

* Preparing the ChEMBL dataset to be used for ChemNet training protocol
* Pretraining ChemCeption on ChEMBL dataset
* Finetuning ChemCeption on Tox21, HIV and Sampl (FreeSolv) datasets
* Training ChemCeption on Tox21, HIV and Sampl (FreeSolv) datasets (without any pretraining)

The repository assumes library [DeepChem](https://github.com/deepchem/deepchem)  has been installed on the system. Please refer to the installation instructions that can be found on the DeepChem page.

**NOTE**: The other model described in Goh et. al, Smiles2Vec, uses RNN models on sequences with padded length 270. The backpropagation is slow in this case, and the training time amounted to roughly ~1 epoch a day. With Smiles2Vec trained for 50 epochs as mentioned in the paper, the training time was infeasible within the scope of Google Summer of Code. I will update the scripts and results once I manage to get training completed. 

Pretrained weights are available [here](https://drive.google.com/drive/folders/1APwvGvxuZPPBZ6Vzx2WaGt-X6gc10opZ)
