# SAMPPI
## Install
### Create a Conda Environment
```
conda create -n SAMPPI python=3.8
conda activate SAMPPI
```
### dependencies
```
torch==1.13.0
torchaudio==0.13.0
torchvision==0.14.0
pandas==1.3.5
python==3.7.12
numpy==1.18.5
scikit-learn==1.0.2
scipy==1.7.3
```
## Generate Feature
```
python ./data/utils/process_csv_to_fasta.py
```
Convert CSV to fasta file first

```
python ./data/utils/AntiBERTy_embedding_generate.py
python ./data/utils/ESM2_embedding_generate.py
python ./data/utils/Prott5_embedding_generate.py
python ./data/utils/PSSM_generate.py
```
One hot, Physicochemical properties and BLOSUM62 will be automatically generated during training. 
Please visit NetSurfP-3.0 online server for RASA generation (https://services.healthtech.dtu.dk/services/NetSurfP-3.0/)

## Train
```
python ./main/S1131/train/train_S1131.py
```
Run this script to train the S1131 model, the same applies to other datasets.

The weights of the trained model can be downloaded in releases (https://github.com/iobio-zjut/SAMPPI/releases/tag/v1.0)

## Predict
```
python ./main/S1131/predict/predict_S1131.py
```
Run this script to predict the S1131 model, the same applies to other datasets
