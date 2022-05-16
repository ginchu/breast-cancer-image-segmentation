# breast-cancer-image-segmentation
Boston College Biomedical Image Analysis Course Spring 2022: Final Project


## Runnable scripts/commands: what are the main commands that you run to get the results
For the dataset:

    wget https://zenodo.org/record/1175282/files/TNBC_NucleiSegmentation.zip -O dataset.zip
    unzip dataset.zip
    
To Run Model:
    
    python3 train.py

## Contribution: who is responsible for which files
Gina: train.py, data.py, unet.py, mlp.py, enet.py, aug_data.py

Ananya: Augmented_data, data_augmentation_code.ipynb, fcn.py (containns 4 FCN models- resnet101, resnet50, resnet34 and resnet18 backbones), fcn_train_aug.py, fcn_data_aug.py, fcn_train_no_aug.py, fcn_data_no_aug.py
