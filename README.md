# breast-cancer-image-segmentation
Boston College Biomedical Image Analysis Course Spring 2022: Final Project


## Runnable scripts/commands: what are the main commands that you run to get the results
- For the dataset:

    wget https://zenodo.org/record/1175282/files/TNBC_NucleiSegmentation.zip -O dataset.zip
    unzip dataset.zip
    
- To run UNET, eNET, or MLP Models:
    
    `python3 train.py`
    
- To run FCN Model (with original data):
    
    `python3 fcn_train_no_aug.py`
    
- To run FCN Model (with augmented data):
    
    `python3 fcn_train_aug.py`
    
    - To run different fcn resnet models, just have to change the model assigned on line 39, can uncomment the desired model

- To run notebooks in augmentation_validation/, open the notebook of interest in Colab, change the wandb login to your own (or comment out all wandb lines) if running a training notebook, and run the notebook.
    - To run the augmentation_validation.ipynb notebook, you will also need to place the model weights into your own Google Drive (find the link to the weights in best_weights_augVal/README.md) and, in the notebook, accordingly set the path pointing to the weights.

## Contribution: who is responsible for which files
Gina:
- train.py
- data.py
- unet.py
- mlp.py
- enet.py
- aug_data.py

Ananya:
- Augmented_data/
- FCN results/
- data_augmentation_code.ipynb
- fcn.py (contains 4 FCN models- resnet101, resnet50, resnet34 and resnet18 backbones)
- fcn_train_aug.py
- fcn_data_aug.py
- fcn_train_no_aug.py
- fcn_data_no_aug.py

Jakob:
- everything in augmentation_validation/
