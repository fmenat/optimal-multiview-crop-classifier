# In the search for an optimal multi-view crop classifier
Public repository of our work "IN THE SEARCH FOR OPTIMAL MULTI-VIEW LEARNING MODELS FOR CROP CLASSIFICATION WITH GLOBAL REMOTE SENSING DATA"

> Now we have created a package to use multi-view learning models in a easy way in PyTorch: [https://github.com/fmenat/mvlearning](https://github.com/fmenat/mvlearning)

## Data
The data used comes from https://github.com/nasaharvest/cropharvest. However we also share the structures that we used on Google Drive: https://drive.google.com/drive/folders/1aPlctAL8B5dXSdpM55fr3-RUmAHO3quj

### Training
* To train the Input fusion strategy:  
```
python train_singleview.py -s config/singleview_ex.yaml
```
* To train the Ensemble aggregation strategy:  
```
python train_singleview_pool.py -s config/singleviewpool_ex.yaml
```
* To train strategies based on multiple encoders: Feature, Decision and Hybrid fusion strategies:
```
python train_multiview.py -s config/multiview_ex.yaml
```

### Evaluation
* To evaluate the model by its predictions (performance):
```
python evaluate_predictions.py -s config/evaluation_ex.yaml
```


## Installation
Please install the required packages with the following command:
```
pip install -r requirements.txt
```

## Source
Preprint: Mena, Francisco, Diego Arenas, and Andreas Dengel. "*In the Search for Optimal Multi-view Learning Models for Crop Classification with Global Remote Sensing Data.*" [arXiv preprint arXiv:2403.16582](https://arxiv.org/abs/2403.16582) (2024).
