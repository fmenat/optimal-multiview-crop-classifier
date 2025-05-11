# In the search for an optimal multi-view crop classifier
<a href="https://github.com/fmenat/mvlearning">  <img src="https://img.shields.io/badge/Package-mvlearning-blue"/>  </a> 
[![paper](https://img.shields.io/badge/arXiv-2308.05407-D12424)](https://www.arxiv.org/abs/2308.05407) 


## Data
The data used comes from https://github.com/nasaharvest/cropharvest. However we also share the code used to generate the data structures that we used in [data folder](./data).

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

# :scroll: Source

Mena, Francisco, Diego Arenas, and Andreas Dengel. "*In the Search for Optimal Multi-view Learning Models for Crop Classification with Global Remote Sensing Data.*", 2024.
