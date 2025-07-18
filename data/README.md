# Dataset creation code

In order to obtain/create the dataset used execute the following code. The original data comes from
> [CropHarvest](https://github.com/nasaharvest/cropharvest).
> In concrete, the 0.5 version: https://github.com/nasaharvest/cropharvest/releases/tag/v0.5.0

```
python data_creation.py -d DIRECTORY_OF_CROPHARVEST_DATA -c COUNTRIES_TO_EXECUTE -o OUTPUT_DIR
```

* options for COUNTRIES_TO_EXECUTE: [togo, brazil, kenya, global, all]
* This code will generate training and testing data for the selected countries, in two formats (xarray NC file and pickle PKL)

Same for the creation of the dataset with more granularity (multiple crops)

```
python data_creation_multi.py -d DIRECTORY_OF_CROPHARVEST_DATA -c COUNTRIES_TO_EXECUTE -o OUTPUT_DIR
```
