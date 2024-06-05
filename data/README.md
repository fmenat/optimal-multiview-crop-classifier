# Dataset creation code

In order to obtain/create the dataset used execute the following code. The original data is extracted and comes from
> [CropHarvest](https://github.com/nasaharvest/cropharvest)

```
python data_creation.py -d DIRECTORY_OF_CROPHARVEST_DATA -c COUNTRIES_TO_EXECUTE -o OUTPUT_DIR
```

* options for COUNTRIES_TO_EXECUTE: [togo, brazil, kenya, global, all]
* this code will generate training and testing data for the selected countries, in two formats (xarray NC file and pickle PKL)

Same for creation of the dataset with more granularity (multiple crops)

```
python data_creation_multi.py -d DIRECTORY_OF_CROPHARVEST_DATA -c COUNTRIES_TO_EXECUTE -o OUTPUT_DIR
```