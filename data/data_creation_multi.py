import numpy as np
import pandas as pd
import argparse
from utils import storage_set

import cropharvest
from cropharvest.datasets import CropHarvestLabels, CropHarvest, Task

def extracting_set(metadata_cropclass, data_structure):
	"""
		Function to extract the training and testing data for each set
	"""
	X_full_train, Y_full_train = [], []
	for i,value in enumerate(metadata_cropclass.index):
	    new_detailed_label = metadata_cropclass.loc[value]["classification_label"]
	    if not pd.isna(new_detailed_label):
	        x_i, y_i = data_structure[i]
	        X_full_train.append(x_i)
	        Y_full_train.append(new_detailed_label)

	X_full_train = np.asarray(X_full_train)
	Y_full_train = np.asarray(Y_full_train)

	return X_full_train, Y_full_train

def extracting_label(DATA_DIR, indx_data):
	metadata_structure = CropHarvestLabels(DATA_DIR)

	metadata_structure._labels["year"] = metadata_structure._labels["collection_date"].apply(lambda x: x.split("-")[0])
	df_metadata = metadata_structure._labels
	final_indx = []
	for v in df_metadata.to_dict(orient="records"):
		dataset = v["dataset"]
		indx_list = v["index"] #read index directly from data
		final_indx.append(f"{indx_list}_{dataset}")
	df_metadata["indx"] = final_indx

	df_metadata = df_metadata.set_index("indx").loc[indx_data] #filter based on data available in xarray
	df_metadata = df_metadata[~df_metadata["is_test"]] #filter test data
	return df_metadata

def codify_labels(array, labels):
    labels_2_idx = {v: i for i, v in enumerate(labels)} 
    return np.vectorize(lambda x: labels_2_idx[x])(array)

if __name__ == "__main__":
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument(
	    "--data_dir",
	    "-d",
	    required=True,
	    type=str,
	    help="path of the data directory",
	)
	arg_parser.add_argument(
	    "--country",
	    "-c",
	    required=False,
	    type=str,
	    help="country to store, options [global, kenya, togo, brazil, all]",
	    default="global"
	)
	arg_parser.add_argument(
	    "--out_dir",
	    "-o",
	    required=True,
	    type=str,
	    help="path of the output directory to store the data",
	)
	arg_parser.add_argument(
	    "--infer_nan",
	    "-i",
	    required=False,
	    type=str,
	    help="whether to infer the nan values or not [yes, no]",
	    default="no"
	)
	args = arg_parser.parse_args()
	DATA_DIR = args.data_dir

	if args.country != "global":
		raise Exception("For now, only supported global data")

	country, crop = "global", "multicrop"
	print("CREATING AND SAVING DATA FROM (%s, %s)"%(country, crop))

	structure = CropHarvest(DATA_DIR, download=True)
	indx_data = [str(f).split("/")[-1].split('.h5')[0] for f in structure.filepaths]

	df_metadata = extracting_label(DATA_DIR, indx_data)
	if args.infer_nan == "yes":
		df_metadata.loc[(pd.isna(df_metadata["classification_label"]) 
		                         & ~df_metadata["is_crop"]),"classification_label"] = "non_crop"
		df_metadata.loc[(pd.isna(df_metadata["classification_label"]) 
		                         & df_metadata["is_crop"]),"classification_label"] = "unclassified_crop" 	
	
	X_full_train, Y_full_train = extracting_set(df_metadata, structure)
	print(f"Dataset with {len(X_full_train)} examples and shape {X_full_train.shape}")

	LABELS = df_metadata["classification_label"].dropna().unique()
	Y_full_train = codify_labels(Y_full_train, LABELS)
	storage_set(args.out_dir, country, f"{crop}-infer" if args.infer_nan == "yes" else crop,
			train_data=[X_full_train, Y_full_train], 
			target_names=LABELS)
