import numpy as np
import pandas as pd
import argparse
from utils import storage_set

import cropharvest
from cropharvest.datasets import CropHarvest


def extracting_set(data_load):
	"""
		Function to extract the training and testing data for each set
	"""
	X_full_train, Y_full_train = data_load.as_array(flatten_x=False)
	print("Total training set ", X_full_train.shape)

	aux_ids = []
	test_inputs = []
	test_labels = []
	for i, (test_id, test_instance) in enumerate(data_load.test_data(flatten_x=False)):
	    print("DATA FROM TESTING REGION= ",i,test_id)
	    aux_ids.append(test_id)
	    
	    labels = test_instance.y	    
	    mask_used = labels!= -1
	    print("Total data", len(labels), "Data with labels", len(labels[mask_used]))
	    
	    test_inputs.append(test_instance.x[mask_used])
	    test_labels.append(test_instance.y[mask_used])
	X_full_test = np.concatenate(test_inputs)
	Y_full_test = np.concatenate(test_labels)
	print("Total test set ", X_full_test.shape)
	return X_full_train, Y_full_train , X_full_test, Y_full_test


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
	    required=True,
	    type=str,
	    help="country to store, options [global, kenya, togo, brazil, all]",
	)
	arg_parser.add_argument(
	    "--out_dir",
	    "-o",
	    required=True,
	    type=str,
	    help="path of the output directory to store the data",
	)
	args = arg_parser.parse_args()
	DATA_DIR = args.data_dir

	if args.country != "global":
		bench_class = CropHarvest.create_benchmark_datasets(DATA_DIR, balance_negative_crops=False, normalize = False)
		bench_class = {"kenya": bench_class[0], "brazil": bench_class[1], "togo": bench_class[2]}

		for country in ["kenya", "togo", "brazil"]:
			if args.country == country or args.country =="all":
				data_benchmark = bench_class[country]
		
			country = data_benchmark.task.test_identifier.split("_")[0]
			crop = data_benchmark.task.target_label
			print("CREATING AND SAVING DATA FROM (%s, %s)"%(country, crop))

			X_full_train, Y_full_train , X_full_test, Y_full_test = extracting_set(data_benchmark)
			storage_set(args.out_dir, country, crop,
				train_data=[X_full_train, Y_full_train],
				test_data=[X_full_test, Y_full_test])

	if args.country == "global" or args.country =="all":
		country,crop = "global", "crop"
		print("CREATING AND SAVING DATA FROM (%s, %s)"%(country, crop))

		data = CropHarvest(DATA_DIR, download=True)
		data.task.normalize = False
		X_full_train, Y_full_train = data.as_array(flatten_x=False)

		print(f"Data set {data} with {len(X_full_train)} examples and shape {X_full_train.shape}")
		storage_set(args.out_dir, country, crop,
			train_data=[X_full_train, Y_full_train])
