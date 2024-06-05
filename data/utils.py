import numpy as np
import sys
sys.path.insert(1, '../')

from src.datasets.views_structure import DataViews #to store data

FEATURES_ = [
		"S2_1","S2_2","S2_3","S2_4","S2_5","S2_6","S2_7","S2_8","S2_9","S2_10","S2_11",
		"S1_VV","S1_VH",
		"ERA5_precip","ERA5_temp",
		"DEM_eleva","DEM_slope",
		"S2i_nvdi"
	]

def storage_set(path_out_dir, country, crop, train_data, test_data = [], target_names=["negative", "positive"]):
	"""
		Function to store the data. This function assumes that the features/channels of 
		input data X comes in the same order as FEATURES_
	"""
	X_full_train, Y_full_train = train_data
	if len(test_data) != 0 :
		X_full_test, Y_full_test = test_data

	indx_sh = np.arange(len(X_full_train)) #just to shuffle training index
	np.random.shuffle(indx_sh)
	X_full_train = X_full_train[indx_sh]
	Y_full_train = Y_full_train[indx_sh]

	N, T, D = X_full_train.shape
	idxs_S2 = [d for d in np.arange(D) if "S2_" in FEATURES_[d]]
	idxs_S1 = [d for d in np.arange(D) if "S1_" in FEATURES_[d]]
	idxs_ERA5 = [d for d in np.arange(D) if "ERA5_" in FEATURES_[d]]
	idxs_DEM = [d for d in np.arange(D) if "DEM_" in FEATURES_[d]]
	idxs_VI = [d for d in np.arange(D) if "S2i_" in FEATURES_[d]] 

	data_views = DataViews()
	data_views.add_view(X_full_train[:,:,idxs_S2], indx_sh, "S2")
	data_views.add_view(X_full_train[:,:,idxs_S1], indx_sh, "S1")
	data_views.add_view(X_full_train[:,:,idxs_ERA5], indx_sh, "weather")
	data_views.add_view(X_full_train[:,0,idxs_DEM], indx_sh, "DEM")
	data_views.add_view(X_full_train[:,:,idxs_VI], indx_sh, "VI")
	data_views.add_target(Y_full_train, indx_sh, target_names=target_names)

	if len(test_data) != 0 : #test data is given, then store 

		data_views.save(f"{path_out_dir}/cropharvest_{country}_{crop}_train", xarray=True)
		data_views.save(f"{path_out_dir}/cropharvest_{country}_{crop}_train", xarray=False)

		indx_sh = np.arange(len(X_full_test)) #just to shuffle testing index
		np.random.shuffle(indx_sh)
		X_full_test = X_full_test[indx_sh]
		Y_full_test = Y_full_test[indx_sh]

		N2, T, D = X_full_test.shape
		data_aux = DataViews()
		data_aux.add_view(X_full_test[:,:,idxs_S2], N+indx_sh, "S2")
		data_aux.add_view(X_full_test[:,:,idxs_S1], N+indx_sh, "S1")
		data_aux.add_view(X_full_test[:,:,idxs_ERA5], N+indx_sh, "weather")
		data_aux.add_view(X_full_test[:,0,idxs_DEM], N+indx_sh, "DEM")
		data_aux.add_view(X_full_test[:,:,idxs_VI], N+indx_sh, "VI")
		data_aux.add_target(Y_full_test, N+indx_sh, target_names=target_names)

	else: #if test data no given, create (usually for global)

		mask_train = np.random.rand(N) <= 0.70
		indx_train = np.arange(N)[~mask_train]
		data_views.set_test_mask(indx_train)

		train_views = data_views.generate_full_view_data(train = True)
		data_aux = DataViews(train_views["views"], train_views["identifiers"], view_names = train_views["view_names"])
		data_aux.add_target(train_views["target"], train_views["identifiers"], target_names=target_names)
		
		data_aux.save(f"{path_out_dir}/cropharvest_{country}_{crop}_train", xarray=True)
		data_aux.save(f"{path_out_dir}/cropharvest_{country}_{crop}_train", xarray=False)
		print("The size of each view in training ", {k: v.shape for k,v in zip(train_views["view_names"],train_views["views"])})

		test_views = data_views.generate_full_view_data(train = False)
		data_aux = DataViews(test_views["views"], test_views["identifiers"], view_names = test_views["view_names"])
		data_aux.add_target(test_views["target"], test_views["identifiers"], target_names=target_names)
	
		
	print(f"data stored in {path_out_dir}/cropharvest_{country}_{crop}_test")
	data_aux.save(f"{path_out_dir}/cropharvest_{country}_{crop}_test", xarray=True)
	data_aux.save(f"{path_out_dir}/cropharvest_{country}_{crop}_test", xarray=False)