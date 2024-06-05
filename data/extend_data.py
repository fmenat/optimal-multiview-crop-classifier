import argparse
import xarray as xr
from pathlib import Path
import pandas as pd

def extend_dataset(file_name, df_metadata, out_name):
    arr = xr.open_dataset(file_name)

    meta_4_ids = df_metadata.iloc[arr.identifier.values]

    arr["year"] = xr.DataArray(data=meta_4_ids["year"].values.astype(int), 
                               dims=["identifier"], coords={"identifier":arr.identifier.values})
    arr["country"] = xr.DataArray(data=meta_4_ids["country"].values, 
                               dims=["identifier"], coords={"identifier":arr.identifier.values})
    arr["continent"] = xr.DataArray(data=meta_4_ids["continent"].values, 
                               dims=["identifier"], coords={"identifier":arr.identifier.values})

    arr.to_netcdf(f"{out_name}_ext.nc", engine="h5netcdf") 

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
	    "--metadata_dir",
	    "-m",
	    required=True,
	    type=str,
	    help="path of the data directory",
	)
	args = arg_parser.parse_args()
	DATA_DIR = args.data_dir

	out_name = Path(DATA_DIR).stem
	df_metadata = pd.read_csv(args.metadata_dir, index_col=0)
	#small filter:
	df_metadata.drop(index=[df_metadata.index[43272], df_metadata.index[65244]], inplace=True)
	extend_dataset(DATA_DIR, df_metadata, out_name)