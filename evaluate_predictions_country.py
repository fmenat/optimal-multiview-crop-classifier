import yaml
import argparse
import os
import sys
import time
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib
matplotlib.rc('font', **{"size":14})

from src.datasets.views_structure import DataViews, load_structure
from src.metrics.metrics import ClassificationMetrics, SoftClassificationMetrics
from src.visualizations.utils import save_results, get_values_xarray
from src.visualizations.tools import plot_prob_dist_bin, plot_conf_matrix

def classification_metric(
                preds_p_run,
                indexs_p_run,
                data_ground_truth,
                ind_save,
                show=True,
                plot_runs = False,
                train_data = [],
                include_metrics = [],
                dir_folder = ""
                ):
    R = len(preds_p_run)

    df_runs_diss = {"year": [], "continent": []}
    time_runs = []
    for r in range(R):
        true_info, y_pred = get_values_xarray(data_ground_truth, indexs_p_run[r], values=["target", "year", "country", "continent"]), preds_p_run[r]
        y_true = true_info["target"]
        y_true = np.squeeze(y_true)
        y_pred_prob = np.squeeze(y_pred)
        y_pred_prob_no_missing = y_pred_prob[y_true != -1]
        y_true_no_missing = y_true[y_true != -1]

        y_pred_no_missing = np.argmax(y_pred_prob_no_missing, axis = -1)

        for info_value in df_runs_diss.keys():
            info_results = []
            for v in np.unique(true_info[info_value]):
                mask_info = true_info[info_value] == v
                d_me = ClassificationMetrics()
                info_results.append( d_me(y_pred_no_missing[mask_info], y_true_no_missing[mask_info]) )
            df_des = pd.DataFrame(info_results)
            df_des.index = np.unique(true_info[info_value])
            df_runs_diss[info_value].append(df_des)

        if plot_runs:
            print(f"Run {r} being shown")
            print(df_des.round(4).to_markdown())

    for v in df_runs_diss:
        df_concat_diss = pd.concat(df_runs_diss[v]).groupby(level=0)
        df_mean_diss = df_concat_diss.mean()
        df_std_diss = df_concat_diss.std()

        save_results(f"{dir_folder}/plots/{ind_save}/preds_ind_mean_{v}", df_mean_diss)
        save_results(f"{dir_folder}/plots/{ind_save}/preds_ind_std_{v}", df_std_diss)
        if show:
            print(df_mean_diss.round(4).to_markdown())
    return df_mean_diss, df_std_diss

def load_data_sup(data_name, method_name, dir_folder="", **args):
    files_load = [str(v) for v in Path(f"{dir_folder}/pred/{data_name}/{method_name}").glob(f"*.csv")]
    files_load.sort()

    preds_p_run = []
    indxs_p_run = []
    for file_n in files_load:
        data_views = pd.read_csv(file_n, index_col=0) #load_structure(file_n)
        preds_p_run.append(data_views.values)
        indxs_p_run.append(list(data_views.index))
    return preds_p_run,indxs_p_run

def calculate_metrics(data_tr,data_te,data_name, method, **args):
    preds_p_run_tr, indexs_p_run_tr = load_data_sup(data_name+"/train", method, **args )
    preds_p_run_te, indexs_p_run_te = load_data_sup(data_name+"/test", method, **args)

    classification_metric(
                        preds_p_run_te,
                        indexs_p_run_te,
                        data_te,
                        ind_save=f"{data_name}/{method}/",
                        show=True,
                        train_data = [preds_p_run_tr,indexs_p_run_tr, data_tr],
                        **args
                        )

def ensemble_avg(method_names, data_tr,data_te,data_name, method="EnsembleAVG", **args):
    preds_p_run_tr, indexs_p_run_tr = [], []
    preds_p_run_te, indexs_p_run_te = [], []
    for method_n in method_names:
        preds_p_run_tr_a, indexs_p_run_tr = load_data_sup(data_name+"/train", method_n, **args )
        preds_p_run_te_a, indexs_p_run_te = load_data_sup(data_name+"/test", method_n, **args )
        preds_p_run_tr.append(preds_p_run_tr_a)
        preds_p_run_te.append(preds_p_run_te_a)
    preds_p_run_tr = np.mean(preds_p_run_tr, axis = 0)
    preds_p_run_te = np.mean(preds_p_run_te, axis = 0)

    classification_metric(
                        preds_p_run_te,
                        indexs_p_run_te,
                        data_te,
                        ind_save=f"{data_name}/{method}/",
                        show=True,
                        train_data = [preds_p_run_tr,indexs_p_run_tr, data_tr],
                        **args
                        )

def main_evaluation(config_file):
    input_dir_folder = config_file["input_dir_folder"]
    output_dir_folder = config_file["output_dir_folder"]
    data_name = config_file["data_name"]
    include_metrics = ["f1 bin", "p bin"]

    data_tr = xr.open_dataset(f"{input_dir_folder}/{data_name}_train_ext.nc")
    data_te = xr.open_dataset(f"{input_dir_folder}/{data_name}_test_ext.nc")

    if config_file.get("methods_to_plot"):
        methods_to_plot = config_file["methods_to_plot"]
    else:
        methods_to_plot = sorted(os.listdir(f"{output_dir_folder}/pred/{data_name}/test"))

    pool_names = {}
    for method in methods_to_plot:
        print(f"Evaluating method {method}")
        calculate_metrics(data_tr, data_te,
                        data_name,
                        method,
                        include_metrics=include_metrics,
                        plot_runs=config_file.get("plot_runs"),
                        dir_folder=output_dir_folder,
                        )
        if method.lower().startswith("pool"):
            key_pool = method.lower().split("_")[0].split("pool")[-1]
            if key_pool =="":
                key_pool = "_"
            if key_pool in pool_names:
                pool_names[key_pool].append(method)
            else:
                pool_names[key_pool] = [method]
    if len(pool_names) != 0:
        for key_pool in pool_names:
            print(f"Evaluating Ensemble method with {key_pool}")
            ensemble_avg(pool_names[key_pool], data_tr,data_te,data_name,
                    include_metrics=include_metrics,
                    plot_runs=config_file.get("plot_runs"),
                    dir_folder=output_dir_folder,
                    method="EnsembleAVG"+key_pool,
                    )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--settings_file",
        "-s",
        action="store",
        dest="settings_file",
        required=True,
        type=str,
        help="path of the settings file",
    )
    args = arg_parser.parse_args()
    with open(args.settings_file) as fd:
        config_file = yaml.load(fd, Loader=yaml.SafeLoader)

    main_evaluation(config_file)
