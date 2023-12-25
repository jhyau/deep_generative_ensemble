from sklearn.datasets import load_diabetes
import pickle
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import os
import torch


from synthcity.metrics.eval_performance import (
    PerformanceEvaluatorMLP,
    PerformanceEvaluatorXGB,
)
from synthcity.utils import reproducibility
from synthcity.plugins import Plugins
import synthcity.logger as log
from synthcity.plugins.core.dataloader import GenericDataLoader
from DGE_utils import metric_different_datasets, mean_across_pandas, add_std, get_folder_names

reproducibility.clear_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


assert device.type == 'cuda'
Plugins(categories=["generic"]).list()


# Set up params for boosting experiment
from DGE_data import get_real_and_synthetic, get_real_and_synthetic_with_multiple_models
model_name = 'ctgan'

n_models = 30  # number of models in ensemble, for each run. 20
max_n = 2000  # maximum number of data points to use for training generative model.
nsyn = 2000  # number of synthetic data points per synthetic dataset. Defaults to same as generative training size if None

num_runs = 10 # Number of runs. Don't choose to large, since total number of synthetic datasets is num_runs*n_models. 10

# Per section 4.1, 10 runs with different seeds

# Whether to load and save models and synthetic datasets
load = True  # results
load_syn = True  # data
save = True  # save results and data

verbose = True
plot = False

from DGE_experiments import predictive_experiment, predictive_experiment_stacking, boosting_DGE
import pandas as pd
import time

all_means = {}
all_stds = {}

datasets = ['moons', 'circles', 'breast_cancer', 'adult', 'covid']
#datasets = ['breast_cancer', 'adult', 'covid']
test_only_dataset = ['adult']

model_type = 'deepish_mlp'
boosting = "SAMME"

print("Generative model to run: ", model_name)
print("Downstream classifier model type: ", model_type)
print("boosting method: ", boosting)
print("n_models: ", n_models)
print("num_runs: ", num_runs)
print("datasets: ", datasets)
print("model string: ", model_name)
print("verbose: ", verbose)
print("plot: ", plot)

# Metrics and overall logging folder
finalOutput = f"./results/boosting_{boosting}_nModels{n_models}_{model_name}_{model_type}_runs{num_runs}"

if not os.path.exists(finalOutput):
    os.makedirs(finalOutput)

with open(os.path.join(finalOutput, "exp_info.txt"), "w") as f:
    f.write(f"Generative model to run: {model_name} \n")
    f.write(f"Downstream classifier model type: {model_type} \n")
    f.write(f"boosting method: {boosting} \n")
    f.write(f"n_models: {n_models} \n")
    f.write(f"max_n for training gen mode: {max_n} \n")
    f.write(f"nsyn for synthetic datapoint to generate per dataset: {nsyn} \n")
    f.write(f"num_runs: {num_runs} \n")
    f.write(f"datasets: {datasets} \n")
    f.write(f"model string: {model_name} \n")
    f.write(f"load: {load} \n")
    f.write(f"load_syn: {load_syn} \n")
    f.write(f"save: {save} \n")
    f.write(f"verbose: {verbose} \n")
    f.write(f"plot: {plot} \n")

start_time = time.time()
for dataset in datasets:
    workspace_folder, results_folder = get_folder_names(
        dataset, model_name, max_n=max_n, nsyn=nsyn)
    
    # For toy runs
    #workspace_folder = f"{workspace_folder}_{boosting}_nModels{n_models}_full_exp_toy"
    #results_folder = f"{results_folder}_{boosting}_nModels{n_models}_full_exp_toy"
    workspace_folder = f"{workspace_folder}_{boosting}_nModels{n_models}_w_origDGE_benchmark"
    results_folder = f"{results_folder}_{boosting}_nModels{n_models}_w_origDGE_benchmark"
    if not os.path.exists(workspace_folder):
        os.makedirs(workspace_folder)

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    print(f'Dataset {dataset}\n')
    print("workspace_folder: ", workspace_folder)
    print("results_folder: ", results_folder)

    means, stds, _ = boosting_DGE(dataset, model_name, num_runs=num_runs, num_iter=n_models, boosting=boosting, p_train=0.8,
                                  max_n=max_n, nsyn=nsyn, reduce_to=20000, task_type=model_type, workspace_folder=workspace_folder,
                                  save=save, load=load, verbose=verbose, plot=plot)
    print("printing weighted avg means to latex:")
    print(means.to_latex())
    print("printing weighted stds to latex:")
    print(stds.to_latex())

    all_means[dataset] = means
    all_stds[dataset] = stds

    print("size of means: ", means.shape)
    print("mean elements: ", means)

end_time = time.time()
time_elapsed = end_time - start_time
print("Time it took to run the experiment: ", time_elapsed)

with open(os.path.join(finalOutput, "exp_info.txt"), "a") as f:
    f.write(f"Time it took to run the experiment: {time_elapsed} \n")


# Metrics
#finalOutput = f"./results/boosting_{boosting}_nModels{n_models}_{model_name}_{model_type}_runs{num_runs}"

#if not os.path.exists(finalOutput):
#    os.makedirs(finalOutput)

# Print results, aggregated over different datasets
means_consolidated = metric_different_datasets(all_means, to_print=False)
if num_runs>1:
    stds_consolidated = metric_different_datasets(all_stds, to_print=False)
    stds_consolidated.drop(columns=['Mean'], inplace=True)
    print(add_std(means_consolidated, stds_consolidated).to_latex())

    with open(os.path.join(finalOutput, "final_metrics.txt"), "w") as f:
        f.write(add_std(means_consolidated, stds_consolidated).to_string() + "\n")
        f.write("*****************************Latex format****************************** \n")
        f.write(add_std(means_consolidated, stds_consolidated).to_latex() + "\n")
else:
    print(means_consolidated.to_latex())
    with open(os.path.join(finalOutput, "final_metrics.txt"), "w") as f:
        f.write(means_consolidated.to_string() + "\n")
        f.write("*****************************Latex format***************************** \n")
        f.write(means_consolidated.to_latex() + "\n")

