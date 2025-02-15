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


reproducibility.clear_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Plugins(categories=["generic"]).list()

assert device.type == 'cuda'

from DGE_data import get_real_and_synthetic

# let's restrict ourselves to classification datasets
datasets = ['moons','circles','adult','breast_cancer','covid','seer'] 
#['moons', 'circles','cal_housing', 'adult', 'diabetes', 'breast_cancer',  'seer', 'cutract' ] 
model_name = 'dpgan'  # synthetic data model
    
p_train = 0.8  # proportion of training data for generative model. Default values if None
n_models = 20  # number of models in ensemble

load = True  # results
load_syn = True  # data
save = True  # save results and data

verbose = False

# for max_n in [10000]:#2000, 5000, 10000]:#, 5000, 10000]:
#     for dataset in datasets:
#         print('Dataset:', dataset)
#         get_real_and_synthetic(dataset=dataset,
#                                 p_train=p_train,
#                                 n_models=n_models,
#                                 model_name=model_name,
#                                 load_syn=load_syn,
#                                 verbose=verbose,
#                                 max_n=max_n)


num_runs = 10
dataset='seer'
for max_n in [2000,5000]:
    nsyn = max_n
    if max_n == 5000:
        num_runs = 1
    X_gt, X_syns = get_real_and_synthetic(dataset=dataset,
                                            p_train=p_train,
                                            n_models=n_models*num_runs,
                                            model_name=model_name,
                                            load_syn=load_syn,
                                            verbose=verbose,
                                            max_n=max_n,
                                            nsyn=nsyn)
