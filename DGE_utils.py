import xgboost
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, mean_absolute_error, log_loss, brier_score_loss
from sklearn.model_selection import KFold, train_test_split
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier, MLPRegressor    
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.utils import reproducibility
from sklearn.ensemble import StackingClassifier, StackingRegressor


import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import torch

from mpl_toolkits.axes_grid1 import make_axes_locatable

from hashlib import sha256

def hash_str2int(s):
    """
    Hash a string to an integer
    """
    
    return int(sha256(s.encode('utf-8')).hexdigest(), 16) % (2**32)


def accuracy_confidence_curve(y_true, y_prob, n_bins=20):
    thresholds = np.linspace(0.5, 0.95, n_bins)
    y_pred = y_prob > 0.5
    correct = y_true == y_pred
    accs = np.zeros(n_bins)
    
    for i, threshold in enumerate(thresholds):
        y_select = (y_prob > threshold) + (y_prob < (1 - threshold))
        accs[i] = np.mean(correct[y_select])
        
    return thresholds, accs

def cat_dl(X_syns, n_limit=None):
    """
    Concatenate a list of GenericDataLoader objects into one GenericDataLoader object
    """
    if n_limit is not None:
        X_syn_cat = pd.concat([X_syns[i][:n_limit]
                              for i in range(len(X_syns))], axis=0)
    else:
        X_syn_cat = pd.concat([X_syns[i].dataframe()
                              for i in range(len(X_syns))], axis=0)
    X_syn_cat = GenericDataLoader(X_syn_cat, target_column="target")
    X_syn_cat.targettype = X_syns[0].targettype
    return X_syn_cat


def parallel_for(func, args_list, max_workers=4):
    from concurrent.futures import ThreadPoolExecutor

    # Create a ThreadPoolExecutor with the desired number of threads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use the executor to map the function over the list of arguments
        results = executor.map(func, args_list)

    return results


def init_model(model_type, targettype):
    """
    Initialize a model of the given type.
    """
    if model_type == 'lr':
        if targettype == 'classification':
            model = sklearn.linear_model.LogisticRegression()
        else:
            model = sklearn.linear_model.LinearRegression()
    elif model_type == 'smallest_mlp':
        if targettype == 'classification':
            model = MLPClassifier(hidden_layer_sizes=(50))
        else:
            model = MLPRegressor(hidden_layer_sizes=(50))
    elif model_type == 'mlp':
        if targettype == 'classification':
            model = MLPClassifier(hidden_layer_sizes=(100))
        else:
            model = MLPRegressor(hidden_layer_sizes=(100))
    elif model_type == 'deepish_mlp':
        if targettype == 'classification':
            model = MLPClassifier(
                hidden_layer_sizes=(100, 100))
        else:
            model = MLPRegressor(
                hidden_layer_sizes=(100, 100))
    
    elif model_type == 'deep_mlp':
        if targettype == 'classification':
            model = MLPClassifier(
                hidden_layer_sizes=(100, 100, 100))
        else:
            model = MLPRegressor(
                hidden_layer_sizes=(100, 100, 100))
    elif model_type == 'largest_mlp':
        if targettype == 'classification':
            model = MLPClassifier(
                hidden_layer_sizes=(500, 500, 500))
        else:
            model = MLPRegressor(
                hidden_layer_sizes=(500, 500, 500))
    elif model_type == 'rf':
        # default 100 trees
        if targettype == 'classification':
            model = sklearn.ensemble.RandomForestClassifier()
        else:
            model = sklearn.ensemble.RandomForestRegressor()
    elif model_type == 'knn':
        # default 5 neighbors
        if targettype == 'classification':
            model = sklearn.neighbors.KNeighborsClassifier()
        else:
            model = sklearn.neighbors.KNeighborsRegressor()
    elif model_type == 'svm':
        # default rbf kernel
        if targettype == 'classification':
            model = sklearn.svm.SVC(probability=True)
        else:
            model = sklearn.svm.SVR()
    elif model_type == 'xgboost':
        if targettype == 'classification':
            model = xgboost.XGBClassifier()
        else:
            model = xgboost.XGBRegressor()
    else:
        raise ValueError('Unknown model type')
    
    # Wrap the model in a pipeline to scale the data
    #! Add scaling only of continuous and encoding of categorical
    model = Pipeline([('scaler', StandardScaler()), ('model', model)])
    return model


def supervised_task(X_gt, X_syn, model=None, model_type='mlp', verbose=False):

    print("supervised task: ", X_gt.targettype)
    print("model type: ", model_type)
    if type(model) == str or model is None:
        model = init_model(model_type, X_syn.targettype)
        X, y = X_syn.unpack(as_numpy=True)
        model.fit(X, y.reshape(-1, 1))
        
    if X_gt.targettype == 'regression':
        pred = model.predict(X_gt.unpack(as_numpy=True)[0])
    else:
        pred = model.predict_proba(X_gt.unpack(as_numpy=True)[0])[:, 1]    
    print("shape of pred in supervised task: ", pred.shape)
    return pred, model


def roc_auc_score_rob(y_true, y_score, throw_error_if_nan=True):
    """
    Robust version of sklearn.metrics.roc_auc_score
    """
    if len(np.unique(y_true))>1 or throw_error_if_nan:
        roc_auc_score(y_true, y_score)
    else:
        return np.nan

def compute_metrics(y_test, yhat_test, targettype='classification'):
    if targettype == 'classification':
        y_test = y_test.astype(bool)
        yhat_test = yhat_test.astype(float) 
        metrics = ['AUC', 'Acc', 'F1', 'Precision', 'Recall', 'NLL', 'Brier',]
        scores = [roc_auc_score(y_test, yhat_test), accuracy_score(y_test, yhat_test>0.5),
                  f1_score(y_test, yhat_test>0.5), precision_score(
                      y_test, yhat_test>0.5), recall_score(y_test, yhat_test>0.5),
                  log_loss(y_test, yhat_test, labels=[0,1]), brier_score_loss(y_test, yhat_test)]
    elif targettype == 'regression':
        metrics = ['RMSE', 'MAE']
        scores = [np.sqrt(mean_squared_error(
            y_test, yhat_test)), mean_absolute_error(y_test, yhat_test)]
    else:
        raise ValueError('unknown target type')
    
    #scores = np.round(scores, 3)
    scores = np.array(scores).reshape(1, -1)
    scores = pd.DataFrame(scores, columns=metrics)
    return scores


def tt_predict_performance(X_test, X_train, model=None, model_type='mlp', subset=None, verbose=False):
    """compute train_test performance for different metrics"""
    # import metrics

    x_train, y_train = X_train.unpack(as_numpy=True)
    if subset is not None:
        X_test = subset(X_test)
    x_test, y_test = X_test.unpack(as_numpy=True)
    
    if model is None:
        model = init_model(model_type, X_test.targettype)
        model.fit(x_train, y_train)

    if X_test.targettype == 'regression':
        yhat_test = model.predict(x_test)
    else:
        yhat_test = model.predict_proba(x_test)[:, 1]
    
    scores = compute_metrics(y_test, yhat_test, X_test.targettype)
    return scores, model


def aggregate_predictive(X_gt, X_syns, task=tt_predict_performance, models=None, task_type='', workspace_folder=None, results_folder=None, load=True, save=True,
                         approach='DGE', relative=False, run_for_all=True, verbose=False, K=None, subset=None):
    """
    aggregate predictions from different synthetic datasets
    """

    results = []
    stds = []
    trained_models = []
    fileroot = os.path.join(workspace_folder, f'model_eval_{task_type}')
    
    if K is None:
        K = len(X_syns)
    
    if not os.path.exists(fileroot) and save:
        os.makedirs(fileroot)
    
    if run_for_all:
        range_limit = len(X_syns)
    else:
        range_limit = 1
    
    for i in range(range_limit):
        filename = f'{fileroot}_{i}.pkl'
        if models is None:
            if os.path.exists(filename) and load:
                model = pickle.load(open(filename, "rb"))
            else:
                model = None
                if verbose:
                    print(f'Train model {i+1}/{len(X_syns)} and save as {filename}')
        else:
            model = models[i]
        reproducibility.enable_reproducible_results(random_state=i+2022)
        X_train = X_syns[i].train()
        if approach == 'Naive':
            X_test = X_syns[i].test()
        elif 'alternative' in approach:
            X_syns_not_i = [X_syns[j] for j in range(len(X_syns)) if j != i][:K-1]
        elif 'DGE' in approach:
            X_syns_not_i = [X_syns[j] for j in range(len(X_syns)) if j != i][:K-1]
            X_syns_not_i[0].targettype = X_syns[0].targettype
            X_test = cat_dl(X_syns_not_i)
        elif approach == 'Oracle':
            X_test = X_gt.test()
        else:
            raise ValueError('Unknown approach')

        if not 'alternative' in approach:
            X_test.targettype = X_syns[0].targettype
            X_train.targettype = X_syns[0].targettype

            res, model = task(X_test, X_train, model, task_type, subset=subset, verbose=verbose)

            if relative and approach != 'Oracle':
                X_test = X_gt.test()
                X_test.targettype = X_syns[0].targettype
                res_oracle, model = task(X_test, X_train, model, task_type, subset=subset, verbose=verbose)

                if relative in ['l2']:
                    res = (res - res_oracle)**2
                elif relative == 'l1':
                    res = (res - res_oracle).abs()
                else:
                    raise ValueError('Unknown relative metric')

        else:
            if relative:
                raise ValueError('Relative not implemented for DGE_alternative')
            res = []
            for j in range(len(X_syns_not_i)):
                X_test = X_syns_not_i[j].test()
                X_test.targettype = X_syns[0].targettype
                X_train.targettype = X_syns[0].targettype
                res.append(task(X_test, X_train, model, task_type, subset=subset, verbose=verbose)[0])
            res, std = meanstd(pd.concat(res, axis=0))
            stds.append(std)

        results.append(res)
        trained_models.append(model)
        # save model to disk as pickle
        if models is None and save:
            pickle.dump(model, open(filename, "wb"))

    results = pd.concat(results, axis=0)
    if approach != 'DGE_alternative':
        return *meanstd(results), trained_models, results
    else:
        stds = pd.concat(stds, axis=0)
        stds = stds.mean(axis=0).to_frame().T
        means, stds2 = meanstd(results)
        return means, (stds**2+stds2**2)**0.5, trained_models, None



def meanstd(A):
    if type(A) == pd.DataFrame:
        return A.mean(axis=0).to_frame().T, A.std(axis=0).to_frame().T
    else:
        return np.mean(A, axis=0), np.std(A, axis=0)


def weighted_meanstd(A, weights):
    print("calculated weighted means/stds with numpy")
    print(f"A shape: {len(A)}, weight shape: {len(weights)}")
    weighted_mean = np.average(A, weights=weights, axis=0)
    weighted_variance = np.average((A - weighted_mean)**2, weights=weights, axis=0)
    return weighted_mean, np.sqrt(weighted_variance)


def aggregate(X_gt, X_syns, task, models=None, task_type='', load=True, save=True, workspace_folder=None, filename='', verbose=False):
    """
    aggregate predictions from different synthetic datasets
    """

    results = []
    trained_models = []
    fileroot = f'{workspace_folder}/{task.__name__}_{task_type}'
    print("fileroot in aggregate: ", fileroot)
    print("task function name: ", task.__name__)
    print("task type: ", task_type)
    
    if (save or load) and not os.path.exists(fileroot):
        os.makedirs(fileroot)

    for i in range(len(X_syns)):
        full_filename = f'{fileroot}_{filename}_{i}.pkl'
        if models is None:
            if verbose:
                print(f'Saving model as {full_filename}')

            if os.path.exists(full_filename) and load:
                print("loading existing model...")
                model = pickle.load(open(full_filename, "rb"))
            else:
                print("Train model in aggregate")
                model = None
                if verbose:
                    print(f'Train model {i+1}/{len(X_syns)}')
                seed = hash_str2int(full_filename)
                reproducibility.enable_reproducible_results(random_state=seed)
        else:
            model = models[i]

        res, model = task(X_gt, X_syns[i], model, task_type, verbose)
        results.append(res)
        trained_models.append(model)
        # save model to disk as pickle
        if models is None and save:
            pickle.dump(model, open(full_filename, "wb"))

    return *meanstd(results), trained_models


def aggregate_stacking_folds(X_gt, X_syns, task, meta_model='lr', mixed_models=False, estimators_list=None, task_type='', load=True, save=True, workspace_folder=None, filename='', verbose=False, cv=5, stack_method='predict_proba'):
    """
    X_syns is a list of repeated datasets (all datasets in the list are the same)
    default if 5 folds
    """
    print("In aggregate stacking folds")
    X, y = X_syns[0].unpack(as_numpy=True)
    if not mixed_models:
        model_str = task_type
        estimators = []
        for i in range(len(X_syns)):
            estimators.append((f'{task_type}_{i}', init_model(task_type, X_syns[0].targettype)))
    else:
        model_str = ''
        for x in task_type:
            model_str += x + "_"
            
        estimators = []
        for i in range(len(X_syns)):
            # Cycle through the list of task type models
            index = i % len(task_type)
            model_type = task_type[index]
            print("model in estimators stack: ", model_type)
            if estimators_list is not None:
                # Access the named_estimators_, Bunch type in sklearn
                # a.named_estimators_['deepish_mlp_0']
                name = f'{model_type}_{i}'
                estimators.append((name, estimators_list[name]))
            else:
                estimators.append((f'{model_type}_{i}', init_model(model_type, X_syns[0].targettype)))

    fileroot = f'{workspace_folder}_stacking_folds/StackingClassifier_{model_str}_{cv}folds_{len(X_syns)}_classifiers_meta_{meta_model}'
    if (save or load) and not os.path.exists(fileroot):
        os.makedirs(fileroot)

    full_filename = f'{fileroot}_{filename}_{len(X_syns)}_concat.pkl'

    print("StackingClassifier full path: ", full_filename)
    if X_gt.targettype == 'regression':
        stack_method = 'predict'
    else:
        stack_method = 'predict_proba'
    if verbose:
        verbosity = 3
    else:
        verbosity = 0
   
    if estimators_list is not None:
        print("Using trained/fitted estimators...")
        cv = "prefit"
    
    if os.path.exists(full_filename) and load:
        print("load existing StackingClassifier ensemble of estimators")
        clf = pickle.load(open(full_filename, "rb"))
    else:
        print("training new StackingClassifier ensemble with cv: ", cv)
        if X_gt.targettype == 'regression':
            clf = StackingRegressor(estimators=estimators, final_estimator=init_model(meta_model, X_gt.targettype), cv=cv, stack_method=stack_method, verbose=verbosity)
        else:
            clf = StackingClassifier(estimators=estimators, final_estimator=init_model(meta_model, X_gt.targettype), cv=cv, stack_method=stack_method, verbose=verbosity)
        clf.fit(X, y.reshape(-1,1))

    if not os.path.exists(full_filename) and save:
        pickle.dump(clf, open(full_filename, "wb"))

    if X_gt.targettype == 'regression':
        meta_pred = clf.predict(X_gt.unpack(as_numpy=True)[0])
    else:
        meta_pred = clf.predict_proba(X_gt.unpack(as_numpy=True)[0])[:, 1]
    return meta_pred, clf.named_estimators_


def supervised_task_stacking(X_gt, X_syn, model=None, model_type='mlp', verbose=False):

    print("supervised task: ", X_gt.targettype)
    print("model type: ", model_type)
    X, y = X_syn.unpack(as_numpy=True)
    if type(model) == str or model is None:
        print("initializing new model in supervised task stacking")
        model = init_model(model_type, X_syn.targettype)
        #X, y = X_syn.unpack(as_numpy=True)
        model.fit(X, y.reshape(-1, 1))
        
    if X_gt.targettype == 'regression':
        # Predict on train data for meta learner later
        train_pred = model.predict(X)
        pred = model.predict(X_gt.unpack(as_numpy=True)[0])
    else:
        train_pred = model.predict_proba(X)[:, 1]
        pred = model.predict_proba(X_gt.unpack(as_numpy=True)[0])[:, 1]    
    return train_pred, y.reshape(-1,1), pred, model


def aggregate_stacking(X_gt, X_syns, task, meta_model='lr', mixed_models=False, models=None, task_type='', load=True, save=True, workspace_folder=None, filename='', verbose=False):
    """
    aggregate predictions from different synthetic datasets with stacking with a meta learning model won't work for the meta learning model
    each downstream classifier must be trained on the same synthetic train data for meta learner to be able to work, so combine all synthetic datasets when training classifiers in this case
    task needs to be one of the tasks modified to include stacking changes like supervised_task_stacking
    If mixed_models is True, use different models here, provide a list of downstream models in task_type 
    Else use the same model type

    TODO: change so each level-0 estimator still only trains on one synthetic dataset, but have each evaluate on the entire concatenated synthetic dataset
    OR
    split each synthetic dataset into train/val so there is a val set where none of the level-0 estimators have seen the data (in case using the full concatenated
    synthetic dataset has any fear of skewing towards the level-0 estimator that was trained on a given data(?))
    """
    if task.__name__.find("stacking") == -1:
        raise ValueError('task is not for stacking method')

    print("X_gt size: ", X_gt.shape)
    train_results = []
    #train_labels = []
    results = []
    trained_models = []
    fileroot = f'{workspace_folder}_stacking/{task.__name__}_{task_type}_stacking_{len(X_syns)}_classifiers'

    #if (save or load) and not os.path.exists(fileroot):
    #    os.makedirs(fileroot)

    for i in range(len(X_syns)):
        if mixed_models:
            # Cycle through the list of task type models
            index = i % len(task_type)
            model_type = task_type[index]
            fileroot = f'{workspace_folder}_mixed_stacking/{task.__name__}_{model_type}_mix_stacking_{len(X_syns)}_classifiers'

        if (save or load) and not os.path.exists(fileroot):
            os.makedirs(fileroot)
        
        full_filename = f'{fileroot}_{filename}_{i}.pkl'
        if models is None:
            if verbose:
                print(f'Saving model as {full_filename} in stacking aggregate')

            if os.path.exists(full_filename) and load:
                print("loading existing model in aggregate stacking...")
                model = pickle.load(open(full_filename, "rb"))
            else:
                print("training new model in aggregate stacking...")
                model = None
                if verbose:
                    print(f'Train model {i+1}/{len(X_syns)}')
                seed = hash_str2int(full_filename)
                reproducibility.enable_reproducible_results(random_state=seed)
        else:
            print("models is not none, using passed in models...")
            model = models[i]

        if mixed_models:
            print("mixed downstream models")
            train_res, train_label, res, model = task(X_gt, X_syns[i], model, model_type, verbose)
        else:
            print("same downstream models")
            train_res, train_label, res, model = task(X_gt, X_syns[i], model, task_type, verbose)
        train_results.append(train_res)
        #train_labels.append(train_label)
        results.append(res)
        trained_models.append(model)
        # save model to disk as pickle
        if models is None and save:
            pickle.dump(model, open(full_filename, "wb"))

    print("res size: ", res.shape)
    print("train_label shape: ", train_label.shape)
    # After training the len(X_syns) of downstream models, need to train the meta learning model
    meta_filename = f'{fileroot}_{filename}_meta_learner.pkl'
    if os.path.exists(meta_filename) and load:
        print("loading existing meta learning model...")
        meta_learn_model = pickle.load(open(meta_filename, "rb"))
    else:
        print("training meta learning model...")
        meta_learn_model = init_model(meta_model, X_syns[0].targettype)
        # For training, pass along trained model's outputs for each trained model on same dataset, concatenate them all into one array where each feature is each model's prediction, and train data label. Reshape bc only one feature in labels
        if len(X_syns) > 1:
            all_train_preds = np.vstack(train_results).T
        else:
            all_train_preds = train_results[0].reshape(-1, 1)
        all_train_y = train_label #np.concatenate(train_labels, axis=0).reshape(-1,1)
        print("size of all synthetic train data pred results from downstream model: ", all_train_preds.shape)
        print("size of all synthetic train data y: ", all_train_y.shape)
        meta_learn_model.fit(all_train_preds, all_train_y)
        if save:
            pickle.dump(meta_learn_model, open(meta_filename, "wb"))
    # For test/inference predictions
    print("Shape of results to be input to meta learner for predictions: ", np.vstack(results).T.shape)
    if X_syns[0].targettype == 'regression':
        meta_pred = meta_learn_model.predict(np.vstack(results).T)
    else:
        meta_pred = meta_learn_model.predict_proba(np.vstack(results).T)[:, 1]
    # Add trained meta learner to trained_models lister
    trained_models.append(meta_learn_model)
    print("size of meta_pred: ", meta_pred.shape)
    print("size of one result in list of results: ", results[0].shape)
    return meta_pred, *meanstd(results), trained_models


def tsne(X):
    """
    Perform t-SNE dimensionality reduction to two dimensions
    """
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(X)
    return X_2d


def aggregate_imshow(X_gt, X_syns, task, models=None, task_type='', results_folder=None, workspace_folder='workspace', load=True, save=True, filename='', baseline_contour=None):
    """
    Aggregate and plot predictions from different synthetic datasets, on a 2D space. E.g., density estimation, predictions.
    """

    xmin = ymin = np.min(X_gt.train().unpack(as_numpy=True)[0])
    xmax = ymax = np.max(X_gt.train().unpack(as_numpy=True)[0])
    
    steps = 400
    X_grid = np.linspace(xmin, xmax, steps)
    Y_grid = np.linspace(ymin, ymax, steps)

    X_grid, Y_grid = np.meshgrid(X_grid, Y_grid)
    Z_grid = pd.DataFrame(np.c_[X_grid.ravel(), Y_grid.ravel()])
    Z_grid['target'] = -1
    Z_grid = GenericDataLoader(Z_grid, target_column="target", train_size=0.01)
    Z_grid.targettype = X_syns[0].targettype

    y_pred_mean, y_pred_std, models = aggregate(
        Z_grid, X_syns,
        task=task,
        models=models,
        task_type=task_type,
        load=load,
        save=save,
        workspace_folder=workspace_folder,
        filename=filename)

    contour = [X_grid, Y_grid, y_pred_mean.reshape(steps, steps)]
        
    for y, stat in zip((y_pred_mean, y_pred_std), ('mean', 'std')):
        fig = plt.figure(figsize=(3, 2.5), dpi=300, tight_layout=True)
        ax = plt.axes()
        if 'oracle' in filename.lower():
            plt.contour(X_grid, Y_grid, contour[2], levels = [0.5], colors='w', linestyles=':')
        else:
            plt.contour(X_grid, Y_grid, contour[2], levels = [0.5], colors='r', linestyles='--')

        if baseline_contour is not None:
            plt.contour(baseline_contour[0], baseline_contour[1], baseline_contour[2], levels = [0.5], colors='w', linestyles=':')

        im = ax.imshow(y.reshape(steps, steps),
                   cmap='viridis', extent=[xmin, xmax, ymin, ymax], origin='lower')

        ax.set_aspect('equal', 'box')
        
        # if 'gaussian' in results_folder:
        #     plt.vlines(0, ymin, ymax, colors='r', linestyles='dashed')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        #cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        plt.colorbar(im, cax=cax) 
        
        
        if save:
            filename_base = results_folder+f'{task.__name__}_{task_type}_{filename}'
            filename_full = filename_base+stat+'.png'
            print(f'Saving {filename_full}')
            fig.savefig(f'{filename_full}', bbox_inches="tight")
        
        plt.show()

        X_train, y_train = X_gt.train().unpack(as_numpy=True)
            
    if len(np.unique(y_train)) == 2 and 'oracle' in filename.lower():
        fig = plt.figure(figsize=(3, 2.5), dpi=300, tight_layout=True)
        ax = plt.axes()
        ax.set_aspect('equal', 'box')
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        y_train = y_train.astype(bool)
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='.')
        plt.tight_layout()
        # if 'gaussian' in results_folder:
        #     plt.vlines(0, ymin, ymax, colors='r', linestyles='dashed')
        

        if save:
            fig.savefig(f'{filename_base}_samples.png', bbox_inches='tight')
            
        plt.show()

    return y_pred_mean, y_pred_std, models, contour



######## paper/results/ formatting

def get_folder_names(dataset, model_name, max_n, nsyn):
    workspace_folder = os.path.join(
        "workspace", dataset, model_name, f'nmax_{max_n}_nsyn_{nsyn}')
    results_folder = os.path.join(
        "results", f'{dataset}_{model_name}_nmax_{max_n}_nsyn_{nsyn}')
    return workspace_folder, results_folder


def mean_across_pandas(dfs, precision=3):
    dfs = pd.concat(dfs.values())
    df_mean = dfs.groupby(level=0).mean()
    df_mean = df_mean.round(precision)
    print(df_mean.to_latex(float_format=lambda x: f'%.{precision}f' % x))
    return df_mean

# use scores but report per dataset
def metric_different_datasets(dfs, metric='AUC', to_print=True, precision=3):
    df_all_datasets = pd.concat([score[metric] for score in dfs.values()], axis=1)
    df_all_datasets.columns = dfs.keys()
    df_all_datasets.round(precision)

    for dataset in zip(['moons', 'circles', 'adult', 'breast_cancer', 'seer', 'covid'][::-1],
                       ['Moons', 'Circles', 'Adult Income', 'Breast Cancer', 'SEER', 'COVID-19'][::-1]):
        try:
            df_all_datasets.insert(0, dataset[1], df_all_datasets.pop(dataset[0]))
        except:
            continue
    df_all_datasets['Mean'] = df_all_datasets.mean(axis=1)

    if to_print:
        print(df_all_datasets.to_latex(float_format=lambda x: f'%.{precision}f' % x))

    return df_all_datasets


def add_std(df, std, precision=3):
    formatting_string = '{.'+str(precision)+'f'
    df = df.round(precision)
    std = std.round(precision)
    df.style.format(precision=3, formatter=formatting_string)
    std.style.format(precision=3, formatter=formatting_string)
    df= df.astype(str)
    std = std.astype(str)
    for column in df.columns:
        if column in std.columns:
            df[column] = df[column] + ' ± ' + std[column]
    return df

