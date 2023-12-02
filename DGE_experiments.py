import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import os
import pickle
from scipy.special import xlogy

from synthcity.plugins.core.dataloader import GenericDataLoader

from DGE_utils import supervised_task, aggregate_imshow, aggregate, aggregate_predictive, cat_dl, compute_metrics, accuracy_confidence_curve, aggregate_stacking, supervised_task_stacking, aggregate_stacking_folds, weighted_meanstd
from DGE_data import get_real_and_synthetic, load_real_data, generate_synthetic_boosting

############################################################################################################
# Boosting synthetic data generations from downstream task performance

def boosting_DGE(dataset, model_name, num_runs=10, num_iter=20, boosting="SAMME.R", p_train=0.8, max_n=2000, nsyn=None, reduce_to=20000, task_type='mlp', workspace_folder='workspace', save=True, load=True, verbose=True):
    """Apply boosting to DGE
    1. initialize all weights to be equal for each train example, 1/num examples
    2. train generative model
    3. generate synthetic dataset
    4. train downstream model on synthetic dataset
    5. evaluate on real train data
    6. adjust the weights for the train data, giving higher weight to the examples that did not do well in downstream model, and iterate
    7. keep track of the "significance" weight for each iteration
    """
    print("Boosting DGE")
    # Default learning rate for ctgan in training is 1e-3
    n_models = num_iter
    # Load real data
    lr = 1e-3
    data_folder = os.path.join("synthetic_data",dataset,model_name)
    X_gt = load_real_data(dataset, p_train=p_train, max_n=max_n, reduce_to=reduce_to)
    if dataset == 'covid':
        X_gt['target'] = (X_gt['target']-1).astype(bool)

    X_train, X_test = X_gt.train(), X_gt.test()
    X_train.targettype = X_gt.targettype
    X_test.targettype = X_gt.targettype
    X_gt.dataset = dataset

    n_train = X_train.shape[0]
    if nsyn is None:
        nsyn = n_train

    print(f"n_train: {n_train}, nsyn: {nsyn}")
    n_classes = len(np.unique(X_train.dataframe()['target'].values))
    d = X_test.unpack(as_numpy=True)[0].shape[1]
    print("targettype: ", X_gt.targettype)
    if not X_gt.targettype in ['regression', 'classification']:
        raise ValueError('X_gt.targettype must be regression or classification.')
   
    each_run_significance = []
    each_run_X_syns = []
    each_run_trained_downstream_models = []
    reproducible_state = 0
    for run in range(num_runs):
        print(f"Run {run} / {num_runs}")
        significance = []
        # Keep track of the synthetic datasets and trained downstream models
        X_syns = []
        trained_downstream_models = []

        # Initialize all data weights to be equal: for first iteration, don't pass in any data weights
        init_weights = (1 / n_train) * np.ones(n_train)
        data_weights = None

        for i in range(num_iter):
            # Need to make sure that in each iterative step, the weights are corresponding to the examples! e.g. no shuffling or reordering of the examples, or need to make sure to keep indices consistent
            # Train generative model and generate synthetic dataset
            print(f"Boosting iter: {i} / {num_iter}")
            filename = f"{data_folder}/Xsyn_n{n_train}_seed{i}_boosting_run{run}.pkl"
            X_syn = generate_synthetic_boosting(model_name, num_iter, save, verbose, X_train, reproducible_state, filename, data_weights=data_weights)
            X_syn = GenericDataLoader(X_syn[:nsyn], target_column="target")
            X_syn.targettype = X_gt.targettype
            X_syns.append(X_syn)
            
            X_syns[i].dataset = dataset
            X_syns[i].targettype = X_gt.targettype
            if dataset == 'covid':
                X_syns[i]['target'] = (X_syns[i]['target']-1).astype(bool)
            reproducible_state += 1

            # Train downstream model on synthetic dataset and evaluate on real train dataset
            run_label = f'run{run}_boosting_iter{i}'
            y_pred_mean, y_pred_std, models = aggregate(
                    X_train, [X_syn], supervised_task, models=None, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save, filename=f'DGE_{i}_', verbose=verbose)
            
            assert(len(models) == 1)
            trained_downstream_models.extend(models)

            # Identify the examples that were incorrect
            y_true = X_train.dataframe()['target'].values
            #scores = compute_metrics(y_true, y_pred, X_test.targettype)
            if X_gt.targettype == "classification":
                acc = accuracy_score(y_true, y_pred_mean>0.5)
                print(f"Boosting iter {i} accuracy: ", acc)
                bool_y_pred_mean = y_pred_mean>0.5
                incorrect = y_true != bool_y_pred_mean
            elif X_gt.targettype == 'regression':
                mse = np.sqrt(mean_squared_error(y_true, y_pred_mean))
                mae = mean_absolute_error(y_true, y_pred_mean)
            else:
                raise Exception("unknown task not implemented yet")
            
            # Calculate significance weight for this iteration, but for SAMME.R, all models have equal weight of 1. Instead, weight class probabilities
            # SAMME: significance_weight = learning_rate * log ((1 - total_error) / total_error) + log(n_classes - 1) where total_error = sum of weights of incorrect samples
            # SAMME.R: boosts weighted prob estimates to update the model instead of classification itself: (n_classes - 1)*(log x - (1/n_classes) * sum(log x_hat))
            if boosting == "SAMME.R":
                print("boost with SAMME.R")
                # y coding: y_k = 1 if c==k else -1 / (n_classes - 1) where c,k are indices with c being index corresponding to true class label
                y_codes = np.array([-1.0 / (n_classes - 1), 1.0])
                y_coding = y_codes.take(np.unique(y_true) == y_true[:, np.newaxis])
                # For binary classification, y_pred_mean only predicts 1 prob instead of 2 prob (one per class)
                #y_coding = y_codes.take(np.unique(y_true) == y_true)
                y_pred_mean_newaxis = y_pred_mean[:, np.newaxis]
                print("y_pred_mean_newaxis size: ", y_pred_mean_newaxis.shape)
                print("y_coding size: ", y_coding.shape)
                estimator_weight = -1.0 * lr * ((n_classes - 1.0) / n_classes) * xlogy(y_coding, y_pred_mean_newaxis).sum(axis=1)
                significance.append(1)
                
                # Adjust weights
                if data_weights is None:
                    # bitwise or to boost data weights
                    data_weights = init_weights * np.exp(estimator_weight * ((init_weights > 0) | (estimator_weight < 0)))
                else:
                    data_weights *= np.exp(estimator_weight * ((data_weights > 0) | (estimator_weight < 0)))
            else:
                print("boost with SAMME")
                # For SAMME
                # Sum up the weights of the incorrect examples
                if data_weights is None:
                    estimator_error = np.mean(np.average(incorrect, weights=init_weights, axis=0))
                else:
                    estimator_error = np.mean(np.average(incorrect, weights=data_weights, axis=0))
                estimator_weight = lr * (np.log((1.0 - estimator_error) / estimator_error) + np.log(n_classes - 1.0))
                significance.append(estimator_weight)

                # Adjust weights based on which examples were correct or incorrect and normalize
                # new weight for incorrect sample = orig_weights * e^(significance)
                # new weight for correct sample: orig_weight * e^(-significant)
                if data_weights is None:
                    # bitwise or to boost data weights
                    data_weights = np.exp(np.log(init_weights) + estimator_weight * incorrect * (init_weights > 0))
                else:
                    data_weights = np.exp(np.log(data_weights) + estimator_weight * incorrect * (data_weights > 0))
            
            # Normalize the weights
            data_weights /= np.sum(data_weights)
            print("normalized data_weights: ", data_weights)
            print("are all weights equal?: ", np.sum(np.where(np.isclose(data_weights, (1.0 / 2000)), 1, 0)))
            print("estimator weight: ", estimator_weight)
        
        # Add to list of runs
        print(f"Finished run {run} / {num_runs}")
        each_run_significance.append(significance)
        each_run_X_syns.append(X_syns)
        each_run_trained_downstream_models.append(trained_downstream_models)
    # add compute metrics and final evaluation
    print("Start final evaluation...")
    #Ks = [20, 10, 5]
    Ks = []
    for k in [20, 10, 5]:
        if k <= n_models:
            Ks.append(k)
    y_DGE_approaches = ['DGE$_{'+str(K)+'}$' for K in Ks]
    y_naive_approaches = ['Naive (S)', 'Naive (E)']
    keys = ['Oracle'] + y_naive_approaches + y_DGE_approaches[::-1] + ['DGE$_{20}$ (concat)']
    y_preds = dict(zip(keys, [[] for _ in keys]))
    keys_for_plotting = ['Oracle', 'Naive'] + y_DGE_approaches[::-1]
    y_preds_for_plotting = dict(zip(keys_for_plotting, [None]*len(keys_for_plotting)))

     # Oracle
    X_oracle = X_gt.train()
    X_oracle.targettype = X_syns[0].targettype

    X_oracle = [X_oracle] * n_models
    for run in range(num_runs):

        run_label = f'run_{run}'

        print("run: ", run)
        # Oracle ensemble

        y_pred_mean, _, models = aggregate(
            X_test, X_oracle, supervised_task, models=None, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save, filename=f'oracle_{run_label}_', verbose=verbose)
        y_preds['Oracle'].append(y_pred_mean)

        # Single dataset single model
        for approach in y_naive_approaches:
            # Get first synthetic dataset from each run
            if approach == 'Naive (S)':
                X_syn_run = [each_run_X_syns[run][0]]
            else:
                X_syn_run = [each_run_X_syns[run][0]] * n_models

            y_pred_mean, y_pred_std, models = aggregate(
                X_test, X_syn_run, supervised_task, models=None, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save, filename=f'naive_m{run}_', verbose=verbose)
            y_preds[approach].append(y_pred_mean)

        # Boosted DGE
        for K, approach in zip(Ks, y_DGE_approaches):
            # Make prediction with each downstream model, then do weighted sum/avg
            y_hat = []
            for j in range(K):
                model = each_run_trained_downstream_models[run][j]
                if X_gt.targettype == 'regression':
                    pred = model.predict(X_test.unpack(as_numpy=True)[0])
                else:
                    pred = model.predict_proba(X_test.unpack(as_numpy=True)[0])[:, 1]
                y_hat.append(pred)
            print(f"shape of {K} DGE y predictions: ", y_hat[0].shape)
            print(f"model/estimator weights shape: {len(each_run_significance[run])}")
            y_weighted_pred_mean, y_weighted_stds = weighted_meanstd(y_hat, each_run_significance[run])
            print("weighted means: ", y_weighted_pred_mean)
            y_preds[approach].append(y_weighted_pred_mean)

        # Data aggregated
        X_syn_cat = pd.concat([each_run_X_syns[run][i].dataframe() for i in range(n_models)], axis=0)
        X_syn_cat = GenericDataLoader(X_syn_cat, target_column="target")
        X_syn_cat.targettype = each_run_X_syns[run][0].targettype
        X_syn_cat = [X_syn_cat]
        y_pred_mean, _, _ = aggregate(
            X_test, X_syn_cat, supervised_task, models=None, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save, filename=f'concat_run{run}', verbose=verbose)
        y_preds['DGE$_{n_models}$ (concat)'].append(y_pred_mean)

    # Evaluation
    y_true = X_test.dataframe()['target'].values
    # Compute metrics

    scores_mean = {}
    scores_std = {}

    scores_all = []
    for approach in y_preds.keys():
        scores = []
        for y_pred in y_preds[approach]:
            scores.append(compute_metrics(y_true, y_pred, X_test.targettype))

        scores = pd.concat(scores, axis=0)
        scores_mean[approach] = np.mean(scores, axis=0)
        scores_std[approach] = np.std(scores, axis=0)
        scores['Approach'] = approach
        scores_all.append(scores)

    scores_all = pd.concat(scores_all, axis=0)
    scores_mean = pd.DataFrame.from_dict(
        scores_mean, orient='index', columns=scores.columns.drop('Approach'))
    scores_std = pd.DataFrame.from_dict(
        scores_std, orient='index', columns=scores.columns.drop('Approach'))

    return scores_mean, scores_std, scores_all



############################################################################################################
# Model training. Predictive performance


def predictive_experiment(X_gt, X_syns, n_models=20, task_type='mlp', results_folder=None, workspace_folder='workspace', load=True, save=True, plot=False, outlier=False, verbose=True, include_concat=True):
    """Compares predictions by different approaches.

    Args:
        X_gt: real data
        X_test (GenericDataLoader): Test data.
        X_syns (List(GenericDataLoader)): List of synthetic datasets.
        X_test (GenericDataLoader): Real data
        load (bool, optional): Load results, if available. Defaults to True.
        save (bool, optional): Save results when done. Defaults to True.

    Returns:

    """
    if save and results_folder is None:
        raise ValueError('results_folder must be specified when save=True.')

    X_test = X_gt.test()
    d = X_test.unpack(as_numpy=True)[0].shape[1]

    if type(outlier) == type(lambda x: 1):
        print('Using subset for evaluation')
        subset = outlier
        X_test = subset(X_test)
        plot = False
    elif outlier:
        raise ValueError('outlier boolean is no longer supported')

    print("targettype: ", X_gt.targettype)
    print("include_concat: ", include_concat)
    X_test.targettype = X_gt.targettype

    if not X_gt.targettype in ['regression', 'classification']:
        raise ValueError('X_gt.targettype must be regression or classification.')

    
    # DGE (k=5, 10, 20)
    #n_models = 20  # maximum K
    num_runs = len(X_syns)//n_models

    print("n_models: ", n_models)
    print("num_runs: ", num_runs)
    print("list size of synthetic datasets: ", len(X_syns))

    if num_runs > 1 and verbose:
        print('Computing means and stds')

    Ks = [20, 10, 5]
    y_DGE_approaches = ['DGE$_{'+str(K)+'}$' for K in Ks]
    y_naive_approaches = ['Naive (S)', 'Naive (E)']
    keys = ['Oracle'] + y_naive_approaches + y_DGE_approaches[::-1] + ['DGE$_{20}$ (concat)']
    y_preds = dict(zip(keys, [[] for _ in keys]))
    keys_for_plotting = ['Oracle', 'Naive'] + y_DGE_approaches[::-1]
    if include_concat:
        keys_for_plotting += ['DGE$_{20}$ (concat)']
    y_preds_for_plotting = dict(zip(keys_for_plotting, [None]*len(keys_for_plotting)))


    # Oracle
    X_oracle = X_gt.train()
    X_oracle.targettype = X_syns[0].targettype

    X_oracle = [X_oracle] * n_models

    # Oracle ensemble

    for run in range(num_runs):

        run_label = f'run_{run}'

        print("run: ", run)
        # Oracle ensemble

        y_pred_mean, _, models = aggregate(
            X_test, X_oracle, supervised_task, models=None, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save, filename=f'oracle_{run_label}_', verbose=verbose)

        if d == 2 and plot and run == 0:
            _, _, _, contour = aggregate_imshow(
                X_test, X_oracle, supervised_task, models=models, results_folder=results_folder, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save, filename='oracle')

        if run == 0 and plot:
            y_preds_for_plotting['Oracle'] = y_pred_mean

        y_preds['Oracle'].append(y_pred_mean)

        # Single dataset single model
        for approach in y_naive_approaches:
            if approach == 'Naive (S)':
                X_syn_run = [X_syns[run]]
            else:
                X_syn_run = [X_syns[run]] * n_models

            y_pred_mean, y_pred_std, models = aggregate(
                X_test, X_syn_run, supervised_task, models=None, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save, filename=f'naive_m{run}_', verbose=verbose)

            if run == 0 and plot and approach == 'Naive (E)':
                if d == 2:
                    aggregate_imshow(X_test, X_syn_run, supervised_task, models=models, results_folder=results_folder,
                                     task_type=task_type, load=load, save=save, filename=f'naive_m{run}_', baseline_contour=contour)

                y_preds_for_plotting['Naive'] = y_pred_mean

            y_preds[approach].append(y_pred_mean)

        # DGE
        starting_dataset = run*n_models
        models = None
        for K, approach in zip(Ks, y_DGE_approaches):
            y_pred_mean, y_pred_std, models = aggregate(
                X_test, X_syns[starting_dataset:starting_dataset+K], supervised_task, models=models, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save, filename=f'DGE_{run_label}_', verbose=verbose)

            if d == 2 and plot and run == 0:
                aggregate_imshow(
                    X_test, X_syns[starting_dataset:starting_dataset+K], supervised_task, models=models, workspace_folder=workspace_folder, results_folder=results_folder, task_type=task_type, load=load, save=save, filename=f'DGE_K{K}_{run_label}_', baseline_contour=contour)

            y_preds[approach].append(y_pred_mean)

            # for plotting calibration and confidence curves later
            if run == 0 and plot:
                y_preds_for_plotting[approach] = y_pred_mean

        # Data aggregated
        X_syn_cat = pd.concat([X_syns[i].dataframe() for i in range(starting_dataset, starting_dataset+20)], axis=0)
        X_syn_cat = GenericDataLoader(X_syn_cat, target_column="target")
        X_syn_cat.targettype = X_syns[0].targettype
        X_syn_cat = [X_syn_cat]
        y_pred_mean, _, _ = aggregate(
            X_test, X_syn_cat, supervised_task, models=None, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save, filename=f'concat_run{run}', verbose=verbose)

        if include_concat and run == 0 and plot:
            y_preds_for_plotting['DGE$_{20}$ (concat)'] = y_pred_mean

        if plot and d == 2 and run==0:
            aggregate_imshow(X_test, X_syn_cat * n_models, supervised_task, models=None, results_folder=results_folder, workspace_folder=workspace_folder,
                                task_type=task_type, load=load, save=save, filename=f'concat_all', baseline_contour=contour)

        y_preds['DGE$_{20}$ (concat)'].append(y_pred_mean)


    # Evaluation
    # Plotting

    y_true = X_test.dataframe()['target'].values

    if X_syns[0].targettype is 'classification' and plot:
        # Consider calibration of different approaches
        fig = plt.figure(figsize=(3, 3), tight_layout=True, dpi=300)
        for key, y_pred in y_preds_for_plotting.items():
            print(key, y_pred.shape)
            prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
            plt.plot(prob_pred, prob_true, label=key)

        plt.xlabel = 'Mean predicted probability'
        plt.ylabel = 'Fraction of positives'
        plt.tight_layout()
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
        plt.legend()

        if save:
            filename = results_folder+'_calibration_curve.png'
            fig.savefig(filename, dpi=300)

        plt.show()
        plt.close()

        plt.figure(figsize=(3, 3), dpi=300)
        for key, y_pred in y_preds_for_plotting.items():
            thresholds, prob_true = accuracy_confidence_curve(y_true, y_pred, n_bins=20)
            plt.plot(thresholds, prob_true, label=key)

        plt.xlabel = r'Confidence threshold \tau'
        plt.ylabel = r'Accuracy on examples \hat{y}'

        plt.legend()
        plt.tight_layout()
        if save:
            filename = results_folder+'_confidence_accuracy_curve.png'
            plt.savefig(filename, dpi=200, bbox_inches='tight')

        if plot:
            plt.show()

        plt.close()

    # Compute metrics

    scores_mean = {}
    scores_std = {}

    scores_all = []
    for approach in y_preds.keys():
        scores = []
        for y_pred in y_preds[approach]:
            scores.append(compute_metrics(y_true, y_pred, X_test.targettype))

        scores = pd.concat(scores, axis=0)
        scores_mean[approach] = np.mean(scores, axis=0)
        scores_std[approach] = np.std(scores, axis=0)
        scores['Approach'] = approach
        scores_all.append(scores)

    scores_all = pd.concat(scores_all, axis=0)
    scores_mean = pd.DataFrame.from_dict(
        scores_mean, orient='index', columns=scores.columns.drop('Approach'))
    scores_std = pd.DataFrame.from_dict(
        scores_std, orient='index', columns=scores.columns.drop('Approach'))

    return scores_mean, scores_std, scores_all


def predictive_experiment_stacking(X_gt, X_syns, n_models=20, task_type='mlp', mixed_models=False, meta_model='lr', results_folder=None, workspace_folder='workspace', load=True, save=True, plot=False, outlier=False, verbose=True, include_concat=True):
    """Compares predictions by different approaches.

    Args:
        X_gt: real data
        X_test (GenericDataLoader): Test data.
        X_syns (List(GenericDataLoader)): List of synthetic datasets.
        X_test (GenericDataLoader): Real data
        load (bool, optional): Load results, if available. Defaults to True.
        save (bool, optional): Save results when done. Defaults to True.

    Returns:

    """
    if save and results_folder is None:
        raise ValueError('results_folder must be specified when save=True.')

    X_test = X_gt.test()
    d = X_test.unpack(as_numpy=True)[0].shape[1]

    if type(outlier) == type(lambda x: 1):
        print('Using subset for evaluation')
        subset = outlier
        X_test = subset(X_test)
        plot = False
    elif outlier:
        raise ValueError('outlier boolean is no longer supported')

    print("targettype: ", X_gt.targettype)
    print("include_concat: ", include_concat)
    X_test.targettype = X_gt.targettype

    if not X_gt.targettype in ['regression', 'classification']:
        raise ValueError('X_gt.targettype must be regression or classification.')


    # DGE (k=5, 10, 20)
    #n_models = 20  # maximum K
    num_runs = len(X_syns)//n_models

    print("n_models: ", n_models)
    print("num_runs: ", num_runs)
    print("list size of synthetic datasets: ", len(X_syns))
    print("mixed_models: ", mixed_models)
    print("meta_model: ", meta_model)

    if num_runs > 1 and verbose:
        print('Computing means and stds')

    #Ks = [20, 10, 5]
    Ks = []
    for k in [20, 10, 5]:
        if k <= n_models:
            Ks.append(k)
    print("Ks to ensemble with: ", Ks)
    y_DGE_approaches = ['DGE$_{'+str(K)+'}$' for K in Ks]
    y_naive_approaches = ['Naive (S)', 'Naive (E)']
    keys = ['Oracle'] + y_naive_approaches + y_DGE_approaches[::-1] + ['DGE$_{20}$ (concat)']
    y_preds = dict(zip(keys, [[] for _ in keys]))
    stacking_y_preds = dict(zip(keys, [[] for _ in keys]))
    actual_meta_preds = dict(zip(keys, [[] for _ in keys]))
    keys_for_plotting = ['Oracle', 'Naive'] + y_DGE_approaches[::-1]
    if include_concat:
        keys_for_plotting += ['DGE$_{20}$ (concat)']
    y_preds_for_plotting = dict(zip(keys_for_plotting, [None]*len(keys_for_plotting)))


    # Oracle
    X_oracle = X_gt.train()
    X_oracle.targettype = X_syns[0].targettype

    X_oracle = [X_oracle] * n_models

    # Oracle ensemble

    for run in range(num_runs):

        run_label = f'run_{run}'

        print("run: ", run)
        # Oracle ensemble

        stacking_pred, y_pred_mean, _, models = aggregate_stacking(
            X_test, X_oracle, supervised_task_stacking, meta_model=meta_model, mixed_models=mixed_models, models=None, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save, filename=f'oracle_{run_label}_', verbose=verbose)

        meta_pred, trained_estimators = aggregate_stacking_folds(X_test, X_oracle, supervised_task_stacking, meta_model=meta_model, mixed_models=mixed_models, estimators_list=None, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save, filename=f'actual_stacking_ensemble_folds_oracle_{run_label}_', verbose=verbose)

        if d == 2 and plot and run == 0:
            _, _, _, contour = aggregate_imshow(
                X_test, X_oracle, supervised_task, models=models, results_folder=results_folder, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save, filename='oracle')

        if run == 0 and plot:
            y_preds_for_plotting['Oracle'] = y_pred_mean

        y_preds['Oracle'].append(y_pred_mean)
        stacking_y_preds['Oracle'].append(stacking_pred)
        actual_meta_preds['Oracle'].append(meta_pred)

        # Single dataset single model
        for approach in y_naive_approaches:
            if approach == 'Naive (S)':
                X_syn_run = [X_syns[run]]
            else:
                X_syn_run = [X_syns[run]] * n_models

            stacking_pred, y_pred_mean, y_pred_std, models = aggregate_stacking(
                X_test, X_syn_run, supervised_task_stacking, meta_model=meta_model, mixed_models=mixed_models, models=None, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save, filename=f'naive_m{run}_', verbose=verbose)

            meta_pred, trained_estimators = aggregate_stacking_folds(X_test, X_syn_run, supervised_task_stacking, meta_model=meta_model, mixed_models=mixed_models, estimators_list=None, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save, filename=f'actual_stacking_ensemble_folds_naive_m{run}_', verbose=verbose)

            print("size of y_pred_mean: ", y_pred_mean.shape)
            print("size of stacking_pred: ", stacking_pred.shape)

            if run == 0 and plot and approach == 'Naive (E)':
                if d == 2:
                    aggregate_imshow(X_test, X_syn_run, supervised_task, models=models, results_folder=results_folder,
                                     task_type=task_type, load=load, save=save, filename=f'naive_m{run}_', baseline_contour=contour)

                y_preds_for_plotting['Naive'] = y_pred_mean

            y_preds[approach].append(y_pred_mean)
            stacking_y_preds[approach].append(stacking_pred)
            actual_meta_preds[approach].append(meta_pred)

        starting_dataset = run*n_models
        # Data aggregated
        #X_syn_cat = pd.concat([X_syns[i].dataframe() for i in range(starting_dataset, starting_dataset+20)], axis=0)
        X_syn_cat = pd.concat([X_syns[i].dataframe() for i in range(starting_dataset, starting_dataset+n_models)], axis=0)
        X_syn_cat = GenericDataLoader(X_syn_cat, target_column="target")
        X_syn_cat.targettype = X_syns[0].targettype
        X_syn_cat_list = [X_syn_cat]
        stacking_pred, y_pred_mean, _, _ = aggregate_stacking(
            X_test, X_syn_cat_list, supervised_task_stacking, meta_model=meta_model, mixed_models=mixed_models, models=None, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save, filename=f'concat_run{run}', verbose=verbose)

        meta_pred, trained_estimators = aggregate_stacking_folds(
            X_test, X_syn_cat_list, supervised_task_stacking, meta_model=meta_model, mixed_models=mixed_models, estimators_list=None, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save, filename=f'actual_stacking_ensemble_folds_concat_run{run}', verbose=verbose)

        if include_concat and run == 0 and plot:
            y_preds_for_plotting['DGE$_{20}$ (concat)'] = y_pred_mean

        if plot and d == 2 and run==0:
            aggregate_imshow(X_test, X_syn_cat_list * n_models, supervised_task, models=None, results_folder=results_folder, workspace_folder=workspace_folder,
                                task_type=task_type, load=load, save=save, filename=f'concat_all', baseline_contour=contour)

        y_preds['DGE$_{20}$ (concat)'].append(y_pred_mean)
        stacking_y_preds['DGE$_{20}$ (concat)'].append(stacking_pred)
        actual_meta_preds['DGE$_{20}$ (concat)'].append(meta_pred)

        # DGE modified to allow for stacking ensemble meta learner, so each downstream classifier needs to be trained on the same dataset, use concatenated synthetic dataset
        models = None
        estimators_list = None
        for K, approach in zip(Ks, y_DGE_approaches):
            syn_list = [X_syn_cat] * K
            stacking_pred, y_pred_mean, y_pred_std, models = aggregate_stacking(
                X_test, syn_list, supervised_task_stacking, meta_model=meta_model, mixed_models=mixed_models, models=models, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save, filename=f'DGE_{run_label}_', verbose=verbose)

            meta_pred, estimators_list = aggregate_stacking_folds(X_test, syn_list, supervised_task_stacking, meta_model=meta_model, mixed_models=mixed_models, estimators_list=estimators_list, workspace_folder=workspace_folder, task_type=task_type, load=load, save=save, filename=f'actual_stacking_ensemble_folds_DGE_{run_label}_', verbose=verbose)

            if d == 2 and plot and run == 0:
                aggregate_imshow(
                    X_test, syn_list, supervised_task, models=models, workspace_folder=workspace_folder, results_folder=results_folder, task_type=task_type, load=load, save=save, filename=f'DGE_K{K}_{run_label}_', baseline_contour=contour)

            y_preds[approach].append(y_pred_mean)
            stacking_y_preds[approach].append(stacking_pred)
            actual_meta_preds[approach].append(meta_pred)

            # for plotting calibration and confidence curves later
            if run == 0 and plot:
                y_preds_for_plotting[approach] = y_pred_mean


    # Evaluation
    # Plotting

    y_true = X_test.dataframe()['target'].values

    if X_syns[0].targettype is 'classification' and plot:
        # Consider calibration of different approaches
        fig = plt.figure(figsize=(3, 3), tight_layout=True, dpi=300)
        for key, y_pred in y_preds_for_plotting.items():
            print(key, y_pred.shape)
            prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
            plt.plot(prob_pred, prob_true, label=key)

        plt.xlabel = 'Mean predicted probability'
        plt.ylabel = 'Fraction of positives'
        plt.tight_layout()
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
        plt.legend()

        if save:
            filename = results_folder+'_calibration_curve.png'
            fig.savefig(filename, dpi=300)

        plt.show()
        plt.close()

        plt.figure(figsize=(3, 3), dpi=300)
        for key, y_pred in y_preds_for_plotting.items():
            thresholds, prob_true = accuracy_confidence_curve(y_true, y_pred, n_bins=20)
            plt.plot(thresholds, prob_true, label=key)

        plt.xlabel = r'Confidence threshold \tau'
        plt.ylabel = r'Accuracy on examples \hat{y}'

        plt.legend()
        plt.tight_layout()
        if save:
            filename = results_folder+'_confidence_accuracy_curve.png'
            plt.savefig(filename, dpi=200, bbox_inches='tight')

        if plot:
            plt.show()

        plt.close()

    # Compute metrics

    scores_mean = {}
    scores_std = {}

    scores_all = []
    assert(y_preds.keys() == stacking_y_preds.keys())
    print("prediction keys/approaches: ", y_preds.keys())
    meta_scores_mean = {}
    meta_scores_std = {}
    meta_scores_all = []

    actual_stacking_folds_mean = {}
    actual_stacking_folds_std = {}
    actual_stacking_folds_all = []
    for approach in y_preds.keys():
        scores = []
        meta_scores = []
        actual_stacking_folds_scores = []
        for y_pred in y_preds[approach]:
            scores.append(compute_metrics(y_true, y_pred, X_test.targettype))
        
        for meta_y_pred in stacking_y_preds[approach]:
            meta_scores.append(compute_metrics(y_true, meta_y_pred, X_test.targettype))

        for folds_y_pred in actual_meta_preds[approach]:
            actual_stacking_folds_scores.append(compute_metrics(y_true, folds_y_pred, X_test.targettype))

        scores = pd.concat(scores, axis=0)
        scores_mean[approach] = np.mean(scores, axis=0)
        scores_std[approach] = np.std(scores, axis=0)
        scores['Approach'] = approach
        scores_all.append(scores)

        # For the meta learner results
        meta_scores = pd.concat(meta_scores, axis=0)
        meta_scores_mean[approach] = np.mean(meta_scores, axis=0)
        meta_scores_std[approach] = np.std(meta_scores, axis=0)
        meta_scores['Approach'] = approach
        meta_scores_all.append(meta_scores)

        # For stacking ensemble with folds
        actual_stacking_folds_scores = pd.concat(actual_stacking_folds_scores, axis=0)
        actual_stacking_folds_mean[approach] = np.mean(actual_stacking_folds_scores, axis=0)
        actual_stacking_folds_std[approach] = np.std(actual_stacking_folds_scores, axis=0)
        actual_stacking_folds_scores['Approach'] = approach
        actual_stacking_folds_all.append(meta_scores)

    scores_all = pd.concat(scores_all, axis=0)
    scores_mean = pd.DataFrame.from_dict(
        scores_mean, orient='index', columns=scores.columns.drop('Approach'))
    scores_std = pd.DataFrame.from_dict(
        scores_std, orient='index', columns=scores.columns.drop('Approach'))

    # For meta learner results
    meta_scores_all = pd.concat(meta_scores_all, axis=0)
    meta_scores_mean = pd.DataFrame.from_dict(meta_scores_mean, orient='index', columns=meta_scores.columns.drop('Approach'))
    meta_scores_std = pd.DataFrame.from_dict(meta_scores_std, orient='index', columns=meta_scores.columns.drop('Approach'))

    # For stacking ensemble with folds and meta learner
    # For meta learner results
    actual_stacking_folds_all = pd.concat(actual_stacking_folds_all, axis=0)
    actual_stacking_folds_mean = pd.DataFrame.from_dict(actual_stacking_folds_mean, orient='index', columns=actual_stacking_folds_scores.columns.drop('Approach'))
    actual_stacking_folds_std = pd.DataFrame.from_dict(actual_stacking_folds_std, orient='index', columns=actual_stacking_folds_scores.columns.drop('Approach'))

    return scores_mean, scores_std, scores_all, meta_scores_mean, meta_scores_std, meta_scores_all, actual_stacking_folds_mean, actual_stacking_folds_std, actual_stacking_folds_all

##############################################################################################################

# Model evaluation and selection experiments


def model_evaluation_experiment(X_gt, X_syns, model_type, relative=False, workspace_folder=None, load=True, save=True, outlier=False, verbose=False):
    means = []
    stds = []
    res = {}
    approaches = ['Oracle', 'Naive', 'DGE$_5$', 'DGE$_{10}$', 'DGE$_{20}$']
    K = [None, None, 5, 10, 20]
    if type(outlier) == type(lambda x: 1):
        subset = outlier
    elif outlier == True:
        raise ValueError('Subset not properly defined')
    else:
        subset = None
    folder = os.path.join(workspace_folder, 'Naive')

    for i, approach in enumerate(approaches):
        if verbose:
            print('Approach: ', approach)
        mean, std, _, all = aggregate_predictive(
            X_gt, X_syns, models=None, task_type=model_type, workspace_folder=folder, load=load, save=save, approach=approach, relative=relative, subset=subset, verbose=verbose, K=K[i])
        means.append(mean)
        stds.append(std)
        all['Approach'] = approach
        res[approach] = all

    means = pd.concat(means, axis=0)
    stds = pd.concat(stds, axis=0)
    res = pd.concat(res, axis=0)
    if relative == 'rmse':
        means = np.sqrt(means)

    means.index = approaches
    stds.index = approaches
    means.index.Name = 'Approach'
    stds.index.Name = 'Approach'

    return means, stds, res


def model_selection_experiment(X_gt, X_syns, relative='l1', workspace_folder='workspace', load=True, save=True, outlier=False, model_types=None):
    
    if model_types is None:
        model_types = ['lr', 'mlp', 'deep_mlp', 'rf', 'knn', 'svm', 'xgboost']
    
    all_stds = []
    all_means = []
    output_means = {}
    output_stds = {}

    for i, model_type in enumerate(model_types):
        mean, std, _ = model_evaluation_experiment(
            X_gt, X_syns, model_type, workspace_folder=workspace_folder, relative=relative, load=load, save=save, outlier=outlier)
        all_means.append(mean)
        all_stds.append(std)

    for metric in mean.columns:
        means = []
        stds = []
        for i, model_type in enumerate(model_types):
            means.append(all_means[i][metric])
            stds.append(all_stds[i][metric])

        means = pd.concat(means, axis=1)
        stds = pd.concat(stds, axis=1)
        means.columns = model_types
        stds.columns = model_types
        approaches = mean.index
        means.index = approaches
        stds.index = approaches

        # sort based on oracle
        sorting = [model_types[i] for i in means.loc['Oracle'].argsort()]
        means_sorted = means.loc[:, sorting]
        stds_sorted = stds.loc[:, sorting]

        for approach in approaches:
            sorting_k = means_sorted.loc[approach].argsort()
            sorting_k = sorting_k.argsort()
            means_sorted.loc[approach+' rank'] = 7-sorting_k.astype(int)

        output_means[metric] = means_sorted
        output_stds[metric] = stds_sorted

    return output_means, output_stds

from DGE_utils import tt_predict_performance, cat_dl

import pandas as pd
from sklearn.model_selection import KFold


def cross_val(X_gt, X_syns, workspace_folder=None, results_folder=None, 
             save=True, load=True, task_type='mlp', 
            cross_fold=5, verbose=False):
    """Compares predictions by different approaches using cross validation.

    Args:
        X_test (GenericDataLoader): Test data.
        X_syns (List(GenericDataLoader)): List of synthetic datasets.
        X_test (GenericDataLoader): Real data
        load (bool, optional): Load results, if available. Defaults to True.
        save (bool, optional): Save results when done. Defaults to True.

    Returns:

    """

    if save and results_folder is None:
        raise ValueError('results_folder must be specified when save=True.')

    X_test_r = X_gt.test()

    X_test_r.targettype = X_gt.targettype

    if not X_gt.targettype in ['regression', 'classification']:
        raise ValueError('X_gt.targettype must be regression or classification.')

    # DGE (k=5, 10, 20)
    n_models = 20  # maximum K
    num_runs = len(X_syns)//n_models

    if num_runs > 1 and verbose:
        print('Computing means and stds')

    keys = ['Oracle', 'Naive', 'DGE$_{20}$', 'DGE$_{20}$ (concat)']
    #keys = keys[-2:]

    # Oracle
    X_oracle = X_gt.train()

    # Oracle ensemble
    scores_r_all = []
    scores_s_all = []

    for run in range(num_runs):

        run_label = f'run_{run}'
        starting_dataset = run*n_models
        scores_s = {}
        scores_r = {}

        for approach in keys:
            kf = KFold(n_splits=cross_fold, shuffle=True, random_state=0)
            print(approach)
            if 'oracle' in approach.lower():
                X_syn_run = X_oracle
            elif approach == 'Naive':
                X_syn_run = X_syns[run]
            elif approach.startswith('DGE') and not 'concat' in approach:
                K = 20
                X_syn_run = X_syns[starting_dataset:starting_dataset+K]
            elif approach == 'DGE$_{20}$ (concat)':
                X_syn_cat = pd.concat([X_syns[i].dataframe() for i in range(
                    starting_dataset, starting_dataset+20)], axis=0)
            else:
                raise ValueError(f'Unknown approach {approach}')

            scores_s[approach] = [0] * cross_fold
            scores_r[approach] = [0] * cross_fold
            for i, (train_index, test_index) in enumerate(kf.split(X_syn_run)):
                
                if verbose:
                    print('Run', run, 'approach', approach, 'split', i)
                
                if type(X_syn_run) == type([]):
                    X_train = cat_dl([X_syn_run[i] for i in train_index])
                    X_test_s = cat_dl([X_syn_run[i] for i in test_index])
                else:
                    if type(X_syn_run)==pd.DataFrame:
                        pass
                    else:
                        X_syn_run = X_syn_run.dataframe()
                    
                    x_train, x_test = X_syn_run.loc[train_index], X_syn_run.loc[test_index]
                    X_train = GenericDataLoader(x_train, target_column="target")
                    X_test_s = GenericDataLoader(x_test, target_column="target")

                X_test_s.targettype = X_syns[0].targettype
                X_train.targettype = X_syns[0].targettype

                filename = os.path.join(
                    workspace_folder, f'cross_validation_{task_type}_{approach}_{run_label}_split_{i}.pkl')
                
                if load and os.path.exists(filename):
                    with open(filename, 'rb') as f:
                        model = pickle.load(f)

                elif load and approach == 'DGE$_{20}$':
                    # for compatibility with old files
                    alt_filename = os.path.join(workspace_folder, f'cross_validation_{task_type}_'+'DGE$_{20]$'+f'_{run_label}_split_{i}.pkl')
                    if os.path.exists(alt_filename):
                        with open(alt_filename, 'rb') as f:
                            model = pickle.load(f)
                    else:
                        model = None
                
                else:
                    model = None
                scores_s[approach][i], model = tt_predict_performance(
                    X_test_s, X_train, model=model, model_type=task_type, subset=None, verbose=False)
                scores_r[approach][i], _ = tt_predict_performance(
                    X_test_r, X_train, model=model, model_type=task_type, subset=None, verbose=False)

                scores_s[approach][i]['run'] = run
                scores_r[approach][i]['run'] = run
                scores_s[approach][i]['split'] = i
                scores_r[approach][i]['split'] = i
                scores_s[approach][i]['approach'] = approach
                scores_r[approach][i]['approach'] = approach

                if save and not os.path.exists(filename):
                    with open(filename, 'wb') as f:
                        pickle.dump(model, f)

            scores_s[approach] = pd.concat(scores_s[approach], axis=0)
            scores_r[approach] = pd.concat(scores_r[approach], axis=0)

        scores_s_all.append(pd.concat(scores_s))
        scores_r_all.append(pd.concat(scores_r))

    scores_s_all = pd.concat(scores_s_all, axis=0)
    scores_r_all = pd.concat(scores_r_all, axis=0)

    scores_s_mean = scores_s_all.groupby(['run', 'approach']).mean()
    scores_r_mean = scores_r_all.groupby(['run', 'approach']).mean()

    return scores_s_mean, scores_r_mean

