
###############################################################################################
# DEPENDENCIES
###############################################################################################
import os
import time
import json
import argparse
import catboost as cb
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

import ios
from tools_comparison import ToolsComparison
import setup
import tools_comparison
import constants 

###############################################################################################
# CONSTANTS
###############################################################################################

VERSION = 3

DIR_RESULTS = f'../../results/v{VERSION}'
DIR_TOOLS = ios.path_join(DIR_RESULTS, 'gender')
DIR_TOOL_W = ios.path_join(DIR_TOOLS, 'tool_w')
DIR_TOOL_R = ios.path_join(DIR_TOOLS, 'tool_r')
DIR_TOOL_N = ios.path_join(DIR_TOOLS, 'tool_n')

DIR_GT = f'../../data/v{VERSION}/annotations/gender'
DIR_NAMES = ios.path_join(DIR_RESULTS, 'names/chinese')
DIR_PRED = ios.path_join(DIR_TOOLS, 'predictions')

FIGSIZE = (4,3)


###############################################################################################
# HELPERS
###############################################################################################

def validation(gt_agg, features_id):
    if gt_agg not in setup.GT_AGGS:
        raise ValueError(f'Invalid variable: {gt_agg}')
    if features_id < 0 or features_id >= len(setup.FEATURE_NAMES):
        raise ValueError(f'Invalid features_id: {features_id} (0-{len(setup.FEATURE_NAMES)-1})')

def _train(X_train, y_train, X_test, y_test, cat_features, loss_function, eval_metric, cv=setup.CV, stratified=False, early_stopping_rounds=setup.EARLY_STOPPING, random_seed=setup.RANDOM_STATE):
    # Pools
    train_pool = cb.Pool(X_train, label=y_train, cat_features=cat_features) 
    test_pool = cb.Pool(X_test, label=y_test, cat_features=cat_features) 

    # Model
    model = cb.CatBoostRegressor(verbose=False,   
                                 cat_features=cat_features,
                                 eval_metric = eval_metric,
                                 loss_function = loss_function,
                                 random_seed = random_seed) 

    # Tuning
    grid_search_result = model.grid_search(setup.GRID, 
                                           train_pool, cv=cv, 
                                           stratified = stratified,
                                           partition_random_seed=random_seed, verbose=False)

    # best model
    best_model = model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=early_stopping_rounds)

    return best_model, grid_search_result

def _eval(model, X_test, y_test):
    pred = model.predict(X_test)
    rmse = (np.sqrt(mean_squared_error(y_test, pred)))
    r2 = r2_score(y_test, pred)
    print("Testing performance")
    print('RMSE: {:.2f}'.format(rmse))
    print('R2: {:.2f}'.format(r2))
    return pred, rmse, r2

def _save_plot(fig, fn):
    fig.savefig(fn, bbox_inches='tight')
    plt.close()

def plot_true_vs_pred(y_test, pred, title="Validation", fn=None, **kwargs):
    fig,ax = plt.subplots(1,1,figsize=FIGSIZE)
    ax.scatter(x=y_test, y=pred)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predictions')
    ax.axis('equal')
    ax.axis('square')
    ax.plot([-100, 100], [-100, 100], lw=1, ls='--', c='grey');
    ax.set_title(title)
    y = 0.95
    if kwargs:
        for key, value in kwargs.items():
            ax.text(0.05, y, f'{key}: {value:.2f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')
            y -= 0.05
    _save_plot(fig, fn)

def plot_feature_importance(model, sorted_feature_importance, sorted_feature_names, fn=None):
    # Feature imoprtance
    
    width = 6
    height = 2 + (len(sorted_feature_names) - 0) * (12 - 2) / (len(setup.FEATURE_NAMES) - 0)

    fig,ax = plt.subplots(1,1,figsize=(width,height))
    ax.barh(sorted_feature_names, 
            model.feature_importances_[sorted_feature_importance], 
            color='turquoise')
    ax.set_xlabel("CatBoost Feature Importance")
    _save_plot(fig, fn)

def plot_shap(model, X_test, sorted_feature_names, fn=None):
    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    width = 6
    height = 2 + (len(sorted_feature_names) - 0) * (12 - 2) / (len(setup.FEATURE_NAMES) - 0)

    fig,ax = plt.subplots(1,1,figsize=(width,height))
    shap.summary_plot(shap_values, X_test, feature_names = sorted_feature_names)
    _save_plot(fig, fn)

def save_json(obj, fn):
    with open(fn, "w") as file:
        json.dump(obj, file, indent=4)

def validate_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

###############################################################################################
# RUN
###############################################################################################
def run(gt_agg, features_id, random_seed, results_dir):

    # 1. Read params (features)
    output_dir = tools_comparison.get_path_comparison(results_dir, random_seed, gt_agg=gt_agg)
    validate_path(output_dir)

    # 2. Read GT (train/test sets)
    fn_gt = os.path.join(output_dir, constants.FN_GT_TRAN_TEST_SAMPLE)
    print(fn_gt)
    data = ios.load_csv(fn_gt)

    # 3. Info variables
    var_y = f"ngender_{gt_agg}"
    feature_names = setup.FEATURE_NAMES[features_id]
    cat_features = [c for c in feature_names if c in setup.CAT_FEATURES]

    ios.printf(f"[INFO] y variable: {var_y}")
    ios.printf(f"[INFO] features: {feature_names}")
    ios.printf(f"[INFO] cat_features: {cat_features}")

    # 4. train/test sets
    df_train = data.query("ml_sample == 'train'").drop(columns=['ml_sample'])
    df_test = data.query("ml_sample == 'test'").drop(columns=['ml_sample'])
    # train
    X_train = df_train[feature_names]    # Features
    y_train = df_train[[var_y]]  # Target variable
    # test
    X_test = df_test[feature_names]    # Features
    y_test = df_test[[var_y]]  # Target variable
    # all 
    X = data[feature_names]    # Features
    y = data[[var_y]]  # Target variable

    # Train 
    best_model, grid_search_result = _train(X_train, y_train, X_test, y_test, 
                                            cat_features = cat_features, 
                                            loss_function = setup.LOSS_FUNCTION_REG, 
                                            eval_metric = setup.EVAL_METRIC_REG,
                                            cv = setup.CV, 
                                            stratified = True,
                                            early_stopping_rounds = setup.EARLY_STOPPING,
                                            random_seed = random_seed)

    # Evaluation
    y_pred, rmse, r2 = _eval(best_model, X_test, y_test)
    all_pred, all_rmse, all_r2 = _eval(best_model, X, y)

    ##### Saving Plots #####

    f_name = f"{gt_agg}_{features_id}"

    # Plot validation
    fn = os.path.join(output_dir, f'{f_name}_predictions_val.pdf')
    plot_true_vs_pred(y_test, y_pred, title="Validation", fn=fn, **{'rmse': rmse, 'r2': r2})

    # Plot all data
    fn = os.path.join(output_dir, f'{f_name}_predictions_all.pdf')
    plot_true_vs_pred(y, all_pred, title="All data", fn=fn, **{'rmse': all_rmse, 'r2': all_r2})

    # Plot feature importance
    fn = os.path.join(output_dir, f'{f_name}_feature_importance.pdf')
    sorted_feature_importance = best_model.feature_importances_.argsort()
    sorted_feature_names = [feature_names[id] for id in sorted_feature_importance]
    plot_feature_importance(best_model, sorted_feature_importance, sorted_feature_names, fn=fn)

    # Plot SHAP
    fn = os.path.join(output_dir, f'{f_name}_shap.pdf')
    plot_shap(best_model, X_test, sorted_feature_names, fn=fn)

    ##### Saving Model #####
    
    # model
    fn = os.path.join(output_dir, f"{f_name}.cbm")
    best_model.save_model(fn)

    ##### Saving Results #####
    # JSON
    obj = { 'y': gt_agg, 
            'features_id': features_id,
            'features': feature_names, 
            'nfeatures': len(feature_names), 
            'cat_features': cat_features,
            'best_params': grid_search_result['params'],
            'best_score': best_model.get_best_score(),
            'best_iteration': best_model.get_best_iteration(),
            'test_size': setup.TEST_SIZE,
            'random_state': random_seed,
            'loss_function': setup.LOSS_FUNCTION_REG,
            'eval_metric': setup.EVAL_METRIC_REG,
            'train_eval': best_model.get_best_score()['learn'][setup.LOSS_FUNCTION_REG], 
            'test_r2': r2, 
            'test_rmse': rmse,
            'all_r2': all_r2,
            'all_rmse': all_rmse,
            'f_name':f_name,
            'y_test': y_test.values.squeeze().tolist(),
            'y_pred': y_pred.squeeze().tolist()
            }
    
    fn = os.path.join(output_dir, f"{f_name}.json")
    save_json(obj, fn)

###############################################################################################
# MAIN
###############################################################################################

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Training to predict gender (numeric).')

    # Add arguments
    parser.add_argument('--gt_agg', help='Dependent variable (y)', type=str, required=True, choices=setup.GT_AGGS)
    parser.add_argument('--features_id', help='Features combination (0, 382)', type=int, required=True, choices=range(len(setup.FEATURE_NAMES)))
    parser.add_argument('--output_dir', help='../../results/v3/gender/predictions/comparison/', type=str, required=True)
    parser.add_argument('--seed', help='Random seed to use and to differentiate each set of runs or epoch ids', type=int, required=True)

    # Parse the arguments
    args = parser.parse_args()
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))

    start_time = time.time()
    validation(args.gt_agg, args.features_id)
    run(args.gt_agg, args.features_id, args.seed, args.output_dir)

    end_time = time.time()
    print('Execution time: {:.2f} seconds'.format(end_time - start_time))