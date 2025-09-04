
###############################################################################################
# DEPENDENCIES
###############################################################################################
import os
import time
import json
import argparse
import catboost as cb
# from code.scripts.batch_comparison_regression import LOSS_FUNCTION
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, r2_score, accuracy_score

import ios
from tools_comparison import ToolsComparison
import tools_comparison
import categorical
import setup
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

def validation(features_id, strategy_id):
    if features_id < 0 or features_id >= len(setup.FEATURE_NAMES):
        raise ValueError(f'Invalid features_id: {features_id} (0-{len(setup.FEATURE_NAMES)-1})')
    if strategy_id not in setup.STRATEGY.keys():
        raise ValueError(f'Invalid strategy_id: {strategy_id} (1-{len(setup.STRATEGY)})')

def _train(X_train, y_train, X_test, y_test, cat_features, loss_function, eval_metric, cv=setup.CV, stratified=True, early_stopping_rounds=setup.EARLY_STOPPING, random_seed=setup.RANDOM_STATE):
    # Pools
    train_pool = cb.Pool(X_train, label=y_train, cat_features=cat_features) 
    test_pool = cb.Pool(X_test, label=y_test, cat_features=cat_features) 

    # Model
    model = cb.CatBoostClassifier(verbose = False, 
                                  cat_features = cat_features,
                                  eval_metric = eval_metric,
                                  loss_function = loss_function,
                                  random_seed = random_seed) 

    # Tuning
    grid_search_result = model.grid_search(setup.GRID, train_pool, cv=cv,
                                            partition_random_seed=random_seed, 
                                            stratified=stratified, verbose=False)

    # best model
    best_model = model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=early_stopping_rounds)

    return best_model, grid_search_result


def _eval(model, X_test, y_test):
    # Predict class labels and probabilities
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # labels
    y_pred_labels = np.vectorize(lambda x: setup.LABELS[x])(y_pred) # Keep original if not in dict
    y_test_labels = np.vectorize(lambda x: setup.LABELS[x])(y_test) # Keep original if not in dict

    # print("y_pred_proba: ", y_pred_proba)
    # print('y_pred_labels', y_pred_labels)
    # print('y_test_labels', y_test_labels)

    # Compute Accuracy
    accuracy = accuracy_score(y_test_labels, y_pred_labels)

    # Compute F1-score (macro for multi-class)
    f1 = f1_score(y_test_labels, y_pred_labels, average='macro', labels=setup.LABELS)

    # Compute Precision & Recall (macro for multi-class)
    precision = precision_score(y_test_labels, y_pred_labels, average='macro', labels=setup.LABELS)
    recall = recall_score(y_test_labels, y_pred_labels, average='macro', labels=setup.LABELS)

    # Compute AUC (only if y_pred_proba has multiple columns for multi-class)
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

    # Compute Confusion Matrix
    conf_matrix = confusion_matrix(y_test_labels, y_pred_labels, labels=setup.LABELS)

    # Print Metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC: {auc:.4f}")
    print("Confusion Matrix:\n", conf_matrix)

    return y_pred, y_pred_proba, accuracy, f1, precision, recall, auc, conf_matrix

def _save_plot(fig, fn):
    fig.savefig(fn, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(conf_matrix, labels, title="Validation", fn=None):
    # Plot Confusion Matrix
    fig,ax = plt.subplots(1,1,figsize=FIGSIZE)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix {title}")
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


def save_json(obj, fn):
    with open(fn, "w") as file:
        json.dump(obj, file, indent=4)

def validate_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

###############################################################################################
# RUN
###############################################################################################
def run(features_id, strategy_id, random_seed, results_dir):
    
    # 1. Read params (strategy, features)
    cuts = setup.STRATEGY[strategy_id]['cuts']
    gt_agg = setup.STRATEGY[strategy_id]['gt_agg']
    output_dir = tools_comparison.get_path_comparison(results_dir, random_seed, strategy_id=strategy_id)
    validate_path(output_dir)

    # 2. Read GT (train/test sets)
    fn_gt = os.path.join(output_dir, constants.FN_GT_TRAN_TEST_SAMPLE)
    data = ios.load_csv(fn_gt)

    # 3. Encode target variales
    var_y = f'gender_{gt_agg}'
    var_y_encoded = f"{var_y}_encoded"
    data[var_y_encoded] = data[var_y].replace(setup.LABELS_CODE)

    # categroical features
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
    y_train = df_train[[var_y_encoded]]  # Target variable
    # test
    X_test = df_test[feature_names]    # Features
    y_test = df_test[[var_y_encoded]]  # Target variable
    # all 
    X = data[feature_names]    # Features
    y = data[[var_y_encoded]]  # Target variable

    # Training model
    best_model, grid_search_result = _train(X_train, y_train, X_test, y_test, 
                                            cat_features = cat_features, 
                                            loss_function = setup.LOSS_FUNCTION_CLA, 
                                            eval_metric = setup.EVAL_METRIC_CLA,
                                            cv = setup.CV, 
                                            stratified = True, 
                                            early_stopping_rounds = setup.EARLY_STOPPING, 
                                            random_seed = random_seed
                                            )

    # Evaluation
    y_pred, y_pred_proba, accuracy, f1, precision, recall, auc, conf_matrix = _eval(best_model, X_test, y_test)
    all_y_pred, all_y_pred_proba, all_accuracy, all_f1, all_precision, all_recall, all_auc, all_conf_matrix = _eval(best_model, X, y)

    ##### Saving Plots #####

    f_name = f'{gt_agg}_{features_id}_{strategy_id}'

    # Plot validation
    fn = os.path.join(output_dir, f'{f_name}_confusion_val.pdf')
    plot_confusion_matrix(conf_matrix, setup.LABELS, title="Validation", fn=fn)

    # Plot all data
    fn = os.path.join(output_dir, f'{f_name}_confusion_all.pdf')
    plot_confusion_matrix(all_conf_matrix, setup.LABELS, title="All data", fn=fn)

    # Plot feature importance
    fn = os.path.join(output_dir, f'{f_name}_feature_importance.pdf')
    sorted_feature_importance = best_model.feature_importances_.argsort()
    sorted_feature_names = [feature_names[id] for id in sorted_feature_importance]
    plot_feature_importance(best_model, sorted_feature_importance, sorted_feature_names, fn=fn)

    ##### Saving Model #####
    
    # model
    fn = os.path.join(output_dir, f"{f_name}.cbm")
    best_model.save_model(fn)

    ##### Saving Results #####
    # JSON
    obj = {'strategy_id': strategy_id,
           'features_id': features_id,
           'y': var_y, 
           'features': feature_names, 
           'nfeatures': len(feature_names), 
           'cat_features': cat_features,
           'best_params': grid_search_result['params'],
           'best_score': best_model.get_best_score(),
           'best_iteration': best_model.get_best_iteration(),
           'test_size': setup.TEST_SIZE,  
           'random_state': random_seed,
           'loss_function': setup.LOSS_FUNCTION_CLA,
           'eval_metric': setup.EVAL_METRIC_CLA,
           'train_eval': best_model.get_best_score()['learn'][setup.LOSS_FUNCTION_CLA], 
           'test_accuracy': accuracy,
           'test_f1': f1,
           'test_precision': precision,
           'test_recall': recall,
           'test_auc': auc,
           'all_accuracy': all_accuracy,
           'all_f1': all_f1,
           'all_precision': all_precision,
           'all_recall': all_recall,
           'all_auc': all_auc,
           'f_name':f_name,
           'labels': setup.LABELS,
           'y_test': y_test.values.squeeze().tolist(),
           'y_pred': y_pred.squeeze().tolist(),
           'y_pred_proba': y_pred_proba.squeeze().tolist()
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
    parser.add_argument('--features_id', help='Features combination (0, 382)', type=int, required=True, choices=range(len(setup.FEATURE_NAMES)))
    parser.add_argument('--strategy_id', help='Strategy id for thresholding', type=int, required=True, choices=setup.STRATEGY.keys())
    parser.add_argument('--output_dir', help='../../results/v3/gender/predictions/comparison', type=str, required=True)
    parser.add_argument('--seed', help='Random seed to use and to differentiate each set of runs or epoch ids', type=int, required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))

    start_time = time.time()

    validation(args.features_id, args.strategy_id)
    run(args.features_id, args.strategy_id, args.seed, args.output_dir)

    end_time = time.time()
    print('Execution time: {:.2f} seconds'.format(end_time - start_time))