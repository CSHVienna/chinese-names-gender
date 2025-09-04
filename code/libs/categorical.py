from collections import Counter
from re import sub
from tkinter import NE
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


##################################################################################################################
# CONSTANTS
##################################################################################################################

MALE = 'male'
FEMALE = 'female'
NEUTRAL = 'neutral'
UNKNOWN = 'unknown'
MOSTLY_FEMALE = 'mostly_female'
MOSTLY_MALE = 'mostly_male'

GENDER_LABELS_2 = [MALE, FEMALE]                                            # DO NOT CHANGE (ORDER MATTERS)
GENDER_LABELS_3 = [MALE, NEUTRAL, FEMALE]                                   # DO NOT CHANGE (ORDER MATTERS)
GENDER_LABELS_5 = [MALE, 'mostly_male', NEUTRAL, 'mostly_female', FEMALE]   # DO NOT CHANGE (ORDER MATTERS)

GENDER_LABELS_3_DICT = {g:i for i,g in enumerate(GENDER_LABELS_3)}

##################################################################################################################
# HELPERS
##################################################################################################################

def _mode(values):
    return Counter(values).most_common(1)[0][0]

def majority_voting(df: pd.DataFrame) -> pd.Series:
    """
    Perform majority voting on the columns provided.
    """
    return df.apply(_mode, axis=1)


##################################################################################################################
# EVALUTAION / PERFORMANCE
##################################################################################################################

def comparison_metrics(data, col_true, cols_pred, labels):
    index = ['precision', 'recall', 'f1', 'auc', 'accuracy', 
             'mean_precision', 'mean_recall','mean_f1', 'mean_auc',
             ] + [f'n_{label}' for label in labels]
    df_results = pd.DataFrame(columns=cols_pred, index=index)
    obj = {}
    d = 2 # decimal places

    df = data.copy()
    df.dropna(subset=cols_pred + [col_true], inplace=True)

    for col_pred in [col_true] + cols_pred:
        y_true = df[col_true]
        y_pred = df[col_pred]

        precision = precision_score(y_true, y_pred, labels=labels, average=None)
        recall = recall_score(y_true, y_pred, labels=labels, average=None)
        f1 = f1_score(y_true, y_pred, labels=labels, average=None)
        auc = roc_auc_score(label_binarize(y_true, classes=labels), label_binarize(y_pred, classes=labels), average=None, multi_class='ovr')
        accuracy = accuracy_score(y_true, y_pred)
        class_counts = Counter(y_pred)

        obj[col_pred] = [precision.round(d),recall.round(d), f1.round(d), auc.round(d), round(accuracy,d),
                         precision.mean().round(d),
                         recall.mean().round(d),
                         f1.mean().round(d),
                         auc.mean().round(d),
                         ] + [class_counts[label] for label in labels]

    df_results = pd.DataFrame(obj, index=index)
    return df_results

def get_performance(data, col_true, col_pred, labels, normalize=None):

    df = data.copy()
    df.dropna(subset=[col_pred, col_true], inplace=True)

    y_true = df[col_true]
    y_pred = df[col_pred]

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    precision = precision_score(y_true, y_pred, labels=labels, average=None)
    recall = recall_score(y_true, y_pred, labels=labels, average=None)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None)
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(label_binarize(y_true, classes=labels), label_binarize(y_pred, classes=labels), average=None, multi_class='ovr')

    return cm, precision, recall, f1, auc, accuracy

##################################################################################################################
# HELPERS
##################################################################################################################

def _get_label_by_threshold(value: float, thresholds: list, labels: list) -> str:
    result = [label for (low, high), label in zip(*[thresholds,labels]) if value >= low and value < high]

    if len(result) == 0:
        return None

    return result[0]
    
def _convert_numeric_to_categorical_by_thresholds(values: pd.Series, thresholds: list, labels: list) -> np.array:
    """
    Convert numeric values to categorical values based on the cuts and labels provided.
    """
    return values.apply(lambda value: _get_label_by_threshold(value, thresholds, labels))

##################################################################################################################
# EQUAL BINS
##################################################################################################################

def convert_numeric_to_categorical_equal_bins(values: np.array, num_bins: int, labels: list) -> np.array:
    """
    Convert numeric values to categorical values based on the number of bins provided.
    """
    return pd.cut(values, num_bins, labels=labels, include_lowest=True)

##################################################################################################################
# ARBITRARY THRESHOLDS
##################################################################################################################

def convert_numeric_to_categorical_by_thresholds(df, thresholds, labels, rot_fnc=None, new_trhesholds_fnc=None):
    # GT
    if 'mean' in df.columns and 'mode' in df.columns and 'median' in df.columns:
        df.loc[:,'gender_mean'] = _convert_numeric_to_categorical_by_thresholds(df['mean'], thresholds, labels)
        df.loc[:,'gender_mode'] = _convert_numeric_to_categorical_by_thresholds(df['mode'], thresholds, labels)
        df.loc[:,'gender_median'] = _convert_numeric_to_categorical_by_thresholds(df['median'], thresholds, labels)

    # Individual Tools
    for tool in ['r']:
        thresholds_tool = thresholds.copy()
        if new_trhesholds_fnc is not None:
            thresholds_tool = new_trhesholds_fnc(tool, thresholds)
        df.loc[:,f'gender_{tool}'] = _convert_numeric_to_categorical_by_thresholds(df[f'ngender_{tool}'], thresholds_tool, labels)
        
    # Combined Tools
    for tool in ['rn', 'rw', 'nw', 'rnw']:
        thresholds_tool = thresholds.copy()
        if new_trhesholds_fnc is not None:
            thresholds_tool = new_trhesholds_fnc(tool, thresholds)
        df.loc[:,f'gender_{tool}'] = _convert_numeric_to_categorical_by_thresholds(df[f'mean_ngender_{tool}'], thresholds_tool, labels)

    df.loc[:,'gender_mv_x1'] = majority_voting(df[['gender_r','gender_w','gender_n']])
    df.loc[:,'gender_mv_x2'] = majority_voting(df[['gender_rn','gender_rw','gender_nw']])
    df.loc[:,'gender_mv_x1x2'] = majority_voting(df[['gender_r','gender_w','gender_n','gender_rn','gender_rw','gender_nw']])
    df.loc[:,'gender_mv_all'] = majority_voting(df[['gender_r','gender_w','gender_n','gender_rn','gender_rw','gender_nw', 'gender_rnw']])
    df.loc[:,'gender_rot'] = majority_rule_of_thumb(df, rot_fnc) if rot_fnc is not None else None

    return df

def majority_rule_of_thumb(df: pd.DataFrame, rot_fnc) -> pd.Series:
    """
    Perform majority rule of thumb on the columns provided.
    """
    return df.apply(lambda row: rot_fnc(row), axis=1)

def _get_gender_rot_default(row: pd.Series) -> str:
    return UNKNOWN

##################################################################################################################
# RULE OF THUMB: BEST CASE | mean cuts=[0.445, 0.541]
##################################################################################################################
def _get_gender_rot(row: pd.Series) -> str:
    '''
    Works best for cuts [0.445, 0.541] (gender_mean)
    Highest performance (f1, precision, recall, auc, accuracy)
    But few data points in the neutral class
    '''

    cols = ['gender_r','gender_w','gender_n']

    if row['gender_r'] == MALE:
        # if tool r says MALE
        return MALE

    if row['gender_mv_x1'] == FEMALE: 
        # if all three tools say FEMALE
        return FEMALE

    if row['gender_r'] == FEMALE and row['gender_w'] == MALE and row['gender_n'] == MALE:
        return NEUTRAL

    if row['gender_r'] == NEUTRAL and row['gender_w'] == MALE and row['gender_n'] == FEMALE and row['gender_rn'] == NEUTRAL:
        return NEUTRAL

    if row['gender_r'] == NEUTRAL and row['gender_w'] == MALE and row['gender_n'] == MALE and row['gender_rn'] == NEUTRAL:
        return MALE

    if row['gender_r'] == NEUTRAL and row['gender_w'] == FEMALE and row['gender_n'] == MALE and row['gender_rn'] == MALE:
        return MALE

    if row['gender_r'] == NEUTRAL and row['gender_w'] == FEMALE and row['gender_n'] == MALE and row['gender_rn'] == MALE:
        return MALE
    
    if row['gender_r'] == NEUTRAL and row['gender_w'] == FEMALE and row['gender_n'] == MALE and row['gender_rn'] == NEUTRAL:
        return FEMALE

    if row['gender_r'] == NEUTRAL and row['gender_rn'] == FEMALE:
        return FEMALE
    if row['gender_r'] == NEUTRAL and row['gender_rn'] == MALE:
        return MALE

    if row['gender_r'] == FEMALE and row['gender_rnw'] == FEMALE:
        return FEMALE
    if row['gender_r'] == FEMALE and row['gender_rnw'] == MALE:
        return FEMALE

    return UNKNOWN

def _get_gender_rot_equivalent(row: pd.Series) -> str:
    '''
    Works best for cuts [0.445, 0.541] (gender_mean) - equivalent tools y-axis
    Highest performance (f1, precision, recall, auc, accuracy)
    But few data points in the neutral class
    '''

    cols = ['gender_r','gender_w','gender_n']

    if row['gender_r'] == FEMALE and row['gender_w'] == MALE and row['gender_n'] == MALE: 
        return NEUTRAL

    if row['gender_r'] == NEUTRAL and row['gender_w'] == MALE and row['gender_n'] == FEMALE and row['gender_rn'] == NEUTRAL: 
        return NEUTRAL
        
    if row['gender_r'] == MALE and row['gender_rn'] == MALE: 
        return MALE

    if row['gender_r'] == NEUTRAL and row['gender_w'] == FEMALE and row['gender_n'] == MALE: 
        return FEMALE

    if row['gender_r'] == NEUTRAL and row['gender_rn'] == FEMALE: 
        return FEMALE

    if row['gender_mv_x1'] == FEMALE: 
        # if all three tools say FEMALE
        return FEMALE
    
    if row['gender_mv_x1'] == MALE: 
        # if all three tools say MALE
        return MALE

    

    return UNKNOWN

##################################################################################################################
# RULE OF THUMB: BEST CASE | mean cuts=[0.45, 0.71] 
##################################################################################################################

def _get_gender_rot_all(row: pd.Series) -> str:
    '''
    Works for cuts cuts=[0.45, 0.71] (gender_mean)
    Highest performance (f1, precision, recall, auc, accuracy)
    But few data points in the neutral class
    '''
    cols = ['gender_r','gender_w','gender_n', 'gender_rn', 'gender_rw', 'gender_nw', 'gender_rnw']

    counts = Counter(row[cols].values)

    if row['gender_r'] == FEMALE and row['gender_w'] == FEMALE and row['gender_n'] == FEMALE:
        return FEMALE
    if row['gender_r'] == MALE and row['gender_w'] == MALE and row['gender_n'] == MALE:
        return MALE




    if row['gender_r'] == NEUTRAL and row['gender_w'] == FEMALE and row['gender_n'] == MALE and row['gender_rn'] == NEUTRAL and row['gender_rw'] == FEMALE  and row['gender_nw'] == NEUTRAL:
        return NEUTRAL

    if row['gender_r'] == MALE and row['gender_w'] == FEMALE and row['gender_n'] == MALE and row['gender_rn'] == MALE:
        return MALE

    if row['gender_r'] == NEUTRAL and row['gender_w'] == FEMALE and row['gender_n'] == MALE and row['gender_rn'] == NEUTRAL:
        return FEMALE

    if row['gender_r'] == NEUTRAL and row['gender_w'] == MALE and row['gender_n'] == MALE and row['gender_rn'] == NEUTRAL:
        return NEUTRAL

    if row['gender_r'] == NEUTRAL and row['gender_w'] == MALE and row['gender_n'] == MALE and row['gender_rn'] == MALE:
        return MALE

    if row['gender_r'] == MALE and row['gender_w'] == FEMALE and row['gender_n'] == FEMALE and row['gender_rn'] == NEUTRAL:
        return MALE
        
    if row['gender_r'] == MALE and row['gender_w'] == FEMALE and row['gender_n'] == FEMALE and row['gender_rn'] == MALE:
        return NEUTRAL



    if row['gender_r'] == FEMALE and row['gender_w'] == MALE and row['gender_n'] == FEMALE:
        return FEMALE

    if row['gender_r'] == NEUTRAL and row['gender_w'] == FEMALE and row['gender_n'] == FEMALE:
        return FEMALE

    if row['gender_r'] == NEUTRAL and row['gender_w'] == FEMALE and row['gender_n'] == MALE:
        return NEUTRAL




    if row['gender_r'] == NEUTRAL and row['gender_w'] == MALE and row['gender_n'] == FEMALE:
        return NEUTRAL
    
    if row['gender_r'] == MALE and row['gender_w'] == FEMALE and row['gender_n'] == MALE:
        return NEUTRAL

    if row['gender_r'] == MALE and row['gender_r'] == row['gender_mv_all']:
        return MALE 

    return UNKNOWN


def _get_gender_rot_all_equivalent(row: pd.Series) -> str:
    '''
    Works for cuts cuts=[0.45, 0.71] (gender_mean)
    Highest performance (f1, precision, recall, auc, accuracy)
    But few data points in the neutral class
    '''
    cols = ['gender_r','gender_w','gender_n', 'gender_rn', 'gender_rw', 'gender_nw', 'gender_rnw']

    counts = Counter(row[cols].values)

    if row['gender_r'] == row['gender_w'] and row['gender_w'] == row['gender_n'] and row['gender_n'] == FEMALE:
        return FEMALE
    if row['gender_r'] == row['gender_w'] and row['gender_w'] == row['gender_n'] and row['gender_n'] == MALE:
        return MALE

    if row['gender_r'] == NEUTRAL and row['gender_w'] == FEMALE and row['gender_n'] == FEMALE:
        return FEMALE
    if row['gender_r'] == NEUTRAL and row['gender_w'] == FEMALE and row['gender_n'] == MALE and row['gender_rn'] == MALE:
        return NEUTRAL
    if row['gender_r'] == MALE and row['gender_w'] == FEMALE and row['gender_n'] == FEMALE and row['gender_rn'] == NEUTRAL:
        return NEUTRAL
    if row['gender_r'] == NEUTRAL and row['gender_w'] == FEMALE and row['gender_n'] == MALE and row['gender_rn'] == NEUTRAL:
        return FEMALE
    if row['gender_r'] == FEMALE and row['gender_w'] == MALE and row['gender_n'] == FEMALE:
        return FEMALE
    if row['gender_r'] == MALE and row['gender_w'] == FEMALE and row['gender_n'] == FEMALE and row['gender_rn'] == MALE:
        return MALE
    if row['gender_r'] == MALE and row['gender_w'] == FEMALE and row['gender_n'] == MALE:
        return MALE
    if row['gender_r'] == MALE and row['gender_w'] == MALE and row['gender_n'] == FEMALE:
        return MALE
    if row['gender_r'] == NEUTRAL and row['gender_w'] == MALE:
        return NEUTRAL

    return UNKNOWN

    
    
##################################################################################################################
# RULE OF THUMB: BEST CASE | median cuts=[0.3, 0.7] 
##################################################################################################################

def _get_gender_rot_median(row: pd.Series) -> str:
    '''
    Works for cuts cuts=[0.3, 0.7] (gender_mean)
    Highest performance (f1, precision, recall, auc, accuracy)
    But few data points in the neutral class
    '''

    if row['gender_r'] == NEUTRAL and row['gender_w'] == FEMALE and row['gender_n'] == FEMALE and row['gender_rn'] == NEUTRAL and row['gender_rw'] == FEMALE and row['gender_nw'] == FEMALE and row['gender_rnw'] == NEUTRAL:
        return NEUTRAL
    if row['gender_r'] == NEUTRAL and row['gender_w'] == FEMALE and row['gender_n'] == MALE and row['gender_rn'] == NEUTRAL and row['gender_rw'] == FEMALE and row['gender_nw'] == FEMALE and row['gender_rnw'] == NEUTRAL:
        return FEMALE
    if row['gender_r'] == NEUTRAL and row['gender_w'] == FEMALE and row['gender_n'] == MALE and row['gender_rn'] == NEUTRAL and row['gender_rw'] == FEMALE and row['gender_nw'] == FEMALE and row['gender_rnw'] == FEMALE:
        return FEMALE
    if row['gender_r'] == NEUTRAL and row['gender_w'] == FEMALE and row['gender_n'] == MALE and row['gender_rn'] == NEUTRAL and row['gender_rw'] == FEMALE and row['gender_nw'] == FEMALE:
        return NEUTRAL
    if row['gender_r'] == NEUTRAL and row['gender_w'] == FEMALE and row['gender_n'] == MALE and row['gender_rn'] == NEUTRAL and row['gender_rw'] == FEMALE:
        return FEMALE
    if row['gender_r'] == NEUTRAL and row['gender_w'] == FEMALE and row['gender_n'] == MALE and row['gender_rn'] == NEUTRAL and row['gender_rw'] == NEUTRAL:
        return MALE
    if row['gender_r'] == FEMALE and row['gender_w'] == FEMALE and row['gender_n'] == FEMALE:
        return FEMALE
    if row['gender_r'] == MALE and row['gender_w'] == MALE and row['gender_n'] == MALE:
        return MALE
    if row['gender_r'] == FEMALE and row['gender_w'] == MALE and row['gender_n'] == FEMALE:
        return FEMALE
    if row['gender_r'] == NEUTRAL and row['gender_w'] == MALE and row['gender_n'] == MALE and row['gender_rnw'] == MALE:
        return MALE
    if row['gender_r'] == MALE:
        return MALE
    if row['gender_r'] == NEUTRAL and row['gender_mv_all'] == FEMALE:
        return FEMALE
    if row['gender_mv_all'] == MALE:
        return MALE
    if row['gender_mv_all'] == NEUTRAL:
        return NEUTRAL
    return UNKNOWN


def _get_gender_rot_median_equivalent(row: pd.Series) -> str:
    '''
    Works for cuts cuts=[0.45, 0.71] (gender_mean)
    Highest performance (f1, precision, recall, auc, accuracy)
    But few data points in the neutral class
    '''
    cols = ['gender_r','gender_w','gender_n',
            'gender_rn','gender_rw','gender_nw',
            'gender_rnw']
    unique_values = set(row[cols].values)

    if len(unique_values) == 1:
        return unique_values.pop()
    
    if row['gender_r'] == NEUTRAL and row['gender_w'] == FEMALE and row['gender_n'] == MALE and row['gender_rn'] == MALE:
        return MALE

    if row['gender_r'] == NEUTRAL and row['gender_w'] == FEMALE and row['gender_n'] == FEMALE and row['gender_rn'] == FEMALE:
        return NEUTRAL

    if row['gender_r'] == NEUTRAL and row['gender_w'] == FEMALE and row['gender_n'] == FEMALE and row['gender_rn'] == NEUTRAL and row['gender_rw'] == FEMALE:
        return FEMALE

    if row['gender_r'] == NEUTRAL and row['gender_w'] == FEMALE and row['gender_n'] == FEMALE and row['gender_rn'] == NEUTRAL and row['gender_rw'] == NEUTRAL:
        return NEUTRAL

    if row['gender_r'] == NEUTRAL and row['gender_w'] == FEMALE and row['gender_n'] == MALE and row['gender_rn'] == NEUTRAL and row['gender_rw'] == FEMALE:
        return FEMALE

    if row['gender_r'] == NEUTRAL and row['gender_w'] == FEMALE and row['gender_n'] == MALE and row['gender_rn'] == NEUTRAL and row['gender_rw'] == NEUTRAL:
        return NEUTRAL

    if row['gender_r'] == FEMALE and row['gender_w'] == MALE and row['gender_n'] == MALE and row['gender_rn'] == NEUTRAL:
        return NEUTRAL

    if row['gender_r'] == NEUTRAL and row['gender_w'] == MALE and row['gender_n'] == FEMALE and row['gender_rn'] == FEMALE:
        return NEUTRAL

    if row['gender_r'] == NEUTRAL and row['gender_w'] == MALE and row['gender_n'] == FEMALE and row['gender_rn'] == NEUTRAL:
        return NEUTRAL

    if row['gender_r'] == FEMALE and row['gender_w'] == MALE and row['gender_n'] == FEMALE:
        return FEMALE

    if row['gender_r'] == MALE and row['gender_w'] == FEMALE and row['gender_n'] == FEMALE:
        return NEUTRAL
    
    if row['gender_r'] == MALE and row['gender_w'] == FEMALE and row['gender_n'] == MALE:
        return MALE
    
    if row['gender_r'] == MALE and row['gender_w'] == MALE:
        return MALE
    
    if row['gender_r'] == NEUTRAL and row['gender_w'] == MALE and row['gender_n'] == MALE:
        return NEUTRAL
    
    return UNKNOWN
    

##################################################################################################################
# RULE OF THUMB: BEST CASE | mode cuts=[0.4, 0.6] 
##################################################################################################################

def _get_gender_rot_mode(row: pd.Series) -> str:

    if row['gender_r'] == FEMALE and row['gender_w'] == MALE and row['gender_n'] == FEMALE and row['gender_rn'] == NEUTRAL:
        return FEMALE 
    if row['gender_r'] == FEMALE and row['gender_w'] == MALE and row['gender_n'] == FEMALE and row['gender_rn'] == FEMALE and row['gender_rw'] == NEUTRAL:
        return FEMALE 
    if row['gender_r'] == FEMALE and row['gender_w'] == FEMALE and row['gender_n'] == FEMALE:
        return FEMALE 
    if row['gender_r'] == FEMALE and row['gender_w'] == FEMALE and row['gender_n'] == MALE:
        return FEMALE 
    if row['gender_r'] == FEMALE and row['gender_w'] == MALE and row['gender_n'] == FEMALE:
        return NEUTRAL 
    if row['gender_r'] == FEMALE and row['gender_w'] == MALE and row['gender_n'] == MALE:
        return NEUTRAL 
    if row['gender_r'] == MALE and row['gender_w'] == FEMALE and row['gender_n'] == MALE:
        return MALE
    if row['gender_r'] == NEUTRAL and row['gender_w'] == FEMALE and row['gender_n'] == FEMALE:
        return FEMALE
    if row['gender_r'] == NEUTRAL and row['gender_w'] == FEMALE and row['gender_n'] == MALE:
        return FEMALE
    if row['gender_r'] == NEUTRAL and row['gender_w'] == MALE and row['gender_n'] == FEMALE:
        return FEMALE
    if row['gender_r'] == MALE and row['gender_w'] == FEMALE and row['gender_n'] == FEMALE:
        return FEMALE

    if row['gender_r'] == MALE and row['gender_w'] == MALE and row['gender_n'] == FEMALE:
        return MALE
    if row['gender_r'] == MALE and row['gender_w'] == MALE and row['gender_n'] == MALE:
        return MALE
    if row['gender_r'] == NEUTRAL and row['gender_w'] == MALE and row['gender_n'] == MALE:
        return NEUTRAL

    return UNKNOWN

def _get_gender_rot_mode_equivalent(row: pd.Series) -> str:

    if row['gender_r'] == MALE and row['gender_w'] == FEMALE and row['gender_n'] == MALE and row['gender_mv_x2'] == FEMALE and row['gender_mv_x1x2'] == MALE and row['gender_mv_all'] == MALE and row['gender_rnw']==NEUTRAL:
        return MALE
    if row['gender_r'] == MALE and row['gender_w'] == FEMALE and row['gender_n'] == MALE and row['gender_mv_x2'] == FEMALE and row['gender_mv_x1x2'] == MALE and row['gender_mv_all'] == MALE and row['gender_rnw'] == FEMALE:
        return MALE
    if row['gender_r'] == MALE and row['gender_w'] == FEMALE and row['gender_n'] == MALE and row['gender_mv_x2'] == FEMALE and row['gender_mv_x1x2'] == MALE and row['gender_mv_all'] == MALE and row['gender_rnw'] == NEUTRAL:
        return NEUTRAL
    if row['gender_r'] == MALE and row['gender_w'] == FEMALE and row['gender_n'] == MALE and row['gender_mv_x2'] == FEMALE and row['gender_mv_x1x2'] == MALE and row['gender_mv_all'] == FEMALE:
        return NEUTRAL

    if row['gender_r'] == MALE and row['gender_w'] == FEMALE and row['gender_n'] == MALE and row['gender_rn'] == NEUTRAL:
        return FEMALE

    if row['gender_r'] == NEUTRAL and row['gender_w'] == MALE and row['gender_n'] == FEMALE:
        return FEMALE
    if row['gender_r'] == MALE and row['gender_w'] == FEMALE and row['gender_n'] == FEMALE:
        return NEUTRAL
    if row['gender_r'] == NEUTRAL and row['gender_w'] == MALE and row['gender_n'] == MALE:
        return NEUTRAL
    if row['gender_r'] == MALE and row['gender_w'] == MALE and row['gender_n'] == FEMALE:
        return MALE
    if row['gender_r'] == MALE and row['gender_w'] == FEMALE and row['gender_n'] == MALE:
        return MALE
    if row['gender_r'] == FEMALE and row['gender_w'] == MALE and row['gender_n'] == MALE:
        return NEUTRAL
    if row['gender_r'] == FEMALE and row['gender_w'] == MALE and row['gender_n'] == FEMALE:
        return FEMALE
    if row['gender_r'] == FEMALE and row['gender_w'] == FEMALE and row['gender_n'] == MALE:
        return FEMALE

    if row['gender_r'] == NEUTRAL and row['gender_w'] == FEMALE:
        return FEMALE
    if row['gender_r'] == FEMALE and row['gender_w'] == FEMALE and row['gender_n'] == FEMALE:
        return FEMALE
    if row['gender_r'] == MALE and row['gender_w'] == MALE and row['gender_n'] == MALE:
        return MALE
    
    return UNKNOWN

    # if row['gender_r'] == MALE and row['gender_w'] == FEMALE and row['gender_n'] == FEMALE and row['gender_rn'] == MALE:
    #     return NEUTRAL
    
    # if row['gender_r'] == MALE and row['gender_w'] == FEMALE and row['gender_n'] == FEMALE and row['gender_rn'] == NEUTRAL:
    #     return MALE
    
    # if row['gender_r'] == MALE and row['gender_w'] == FEMALE and row['gender_n'] == MALE and row['gender_rn'] == NEUTRAL:
    #     return NEUTRAL
    


    # if row['gender_r'] == NEUTRAL and row['gender_w'] == FEMALE and row['gender_n'] == MALE and row['gender_rn'] == NEUTRAL and row['gender_rw'] == FEMALE and row['gender_nw'] == FEMALE:
    #     return FEMALE
    
    # if row['gender_r'] == NEUTRAL and row['gender_w'] == FEMALE and row['gender_n'] == MALE and row['gender_rn'] == NEUTRAL and row['gender_rw'] == FEMALE and row['gender_nw'] == NEUTRAL:
    #     return NEUTRAL
    
    # if row['gender_r'] == NEUTRAL and row['gender_nw'] == FEMALE and row['gender_rnw'] == NEUTRAL:
    #     return NEUTRAL
    
    
    
    # if row['gender_r'] == NEUTRAL and row['gender_w'] == MALE and row['gender_n'] == MALE and row['gender_rn'] == MALE:
    #     return MALE
    
    # if row['gender_r'] == NEUTRAL and row['gender_w'] == MALE and row['gender_n'] == MALE and row['gender_rn'] == NEUTRAL:
    #     return NEUTRAL
    
    # if row['gender_r'] == NEUTRAL and row['gender_w'] == MALE:
    #     return NEUTRAL
    

    # if counts[FEMALE] >= len(cols) or counts.most_common(1)[0][0] == FEMALE:
    #     return FEMALE
    
    # if counts[MALE] >= len(cols) or counts.most_common(1)[0][0] == MALE:
    #     return MALE
    
    # if counts[NEUTRAL] >= len(cols) or counts.most_common(1)[0][0] == NEUTRAL:
    #     return NEUTRAL

    # if row['gender_r'] == FEMALE:
    #     return FEMALE

    # if row['gender_r'] == MALE and row['gender_w'] == MALE:
    #     return MALE
    
    
    # if row['gender_r'] == MALE and row['gender_w'] == FEMALE and row['gender_n'] == MALE and row['gender_rn'] == MALE and row['gender_rw'] == NEUTRAL:
    #     return NEUTRAL

    # if row['gender_r'] == MALE and row['gender_w'] == FEMALE and row['gender_n'] == MALE and row['gender_rn'] == MALE and row['gender_rw'] == FEMALE:
    #     return MALE
    
    