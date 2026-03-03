from collections import Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score

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
    Perform majority voting on the columns provided using the mode function.
    """
    return df.apply(_mode, axis=1)


def infer_gender_by_majority_vote(male_label_count, female_label_count, neutral_label_count, threshold=0.1):
    """
    Infer the gender by majority vote using the ratio of male, female and neutral labels.
    It does not use the mode function, but rather looks for the label that is majority taking into account the ratio of the other two labels.
    It also considers the case where both female and male together are the majority, then the neutral label is inferred.
    """
    ratio_male = male_label_count / (male_label_count + female_label_count + neutral_label_count)
    ratio_female = female_label_count / (male_label_count + female_label_count + neutral_label_count)
    ratio_neutral = neutral_label_count / (male_label_count + female_label_count + neutral_label_count)

    if ratio_neutral >= 0.5 or (ratio_male + ratio_female >= 0.5 and abs(ratio_male - ratio_female) < threshold) :
        return 'neutral'
    elif ratio_male > ratio_female:
        return 'male'
    else:
        return 'female'

def aggregate_gender(values):
    """
    Generic reducer for gender-like columns.
    Assumes values like 'male', 'female', 'neutral', 'unknown' (plus NaN).
    """
    uniques = {v for v in values if pd.notna(v)}
    if not uniques:
        return np.nan

    # Only one unique value -> return it
    if len(uniques) == 1:
        return next(iter(uniques))

    # If multiple values, but only one is not "unknown" -> return that one
    non_unknown = {v for v in uniques if v != "unknown" and v != 'unisex'}
    if len(non_unknown) == 1:
        return next(iter(non_unknown))

    # If we have both male and female (with or without unknown/neutral) -> neutral
    if "male" in uniques and "female" in uniques:
        return "neutral"

    # Fallback: if neutral is present among multiple values -> neutral
    if "neutral" in uniques or 'unisex' in uniques:
        return "neutral"

    # Last resort: pick a stable value (sorted for determinism)
    return sorted(uniques)[0]



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

    try:
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    except ValueError:
        cm = None
    try:    
        precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)    
    except ValueError:
        precision = None
    try:
        recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    except ValueError:
        recall = None
    try:
        f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    except ValueError:
        f1 = None
    try:
        # roc_auc requires y_scores (probabilities) not y_pred (labels)
        # y_true_binarized = label_binarize(y_true, classes=labels)
        # y_pred_binarized = label_binarize(y_pred, classes=labels)
        # auc = roc_auc_score(y_true_binarized, y_pred_binarized, average=None, multi_class='ovr')
        auc = None
    except ValueError:
        auc = None
    try:
        accuracy = accuracy_score(y_true, y_pred)
    except ValueError:
        accuracy = None

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
# PAIRWISE AGREEMENT BETWEEN GENDER INFERENCE METHODS
##################################################################################################################

def agreement_rate(s1, s2):
    """
    Calculate the agreement rate between two series.
    Simply the mean of the equality of the two series (accuracy)
    """
    return (s1 == s2).mean()

def pairwise_agreement(df: pd.DataFrame, method_cols: list) -> pd.DataFrame:
    """
    Calculate the pairwise agreement between the columns provided.
    Pairwise agreement rate (accuracy): proportion of rows where the two columns match
    """
    n = len(method_cols)
    agreement = pd.DataFrame(index=method_cols, columns=method_cols, dtype=float)
    for i, c1 in enumerate(method_cols):
        for j, c2 in enumerate(method_cols):
            df_ = df.dropna(subset=[c1, c2])
            agreement.loc[c1, c2] = agreement_rate(df_[c1], df_[c2])
    return agreement

def pairwise_kappa(df: pd.DataFrame, method_cols: list, labels: list=None) -> pd.DataFrame:
    """
    Calculate the pairwise kappa between the columns provided.
    Cohen's Kappa — agreement corrected for chance (κ = (p_o - p_e) / (1 - p_e))
    """
    unique_values = pd.unique(df[method_cols].values.ravel())
    unique_values = unique_values[pd.notna(unique_values)]

    labels = labels if labels is not None else unique_values
    kappa = pd.DataFrame(index=method_cols, columns=method_cols, dtype=float)
    for i, c1 in enumerate(method_cols):
        for j, c2 in enumerate(method_cols):
            df_ = df.dropna(subset=[c1, c2])
            kappa.loc[c1, c2] = cohen_kappa_score(df_[c1], df_[c2], labels=labels)
    return kappa



def comparison_table_to_latex(
    metrics_ch: dict,
    metrics_py: dict,
    reference_col: str,
    *,
    row_labels: dict | None = None,
    row_order: list | None = None,
    title: str = "Evaluation metrics vs. ground truth",
    caption: str = "Agreement (accuracy) and Cohen's Kappa of each method compared to the reference ground-truth.",
    decimal_places: int = 2,
    group_header_labels: tuple = ("Chinese", "Pinyin"),
) -> str:
    """
    Build a LaTeX table from two dicts of symmetric agreement/kappa matrices vs. a reference column.

    Each dict has the form {'agreement': dataframe, 'kappa': dataframe}. For each dataframe,
    the column given by `reference_col` is taken (values of each method vs. ground truth).
    Rows are the index of the dataframe excluding the reference (the methods being evaluated).

    Parameters
    ----------
    metrics_ch : dict
        First group: {'agreement': df, 'kappa': df} (e.g. Chinese).
        Optional key 'n': number of records to show in header (e.g. "Chinese (n=123)").
    metrics_py : dict
        Second group: {'agreement': df, 'kappa': df} (e.g. Pinyin).
        Optional key 'n': number of records to show in header (e.g. "Pinyin (n=456)").
    reference_col : str
        Name of the ground-truth column in the matrices.
    row_labels : dict, optional
        Map from index (method) names to display labels for the table rows.
    row_order : list, optional
        Order of row names; only these rows are included, in this order.
    title : str
        LaTeX table title (e.g. \\caption{}).
    caption : str
        LaTeX table caption text.
    decimal_places : int
        Number of decimal places for numeric cells.
    group_header_labels : tuple of str
        Labels for the top-level header row (merged over 2 columns each).
        Default ("Chinese", "Pinyin").

    Returns
    -------
    str
        LaTeX table as a string (including \\begin{table}, \\caption, \\label optional).
    """
    r_ch, a_ch, k_ch = metrics_ch["rmse"], metrics_ch["agreement"], metrics_ch["kappa"]
    r_py, a_py, k_py = metrics_py["rmse"], metrics_py["agreement"], metrics_py["kappa"]

    def col_vs_ref(df, ref):
        s = df[ref].drop(index=ref, errors="ignore")
        return s.reindex(index=[i for i in df.index if i != ref])

    acc_ch = col_vs_ref(a_ch, reference_col)
    kap_ch = col_vs_ref(k_ch, reference_col)
    
    acc_py = col_vs_ref(a_py, reference_col)
    kap_py = col_vs_ref(k_py, reference_col)

    rows = acc_ch.index.union(acc_py.index).drop_duplicates(keep="first")
    if row_order is not None:
        rows = [r for r in row_order if r in rows]
    else:
        rows = list(rows)

    rmse_py = pd.Series(index=rows, data={r: None if r != 'pred' else r_py[0] for r in rows}, dtype=float)
    rmse_ch = pd.Series(index=rows, data={r: None if r != 'pred' else r_ch for r in rows}, dtype=float)

    table = pd.DataFrame(
        {
            "rmse ch": rmse_ch.reindex(rows),
            "accuracy ch": acc_ch.reindex(rows),
            "kappa ch": kap_ch.reindex(rows),
            "rmse py": rmse_py.reindex(rows),
            "accuracy py": acc_py.reindex(rows),
            "kappa py": kap_py.reindex(rows),
        },
        index=rows,
    )

    # Format numbers and bold max per column (before renaming index)
    fmt = f"{{:0.{decimal_places}f}}"
    out = table.copy()
    for c in out.columns:
        out[c] = out[c].apply(lambda x: fmt.format(x) if pd.notna(x) else "—")
        idx_best = table[c].idxmin() if 'rmse' in c else table[c].idxmax()
        if pd.notna(idx_best) and table[c].notna().any():
            val = fmt.format(table.loc[idx_best, c])
            out.loc[idx_best, c] = "\\textbf{" + val + "}"

    if row_labels:
        out.index = out.index.map(lambda x: row_labels.get(x, x))

    latex_body = out.to_latex(
        index=True,
        escape=False,
        column_format="l" + "r" * len(out.columns),
    )

    # Replace single header row with two-row header: group names (with n= if provided) spanning 2 cols each, then accuracy/kappa
    g1, g2 = group_header_labels
    n_ch = metrics_ch.get("n")
    n_py = metrics_py.get("n")
    h1 = f"{g1} (n={n_ch})" if n_ch is not None else g1
    h2 = f"{g2} (n={n_py})" if n_py is not None else g2
    toprule = "\\toprule\n"
    i = latex_body.find(toprule)
    if i != -1:
        start = i + len(toprule)
        j = latex_body.find("\n", start)
        if j != -1:
            # Replace only the header line; leave \midrule and all data rows untouched
            line_end = " \\\\\n"  # LaTeX line break: space + two backslashes + newline
            new_header = (
                " & \\multicolumn{3}{c}{" + h1 + "} & \\multicolumn{3}{c}{" + h2 + "}" + line_end
                + r" & rmse$\downarrow$ & accuracy$\uparrow$ & kappa$\uparrow$ & rmse$\downarrow$ & accuracy$\uparrow$ & kappa$\uparrow$" + line_end
            )
            latex_body = latex_body[:start] + new_header + latex_body[j + 1:]

    # Wrap with table env, title and caption (short title for list of tables, full caption for text)
    caption_escaped = caption.replace("\\", "\\\\").replace("_", "\\_")
    title_escaped = title.replace("\\", "\\\\").replace("_", "\\_")
    title_escaped = "\\textbf{" + title_escaped + "}"
    
    full = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{"
        + f"{title_escaped} "
        + f"{caption_escaped} "
        + "}\n"
        "\\label{tab:comparison}\n"
        + latex_body
        + "\\end{table}"
    )

    print(f"[INFO] Pinyin aggregation: {r_py[1]}")
    return full




### --------------------------- FROM HERE DOWN IS THE OLD CODE --------------------------- ###

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
    
    