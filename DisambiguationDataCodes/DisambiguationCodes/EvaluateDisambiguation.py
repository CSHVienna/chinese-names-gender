import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def calculate_precision(test_file, var, df):
    """
    Compare predicted author indices with ground truth to count matches/mismatches.

    Parameters
    ----------
    test_file : pd.DataFrame
        Annotation file with columns ['title1_pid','title2_pid','name','Final'].
    var : str
        Column in test_file to use ('name' for Chinese, 'name_en' for English).
    df : pd.DataFrame
        Disambiguation results with columns ['paper_index','author_name','author_index'].

    Returns
    -------
    tuple : (true_match, true_not_match, false_negative, false_positive)
    """
    df = df.set_index(['paper_index', 'author_name']).copy()
    names = test_file[var].to_list()
    title1_pids = test_file['title1_pid'].to_list()
    title2_pids = test_file['title2_pid'].to_list()
    program_results = []

    for i in range(len(names)):
        idx1 = df.loc[title1_pids[i], names[i]]['author_index']
        idx2 = df.loc[title2_pids[i], names[i]]['author_index']
        # In case multiple results are returned, take first
        idx1 = idx1.values[0] if hasattr(idx1, "values") else idx1
        idx2 = idx2.values[0] if hasattr(idx2, "values") else idx2

        if idx1 == idx2:
            program_results.append("match")
        else:
            program_results.append("not_match")

    true_results = np.array(test_file['Final'].to_list())
    program_results = np.array(program_results)

    match_match = np.sum((true_results == 'match') & (program_results == 'match'))
    not_match_not_match = np.sum((true_results == 'not_match') & (program_results == 'not_match'))
    true_match_predict_not_match = np.sum((true_results == 'match') & (program_results == 'not_match'))
    true_not_match_predict_match = np.sum((true_results == 'not_match') & (program_results == 'match'))

    return match_match, not_match_not_match, true_match_predict_not_match, true_not_match_predict_match


def calculate_F1(test_file, var, df):
    """
    Compute Precision, Recall, and F1 Score.

    Parameters
    ----------
    test_file : pd.DataFrame
        Annotation file with ground truth labels.
    var : str
        Column in test_file ('name' or 'name_en').
    df : pd.DataFrame
        Disambiguation results.

    Returns
    -------
    tuple : (precision, recall, f1_score)
    """
    stat = calculate_precision(test_file, var, df)
    precision = stat[0] / (stat[0] + stat[3]) if (stat[0] + stat[3]) > 0 else 0
    recall = stat[0] / (stat[0] + stat[2]) if (stat[0] + stat[2]) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def evaluate_methods(test_file, result_files, var):
    """
    Evaluate multiple disambiguation methods.

    Parameters
    ----------
    test_file : pd.DataFrame
        Ground truth annotation file.
    result_files : dict
        Mapping of method names to file paths.
    var : str
        'name' for Chinese dataset, 'name_en' for English dataset.

    Returns
    -------
    dict : method -> (precision, recall, f1)
    """
    results = {}
    for method, file in result_files.items():
        df = pd.read_csv(file)
        results[method] = calculate_F1(test_file, var, df)
    return results


def plot_results(chinese_results, english_results, output_path="PLOTS/Comparison.pdf"):
    """
    Plot bar charts comparing methods in Chinese and English datasets.

    Parameters
    ----------
    chinese_results : dict
        Method -> (precision, recall, f1) for Chinese names.
    english_results : dict
        Method -> (precision, recall, f1) for Pinyin names.
    output_path : str
        File path to save the PDF plot.
    """
    methods = list(chinese_results.keys())
    metrics = ["Precision", "Recall", "F1 Score"]
    colors = ["black", "gray", "white"]
    edgecolor = "black"
    width = 0.25
    x = np.arange(len(metrics))

    chinese_data = [chinese_results[m] for m in methods]
    english_data = [english_results[m] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'wspace': 0.2})

    # Chinese dataset
    for i, method in enumerate(methods):
        axes[0].bar(x + i * width - width, chinese_data[i], width, label=method,
                    color=colors[i], alpha=1, linewidth=1.5, edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics, fontsize=15)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Score", fontsize=15)
    axes[0].set_title("Comparison in Chinese Character names", fontsize=16)
    axes[0].legend(loc="best", fontsize=12)
    for side in ["top", "right"]:
        axes[0].spines[side].set_visible(False)

    # English dataset
    for i, method in enumerate(methods):
        axes[1].bar(x + i * width - width, english_data[i], width, label=method,
                    color=colors[i], alpha=1, linewidth=1.5, edgecolor=edgecolor)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics, fontsize=15)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Comparison in Pinyin names", fontsize=16)
    axes[1].legend(loc="best", fontsize=12)
    for side in ["top", "right"]:
        axes[1].spines[side].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=500)
    plt.show()