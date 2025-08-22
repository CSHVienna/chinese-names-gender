from tkinter import N
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def __axlines(**kwargs):
    axhline = kwargs.get('axhline', None)
    if axhline:
        plt.axhline(**axhline)
    
    axhlines = kwargs.get('axhlines', None)
    if axhlines:
        for axhline in axhlines:
            plt.axhline(**axhline)
    
    axvline = kwargs.get('axvline', None)
    if axvline:
        plt.axvline(**axvline)

    axvlines = kwargs.get('axvlines', None)
    if axvlines:
        for axvline in axvlines:
            plt.axvline(**axvline)

    axdiagonal = kwargs.get('axdiagonal', None)
    if axdiagonal:
        plt.plot(axdiagonal.pop('x'), axdiagonal.pop('y'), **axdiagonal)

def __save(**kwargs):
    filename = kwargs.get('fn', None)
    dpi = kwargs.get('dpi', 300)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', dpi=dpi)

def __labels(title, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def __finaltouch(**kwargs):
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    plt.xlim(xlim)
    plt.ylim(ylim)

    legend = kwargs.get('legend', None)
    if legend is not None:
        plt.legend(**legend)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def __close():
    plt.show()
    plt.close()

def plot_violin(df, x, y, title='', xlabel='', ylabel='', hue=None, figsize=(10, 6), palette=None, kwargs_violin={}, **kwargs_plot):
    plt.figure(figsize=figsize)
    sns.violinplot(data=df, x=x, y=y, hue=hue, palette=palette, **kwargs_violin)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    __labels(title, xlabel, ylabel)
    __axlines(**kwargs_plot)
    __save(**kwargs_plot)
    __close()


def plot_scatter(data, x, y, title='', xlabel='', ylabel='', hue=None, hue_order=None, figsize=(10, 6), palette=None, corr=False, fit=False, rmse=False, **kwargs):
    plt.figure(figsize=figsize)
    fg = sns.scatterplot(data=data, x=x, y=y, hue=hue, palette=palette, hue_order=hue_order)

    yt = 0.01
    ys = 0.08

    
    df = data.copy().dropna(subset=[x, y])
    diff = data.shape[0] - df.shape[0]
    if diff > 0:
        print(f"[Warning] Dropped {diff} rows with NaN values")

    # Fit linear regression
    if fit:
        slope, intercept = np.polyfit(df[x], df[y], 1)
        print(slope, intercept)
        # Plot the regression line
        plt.plot(df[x], intercept + slope * df[x], color='black', ls='--', lw=1.0)
                #  label=f'y={intercept:.2f}+{slope:.2f}x')
        plt.text(0.98, yt, f'y={intercept:.2f}+{slope:.2f}x', transform=plt.gca().transAxes, ha='right', va='bottom')
        yt += ys

    # Pearson correlation
    if corr:
        corr, _ = pearsonr(df[x], df[y])
        plt.text(0.98, yt, f'Pearson correlation: {corr:.2f}', transform=plt.gca().transAxes, ha='right', va='bottom')
        yt += ys

    # root mean squared error
    if rmse:
        v = root_mean_squared_error(df[x], df[y])
        plt.text(0.98, yt, f'RMSE: {v:.2f}', transform=plt.gca().transAxes, ha='right', va='bottom')
        yt += ys

    
    __labels(title, xlabel, ylabel)
    __axlines(**kwargs)
    __finaltouch(**kwargs)
    __save(**kwargs)
    __close()

def plot_roc_auc(auc, fpr, tpr, labels, figsize, **kwargs):
    plt.figure(figsize=figsize)

    n_classes = len(labels)
    for i in range(n_classes):
        auci = round(auc[i],2) if auc is not None else 'N/A'
        plt.plot(fpr[i], tpr[i], label=f'{labels[i]} (auc = {auci})')

    plt.plot([0, 1], [0, 1], 'k--')
    
    kwargs['xlim'] = [0.0, 1.0]
    kwargs['ylim'] = [0.0, 1.0]
    kwargs['legend'] = {'loc': 'lower right'}

    title = 'ROC-AUC Curve'
    xlabel = 'False Positive Rate'
    ylabel = 'True Positive Rate'
    __labels(title, xlabel, ylabel)
    __axlines(**kwargs)
    __finaltouch(**kwargs)
    __save(**kwargs)
    __close()

def plot_confusion_matrix(cm, labels, title='Confusion Matrix', cmap='Blues', figsize=(10, 6), **kwargs):
    plt.figure(figsize=figsize)
    
    fmt = '.2f' if cm.dtype == np.float64 else 'd'
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', xticklabels=labels, yticklabels=labels)
    
    xlabel = 'Predicted'
    ylabel = 'True'
    
    __labels(title, xlabel, ylabel)
    __axlines(**kwargs)
    __finaltouch(**kwargs)
    __save(**kwargs)
    __close()

def plot_distribution_and_cuts(df, col, bins, figsize, cuts):
    ax = df[col].plot.hist(bins=bins, alpha=0.5, figsize=figsize)

    cummulative = 0
    for cut in cuts:
        ax.axvline(cut)
        s = df.query(f"{col} <= {cut}").shape[0] - cummulative
        ax.text(s=f'{s}', x=cut, y=0.95, ha='right', va='top', transform=ax.transAxes)
        cummulative += s

    s = df.query(f"{col} <= {df[col].max()}").shape[0] - cummulative
    ax.text(s=f'{s}', x=1., y=0.95, ha='right', va='top', transform=ax.transAxes)