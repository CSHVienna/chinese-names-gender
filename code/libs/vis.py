from curses import textpad
from socket import AF_X25

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np
from sklearn.metrics import root_mean_squared_error
import pandas as pd
from pdf2image import convert_from_path
from sklearn.metrics import r2_score
from matplotlib import rc
from itertools import cycle
import matplotlib.colors as mcolors # do not remove

from catboost import CatBoostRegressor
from catboost import CatBoostClassifier

from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D
from matplotlib.legend import Legend

import ast
from matplotlib.colors import Normalize

from sklearn.metrics import roc_curve, auc, cohen_kappa_score
from sklearn.preprocessing import label_binarize



from libs import constants
from libs import ios


####################################################################################################
# CONSTANTS
####################################################################################################

FIG_DPI = 600

####################################################################################################
# VISUAL HANDLERS
####################################################################################################

def sns_reset():
  sns.reset_orig()
  #matplotlib.use('agg')

def sns_paper_style():
  sns.set_context("paper", font_scale=1.5) #rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":5}) 
  rc('font', family = 'serif')
  
def set_style(font_scale=1.5):
    sns.reset_orig()
    sns.set_context("poster", font_scale=font_scale) #rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":5}) 
    rc('font', family = 'serif')
    
def set_latex():
  # rc('text', usetex=True)
  matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


####################################################################################################
# PLOT HANDLERS
####################################################################################################


def __axlines(**kwargs):
    ax = kwargs.get('ax', plt.gca())

    axhline = kwargs.get('axhline', None)
    if axhline:
        # plt.axhline(**axhline)
        ax.axhline(**axhline)
    
    axhlines = kwargs.get('axhlines', None)
    if axhlines:
        for axhline in axhlines:
            # plt.axhline(**axhline)
            ax.axhline(**axhline)
    
    axvline = kwargs.get('axvline', None)
    if axvline:
        # plt.axvline(**axvline)
        ax.axvline(**axvline)

    axvlines = kwargs.get('axvlines', None)
    if axvlines:
        for axvline in axvlines:
            # plt.axvline(**axvline)
            ax.axvline(**axvline)
            

    axdiagonal = kwargs.get('axdiagonal', None)
    if axdiagonal:
        # plt.plot(axdiagonal.pop('x'), axdiagonal.pop('y'), **axdiagonal)
        ax.plot(axdiagonal.pop('x'), axdiagonal.pop('y'), **axdiagonal)

def __save(**kwargs):
    filename = kwargs.get('fn', None)
    dpi = kwargs.get('dpi', FIG_DPI)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', dpi=dpi)

def __labels(title, xlabel, ylabel, **kwargs):
    ax = kwargs.get('ax', plt.gca())
    # if ax is not None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # else:
    #     plt.title(title)
    #     plt.xlabel(xlabel)
    #     plt.ylabel(ylabel)

def __finaltouch(**kwargs):
    ax = kwargs.get('ax', plt.gca())

    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)

    # if ax is not None:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # else:
    #     plt.xlim(xlim)
    #     plt.ylim(ylim)

    show_legend = kwargs.get('show_legend', True)
    if show_legend:
        legend = kwargs.get('legend', None)
        if legend is not None:
        #     if ax is not None:
            ax.legend(**legend)
            # else:
            #     plt.legend(**legend)
    else:
        ax.legend().remove()

    # ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def __close():
    plt.show()
    plt.close()


def create_plot(nrows, ncols, figsize=(10, 6), **kwargs):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)
    return fig, axes

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


def plot_scatter(data, x, y, title=None, xlabel='', ylabel='', 
                 true_var=None,
                 hue=None, hue_order=None, style=None, style_order=None, 
                 markersize=6,
                 palette=None, corr=False, fit=False, rmse=False, r2=False, diagonal=False, **kwargs):
    

    figsize = kwargs.get('figsize', None)
    if figsize is None:
        figure_width = kwargs.get('figure_width', 10)
        figure_height = kwargs.get('figure_height', 6)
        figsize = (figure_width, figure_height)

    save_and_close = kwargs.get('save_and_close', True)
    ax = kwargs.get('ax', None)
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=figsize)

    fg = sns.scatterplot(data=data, x=x, y=y, hue=hue, 
                         s=markersize,
                         palette=palette, hue_order=hue_order, style=style, style_order=style_order,
                         ax=ax,
                         )

    yt = 0.01
    ys = 0.1

    df = data.copy().dropna(subset=[x, y])
    diff = data.shape[0] - df.shape[0]
    if diff > 0:
        print(f"[Warning] Dropped {diff} rows with NaN values")

    bbox_annotations=dict(
        facecolor="white",
        alpha=0.8,
        edgecolor="none",
        boxstyle='round,pad=0.01'
    )

    # Fit linear regression
    if fit:
        slope, intercept = np.polyfit(df[x], df[y], 1)
        print(slope, intercept)
        # Plot the regression line
        ax.plot(df[x], intercept + slope * df[x], color='black', ls='--', lw=1.0)
                #  label=f'y={intercept:.2f}+{slope:.2f}x')
        ax.text(0.98, yt, f'y={intercept:.2f}+{slope:.2f}x', transform=AF_X25.transAxes, ha='right', va='bottom', bbox=bbox_annotations)
        yt += ys

    # Pearson correlation
    if corr:
        corr, _ = pearsonr(df[x], df[y])
        ax.text(0.98, yt, r"Pearson $\rho$: " + f"{corr:.2f}", transform=ax.transAxes, ha='right', va='bottom', bbox=bbox_annotations)
        yt += ys

    col_true = true_var
    col_pred = y if true_var == x else x
    
    # root mean squared error
    if rmse:
        v = root_mean_squared_error(df[col_true], df[col_pred])
        ax.text(0.98, yt, f'RMSE: {v:.2f}', transform=ax.transAxes, ha='right', va='bottom', bbox=bbox_annotations)
        yt += ys

    # R2
    if r2: 
        v = r2_score(df[col_true], df[col_pred])
        ax.text(0.98, yt, f'$R^2$: {v:.2f}', transform=ax.transAxes, ha='right', va='bottom', bbox=bbox_annotations)
        yt += ys

    # N
    ax.text(0.02, 0.98, f'N: {df.shape[0]}', transform=ax.transAxes, ha='left', va='top', bbox=bbox_annotations)

    # diagonal
    if diagonal:
        ax.plot([0,1], [0,1], ls='--', color='grey', lw=1, zorder=0)
    
    show_legend = kwargs.get("show_legend", True)
    flatten_legend = kwargs.get("flatten_legend", False)

    if hue and style and show_legend:
        legend_kwargs = kwargs.get("legend_kwargs", {"fontsize": 13, 'title_fontsize': 13})
        legend_fontsize = legend_kwargs.get("fontsize", 13)
        title_fontsize = legend_kwargs.get("title_fontsize", legend_fontsize)

        legend_handles, legend_labels = fg.get_legend_handles_labels()
        bbox_to_anchor = legend_kwargs.get("bbox_to_anchor", (0.45, 1.02))
        

        if flatten_legend:

            # Extract entries
            # Expected:
            # single_char, False, True,
            # unknown_features, False, True

            handletextpad = 0.7

            sc_handles = legend_handles[1:3]
            sc_labels = legend_labels[1:3]

            uf_handles = legend_handles[4:6]
            uf_labels = legend_labels[4:6]

            title_handle = Line2D([], [], linestyle="none", marker=None)

            leg1 = ax.legend(
                [title_handle] + sc_handles,
                ["single_char:"] + sc_labels,
                ncol=3,
                loc="lower center",
                bbox_to_anchor=bbox_to_anchor,
                frameon=True,
                fontsize=legend_fontsize,
                title_fontsize=title_fontsize,
                handlelength=0,
                handletextpad=handletextpad,
                columnspacing=0.8,
            )

            ax.add_artist(leg1)

            leg2 = ax.legend(
                [title_handle] + uf_handles,
                ["unknown_features:"] + uf_labels,
                ncol=3,
                loc="lower center",
                bbox_to_anchor=(bbox_to_anchor[0]+1.2, bbox_to_anchor[1]),
                frameon=True,
                fontsize=legend_fontsize,
                title_fontsize=title_fontsize,
                handlelength=0,
                handletextpad=handletextpad,
                columnspacing=0.8,
            )


        else:
            
            ax.legend(
                legend_handles,
                legend_labels,
                ncol=2,
                loc="lower center",
                bbox_to_anchor=bbox_to_anchor,
                borderaxespad=0,
                fontsize=legend_fontsize,
                title_fontsize=legend_fontsize,
            )

    __labels(title, xlabel, ylabel, **kwargs)
    __axlines(**kwargs)
    __finaltouch(**kwargs)

    if save_and_close:
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

def plot_confusion_matrix(cm, labels, title='Confusion Matrix', cmap='Blues', **kwargs): 
    cbar = kwargs.get('cbar', True)
    annot_kws = kwargs.get('annot_kws', {'fontsize': 10})
    annot = kwargs.get('annot', True)
    fmt = '.2f' if cm.dtype == np.float64 else 'd'
    vmin = kwargs.get('vmin', None)
    vmax = kwargs.get('vmax', None)
    ax = kwargs.get('ax', None)
    finalize = kwargs.get('finalize', True)

    if ax is None:
        figsize = kwargs.get('figsize', (10, 6))
        plt.figure(figsize=figsize)

    sns.heatmap(cm, annot=annot, fmt=fmt, cmap=cmap, 
                annot_kws=annot_kws,
                vmin=vmin, vmax=vmax,
                xticklabels=labels, yticklabels=labels, cbar=cbar, ax=ax)
    
    if finalize:
        xlabel = kwargs.get('xlabel', 'Predicted')
        ylabel = kwargs.get('ylabel', 'True')
        __labels(title, xlabel, ylabel)
        __axlines(**kwargs)
        __finaltouch(**kwargs)    
        __save(**kwargs)
        __close()

def plot_distribution_and_cuts(df, col, bins, figsize, cuts, **kwargs):
    ax = df[col].plot.hist(bins=bins, alpha=0.5, figsize=figsize)

    cummulative = 0
    for cut in cuts:
        ax.axvline(cut)
        s = df.query(f"{col} <= {cut}").shape[0] - cummulative
        ax.text(s=f'{s}', x=cut, y=0.95, ha='right', va='top', transform=ax.transAxes)
        cummulative += s

    s = df.query(f"{col} <= {df[col].max()}").shape[0] - cummulative
    ax.text(s=f'{s}', x=1., y=0.95, ha='right', va='top', transform=ax.transAxes)

    # __labels(title, xlabel, ylabel)
    __axlines(**kwargs)
    __finaltouch(**kwargs)
    __save(**kwargs)
    __close()



### PLOTS FOR PREDICTIONS
def plot_confusion_matrix_from_file(fn):
    images = convert_from_path(fn)
    plt.imshow(images[0])
    plt.axis("off")  # Hide axes
    plt.show()
    plt.close()
    
def plot_scatter_from_file(fn):
    # predictions
    images = convert_from_path(fn)
    plt.imshow(images[0])
    plt.axis("off")  # Hide axes
    plt.show()
    plt.close()

def plot_importance_from_file(fn):
    images = convert_from_path(fn)
    plt.imshow(images[0])
    plt.axis("off")  # Hide axes
    plt.show()
    plt.close()

def plot_shap_from_file(fn):
    images = convert_from_path(fn)
    plt.imshow(images[0])
    plt.axis("off")  # Hide axes
    plt.show()
    plt.close()


def replot_confusion_matrix(f_name, test_only=True, **kwargs):
    
    from sklearn.metrics import confusion_matrix

    figure_width = kwargs.pop('figure_width', 4)
    figure_height = kwargs.pop('figure_height', 3)
    figsize = (figure_width, figure_height)
    title = kwargs.pop('title', None)
    cbar = kwargs.pop('cbar', True)

    fn_model = f"{f_name}.json"
    obj = ios.read_json(fn_model)
    labels = obj['labels']

    # Loading model
    if test_only:
        y_true = obj['y_test']
        y_pred = obj['y_pred']
        # y_pred_proba = obj['y_pred_proba']
        y_pred_labels = np.vectorize(lambda x: labels[x])(y_pred) # Keep original if not in dict
        y_true_labels = np.vectorize(lambda x: labels[x])(y_true) # Keep original if not in dict
    else:
        fn_sample = ios.path_join(ios.get_path(f_name), constants.FN_GT_TRAN_TEST_SAMPLE)
        data = ios.load_csv(fn_sample, index_col=0)
        
        # Loading model
        fn_model = f"{f_name}.cbm"
        model = CatBoostClassifier()
        model.load_model(fn_model, format="cbm")
        col_features = model.feature_names_

        # Loading model summary
        y_var = obj['y']

        # predict
        y_pred = data.apply(lambda row: model.predict(row[col_features])[0], axis=1)
        y_pred_labels = np.vectorize(lambda x: labels[x])(y_pred) # Keep original if not in dict
        y_true_labels = data.loc[:, y_var]

    cm = confusion_matrix(y_true=y_true_labels, y_pred=y_pred_labels, labels=labels)
    
    plt.figure(figsize=figsize)
    
    fmt = '.2f' if cm.dtype == np.float64 else 'd'
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=labels, 
                yticklabels=labels,
                cbar=cbar)
    plt.yticks(rotation=90) 


    xlabel = 'Predicted'
    ylabel = 'True'
    
    fn = f"{f_name}_confusion_{'val' if test_only else 'all'}_enhanced.pdf"
    kwargs['fn'] = fn

    __labels(title, xlabel, ylabel)
    __axlines(**kwargs)
    __finaltouch(**kwargs)
    __save(**kwargs)
    __close()



def replot_rocauc(f_name, test_only=True, **kwargs):
    from libs import setup
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import cohen_kappa_score

    figure_width = kwargs.pop('figure_width', 4)
    figure_height = kwargs.pop('figure_height', 3)
    figsize = (figure_width, figure_height)
    title = kwargs.pop('title', None)
    cbar = kwargs.pop('cbar', True)

    fn_model = f"{f_name}.json"
    obj = ios.read_json(fn_model)
    labels = obj['labels']

    # Loading model
    if test_only:
        y_true = obj['y_test']
        y_true_labels = np.vectorize(lambda x: labels[x])(y_true) # Keep original if not in dict
        y_pred_proba = np.array(obj['y_pred_proba'])
        y_pred = obj['y_pred']
        y_pred_labels = np.vectorize(lambda x: labels[x])(y_pred) # Keep original if not in dict
    else:
        fn_sample = ios.path_join(ios.get_path(f_name), constants.FN_GT_TRAN_TEST_SAMPLE)
        data = ios.load_csv(fn_sample, index_col=0)
        
        # Loading model
        fn_model = f"{f_name}.cbm"
        model = CatBoostClassifier()
        model.load_model(fn_model, format="cbm")
        col_features = model.feature_names_

        # Loading model summary
        y_var = obj['y']
        X = data.loc[:,col_features]

        # predict
        y_pred = model.predict(X)
        y_pred_labels = np.vectorize(lambda x: labels[x])(y_pred) # Keep original if not in dict
        y_pred_proba = model.predict_proba(X)
        y_true_labels = data.loc[:, y_var]

    # Map labels to indices 0..K-1 in the specified order
    name_to_idx = {c: i for i, c in enumerate(labels)}
    y_idx = np.array([name_to_idx[y] for y in y_true_labels])
    
    K = len(labels)
    Y = label_binarize(y_idx, classes=np.arange(K))  # (n, K)

    fig, ax = plt.subplots(1,1,figsize=figsize)

    # Per-class ROC
    aucs = {}
    for k, cname in enumerate(labels):
        fpr, tpr, _ = roc_curve(Y[:, k], y_pred_proba[:, k])
        aucs[cname] = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{cname} (AUC={aucs[cname]:.3f})")

    # Macro-average AUC (unweighted mean of class AUCs)
    macro_auc = roc_auc_score(Y, y_pred_proba, average="macro", multi_class="ovr")
    kappa_str = r"Cohen's $\kappa$"
    kappa = cohen_kappa_score(y_true_labels, y_pred_labels, labels=labels)

    yb = 0.3
    d = 0.07
    ax.text(0.98, yb+(d*2), f'N: {Y.shape[0]}', transform=ax.transAxes, ha='right', va='bottom')
    ax.text(0.98, yb+(d*1), f'Macro-AUC: {macro_auc:.3f}', transform=ax.transAxes, ha='right', va='bottom')
    ax.text(0.98, yb, f'{kappa_str}: {kappa:.3f}', transform=ax.transAxes, ha='right', va='bottom')

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.5, color='grey')

    xlabel = "False Positive Rate"
    ylabel = "True Positive Rate"
    title = kwargs.pop('title', None)

    ax.legend(frameon=True, fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    fn = f"{f_name}_rocauc_{'val' if test_only else 'all'}_enhanced.pdf"
    kwargs['fn'] = fn

    __labels(title, xlabel, ylabel)
    __axlines(**kwargs)
    __finaltouch(**kwargs)
    __save(**kwargs)
    __close()




def replot_importance(f_name, regressor=False, **kwargs):
    

    # Loading model
    fn_model = f"{f_name}.cbm"
    model = CatBoostRegressor() if regressor else CatBoostClassifier()
    model.load_model(fn_model, format="cbm")
    col_features = model.feature_names_

    # Loading feature importance
    feature_importance_dict = {}
    for fn, fi in zip(*(col_features,model.get_feature_importance())):
        feature_importance_dict[fn] = fi
    feature_importance_df = pd.DataFrame(feature_importance_dict, index=['features'])


    # Plotting enhance version
    width = kwargs.pop('figure_width', 4)
    height = len(col_features) * 0.3 + 1.1
    height = kwargs.pop('figure_height', height)
    figsize = (width, height)

    f, ax = plt.subplots(figsize=figsize)
    data = pd.DataFrame(feature_importance_df.T).reset_index().rename(columns={'features':'Importance', 'index':'Features'}).sort_values('Importance', ascending=False)
    sns.barplot(data=data, x='Importance', y='Features', color='tab:green', width=0.9, ax=ax)

    ax.set_xlabel("Importance (%)")

    fn = f"{f_name}_feature_importance_enhanced.pdf"

    __axlines()
    __finaltouch()
    __save(fn=fn)
    __close()


def replot_scatter(f_name, regressor=True, **kwargs):
    from catboost import CatBoostRegressor
    from catboost import CatBoostClassifier
    from libs import setup
    from libs import ios
    from libs.inference import categorical

    # Loading data
    fn_sample = ios.path_join(ios.get_path(f_name), "ground_truth_MLsampled.csv")
    data = ios.load_csv(fn_sample)
    
    # Loading model
    fn_model = f"{f_name}.cbm"
    model = CatBoostRegressor() if regressor else CatBoostClassifier()
    model.load_model(fn_model, format="cbm")
    col_features = model.feature_names_

    # Loading model summary
    fn_summary = f"{f_name}.json"
    obj = ios.read_json(fn_summary)
    y_var = "ngender" if regressor else 'gender'
    y_var = f"{y_var}_{obj['y']}"

    # predict
    if not regressor:
        data.loc[:,'pred_code'] = data.apply(lambda row: model.predict(row[col_features])[0], axis=1)
        data.loc[:,'pred_label'] = data.apply(lambda row: setup.LABELS[row.pred_code], axis=1)
    else:
        data.loc[:,'pred'] = data.apply(lambda row: model.predict(row[col_features]), axis=1)
    
    unknown_feature_col = f'{categorical.UNKNOWN}_features'
    data.loc[:,unknown_feature_col] = data.apply(lambda row: not pd.notna(row['ngender_r']) and not pd.notna(row['ngender_n']), axis=1)

    # plot test
    for key, df in {'val':data.query("ml_sample == 'test'"), 'all':data}.items():
        fn = f"{f_name}_predictions_{key}_enhanced.pdf"
        
        plot_scatter(df, 
                    
                    y='pred',  
                    ylabel='Predictions',

                    x=y_var, 
                    xlabel='True values', 
                    
                    true_var = y_var,

                    markersize=50,
                    title='',  
                    hue='single_char', 
                    hue_order=[False, True], 
                    
                    style = unknown_feature_col,
                    style_order = [False, True],
                    # figsize=(4, 3), 
                    palette=None, 
                    corr=True, fit=False, rmse=True, r2=True, diagonal=True,
                    fn=fn,
                    **kwargs
                    )

    plt.tight_layout()
    plt.show()
    plt.close()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def shap_dot_summary(
    shap_vals,                 # (n, p)
    X,                         # DataFrame or array (n, p)
    feature_names,             # list/array length p
    order=None,                # indices, optional
    max_display=20,
    s=18,
    alpha=0.9,
    jitter=0.28,
    xlim=None,
    ax=None,
    title=None,
    colorbar=False,
    cmap=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    # Convert inputs
    shap_vals = np.asarray(shap_vals)
    # Xv = X.values if hasattr(X, "values") else np.asarray(X)
    # Xv = Xv.astype(float)

    if hasattr(X, "iloc"):  # DataFrame
        X_numeric = X.apply(pd.to_numeric, errors="coerce")
        Xv = X_numeric.to_numpy(dtype=float)
    else:
        Xv = np.asarray(X, dtype=float)
    Xv = Xv.astype(float)

    n, p = shap_vals.shape
    if feature_names is None:
        feature_names = np.array([f"f{i}" for i in range(p)])
    else:
        feature_names = np.array(feature_names)

    # Feature order
    if order is None:
        imp = np.mean(np.abs(shap_vals), axis=0)
        order = np.argsort(imp)[::-1]
    order = np.array(order)[:min(max_display, p)]

    shap_vals = shap_vals[:, order]
    Xv = Xv[:, order]
    feature_names = feature_names[order]

    m = shap_vals.shape[1]
    y_positions = np.arange(m)[::-1]

    # --- colormap: prefer SHAP's red_blue if available
    if cmap is None:
        try:
            import shap
            cmap = shap.plots.colors.red_blue
        except Exception:
            cmap = plt.get_cmap("RdBu_r")

    # --- draw dotted horizontal guides
    for y in y_positions:
        ax.hlines(y, xmin=np.nanmin(shap_vals), xmax=np.nanmax(shap_vals),
                  colors="#d9d9d9", linestyles=":", linewidth=1, zorder=0)

    # --- scatter per feature
    # Use one global normalization like SHAP (0..1 works for your features; otherwise use percentiles)
    vmin, vmax = 0.0, 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)

    rng = np.random.default_rng(0)  # reproducible jitter
    for j, y0 in enumerate(y_positions):
        y = y0 + (rng.random(n) - 0.5) * 2 * jitter
        c = cmap(norm(Xv[:, j]))
        ax.scatter(shap_vals[:, j], y, s=s, c=c, alpha=alpha, linewidths=0, zorder=2)


    # --- vertical zero line
    ax.axvline(0, color="#808080", linewidth=1, zorder=1)

    # --- axis formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels(feature_names) #, fontsize=11)
    ax.set_xlabel("SHAP value") #, fontsize=12)
    if title:
        ax.set_title(title) #, fontsize=12)

    # remove spines (SHAP-like)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(1)

    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x") #, labelsize=10)

    if xlim is not None:
        ax.set_xlim(xlim)

    # --- optional shared colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    if colorbar:
        cb = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.03)
        cb.set_label("Feature value", rotation=90)
        cb.set_ticks([vmin, vmax])
        cb.set_ticklabels(["Low", "High"], fontsize=9)
        cb.outline.set_visible(False)

    return fig, ax, sm


def replot_shap(f_name, regressor=True, **kwargs):
    from catboost import CatBoostRegressor
    from catboost import CatBoostClassifier
    import shap
    import json
    from libs import ios
    from libs import setup

    def to_sv_list(shap_values, X):
        # Case 1: SHAP already returns list per class
        if isinstance(shap_values, list):
            sv_list = [np.asarray(sv) for sv in shap_values]
            return sv_list

        sv = np.asarray(shap_values)

        # Case 2: regression style (n, p)
        if sv.ndim == 2:
            return [sv]

        # Case 3: classification style, but axis order may vary
        if sv.ndim == 3:
            n = X.shape[0]

            # (n, p, k)
            if sv.shape[0] == n:
                return [sv[:, :, k] for k in range(sv.shape[2])]

            # (k, n, p)
            if sv.shape[1] == n:
                return [sv[k, :, :] for k in range(sv.shape[0])]

            raise ValueError(
                f"3D shap_values has ambiguous layout: sv.shape={sv.shape}, X.shape={X.shape}. "
                "Expected (n,p,k) or (k,n,p)."
            )

        raise ValueError(f"Unexpected shap_values shape: {sv.shape}")
        
    # Loading data
    fn_sample = ios.path_join(ios.get_path(f_name),constants.FN_GT_TRAN_TEST_SAMPLE)
    data = ios.load_csv(fn_sample)
    
    # model
    # Loading model
    fn_model = f"{f_name}.cbm"
    model = CatBoostRegressor() if regressor else CatBoostClassifier()
    model.load_model(fn_model, format="cbm")
    col_features = model.feature_names_
    feature_names = np.array(col_features)
    

    # SHAP
    X_test = data.query("ml_sample == 'test'")[col_features]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    sv_list = to_sv_list(shap_values, X_test)

    # figure
    height = len(col_features) * 0.3 + 1.
    height = kwargs.pop('figure_height', height)
    
    # making categorical numeric (so they appear in the colorbar):
    for c in feature_names:
        if c.startswith('gender_'):
            X_test.loc[:,c] = X_test.loc[:,c].map({v:i/(len(setup.LABELS)-1) for i,v in enumerate(setup.LABELS)})

    if regressor:
        width = kwargs.pop('figure_width', 4)
        figsize = (width, height)

        fig, ax = plt.subplots(1, 1, figsize=figsize, sharey=True, sharex=True)

        fig, ax, sm = shap_dot_summary(
            shap_vals=sv_list[0],         # (n, p)
            X=X_test,                  # (n, p) DataFrame or array
            feature_names=feature_names,
            ax=ax,
            colorbar=True,
        )
        

    else:
        if not (X_test.shape[1] == len(col_features) == len(model.feature_names_)):
            print("[Warning] X_test.shape[1] != len(col_features) != len(model.feature_names_)")

        with open(f"{f_name}.json", "r") as f:
            labels = json.load(f)["labels"] 

        # global order across classes
        all_sv = np.stack(sv_list, axis=2)  # (n,p,k)
        global_imp = np.mean(np.abs(all_sv), axis=(0,2))
        order = np.argsort(global_imp)[::-1]

        subplot_width = kwargs.pop('subplot_width', 4)
        width = kwargs.pop('figure_width', subplot_width*len(labels))
        figsize = (width, height)
        fig, axes = plt.subplots(1, len(labels), figsize=figsize, sharey=True, sharex=True)

        mappables = []
        for k, lab in enumerate(labels):
            _, _, sm = shap_dot_summary(
                sv_list[k], 
                X_test,
                feature_names=feature_names,
                order=order,
                max_display=10,
                ax=axes[k],
                title=lab,
                colorbar=False,
            )
            mappables.append(sm)

        # One shared colorbar (use the first mappable)
        # Leave space on right
        fig.subplots_adjust(right=0.89)

        # Add colorbar axis manually
        cax = fig.add_axes([0.90, 0.15, 0.005, 0.8])  
        # [left, bottom, width, height] in figure coordinates

        cb = fig.colorbar(mappables[0], cax=cax)
        cb.set_label("Feature value")
        cb.set_ticks([0.0, 1.0])
        cb.set_ticklabels(["Low", "High"], fontsize=9)
        cb.outline.set_visible(False)

        # Add spacing between plots
        plt.subplots_adjust(wspace=0.1)
    

    fn = f"{f_name}_shap_enhanced.pdf"
    __save(fn=fn)
    __close()


####################################################################################################
# PLOTS FOR APS VALIDATION
####################################################################################################

def plot_gender_comparison(
    df,
    method_order,
    method_labels=None,
    xlabel_order=None,
    colors=None,
    hatches=None,
    use_percentages=True,
    annotate_bars=False,
    title=None,
    fn=None,
    figsize=(6, 4),
    n_annotation_kwargs=None,
    annotate_kwargs=None,
    legend_kwargs=None,
    **plot_kwargs,
):
    """
    Grouped bar chart comparing gender distribution across multiple methods.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the method columns.
    method_order : list of str
        Column names in the order you want for legend and bars (e.g. ["gender_clean", "gender_nq_clean", "pred_label"]).
    method_labels : dict, optional
        Map column name -> legend label (e.g. {"pred_label": "Ours"}). If None, column names are used.
    xlabel_order : list of str, optional
        Order of gender categories on x-axis (e.g. ["female", "male", "neutral", "unknown"]).
        Categories in data but not in this list are appended in sorted order.
    colors : dict, optional
        One color per method (e.g. {"py_gender_karimi": "#2ecc71", "py_gender_saeta": "#3498db"}).
    hatches : dict, optional
        One hatch pattern per method (e.g. {"py_gender_karimi": None, "py_gender_saeta": "/"})
    use_percentages : bool, default True
        If True, show % within each method (each method sums to 100%); else show counts.
    annotate_bars : bool, default False
        If True, show values on top of each bar.
    annotate_kwargs : dict, optional
        Passed to ax.text() for bar labels (e.g. fontsize, ha, va).
    legend_kwargs : dict, optional
        Passed to ax.legend().
    title : str, optional
        Axes title.
    figsize : tuple, default (6, 4)
        Figure size.
    n_annotation_kwargs : dict, optional
        Passed to ax.text() for the "N=<count>" annotation (e.g. fontsize, transform, x, y, va, ha).
        Default places it in the top-left in axes coordinates.
    **plot_kwargs
        Additional kwargs passed to plt.subplots() (e.g. constrained_layout=True).

    Returns
    -------
    fig, ax
    """
    

    method_labels = method_labels or {}
    annotate_kwargs = dict(annotate_kwargs or {})
    legend_kwargs = dict(legend_kwargs or {})
    n_annotation_kwargs = dict(n_annotation_kwargs or {})
    colors = colors or {"py_gender_karimi": "#2ecc71", "py_gender_saeta": "#3498db", "py_gender_buskirk": "#e74c3c", "py_gender_mv": "#95a5a6", "ch_pred_label": "#f1c40f"}
    xlabel_order = xlabel_order or ["female", "male", "neutral", "unknown"]

    counts = {col: df[col].value_counts() for col in method_order if col in df.columns}
    all_cats = set().union(*(c.index.astype(str) for c in counts.values()))
    xlabel_order = [c for c in xlabel_order if c in all_cats] + sorted(all_cats - set(xlabel_order))
    n_cats = len(xlabel_order)
    n_methods = len(counts)
    width = 0.8 / n_methods

    if use_percentages:
        totals = {col: counts[col].sum() for col in counts.keys()}
        data = np.array([
            [counts[col].get(cat, 0) / totals[col] * 100 for cat in xlabel_order]
            for col in counts.keys()
        ])
        ylabel = "Percentage (%)"
    else:
        data = np.array([
            [counts[col].get(cat, 0) for cat in xlabel_order]
            for col in counts.keys()
        ])
        ylabel = "Count"

    x = np.arange(n_cats)
    fig, ax = plt.subplots(figsize=figsize, **plot_kwargs)
    for i, col in enumerate(counts.keys()):
        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(
            x + offset, data[i], width,
            label=method_labels.get(col, col),
            color=colors.get(col, "#000000") if colors else None,
            hatch=hatches.get(col, None) if hatches else None,
        )
        if annotate_bars:
            default_annot = {"ha": "center", "va": "bottom", "fontsize": 8}
            default_annot.update(annotate_kwargs)
            for bar, val in zip(bars, data[i]):
                if val != 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.1f}" if use_percentages else f"{int(val)}", **default_annot)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabel_order)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Gender")
    if title is not None:
        ax.set_title(title)
    ax.legend(**legend_kwargs)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    n_records = len(df)
    default_n_kw = {"transform": ax.transAxes, "x": 0.02, "y": 0.98, "va": "top", "ha": "left", "fontsize": 9}
    default_n_kw.update(n_annotation_kwargs)
    ax.text(s=f"N={n_records}", **default_n_kw)
    ax.autoscale(axis="y")
    plt.tight_layout()

    __save(fn=fn)

    return fig, ax



def plot_gender_distributions(df, cols, **kwargs):
    figsize = kwargs.get('figsize', (10, 6))
    nbins = kwargs.get('nbins', 10)
    label_mapping = kwargs.get('label_mapping', {col: col for col in cols})
    fn = kwargs.get('fn', None)
    histogram = kwargs.get('histogram', False)
    percentage = kwargs.get('percentage', True)
    title = kwargs.get('title', None)
    n_annotation_kwargs = kwargs.get('n_annotation_kwargs', {})
    
    cmap_name = kwargs.get('cmap', 'tab20')
    cmap = plt.get_cmap(cmap_name)
    colors = cmap.colors
    color_cycle = cycle(colors)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for col in cols:
        color = next(color_cycle)
        if histogram:
            ax.hist(df[col], bins=nbins, alpha=0.4, label=label_mapping[col], density=percentage, color=color)
        else:
            tmp = df[col].round(1).value_counts(normalize=percentage).sort_index()
            ax.plot(tmp.index, tmp.values, label=label_mapping[col], color=color)

    ax.legend()
    ax.set_xlabel("Gender score")
    ax.set_ylabel("Percentage (%)" if percentage else "Count")
    ax.set_title(title)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    n_records = len(df)
    default_n_kw = {"transform": ax.transAxes, "x": 0.02, "y": 0.02, "va": "bottom", "ha": "left", "fontsize": 9}
    default_n_kw.update(n_annotation_kwargs)
    ax.text(s=f"N={n_records}", **default_n_kw)
    
    __save(fn=fn)
    
    plt.show()
    plt.close

def plot_regression_performance(df, col_x, col_y, metrics, **kwargs):
    figsize = kwargs.get('figsize', (3, 3))
    title = kwargs.get('title', None)
    xlabel = kwargs.get('xlabel', "Ground-truth")
    ylabel = kwargs.get('ylabel', "Predicted")
    markersize = kwargs.get('markersize', 30)
    markeralpha = kwargs.get('markeralpha', 0.5)
    markercolor = kwargs.get('markercolor', 'tab:gray')
    fn = kwargs.get('fn', None)
    fig = kwargs.get('fig', None)
    show_yticklabels = kwargs.get('show_yticklabels', True)

    label_order = kwargs.get('label_order', None)
    color_col = kwargs.get('color_col', None)
    label_col = kwargs.get('label_col', None)

    fig, ax = plt.subplots(1, 1, figsize=figsize) if fig is None else fig

    for label in label_order if label_order is not None else (df[label_col].unique() if label_col is not None else [None]):
        df_subset = df[df[label_col] == label] if label_col is not None else df
        print(label, df_subset.shape)
        ax.scatter(x=df_subset[col_x], y=df_subset[col_y], 
                s=markersize, 
                alpha=markeralpha, 
                zorder=1, 
                edgecolor='white', 
                linewidth=0.5,
                rasterized=True,
                color=df_subset[color_col] if color_col is not None else markercolor,
                label=label if label is not None else kwargs.get('datalabel', None)
                )
        
        
    additional_data = kwargs.get('additional_data', None)
    if additional_data is not None:
        for ad in additional_data:
            ax.scatter(ad.get('x'), ad.get('y'), 
                    s=markersize, 
                    alpha=markeralpha, 
                    zorder=2, 
                    edgecolor='white', 
                    linewidth=0.5,
                    rasterized=True,
                    label=ad.get('label', None),
                    color=ad.get('color', 'red')
                    )
    

    if label_col is None:
        legend_kwargs = {'loc': 'lower left', 'bbox_to_anchor': (0.1, -0.01), 'borderpad': 0., 'frameon': False, 'fontsize': 10, 'handletextpad': 0.05, 'labelspacing': 0.05}
        legend_kwargs.update(kwargs.get('legend_kwargs', {}))
        ax.legend(**legend_kwargs)
    else:

        # Position across the top
        for i, lab in enumerate(label_order) if label_order is not None else (df[label_col].unique() if label_col is not None else [None]):
            
            if df[df[label_col] == lab].shape[0] == 0:
                continue
            
            col = df[df[label_col] == lab][color_col].iloc[0] if color_col is not None else markercolor
            
            x_pos = 0.11 + i * 0.13  # spread evenly in figure coords
            ax.annotate('●', xy=(x_pos, 1.14), xycoords='axes fraction',
                        fontsize=10, color=col, ha='center', va='center')
            ax.annotate(lab, xy=(x_pos, 1.06), xycoords='axes fraction',
                        fontsize=8, ha='center', va='center', color='0.3')

            if i == 0:
                ax.text(s="Ambiguity", x=0.5, y=1.20, transform=ax.transAxes, ha='center', va='center', fontsize=9, color='0.3')
            
        
    ax.plot([0, 1], [0, 1], '--', color='black', alpha=0.2, zorder=0)
    ymax = 0.98
    delta = 0.07
    bbox=dict(
        facecolor='white',
        edgecolor='none',
        pad=0.1,
        alpha=0.5,
    )
    ax.text(0.02, ymax, f"R2: {metrics.get('r2', None):.2f}", transform=ax.transAxes, ha='left', va='top', fontsize=10, zorder=2)
    ax.text(0.02, ymax - delta, f"MAE: {metrics.get('mae', None):.3f}", transform=ax.transAxes, ha='left', va='top', fontsize=10, zorder=2)
    ax.text(0.02, ymax - 2 * delta, f"MSE: {metrics.get('mse', None):.3f}", transform=ax.transAxes, ha='left', va='top', fontsize=10, zorder=2, bbox=bbox)
    ax.text(0.02, ymax - 3 * delta, f"RMSE: {metrics.get('rmse', None):.3f}", transform=ax.transAxes, ha='left', va='top', fontsize=10, zorder=2, bbox=bbox)
    ax.text(0.02, ymax - 4 * delta, f"Pearson $\\rho$: {metrics.get('pearsonr', None):.2f}{metrics.get('pvalue','')}", transform=ax.transAxes, ha='left', va='top', fontsize=10, zorder=2, bbox=bbox)

    
    ax.text(0.98, 0.02, f"N: {df.shape[0]}", transform=ax.transAxes, ha='right', va='bottom', fontsize=10, zorder=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # plt.tight_layout()

    if show_yticklabels is False:
        ax.set_yticklabels([])

    __save(fn=fn)
    

    return fig, ax



###############################################

def plot_py_ch_percentiles(df_py_summary, py_agg_cols, py_agg_labels, n_per_group=4, fn=None):

    def as_list(x):
        if isinstance(x, list):
            return x
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, str):
            return ast.literal_eval(x)
        return []

    # -----------------------------
    # Prepare data
    # -----------------------------

    df_plot = df_py_summary.copy()

    if 'ambiguity' not in df_plot.columns:
        df_plot['ambiguity'] = df_plot['ch_pred_q90'] - df_plot['ch_pred_q10']

    if 'ch_pred_mean' not in df_plot.columns:
        df_plot['ch_pred_mean'] = df_plot['ch_pred_all'].apply(
            lambda x: np.mean(as_list(x))
        )

    df_plot['mean_group'] = pd.cut(
        df_plot['ch_pred_mean'],
        bins=[-np.inf, 0.4, 0.6, np.inf],
        labels=['low < 0.4', 'middle 0.4-0.6', 'high > 0.6']
    )

    def sample_group(g, n=10):
        g = g.sort_values('ch_n', ascending=False)

        top = g.head(min(n // 2, len(g)))

        rest = g.drop(top.index).sample(
            n=min(n - len(top), max(len(g) - len(top), 0)),
            random_state=42
        )

        return pd.concat([top, rest])

    df_plot = (
        df_plot
        .dropna(subset=['mean_group'])
        .groupby('mean_group', group_keys=False, observed=True)
        .apply(lambda g: sample_group(g, n=n_per_group))
        .copy()
    )

    group_order = ['high > 0.6', 'middle 0.4-0.6', 'low < 0.4']

    df_plot['mean_group'] = pd.Categorical(
        df_plot['mean_group'],
        categories=group_order,
        ordered=True
    )

    df_plot = (
        df_plot
        .sort_values(['mean_group', 'ch_pred_mean'], ascending=[True, False])
        .reset_index(drop=True)
    )

    # -----------------------------
    # Color by ambiguity
    # -----------------------------

    norm = Normalize(
        vmin=df_plot['ambiguity'].min(),
        vmax=df_plot['ambiguity'].max()
    )

    cmap = plt.cm.RdYlGn_r

    # -----------------------------
    # Plot
    # -----------------------------

    n_rows = len(df_plot)
    n_cols = len(py_agg_cols)
    width = 2.5
    height = 0.2

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(width * n_cols, height * n_rows + 1.7),
        sharex=True
    )

    if n_rows == 1:
        axes = np.array([axes])

    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    for i, row in df_plot.iterrows():

        vals = np.array(as_list(row['ch_pred_all']), dtype=float)
        vals = vals[~np.isnan(vals)]

        q10 = row['ch_pred_q10']
        q90 = row['ch_pred_q90']

        dot_color = cmap(norm(row['ambiguity']))

        for j, col in enumerate(py_agg_cols):

            ax = axes[i, j]
            aggregate = row[col]

            # Hanzi-level scores
            jitter = np.random.default_rng(42 + i + j).normal(
                0,
                0.018,
                len(vals)
            )

            ax.scatter(
                vals,
                jitter,
                s=20,
                marker='x',
                color='lightgrey',
                alpha=0.7,
                zorder=1,
                clip_on=False
            )

            # 10-90 percentile interval
            # Only meaningful when there are enough Hanzi values
            if len(vals) >= 3 and q90 > q10:
                mid = (q10 + q90) / 2

                err = ax.errorbar(
                    mid,
                    0,
                    xerr=[[mid - q10], [q90 - mid]],
                    fmt='none',
                    ecolor='black',
                    elinewidth=0.9,
                    capsize=2.,
                    capthick=0.9,
                    alpha=0.7,
                    zorder=3
                )

                for cap in err[1]:
                    cap.set_clip_on(False)

                for bar in err[2]:
                    bar.set_clip_on(False)

            # Aggregate dot
            ax.scatter(
                aggregate,
                0,
                s=42,
                color=dot_color,
                edgecolor='black',
                linewidth=0.25,
                zorder=4,
                clip_on=False
            )

            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.20, 0.20)
            ax.set_yticks([])

            for spine in ['left', 'right', 'top', 'bottom']:
                ax.spines[spine].set_visible(False)

            if i < n_rows - 1:
                ax.tick_params(axis='x', bottom=False, labelbottom=False)
            else:
                ax.spines['bottom'].set_visible(True)
                ax.spines['bottom'].set_linewidth(0.6)
                ax.spines['bottom'].set_position(('outward', 6))

                ax.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
                ax.set_xticklabels(['0', '0.25', '0.50', '0.75', '1'])
                ax.tick_params(axis='x', labelsize=8, length=3, width=0.6)

            if i == 0:
                ax.set_title(
                    py_agg_labels.get(col, col),
                    fontsize=10,
                    fontweight='bold'
                )

            if j == 0:
                ax.text(
                    -0.20,
                    0.5,
                    str(row['py_givenname']),
                    transform=ax.transAxes,
                    ha='right',
                    va='center',
                    fontsize=9,
                    fontweight='semibold',
                    color='black',
                    clip_on=False
                )

                ax.text(
                    -0.19,
                    0.5,
                    f"$n_h$={int(row['ch_n'])}",
                    transform=ax.transAxes,
                    ha='left',
                    va='center',
                    fontsize=7.5,
                    color='0.45',
                    clip_on=False
                )
            
            if i == n_rows - 1:
                ax.set_xlabel(r"$\hat{y}_p$", fontsize=10)

    # colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # leave room on the right
    plt.tight_layout()
    plt.subplots_adjust(
        left=0.11,
        right=0.92,   # reserve space for colorbar
        wspace=0.10
    )

    # dedicated colorbar axis
    cax = fig.add_axes([0.93, 0.15, 0.012, 0.70])

    cbar = fig.colorbar(
        sm,
        cax=cax
    )

    cbar.set_label(
        r'Ambiguity', #: q$_{90}$ − q$_{10}$',
        fontsize=9
    )

    cbar.ax.tick_params(
        labelsize=7
    )

    if fn is not None:
        fig.savefig(
            fn,
            dpi=FIG_DPI,
            bbox_inches='tight'
        )

    plt.show()
    plt.close()





def plot_multiclass_roc(
    df,
    true_col,
    score_cols,
    pred_col=None,
    labels=("male", "neutral", "female"),
    figsize=(5, 4),
    fn = None,
):
    """
    Parameters
    ----------
    df : pandas.DataFrame
    true_col : str
        Column with true labels.
    score_cols : dict
        Mapping from label to probability/score column.
        Example:
        {
            "male": "p_male",
            "neutral": "p_neutral",
            "female": "p_female"
        }
    pred_col : str, optional
        Column with predicted class labels. Used for Cohen's kappa.
        If None, prediction is taken as the label with maximum score.
    labels : tuple/list
        Class order.
    """

    y_true = df[true_col].to_numpy()

    y_score = np.column_stack([
        df[score_cols[label]].to_numpy()
        for label in labels
    ])

    y_bin = label_binarize(y_true, classes=labels)

    if pred_col is None:
        y_pred = np.array(labels)[np.argmax(y_score, axis=1)]
    else:
        y_pred = df[pred_col].to_numpy()

    fig, ax = plt.subplots(figsize=figsize)

    aucs = []

    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        ax.plot(
            fpr,
            tpr,
            linewidth=1.6,
            label=f"{label} (AUC={roc_auc:.3f})"
        )

    macro_auc = np.mean(aucs)
    kappa = cohen_kappa_score(y_true, y_pred, labels=labels)

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.7, color="gray")

    ax.text(
        0.98, 0.55,
        f"N: {len(df)}\nMacro-AUC: {macro_auc:.3f}\nCohen's $\\kappa$: {kappa:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=13
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.legend(frameon=True, loc="lower right", fontsize=12)

    fig.tight_layout()

    if fn is not None:
        fig.savefig(fn, dpi=FIG_DPI, bbox_inches="tight")

    return fig, ax



def plot_py_disagreement(df_py_agg, pynames, labels, fn):
    # Required columns:
    # py_givenname, n_h, py_gender_mv, entropy, disagreement,
    # n_l_h_male, n_l_h_neutral, n_l_h_female

    plot_df = df_py_agg.query("py_givenname in @pynames").copy().reset_index(drop=True)

    order_map = {name: i for i, name in enumerate(pynames)}

    plot_df = plot_df.sort_values(
        by='py_givenname',
        key=lambda s: s.map(order_map)
    )

    count_cols = ["n_l_h_male", "n_l_h_neutral", "n_l_h_female"]

    props = plot_df[count_cols].div(plot_df[count_cols].sum(axis=1), axis=0)

    colors = {
        "male": "tab:blue",
        "neutral": "tab:green",
        "female": "tab:red"
    }

    min_alpha = 0.1
    plot_df["alpha"] = 1 - plot_df["entropy"].clip(0, 1) * (1 - min_alpha)

    n = len(plot_df)
    y = np.arange(n)

    fig = plt.figure(figsize=(9.5, max(3.5, 0.42 * n)))

    gs = GridSpec(
        1, 3,
        width_ratios=[0.45, 1.70, 1.95],   # was something like [0.75, 1.45, 1.65]
        left=0.005,
        right=0.995,
        top=0.86,
        bottom=0.14,
        wspace=0.05
    )

    ax_names = fig.add_subplot(gs[0, 0])
    ax_dis = fig.add_subplot(gs[0, 1], sharey=ax_names)
    ax_bar = fig.add_subplot(gs[0, 2], sharey=ax_names)

    # -----------------------
    # Column 1: names + n_h
    # -----------------------
    ax_names.set_xlim(0, 1)
    ax_names.set_ylim(-0.5, n - 0.5)
    ax_names.invert_yaxis()
    ax_names.axis("off")

    # for i, row in plot_df.iterrows():
    for i, name in enumerate(pynames):
        row = plot_df[plot_df.py_givenname == name].iloc[0]
        ax_names.text(
            0.55, i,
            row["py_givenname"],
            ha="right",
            va="center",
            fontweight="bold",
            fontsize=12
        )
        ax_names.text(
            0.58, i,
            rf"$n_h={int(row['n_h'])}$",
            ha="left",
            va="center",
            color="0.55",
            fontsize=8
        )

    # -----------------------
    # Column 2: disagreement
    # -----------------------
    ax_dis.set_xlim(-0.03, 1.00)
    # ax_dis.set_xlim(-0.06, 1.02)
    ax_dis.set_ylim(-0.5, n - 0.5)

    for i in y:
        ax_dis.hlines(
            i, 0, 1,
            color="0.83",
            linestyle="--",
            linewidth=0.9,
            zorder=0
        )

    for i, name in enumerate(pynames):
        row = plot_df[plot_df.py_givenname == name].iloc[0]
        ax_dis.scatter(
            row["disagreement"],
            i,
            s=55,
            color=colors[row["py_gender_mv"]],
            alpha=row["alpha"],
            edgecolor="black",
            linewidth=0.45,
            zorder=3
        )

    ax_dis.set_xlabel("Disagreement", fontsize=13)
    ax_dis.set_yticks([])
    ax_dis.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
    ax_dis.set_xticklabels(["0.0", "0.25", "0.50", "0.75", "1.0"])

    for spine in ["top", "right", "left"]:
        ax_dis.spines[spine].set_visible(False)

    # -----------------------
    # Column 3: stacked bars
    # -----------------------
    smooth = 0.05
    ax_dis.set_xlim(0-smooth, 1+smooth)
    ax_bar.set_xlim(0-smooth, 1+smooth)
    ax_bar.set_ylim(-0.5, n - 0.5)

    left = np.zeros(n)

    for label, col in zip(labels, count_cols):
        ax_bar.barh(
            y,
            props[col],
            left=left,
            height=0.65,
            color=colors[label],
            alpha=0.88,
            edgecolor="none",
            label=label
        )
        left += props[col].to_numpy()

    ax_bar.set_xlabel("Hanzi-label composition", fontsize=13)
    ax_bar.set_yticks([])
    ax_bar.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
    ax_bar.set_xticklabels(["0.0", "0.25", "0.50", "0.75", "1.0"])

    for spine in ["top", "right", "left"]:
        ax_bar.spines[spine].set_visible(False)

    # Shorter tick labels
    ax_dis.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax_bar.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    # -----------------------
    # Shared legend
    # -----------------------
    handles = [
        Patch(facecolor=colors["male"], label="male"),
        Patch(facecolor=colors["neutral"], label="neutral"),
        Patch(facecolor=colors["female"], label="female"),
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.95),
        fontsize=12
    )

    fig.subplots_adjust(left=0.005)

    if fn is not None:
        fig.savefig(fn, dpi=FIG_DPI, bbox_inches="tight")

    plt.show()
    plt.close()




def make_y_positions(y_order, gap_after=None, gap_size=0.6):
    gap_after = set(gap_after or [])
    positions = []
    pos = 0.0

    for method in y_order:
        positions.append(pos)
        pos += 1.0
        if method in gap_after:
            pos += gap_size

    return positions


def plot_gender_facets(
    df,
    method_cols,
    y_name_map=None,
    y_order=None,
    col_name_map=None,
    col_order=("female", "neutral", "male", "unknown"),
    method_color_map=None,
    reference_method="hz_gender",
    gap_after=None,
    gap_size=0.6,
    show_annotations=True,
    annotation_mode="percent",   # percent | count | both
    annotation_fontsize=9,
    annotation_color="0.55",
    figsize=(13, 4),
    fn=None,
):
    y_name_map = y_name_map or {}
    col_name_map = col_name_map or {}
    method_color_map = method_color_map or {}
    n_total = len(df)

    if y_order is None:
        y_order = method_cols

    long = (
        df[method_cols]
        .melt(var_name="method", value_name="gender")
        .dropna()
    )

    y_positions = make_y_positions(
        y_order,
        gap_after=gap_after,
        gap_size=gap_size,
    )

    y_labels = [y_name_map.get(m, m) for m in y_order]

    fig, axes = plt.subplots(
        1,
        len(col_order),
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    if len(col_order) == 1:
        axes = [axes]

    for ax, gender in zip(axes, col_order):
        counts = (
            long[long["gender"].eq(gender)]["method"]
            .value_counts()
            .reindex(y_order, fill_value=0)
        )

        colors = [
            method_color_map.get(method, "tab:blue")
            for method in y_order
        ]

        ax.barh(
            y_positions,
            counts.values,
            height=0.72,
            color=colors,
        )

        if reference_method in counts.index:
            ax.axvline(
                counts.loc[reference_method],
                linestyle="--",
                linewidth=0.9,
                color="0.35",
                alpha=0.8,
                zorder=0,
            )

        if show_annotations:
            
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()

            ref_count = (
                counts.loc[reference_method]
                if reference_method in counts.index
                else None
            )

            x_offset = max(counts.max() * 0.012, 4)

            for y, count in zip(y_positions, counts.values):
                if count == 0:
                    continue

                pct = 100 * count / n_total

                if annotation_mode == "percent":
                    label = f"{pct:.1f}%"

                elif annotation_mode == "count":
                    label = f"{count:,}"

                elif annotation_mode == "both":
                    label = f"{count:,} ({pct:.1f}%)"

                x_text = count + x_offset
                
                txt = ax.text(
                    x_text,
                    y,
                    label,
                    va="center",
                    ha="left",
                    fontsize=annotation_fontsize,
                    color=annotation_color,
                )

                if ref_count is not None:
                    fig.canvas.draw()
                    bbox = txt.get_window_extent(renderer=renderer)

                    ref_x_display = ax.transData.transform((ref_count, y))[0]

                    overlaps_ref_line = bbox.x0 <= ref_x_display <= bbox.x1

                    if overlaps_ref_line:
                        txt.remove()

                        ref_x_data = ax.transData.inverted().transform(
                            (ref_x_display + 5, 0)
                        )[0]

                        ax.text(
                            ref_x_data,
                            y,
                            label,
                            va="center",
                            ha="left",
                            fontsize=annotation_fontsize,
                            color=annotation_color,
                        )

        ax.set_title(col_name_map.get(gender, gender))
        ax.set_xlabel("Count")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

        ax.set_yticks(y_positions)

    axes[0].set_yticklabels(y_labels)
    axes[0].set_ylabel("Gender classifier")

    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)

    axes[0].invert_yaxis()

    if fn:
        plt.savefig(fn, dpi=FIG_DPI, bbox_inches="tight")

    return fig, axes


def plot_gender_facets_percent(
    df,
    method_cols,
    y_name_map=None,
    y_order=None,
    col_name_map=None,
    col_order=("unknown", "male", "neutral", "female"),
    method_color_map=None,
    reference_method="hz_gender",
    gap_after=None,
    gap_size=0.6,
    figsize=(13, 4),
    fn=None,
):
    y_name_map = y_name_map or {}
    col_name_map = col_name_map or {}
    method_color_map = method_color_map or {}

    if y_order is None:
        y_order = method_cols

    long = (
        df[method_cols]
        .melt(var_name="method", value_name="gender")
        .dropna()
    )

    denominators = (
        df[method_cols]
        .notna()
        .sum()
        .reindex(y_order)
    )

    y_positions = make_y_positions(
        y_order,
        gap_after=gap_after,
        gap_size=gap_size,
    )

    y_labels = [y_name_map.get(m, m) for m in y_order]

    fig, axes = plt.subplots(
        1,
        len(col_order),
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    if len(col_order) == 1:
        axes = [axes]

    for ax, gender in zip(axes, col_order):
        counts = (
            long[long["gender"].eq(gender)]["method"]
            .value_counts()
            .reindex(y_order, fill_value=0)
        )

        percents = 100 * counts / denominators

        colors = [
            method_color_map.get(method, "tab:blue")
            for method in y_order
        ]

        ax.barh(
            y_positions,
            percents.values,
            height=0.72,
            color=colors,
        )

        if reference_method in counts.index:
            ref_pct = 100 * counts.loc[reference_method] / denominators.loc[reference_method]

            ax.axvline(
                ref_pct,
                linestyle="--",
                linewidth=0.9,
                color="0.35",
                alpha=0.8,
                zorder=0,
            )

        for y, pct, n in zip(y_positions, percents.values, counts.values):
            if n > 0:
                ax.text(
                    pct + 1.2,
                    y,
                    f"(n={n:,})",
                    va="center",
                    ha="left",
                    fontsize=9,
                    color="0.55",
                )

        ax.set_title(col_name_map.get(gender, gender))
        ax.set_xlabel("Percentage of names")
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))

        ax.set_xlim(0, 100)
        ax.grid(False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_yticks(y_positions)

    axes[0].set_yticklabels(y_labels)
    axes[0].set_ylabel("Gender classifier")

    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)

    axes[0].invert_yaxis()

    if fn:
        plt.savefig(fn, dpi=FIG_DPI, bbox_inches="tight")

    return fig, axes