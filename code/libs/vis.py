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
from pdf2image import convert_from_path
from sklearn.metrics import r2_score
from matplotlib import rc

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


def plot_scatter(data, x, y, title='', xlabel='', ylabel='', 
                 hue=None, hue_order=None, style=None, style_order=None, figsize=(10, 6), 
                 markersize=6,
                 palette=None, corr=False, fit=False, rmse=False, r2=False, diagonal=False, **kwargs):
    
    plt.figure(figsize=figsize)
    fg = sns.scatterplot(data=data, x=x, y=y, hue=hue, 
                         s=markersize,
                         palette=palette, hue_order=hue_order, style=style, style_order=style_order)

    yt = 0.01
    ys = 0.09

    
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
        plt.text(0.98, yt, r"Pearson $\rho$: " + f"{corr:.2f}", transform=plt.gca().transAxes, ha='right', va='bottom')
        yt += ys

    # root mean squared error
    if rmse:
        v = root_mean_squared_error(df[x], df[y])
        plt.text(0.98, yt, f'RMSE: {v:.2f}', transform=plt.gca().transAxes, ha='right', va='bottom')
        yt += ys

    # R2
    if r2: 
        v = r2_score(df[x], df[y])
        plt.text(0.98, yt, f'$R^2$: {v:.2f}', transform=plt.gca().transAxes, ha='right', va='bottom')
        yt += ys

    # diagonal
    if diagonal:
        plt.plot([0,1], [0,1], ls='--', color='grey', lw=1)
    
    if hue and style:
        legend_handles, legend_labels = fg.get_legend_handles_labels()
        plt.legend(legend_handles, legend_labels, ncol=2, 
                   loc="lower center",           # anchor point relative to bbox
                    bbox_to_anchor=(0.45, 1.02),   # (x, y) = center top, slightly above
                    borderaxespad=0,              # remove extra padding
                    )

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

def replot_importance(f_name, regressor=False, **kwargs):
    from catboost import CatBoostRegressor
    from catboost import CatBoostClassifier
    import setup

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
    #height = 2 + (len(model.feature_names_) - 0) * (12 - 2) / (len(setup.FEATURE_NAMES) - 0)
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


def replot_scatter(f_name, regressor=True):
    from catboost import CatBoostRegressor
    from catboost import CatBoostClassifier
    import setup
    import ios
    import categorical

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
                    x=y_var, y='pred',  
                    xlabel='True values', ylabel='Predictions',
                    markersize=50,
                    title='',  
                    hue='single_char', 
                    hue_order=[False, True], 
                    style = unknown_feature_col,
                    style_order = [False, True],
                    figsize=(4, 3), 
                    palette=None, 
                    corr=True, fit=False, rmse=True, r2=True, diagonal=True,
                    fn=fn
                    )

def replot_shap(f_name, regressor=True, **kwargs):
    from catboost import CatBoostRegressor
    from catboost import CatBoostClassifier
    import shap
    import setup
    import ios

    # Loading data
    fn_sample = ios.path_join(ios.get_path(f_name), "ground_truth_MLsampled.csv")
    data = ios.load_csv(fn_sample)
    
    # model
    # Loading model
    fn_model = f"{f_name}.cbm"
    model = CatBoostRegressor() if regressor else CatBoostClassifier()
    model.load_model(fn_model, format="cbm")
    col_features = model.feature_names_
    
    # SHAP
    X_test = data.query("ml_sample == 'test'")[col_features]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # figure
    width = kwargs.pop('figure_width', 4)
    height = len(col_features) * 0.3 + 1.
    height = kwargs.pop('figure_height', height)
    figsize = (width, height)


    # plot_shap_beeswarm(shap_values.data, shap_values, model.feature_names_, xlabel='SHAP value', figsize=figsize)

    shap.summary_plot(shap_values=shap_values, features=X_test, feature_names=col_features, 
                      plot_size=figsize,
                      show=False)
    
    plt.gca().set_xlabel('SHAP value')
    fn = f"{f_name}_shap_enhanced.pdf"
    
    __save(fn=fn)
    __close()