import pandas as pd
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor


from libs import ios
from libs import tools_comparison
from libs import setup

METRICS_SORT_ASC = {
    'test_rmse': True,
    'test_r2': True,
    'test_auc': False,
}

METRIC = {'regression': 'test_rmse', 'classification': 'test_auc'}

METRICS_CLASS = ['test_auc']

SUB_FOLDERS = {'regression': 'rs*/regression/_<VAR_X>/*', 
               'classification': 'rs*/classification/_<VAR_X>/cut_*/*'}

class Evaluation(object):
    def __init__(self, comparison_dir: str, is_classifier: bool = False):
        self.is_classifier = is_classifier
        self.df_results = None
        self.df_best_per_rs = None
        subpath = SUB_FOLDERS[('classification' if self.is_classifier else 'regression')]
        self.comparison_dir = ios.path_join(comparison_dir, subpath)
        
        if ios.is_empty(self.comparison_dir.replace('<VAR_X>', '*')):
            raise ValueError(f'Comparison directory {self.comparison_dir} is empty')

    def read_eval_results(self, ignore_rs: list[int] = None):
        df_results_mean = ios.load_json_files_to_dataframe(self.comparison_dir.replace('<VAR_X>', 'mean'), include_fn=True)
        df_results_median = ios.load_json_files_to_dataframe(self.comparison_dir.replace('<VAR_X>', 'median'), include_fn=True)
        df_results_mode = ios.load_json_files_to_dataframe(self.comparison_dir.replace('<VAR_X>', 'mode'), include_fn=True)
        self.df_results = pd.concat([df_results_mean, df_results_median, df_results_mode], ignore_index=True)

        # A few mods:
        self.df_results.loc[:,'rs'] = self.df_results.fn.apply(lambda v: int(v.split('/rs')[-1].split('/')[0]))
        if ignore_rs is not None:
            self.df_results = self.df_results.query("rs not in @ignore_rs").copy()

        return self.df_results

    def _validate_metric(self, metric: str):
        if metric not in METRICS_SORT_ASC.keys():
            raise ValueError(f'Metric {metric} not supported')

        if metric in METRICS_CLASS and not self.is_classifier:
            raise ValueError(f'Metric {metric} is not supported for regression')
        elif metric not in METRICS_CLASS and self.is_classifier:
            raise ValueError(f'Metric {metric} is not supported for classification')

    def identify_best_inferences(self, metric: str, ascending: bool = None):

        self._validate_metric(metric)

        # regression: ascending True (best: min) - classification: ascending False (best: max)

        # best per rs
        ascending = ascending if ascending is not None else METRICS_SORT_ASC[metric]
        df_group = self.df_results.groupby('rs')[metric]
        idx = df_group.idxmin() if ascending else df_group.idxmax()
        self.df_results.loc[:, 'best_per_rs'] = False
        self.df_results.loc[idx, 'best_per_rs'] = True

        # best overall
        df_group = self.df_results[metric]
        idx = df_group.idxmin() if ascending else df_group.idxmax()
        self.df_results.loc[:, 'best_overall'] = False
        self.df_results.loc[idx, 'best_overall'] = True

        # sort
        self.df_results = self.df_results.sort_values(metric, ascending=ascending)


    def _load_model(self, model_fn: str):
        model = CatBoostClassifier() if self.is_classifier else CatBoostRegressor()
        if model is None:
            raise ValueError('model_fn must contain either "class" or "reg"')
        model.load_model(model_fn, format="cbm")
        ios.printf(f'Model loaded: {model_fn}')
        return model

    def _prepare_features(self, model: CatBoostClassifier | CatBoostRegressor, df_ch_names: pd.DataFrame, resources_dir: str, cache_dir: str):
        # prepare features
        col_features = model.feature_names_
        column_name = 'givenname' if 'givenname' in df_ch_names.columns else 'character' if 'character' in df_ch_names.columns else None
        if column_name is None:
            raise ValueError('Column "givenname" or "character" not found in dataframe')
        
        # prepares inferred genders by the three toolsL W (infer on the fly if not present), N, R. For the latter two, they must be present in the dataframe.
        # it requires: gender_w, NG, ngender_n
        df_ch_names = tools_comparison.prepare_features(df_ch_names, column_name, resources_dir, cache_dir)
        ios.printf(f'Features: {col_features} ({len(col_features)})')
        
        missing_features = set(col_features) - set(df_ch_names.columns)
        if missing_features:
            raise ValueError(f'Missing features in dataframe: {missing_features}')
        
        # preprocessing
        data = df_ch_names[['givenname']+col_features].copy()

        # check single_char
        data.loc[:,'single_char'] = data.givenname.apply(lambda v: len(v) == 1)

        return data, col_features

        
    def predict_gender(self, model_fn: str, df_ch_names: pd.DataFrame, resources_dir: str, cache_dir: str):
        # load model
        model = self._load_model(model_fn)

        # prepare features
        data, col_features = self._prepare_features(model, df_ch_names, resources_dir, cache_dir)

        # predict
        if self.is_classifier:
            data.loc[:,'pred_code'] = data.progress_apply(lambda row: model.predict(row[col_features])[0], axis=1)
            data.loc[:,'pred_label'] = data.apply(lambda row: setup.LABELS[row.pred_code], axis=1)
        else:
            data.loc[:,'pred'] = data.progress_apply(lambda row: model.predict(row[col_features]), axis=1)

        return data


    def predict_gender_per_rs(self, df_ch_names: pd.DataFrame, fn_output_prefix: str, output_root_dir: str, resources_dir: str, cache_dir: str, best_per_rs: bool = False):
        if best_per_rs: 
            df_per_rs = self.df_results.query("best_per_rs == True")
            print(f'Using best per rs: {df_per_rs.shape}')
        else:
            df_per_rs = self.df_results
            print(f'Using all results: {df_per_rs.shape}')

        for _, row in df_per_rs.iterrows():
            fn_prefix = row.fn.replace('.json','')
            model_fn = f"{fn_prefix}.cbm"

            # predict
            data = self.predict_gender(model_fn=model_fn, df_ch_names=df_ch_names, resources_dir=resources_dir, cache_dir=cache_dir)
        
            # save
            postfix = f"{'class' if self.is_classifier else 'reg'}_{ios.os.path.basename(fn_prefix)}"
            fn_output = f"{fn_output_prefix}_{postfix}.csv"
            output_dir = ios.path_join(output_root_dir, f"rs{row.rs}")
            ios.validate_path(output_dir)
            fn = ios.path_join(output_dir, fn_output)
            ios.to_csv(data, fn)
            ios.printf(f'Saved: {fn}')
