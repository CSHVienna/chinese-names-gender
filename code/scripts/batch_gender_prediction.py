# From code/scripts (path = code/):
# export PYTHONPATH=$PYTHONPATH:..
# python batch_gender_prediction.py --input data/processed/df.csv --output_dir results/regression --model_fn model.cbm

import argparse
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor

from libs import ios
from libs import setup
from libs import tools_comparison


def run(input_fn, model_fn, output_dir, resources_dir, cache_dir):
    ###################
    # Load model
    ###################
    is_classifier =  'classification' in model_fn 
    is_regressor =  'regression' in model_fn 
    model = CatBoostClassifier() if is_classifier else CatBoostRegressor() if is_regressor else None
    if model is None:
        raise ValueError('model_fn must contain either "class" or "reg"')
    model.load_model(model_fn, format="cbm")
    ios.printf(f'Model loaded: {model_fn}')

    ###################
    # Load chinese names
    ###################
    df_ch_names = ios.load_csv(input_fn)
    ios.printf(f'Chinese names loaded: {input_fn} ({df_ch_names.shape})')
    print(df_ch_names.head(2))

    # test 
    # df_ch_names = df_ch_names.head(3) # TODO:delete

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
    
    ###################
    # Preprocessing
    ###################
    data = df_ch_names[['givenname']+col_features].copy()
    print("=== BEFORE PRE-PROCESSING 'y'")
    print(data.head(2))

    # check single_char
    data.loc[:,'single_char'] = data.givenname.apply(lambda v: len(v) == 1)

    ###################
    # Predict
    ###################
    # predict and decode (from number to label)
    if is_classifier:
        ## @deprecated: used when the model was not working for single-char (and w was used though it was always male)
        # unknown_id = len(setup.LABELS)
        # data.loc[:,'pred_code'] = data.apply(lambda row: unknown_id if ios.pd.isna(row.ngender_r) or ios.pd.isna(row.ngender_w) else model.predict(row[col_features])[0], axis=1)
        # data.loc[:,'pred_label'] = data.apply(lambda row: categorical.UNKNOWN if row.pred_code == unknown_id else setup.LABELS[row.pred_code], axis=1)
        # <<

        data.loc[:,'pred_code'] = data.apply(lambda row: model.predict(row[col_features])[0], axis=1)
        data.loc[:,'pred_label'] = data.apply(lambda row: setup.LABELS[row.pred_code], axis=1)
    else:
        ## @deprecated: used when the model was not working for single-char (and w was used though it was always male)
        # data.loc[:,'pred'] = data.apply(lambda row: ios.pd.NA if ios.pd.isna(row.ngender_r) or ios.pd.isna(row.ngender_w) else model.predict(row[col_features]), axis=1)
        # <<

        data.loc[:,'pred'] = data.apply(lambda row: model.predict(row[col_features]), axis=1)
    
    print("=== AFTER AFTER-PROCESSING 'y'")
    print(data.sample(2))
    

    # Save
    postfix = f"{'class' if is_classifier else 'reg'}_{ios.os.path.basename(model_fn).split('.')[0]}"
    fname = f"{ios.os.path.basename(input_fn).replace('.csv','')}_{postfix}.csv"
    fn = ios.path_join(output_dir, fname)
    ios.to_csv(data, fn)
    ios.printf(f'Saved: {fn}')

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Training to predict')
    parser.add_argument('--input_fn', type=str, help='Input file with chinese names or characters', required=True)
    parser.add_argument('--output_dir', type=str, help='Output directory to save the results', required=True)
    parser.add_argument('--resources_dir', type=str, help='Resources directory where http metadata is stored (e.g., ../../resources)', required=True)
    parser.add_argument('--cache_dir', type=str, help='Cache directory where to store queries to W tool (e.g., ../../cache)', required=True)
    parser.add_argument('--model_fn', type=str, help='Model to use to predict gender (*.cbm)', required=True)
    args = parser.parse_args()

    # Print arguments (k,v)
    print('Arguments:')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    # do
    run(args.input_fn, args.model_fn, args.output_dir, args.resources_dir, args.cache_dir)
