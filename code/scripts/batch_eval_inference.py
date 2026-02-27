# export PYTHONPATH="${PYTHONPATH}:../"


# read json files with predictions from training
# concatenate
# identify best per random seed
# identify best overall
# for each model and random seed, infer gender for all names
# store predictions for best model and random seed, and overall evaluation metrics

# From code/scripts
# export PYTHONPATH=$PYTHONPATH:../
# python batch_eval_inference.py --comparison_dir ../../results/v3/gender/predictions_ch/comparison --input_fn ../../results/v3/gender/predictions_ch/CNKI_uniqueChineseAllGivenNames_gender.csv --metric test_auc --classification --output_dir ../../results/v3/gender/predictions_ch/best_models --resources_dir ../../resources --cache_dir ../../cache
# python batch_eval_inference.py --comparison_dir ../../results/v3/gender/predictions_ch/comparison --input_fn ../../results/v3/gender/predictions_ch/CNKI_uniqueChineseAllGivenNames_gender.csv --metric test_rmse --output_dir ../../results/v3/gender/predictions_ch/best_models --resources_dir ../../resources --cache_dir ../../cache


import argparse
import tqdm

tqdm.tqdm.pandas()

from libs import ios
from libs import constants
from libs.inference.evaluation import Evaluation

def _add_ntest_ntrain(df, only_best=False):
    if only_best:
        df = df.query("best_per_rs == True")

    loaded_df = {}
    for id, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="Adding ntest and ntrain"):
        fn = ios.path_join(ios.os.path.dirname(row.fn),  constants.FN_GT_TRAN_TEST_SAMPLE)

        if fn in loaded_df:
            tmp = loaded_df[fn]
        else:
            tmp = ios.load_csv(fn)
            loaded_df[fn] = tmp
            
        df.loc[id,'ntest'] = int(tmp.query("ml_sample == 'test'").shape[0])
        df.loc[id,'ntrain'] = int(tmp.query("ml_sample == 'train'").shape[0])

    return df

def run(comparison_dir, is_classifier, metric, input_fn, output_dir, resources_dir, cache_dir):

    # read all evaluation results
    eva = Evaluation(comparison_dir=comparison_dir, is_classifier=is_classifier)
    eva.read_eval_results(ignore_rs=[12345])
    ios.printf(f'Evaluation results: {eva.df_results.shape}')

    # identify best per random seed
    eva.identify_best_inferences(metric=metric)
    eva.df_results = _add_ntest_ntrain(eva.df_results)
    cols = ['features_id', 'y', 'features', 'nfeatures', 'train_eval', metric, 'best_per_rs', 'best_overall', 'rs', 'ntest', 'ntrain']
    cols = ['strategy_id'] + cols if is_classifier else cols
    ios.printf('Best per rs:')
    print(eva.df_results.query("best_per_rs == True")[cols])
    ios.printf('Best overall:')
    print(eva.df_results.query("best_overall == True")[cols])

    # save evaluation
    fn_eval = f"eval_results_{'classification' if is_classifier else 'regression'}.csv"
    fn = ios.path_join(comparison_dir, fn_eval)
    ios.to_csv(eva.df_results, fn)
    ios.printf(f'Evaluation results saved: {fn}')

    # Load chinese names to infer gender
    df_ch_names = ios.load_csv(input_fn)
    ios.printf(f'Chinese names loaded: {input_fn} ({df_ch_names.shape})')

    # For each best model per random seed, infer gender for all names
    fn_output_prefix = f"{ios.os.path.basename(input_fn).replace('.csv','')}"
    eva.predict_gender_per_rs(df_ch_names, fn_output_prefix, output_dir, resources_dir, cache_dir, best_per_rs=True)
    
    # save summary eval
    fn_eval = fn_eval.replace('eval_','eval_best_')
    fn = ios.path_join(output_dir, fn_eval)
    ios.to_csv(eva.df_results.query("best_per_rs == True"), fn)
    ios.printf(f'Evaluation best results saved: {fn}')


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Training to predict')
    parser.add_argument('--comparison_dir', type=str, help='Directory with comparison results', required=True)
    parser.add_argument('--classification', action='store_true', help='Is classification', required=False)
    parser.add_argument('--metric', type=str, help='Metric to use to identify the best model', required=True, choices=['test_rmse', 'test_auc', 'test_r2'])
    parser.add_argument('--input_fn', type=str, help='Input file with chinese names or characters', required=True)
    parser.add_argument('--output_dir', type=str, help='Output directory to save the best model results', required=True)
    parser.add_argument('--resources_dir', type=str, help='Resources directory where http metadata is stored (e.g., ../../resources)', required=True)
    parser.add_argument('--cache_dir', type=str, help='Cache directory where to store queries to W tool (e.g., ../../cache)', required=True)
    args = parser.parse_args()

    # Print arguments (k,v)
    print('Arguments:')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    print()

    # do
    run(args.comparison_dir, args.classification, args.metric, args.input_fn, args.output_dir, args.resources_dir, args.cache_dir)

