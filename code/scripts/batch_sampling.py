
###############################################################################################
# DEPENDENCIES
###############################################################################################
import os
import time
import argparse

import pandas as pd

from libs import ios
from libs.tools_comparison import ToolsComparison
from libs import setup
from libs import tools_comparison
from libs.inference import categorical
from libs import constants

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

def validation(test_fct, strategy_id=None, gt_agg=None):
    if test_fct <= 0 and test_fct >= 1:
        raise ValueError("test_fct must be between 0 and 1")
    
    if strategy_id is None and gt_agg is None:
        raise ValueError("One must be passed: either strategy_id or gt_agg")
    
    if gt_agg in setup.GT_AGGS and strategy_id in setup.STRATEGY.keys():
        raise ValueError("Ambiguity: Only one must be passed; either strategy_id or gt_agg")

    if gt_agg is None and strategy_id not in setup.STRATEGY.keys():
        raise ValueError(f'Invalid strategy_id: {strategy_id} (1-{len(setup.STRATEGY)})')
    
    if strategy_id is None and gt_agg not in setup.GT_AGGS:
        raise ValueError(f'Invalid strategy_id: {strategy_id} (1-{len(setup.STRATEGY)})')
    
###############################################################################################
# RUN
###############################################################################################
def run(strategy_id, gt_agg, test_fct, remove_unnecesary_names, random_seed, results_dir):
    
    # Validate Type 
    validation(test_fct, strategy_id, gt_agg)

    # Reconstruct output dir based on type
    output_dir = tools_comparison.get_path_comparison(results_dir, random_seed, strategy_id=strategy_id, gt_agg=gt_agg)
    ios.validate_path(output_dir)

    # Load GT
    tc = ToolsComparison(dir_tool_w=DIR_TOOL_W, 
                         dir_tool_r=DIR_TOOL_R, 
                         dir_tool_n=DIR_TOOL_N, 
                         dir_gt=DIR_GT, 
                         dir_names=DIR_NAMES, 
                         remove_unnecesary_names=remove_unnecesary_names)

    # If exists, load:
    tc.load_tools_inference(DIR_PRED, tools_comparison.const.FN_CH_GIVEN_NAMES_GENDER)
    if tc.df_ch_genders_gt is None:
        raise ValueError("GT does not exist.")

    # start
    data = tc.df_ch_genders_gt.copy()
    
    var_y = f"gender_{gt_agg}"
    if strategy_id:
        # If it is classification, labels mus be reconstructed
        cuts = setup.STRATEGY[strategy_id]['cuts']
        thresholds = [(-1,cuts[0]),(cuts[0],cuts[1]),(cuts[1],2)]

        data.loc[:,var_y] = categorical._convert_numeric_to_categorical_by_thresholds(data[gt_agg], thresholds, setup.LABELS)
        data.loc[:,var_y] = pd.Categorical(data[var_y], categories=setup.LABELS, ordered=True)
        data.drop(columns=setup.GT_AGGS, inplace=True)
    else:
        data.loc[:,var_y] = data.loc[:,gt_agg].round(1) # toy column to apply stratified sampling

    # Stratified sampling: test_fct% from each group
    stratified_sample = data.groupby(['single_char',var_y], group_keys=False).apply(
        lambda x: x.sample(frac=test_fct, random_state=random_seed), include_groups=False
    )

    if strategy_id is None:
        data.drop(columns = var_y, inplace=True)
        var_y =  f"n{var_y}"
        data.rename(columns={gt_agg:var_y}, inplace=True)

    ##### Saving Results #####
    # CSV
    data.loc[:,'ml_sample'] = 'train'
    data.loc[stratified_sample.index, 'ml_sample'] = 'test'    
    ios.printf(f"[INFO] data sample: {data.ml_sample.value_counts()}")
    
    # clean-up columns
    cols = ['givenname', 
            var_y,
            'single_char', 
            'ngender_r', 'ngender_n',
            'gender_n', 'gender_w', 
            'gender_nw',
            'mean_ngender_rnw', 
            'mean_ngender_rn', 
            'mean_ngender_rw',
            'mean_ngender_nw',
            'ml_sample'
            ]
    data = data[cols]
    print(data.head(2))

    # Storage
    fn = os.path.join(output_dir, constants.FN_GT_TRAN_TEST_SAMPLE)
    ios.to_csv(data, fn, index=False)
    ios.printf(f"[INFO] sample saved in {fn}")


###############################################################################################
# MAIN
###############################################################################################

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Sampling data to traing ML models')

    # Add arguments
    parser.add_argument('--strategy_id', help='Strategy id for thresholding for classification', default=None, type=int, required=False, choices=setup.STRATEGY.keys())
    parser.add_argument('--gt_agg', help='Dependent variable (y) for regression', type=str, default=None, required=False, choices=setup.GT_AGGS)
    parser.add_argument('--test_fct', help='Fraction size of test set', type=float, required=True, default=0.2)
    parser.add_argument('--seed', help='Random seed to use and to differentiate each set of runs or epoch ids', type=int, required=True, default=setup.RANDOM_STATE)
    parser.add_argument('--remove_unnecesary_names', help='True (130 given names in GT) or False (166 given names in GT)', action="store_true", default=False)
    parser.add_argument('--output_dir', help='../../results/v3/gender/predictions/comparison/', type=str, required=True)

    # Parse the arguments
    args = parser.parse_args()
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))

    
    start_time = time.time()
    run(args.strategy_id, args.gt_agg, args.test_fct, args.remove_unnecesary_names, args.seed, args.output_dir)
    end_time = time.time()
    print('Execution time: {:.2f} seconds'.format(end_time - start_time))