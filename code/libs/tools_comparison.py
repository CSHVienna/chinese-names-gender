import pandas as pd
import numpy as np
from itertools import product
from itertools import combinations
from collections import Counter

import ios
import constants as const
import categorical
from categorical import _mode

####################################################################################################
# Functions
####################################################################################################

def scale_minus1_to_1_to_0_to_1(x):
    return None if x in const.NONE else (x + 1) / 2

def scale_1_to_5_to_0_to_1(x):
    return None if x in const.NONE else (x - 1) / 4


####################################################################################################
# Main
####################################################################################################

class ToolsComparison(object):
    '''
    -----
    Chinsese Tool GenderGuesser: male, female
    -----
    ChineseNames R package: https://psychbruce.github.io/ChineseNames/
    NG: name gender (difference in proportions of a character used by male vs. female)
    –1~1
    NG = (Nmale – Nfemale) / (Nmale + Nfemale)
    NG ranges from –1 (completely feminine; 100% used by female) through 0 (gender-neutral; half by female and half by male) to 1 (completely masculine; 100% used by male).
    ------
    Namsor: https://namsor.app/api-documentation/#genderize-chinese-name-batch
    - genderScale: Gender scale ranging from -1 (male) to +1 (female).
    - score: Higher implies a more reliable result, score is not normalized.
    - probabilityCalibrated: Higher implies a more reliable result, ranges from 0 to 1.
    ------
    Groundtruth
    Scale from 1 to 5: 1.Definitely male   2.More likely male   3.Neutral  4.More likely female  5.Definitely female
'''
    def __init__(self, dir_tool_w, dir_tool_r, dir_tool_n, dir_gt, dir_names):
        self.dir_tool_w = dir_tool_w
        self.dir_tool_r = dir_tool_r
        self.dir_tool_n = dir_tool_n
        self.dir_gt = dir_gt
        self.dir_names = dir_names
        
        # unique chinese given names 1 char
        self.fn_ch_gender_w_1ch = ios.path_join(self.dir_tool_w, const.FN_CH_GIVEN_NAMES_GENDER_1CH)
        self.fn_ch_gender_r_1ch = ios.path_join(self.dir_tool_r, const.FN_CH_GIVEN_NAMES_GENDER_1CH)
        self.fn_ch_gender_n_1ch = ios.path_join(self.dir_tool_n, const.FN_CH_GIVEN_NAMES_GENDER_1CH)

        # unique chinese given names 2 chars
        self.fn_ch_gender_w_2ch = ios.path_join(self.dir_tool_w, const.FN_CH_GIVEN_NAMES_GENDER_2CH)
        self.fn_ch_gender_r_2ch = ios.path_join(self.dir_tool_r, const.FN_CH_GIVEN_NAMES_GENDER_2CH)
        self.fn_ch_gender_n_2ch = ios.path_join(self.dir_tool_n, const.FN_CH_GIVEN_NAMES_GENDER_2CH)
        
        # all valid given names (two and single char)
        self.fn_givennames_1ch = ios.path_join(self.dir_names, const.FN_GIVEN_NAMES_1CH)    
        self.fn_givennames_2ch = ios.path_join(self.dir_names, const.FN_GIVEN_NAMES_2CH)
    
        # annotations (ground truth)
        self.fn_gt = ios.path_join(self.dir_gt, const.FN_ANNOTATIONS)
                                   
        self.df_ch_gender_w = None
        self.df_ch_gender_r = None
        self.df_ch_gender_n = None
        self.df_ch_genders = None
        self.df_givennames_1ch = None
        self.df_givennames_2ch = None
        self.df_gt = None
        self.gt_cols = None

    def load_data(self):
        # 1. all unique given names
        tmp_1ch = ios.load_csv(self.fn_givennames_1ch)[['character']].sort_values('character').rename(columns={'character':'givenname'})
        tmp_1ch.loc[:, 'single_char'] = True
        tmp_2ch = ios.load_csv(self.fn_givennames_2ch)[['givenname']].sort_values('givenname')
        tmp_2ch.loc[:, 'single_char'] = False
        tmp_2ch = tmp_2ch.query("givenname.str.len() > 1", engine='python').copy() # remove 1-characters
        self.df_givennames = pd.concat([tmp_1ch, tmp_2ch], ignore_index=True)

        # gender from each tool for all unique given names 
        def __load_tool_data(fn1, fn2, cols=None):
            tmp_1ch = tmp_2ch = None

            if fn1:
                tmp_1ch = ios.load_csv(fn1).sort_values('character').rename(columns={'character':'givenname'})
                if cols:
                    tmp_1ch = tmp_1ch[cols]
                tmp_1ch.loc[:, 'single_char'] = True
            if fn2:
                tmp_2ch = ios.load_csv(fn2).sort_values('givenname')
                if cols:
                    tmp_2ch = tmp_2ch[cols]
                # remove 1-characters
                tmp_2ch = tmp_2ch.query("givenname.str.len() > 1", engine='python').copy()
                tmp_2ch.loc[:, 'single_char'] = False

            if tmp_1ch is None and tmp_2ch is None:
                raise ValueError('Both dataframes are None')

            if tmp_1ch is not None and tmp_2ch is not None:
                df = pd.concat([tmp_1ch, tmp_2ch], ignore_index=True)
                df.sort_values(['single_char','givenname'], ascending=[False,True], inplace=True)
                # df.drop_duplicates(subset=['givenname'],  inplace=True, keep='first') # @deprecated
                return df
            
            if tmp_1ch:
                return tmp_1ch
            
            return tmp_2ch
        
        self.df_ch_gender_w = __load_tool_data(fn1 = None, fn2 = self.fn_ch_gender_w_2ch, cols = None) # W: chinese-tools
        self.df_ch_gender_n = __load_tool_data(fn1 = self.fn_ch_gender_n_1ch, fn2 = self.fn_ch_gender_n_2ch, cols = ['givenname', 'likelyGender', 'genderScale', 'score', 'probabilityCalibrated']) # N: namsor
        self.df_ch_gender_r = __load_tool_data(fn1 = self.fn_ch_gender_r_1ch, fn2 = self.fn_ch_gender_r_2ch, cols = ['givenname', 'NG']) # R: r-package
        
        # GT
        if not ios.path_exists(self.fn_gt):
            assert ToolsComparison.prepare_ground_truth(fn_gt=self.fn_gt)
        self.df_gt = ios.load_csv(self.fn_gt)
        self.df_gt.loc[:,'gt'] = True

    def validate_givennames(self):
        # 1. validate all given names in GT are valid
        assert self.df_gt.givenname.isin(self.df_givennames.givenname).all(), 'Some GT names are invalid.'

        # 2. validate all given names in tools are valid
        assert self.df_ch_gender_w.givenname.isin(self.df_givennames.givenname).all(), 'Some W names are invalid.'
        assert self.df_ch_gender_r.givenname.isin(self.df_givennames.givenname).all(), 'Some R names are invalid.'
        assert self.df_ch_gender_n.givenname.isin(self.df_givennames.givenname).all(), 'Some N names are invalid.'

        # 3. validate all GT given names are in tools
        assert self.df_gt.query("single_char == False").givenname.isin(self.df_ch_gender_w.givenname).all(), 'Some GT names are not in W.'
        assert self.df_gt.givenname.isin(self.df_ch_gender_r.givenname).all(), 'Some GT names are not in R.'
        assert self.df_gt.givenname.isin(self.df_ch_gender_n.givenname).all(), 'Some GT names are not in N.'

        # 4. validate number of names across tools
        assert self.df_ch_gender_r.shape[0] == self.df_ch_gender_n.shape[0], 'Different number of names between R and N.'
        assert self.df_ch_gender_r.query("single_char == True").shape[0] == self.df_ch_gender_n.query("single_char == True").shape[0], 'Different number of names between R and N single char.'
        assert self.df_ch_gender_r.query("single_char == False").shape[0] == self.df_ch_gender_w.shape[0], 'Different number of names between R and W.'
        assert self.df_ch_gender_n.query("single_char == False").shape[0] == self.df_ch_gender_w.shape[0], 'Different number of names between N and W.'

    def compare_unique_chinese_given_names(self):

        # merging w and r
        # NG ranges from –1 (completely feminine; 100% used by female) through 0 (gender-neutral; half by female and half by male) to 1 (completely masculine; 100% used by male).
        # how=left because W only has 2-char givennames, while R has both 1-char and 2-char givennames.
        self.df_ch_genders = pd.merge(self.df_ch_gender_r[['givenname','NG']], self.df_ch_gender_w[['givenname','gender']], on='givenname', suffixes=('_r', '_w'), how='left')

        print(self.df_ch_genders.shape, self.df_ch_gender_r.shape, self.df_ch_gender_w.shape)

        self.df_ch_genders = self.df_ch_genders.set_index('givenname').join(self.df_gt.set_index('givenname'), how='left')
        self.df_ch_genders.loc[:, 'gt'] = self.df_ch_genders.loc[:,'gt'].apply(lambda v: v not in const.NONE)  # ensure gt is boolean

        # renaming columns
        self.df_ch_genders.rename(columns={'gender':'gender_w', 'NG':'ngender_r'}, inplace=True)
        
        # merging with n
        # scale: Gender scale ranging from -1 (male) to +1 (female).
        # prob: Higher implies a more reliable result, ranges from 0 to 1.
        tmp = self.df_ch_gender_n.rename(columns={'likelyGender':'gender_n', 'genderScale':'ngender_n', 'score':'score_n', 'probabilityCalibrated':'prob_n'})
        tmp = tmp[['givenname', 'gender_n','ngender_n','score_n','prob_n']]
        self.df_ch_genders = self.df_ch_genders.join(tmp.set_index('givenname'), how='left')

        # normalizing values (Same scales)
        # r: -1 feminine, 0 neutral, 1 masculine
        # n: -1 male, 1 feminine
        # ---------------------------------------
        # new: 0 male, 1 feminine
        self.df_ch_genders.loc[:,'ngender_w'] = self.df_ch_genders.gender_w.apply(lambda v: 0. if v=='male' else 1.0 if v=='female' else None)
        self.df_ch_genders.loc[:,'ngender_r'] = self.df_ch_genders.ngender_r.apply(lambda v: scale_minus1_to_1_to_0_to_1(v*-1))
        self.df_ch_genders.loc[:,'ngender_n'] = self.df_ch_genders.ngender_n.apply(lambda v: scale_minus1_to_1_to_0_to_1(v))

        

        # metrics
        index_gt = self.df_gt.givenname.unique()
        self.gt_cols = [col for col in self.df_gt.columns if col not in ['NG', 'gender_r', 'gender_w', 'gender_n', 'ngender_w', 'ngender_r', 'ngender_n', 'givenname','gt','single_char', 'source']]
        
        print(len(index_gt))

        for metric in [np.mean, np.median, np.std, _mode]:
            metric_name = metric.__name__.replace("_",'')
            # GT
            self.df_ch_genders.loc[index_gt,metric_name] = self.df_ch_genders.loc[index_gt,:].apply(lambda row: scale_1_to_5_to_0_to_1(metric(row[self.gt_cols].dropna())), axis=1) # @TODO: remove dropna once all GT is finalized
            # TOOLS
            self.df_ch_genders.loc[:,f'{metric_name}_ngender_rnw'] = self.df_ch_genders.apply(lambda row: metric([row['ngender_r'], row['ngender_n'], row['ngender_w']]), axis=1)
            self.df_ch_genders.loc[:,f'{metric_name}_ngender_rn'] = self.df_ch_genders.apply(lambda row: metric([row['ngender_r'], row['ngender_n']]), axis=1)
            self.df_ch_genders.loc[:,f'{metric_name}_ngender_rw'] = self.df_ch_genders.apply(lambda row: metric([row['ngender_r'], row['ngender_w']]), axis=1)
            self.df_ch_genders.loc[:,f'{metric_name}_ngender_nw'] = self.df_ch_genders.apply(lambda row: metric([row['ngender_n'], row['ngender_w']]), axis=1)


        # majority voting
        self.df_ch_genders.loc[:,'gender_nw'] = self.df_ch_genders.apply(lambda row: categorical.NEUTRAL if row['gender_n'] != row['gender_w'] else row['gender_n'], axis=1)

        # only ground truth
        self.df_ch_genders_gt = self.df_ch_genders.query("gt == True").reset_index().copy()



    def save(self, path=None, fn=None):
        if path is None or fn is None:
            raise ValueError('Path and filename must be provided.')
        ios.validate_path(path)
        fn = ios.path_join(path, fn)
        ios.to_csv(self.df_ch_genders, fn)

    @staticmethod
    def prepare_ground_truth(dir_gt=None, fn_gt=None):
        if dir_gt is None:
            raise ValueError('Directory for ground truth must be provided.')
    
        try:
            df_result = pd.DataFrame()
            files = ios.listdir(dir_gt, pattern='(Responses)', ext='.xlsx')
            
            for fn in files:
                ios.printf(f'Reading {fn}')
                df = ios.load_excel(fn).pivot_table(columns='Timestamp')
                print(f'[INFO] {fn} has {df.shape[1]} columns and {df.shape[0]} rows.')

                # changing timestamp to sequential number (annotator id)
                df.rename(columns={c:f'a{id+1}' for id,c in enumerate(df.columns) if c != 'Timestamp'}, inplace=True)

                # renaming columns
                df.index.name = 'givenname'
                df.columns.name = 'annotator'
                df.reset_index(inplace=True)

                # add single character column (bool)
                df.loc[:,'single_char'] = df.givenname.apply(lambda v: len(v) == 1)
                df.loc[:,'source'] = fn.split('/')[-1]
                df.set_index('givenname', inplace=True)
                
                # info
                print(f"[INFO] {df.single_char.value_counts()}")

                # concatenating files
                df_result = pd.concat([df_result, df], ignore_index=False)
                
            # Summary
            print(f'[INFO] Merging {len(files)} files into one dataframe with {df_result.shape[0]} rows and {df_result.shape[1]} columns.')
            print(f"[INFO] {df_result.single_char.value_counts()}")

            # Save the result
            if fn_gt is not None:
                df_result = df_result[[f"a{id+1}" for id in range(len(df_result.columns)-2)]+['single_char','source']]  # keep single_char at the end
                ios.to_csv(df_result.sort_values(['single_char', 'givenname']), ios.path_join(dir_gt, fn_gt))

        except Exception as e:
            print(f'Error reading ground truth files: {e}')
            return False
        
        
        return True
        
    
def prepare_features(data, column_name, dir_resources, cache_dir=None):

    df = data.copy()
    
    # renames
    df.rename(columns={'NG':'ngender_r', 'genderScale':'ngender_n', 'likelyGender':'gender_n'}, inplace=True)

    # Combinations
    # mean_ngender_rn	mean_ngender_rw	mean_ngender_nw	mean_ngender_rnw
    # reg: ngender_r    gender_w        mean_ngender_rw
    combinations = [('r','n'), ('r','w'), ('n','w'), ('r','n','w')]

    # check if all cols exist already
    required_cols = ['ngender_r', 'ngender_n', 'ngender_w', 'gender_w']
    for c in combinations:
        tmp = [f"mean_ngender_{''.join(c)}"]
        required_cols.extend(tmp)
    missing_cols = set(required_cols) - set(df.columns)
    if not missing_cols:
        ios.printf('All columns already exist.')
        return df
    
    ios.printf(f'Recovering Missing columns: {missing_cols}')
    # import sys
    # sys.exit(0)

    # W tool
    if 'gender_w' not in df.columns:
        from gender.chinese_tools import ChineseTools
        ct = ChineseTools(fn_full_names=None,
                        fn_given_names=None, 
                        fn_ch_given_names=None,
                        fn_en_given_names=None,
                        fn_ch_given_names_gender=None,
                        fn_en_given_names_gender=None,
                        fn_full_names_gender=None,
                        resource_dir=dir_resources,
                        cache_dir=cache_dir,
                        )
        df.loc[:,'gender_w'] = df.apply(lambda row: ct.chinese_tools_gender_guesser(name=row[column_name])[2], axis=1)
        df.loc[:,'ngender_w'] = df.gender_w.apply(lambda v: 0. if v=='male' else 1.0 if v=='female' else ios.pd.NA)

    # R tool
    if 'ngender_r' in df.columns:
        df.loc[:,'ngender_r'] = df.ngender_r.apply(lambda v: scale_minus1_to_1_to_0_to_1(v*-1))

    # N tool
    df.loc[:,'ngender_n'] = df.ngender_n.apply(lambda v: scale_minus1_to_1_to_0_to_1(v))

    # NW tool (majority voting)
    df.loc[:,'gender_nw'] = df.apply(lambda row: categorical.NEUTRAL if row['gender_n'] != row['gender_w'] else row['gender_n'], axis=1)

    # Combinations
    for c in combinations:
        df.loc[:,f'mean_ngender_{"".join(c)}'] = df.apply(lambda row: np.mean([row[f'ngender_{v}'] for v in c]), axis=1)
        #df.loc[:,f'median_ngender_{"".join(c)}'] = df.apply(lambda row: np.mean([row[f'ngender_{v}'] for v in c]), axis=1)
        #df.loc[:,f'mode_ngender_{"".join(c)}'] = df.apply(lambda row: np.mean([row[f'ngender_{v}'] for v in c]), axis=1)
    
    return df



# def get_gender_from_NG(ng, smooth=0):
#     threshold = 0 + abs(smooth)
#     return FEMALE if ng < -threshold else MALE if ng > threshold else NEUTRAL


# def get_labels(nclasses=3):
#     if nclasses == 3:
#         return GENDER_LABELS_3
#     if nclasses == 5:
#         return  GENDER_LABELS_5
    
#     raise ValueError('Invalid number of classes.')

# def get_label_5classes(v=None, thresholds=None, labels=GENDER_LABELS_5):
#     if v is None or thresholds is None:
#         return labels
#     if v <= thresholds[0]:
#         return labels[0]
#     if v <= thresholds[1]:
#         return labels[1]
#     if v <= thresholds[2]:
#         return labels[2]
#     if v <= thresholds[3]:
#         return labels[3]
#     if v > thresholds[3]:
#         return labels[4]
#     return None

# def get_label_3classes(v=None, thresholds=None, labels=GENDER_LABELS_3):
#     if v is None or thresholds is None:
#         return labels
#     if v <= thresholds[0]:
#         return labels[0]
#     if v > thresholds[1]:
#         return labels[2]
#     return labels[1]

# def get_gender_majority_voting(row, values=None):
#     from statistics import mode
#     values = [row.gender_r, row.gender_w , row.gender_n] if values is None else values
    
#     # all the same (consensus)
#     if len(set(values)) == 1:
#         return values[0]
    
#     if (row.ngender_r >= 0.3 and row.ngender_r < 0.7) and (row.ngender_n >= 0.3 and row.ngender_n < 0.7):
#         return 'neutral'
#     return f"mostly_{mode(values)}"



# def get_gender_from_NG(ng, smooth=0):
#     threshold = 0 + abs(smooth)
#     return FEMALE if ng < -threshold else MALE if ng > threshold else NEUTRAL


# def get_labels(nclasses=3):
#     if nclasses == 3:
#         return GENDER_LABELS_3
#     if nclasses == 5:
#         return  GENDER_LABELS_5
    
#     raise ValueError('Invalid number of classes.')

# def get_label_5classes(v=None, thresholds=None, labels=GENDER_LABELS_5):
#     if v is None or thresholds is None:
#         return labels
#     if v <= thresholds[0]:
#         return labels[0]
#     if v <= thresholds[1]:
#         return labels[1]
#     if v <= thresholds[2]:
#         return labels[2]
#     if v <= thresholds[3]:
#         return labels[3]
#     if v > thresholds[3]:
#         return labels[4]
#     return None

# def get_label_3classes(v=None, thresholds=None, labels=GENDER_LABELS_3):
#     if v is None or thresholds is None:
#         return labels
#     if v <= thresholds[0]:
#         return labels[0]
#     if v > thresholds[1]:
#         return labels[2]
#     return labels[1]

# def get_gender_majority_voting(row, values=None):
#     from statistics import mode
#     values = [row.gender_r, row.gender_w , row.gender_n] if values is None else values
    
#     # all the same (consensus)
#     if len(set(values)) == 1:
#         return values[0]
    
#     if (row.ngender_r >= 0.3 and row.ngender_r < 0.7) and (row.ngender_n >= 0.3 and row.ngender_n < 0.7):
#         return 'neutral'
#     return f"mostly_{mode(values)}"


    

#     def get_best_thresholds_general(self, column, metric='mean', labels_fnc=get_label_5classes, step=0.1):

#         if column == 'majority_voting':
#             raise ValueError('Majority voting is not a valid column for this function.')

#         # data
#         df = self.df_ch_genders_gt.copy()
#         df.dropna(inplace=True)

#         # Binarize the labels for multi-class ROC AUC
#         gender_classes = labels_fnc()
#         n_classes = len(gender_classes)

#         # Initialize variables to store the best thresholds and the best AUC
#         best_thresholds = None
#         best_auc = -np.inf

#         # RMSE
#         rmse = root_mean_squared_error(df[metric], df[column]) # metric is GT,
        
#         # Create the vector
#         vector = np.arange(0, 1+step, step)
#         for split_points_m in combinations(range(1, len(vector)), n_classes-1):
#             chunks_m = np.split(vector, split_points_m)
#             thresholds_m = [chunk[0] for chunk in chunks_m[1:] if len(chunk) > 0]

#             # Binarize the scores based on the current thresholds
#             y_true = df[metric].apply(lambda v: labels_fnc(v, thresholds_m))
#             y_pred = df[column].apply(lambda v: labels_fnc(v, thresholds_m))

#             y_true = label_binarize(y_true, classes=gender_classes)
#             y_pred = label_binarize(y_pred, classes=gender_classes)

#             # Compute AUC for each class
#             try:
#                 auc = roc_auc_score(y_true, y_pred, average=None)
#                 min_auc = min(auc)
#                 max_auc = max(auc)
                
#                 print(f'Thresholds: {thresholds_m} | Min AUC: {min_auc} | Max AUC: {max_auc}')

#                 # Update the best thresholds if the current combination is better
#                 if min_auc > best_auc:
#                     best_auc = auc
#                     best_thresholds = thresholds_m
#             except Exception as e:
#                 pass
            
#         return best_thresholds, rmse


#     def get_performance_general(self, column=None, metric='mean', labels_fnc=get_label_3classes, bins_gt=[0.2,0.5,0.8], bins_tools=[0.2,0.5,0.8]):

#         df = self.df_ch_genders.copy()
#         df.dropna(inplace=True)

#         # column = 'gender_majority_voting' # 2025-01-28 @TODO: This was here but it doesn't make sense if it is passed as an argument (I'm passing it now in the notebooks)
#         # df.loc[:,'inferred_gender'] = df[column] if column == 'gender_majority_voting' else df[column].apply(lambda v: labels_fnc(v, thresholds)) # from tools
#         # df.loc[:,'GT_gender'] = df[metric].apply(lambda v: labels_fnc(v, thresholds)) # from GT

        
#         # df.loc[:,'GT_gender'] = pd.cut(df[metric], bins=bins_gt, labels=False, include_lowest=True)
#         # df.loc[:,'inferred_gender'] = pd.cut(df[column], bins=bins_tools, labels=False, include_lowest=True)

#         # RMSE with 0-1 values
#         rmse = root_mean_squared_error(df[metric], df[column]) # metric is GT, 

#         # Binarize the labels for multi-class ROC AUC
#         # gender_classes = labels_fnc()
#         n_classes = len(bins_gt) + 1
#         y_pred = pd.cut(df[column], bins=[0, *bins_tools, 1], labels=get_labels(n_classes), include_lowest=True)
#         y_true = pd.cut(df[metric], bins=[0, *bins_gt, 1], labels=get_labels(n_classes), include_lowest=True)
        

#         # Compute precision, recall, accuracy for each class
#         precision = precision_score(y_true, y_pred, average=None)
#         recall = recall_score(y_true, y_pred, average=None)
#         accuracy = accuracy_score(y_true, y_pred)

#         # Compute AUC for each class
#         auc = roc_auc_score(y_true, y_pred, average=None)

#         # Print metrics
#         print(f'Precision: {precision}')
#         print(f'Recall: {recall}')
#         print(f'Accuracy: {accuracy}')
#         print(f'AUC: {auc}')
#         print(f'RMSE: {rmse}')

#         # Compute TPR and FPR for each class and find optimal thresholds
#         fpr = dict()
#         tpr = dict()
#         thresholds = dict()
#         optimal_thresholds = dict()
#         for i in range(n_classes):
#             fpr[i], tpr[i], thresholds[i] = roc_curve(y_true[:, i], y_pred[:, i])
#             # Calculate Youden's J statistic
#             J = tpr[i] - fpr[i]
#             ix = np.argmax(J)
#             optimal_thresholds[i] = thresholds[i][ix]
#             print(f'Optimal threshold for class {i}: {optimal_thresholds[i]}')

#         # Compute the confusion matrix (raw)
#         cm_raw = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
#         cm_recall = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1), normalize='true')
#         cm_precision = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1), normalize='pred')

#         return auc, precision, recall, accuracy, fpr, tpr, cm_raw, cm_recall, cm_precision, gender_classes
    

#     def get_best_thresholds(self, column='ngender_r', metric='mean', labels_fnc=get_label_3classes):
#         # Initialize variables
#         log_auc_rmse = {}

#         # data
#         df = self.df_ch_genders.copy()
#         df.dropna(inplace=True)

#         # Binarize the labels for multi-class ROC AUC
#         gender_classes = labels_fnc()

#         # Define a range of thresholds
#         thresholds_range = np.arange(0, 1, 0.01)

#         # Initialize variables to store the best thresholds and the best AUC
#         best_thresholds = None
#         best_auc = -np.inf

#         # Iterate over all combinations of thresholds
#         for t1, t2 in product(thresholds_range, repeat=2):
#             if t1 >= t2:
#                 continue

#             # RMSE with 0-1 values
#             rmse = root_mean_squared_error(df[metric], df[column]) # metric is GT, 

#             # Binarize the scores based on the current thresholds
#             y_true = df[metric].apply(lambda v: labels_fnc(v, t1, t2))
#             y_pred = df[column].apply(lambda v: labels_fnc(v, t1, t2))

#             y_true = label_binarize(y_true, classes=gender_classes)
#             y_pred = label_binarize(y_pred, classes=gender_classes)

#             # Compute AUC for each class
#             try:
#                 auc = roc_auc_score(y_true, y_pred, average=None)
#                 min_auc = np.min(auc)
#                 max_aux = np.max(auc)

#                 log_auc_rmse[(t1,t2)] = {'min_auc': min_auc, 'max_aux': max_aux, 'rmse': rmse}

#                 # Update the best thresholds if the current combination is better
#                 if min_auc > best_auc:
#                     best_auc = min_auc
#                     best_thresholds = (t1, t2)
#             except Exception as e:
#                 pass

#         return best_thresholds, log_auc_rmse
    
#     def get_performance(self, column='ngender_r', metric='mean', labels_fnc=get_label_3classes, t1=0.4, t2=0.6):

#         df = self.df_ch_genders.copy()
#         df.dropna(inplace=True)

#         df.loc[:,'inferred_gender'] = df[column].apply(lambda v: labels_fnc(v, t1, t2)) # from tools
#         df.loc[:,'GT_gender'] = df[metric].apply(lambda v: labels_fnc(v, t1, t2)) # from GT

#         # Binarize the labels for multi-class ROC AUC
#         gender_classes = labels_fnc()
#         y_pred = label_binarize(df['inferred_gender'], classes=gender_classes)
#         y_true = label_binarize(df['GT_gender'], classes=gender_classes)
#         n_classes = y_pred.shape[1]

#         # Compute precision, recall, accuracy for each class
#         precision = precision_score(y_true, y_pred, average=None)
#         recall = recall_score(y_true, y_pred, average=None)
#         accuracy = accuracy_score(y_true, y_pred)
#         rmse = root_mean_squared_error(df[metric], df[column]) # metric is GT, 

#         # Compute AUC for each class
#         auc = roc_auc_score(y_true, y_pred, average=None)

#         # Print metrics
#         print(f'Precision: {precision}')
#         print(f'Recall: {recall}')
#         print(f'Accuracy: {accuracy}')
#         print(f'AUC: {auc}')
#         print(f'RMSE: {rmse}')

#         # Compute TPR and FPR for each class and find optimal thresholds
#         fpr = dict()
#         tpr = dict()
#         thresholds = dict()
#         optimal_thresholds = dict()
#         for i in range(n_classes):
#             fpr[i], tpr[i], thresholds[i] = roc_curve(y_true[:, i], y_pred[:, i])
#             # Calculate Youden's J statistic
#             J = tpr[i] - fpr[i]
#             ix = np.argmax(J)
#             optimal_thresholds[i] = thresholds[i][ix]
#             print(f'Optimal threshold for class {i}: {optimal_thresholds[i]}')

#         # Compute the confusion matrix (raw)
#         cm_raw = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
#         cm_recall = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1), normalize='true')
#         cm_precision = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1), normalize='pred')

#         return auc, precision, recall, accuracy, fpr, tpr, cm_raw, cm_recall, cm_precision, gender_classes



# def get_majority_voting(df, cols, n_labels):
#     return