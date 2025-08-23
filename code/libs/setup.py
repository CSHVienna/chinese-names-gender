from itertools import combinations

import categorical

GRID = {'iterations': [100, 150, 200, 300],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'depth': [2, 4, 6, 8],
        'l2_leaf_reg': [0.1, 0.2, 0.5, 1, 3]}

RANDOM_STATE = 12345
TEST_SIZE = 0.2
CV = 3
LOSS_FUNCTION_CLA = 'MultiClass'
EVAL_METRIC_CLA = 'MultiClass'
LOSS_FUNCTION_REG = 'RMSE'
EVAL_METRIC_REG = 'RMSE'
EARLY_STOPPING = 50

LABELS = categorical.GENDER_LABELS_3            # DO NOT CHANGE (ORDER MATTERS)
LABELS_CODE = categorical.GENDER_LABELS_3_DICT  # DO NOT CHANGE (ORDER MATTERS)

STRATEGY = {1:{'cuts':[0.33, 0.66], 'gt_agg':'mean'},
            2:{'cuts':[0.32, 0.57], 'gt_agg':'mean'},
            3:{'cuts':[0.24, 0.70], 'gt_agg':'mean'},
            4:{'cuts':[0.30, 0.70], 'gt_agg':'median'},
            5:{'cuts':[0.40, 0.60], 'gt_agg':'mode'},
            6:{'cuts':[0.4, 0.6], 'gt_agg':'mean'},
            7:{'cuts':[0.2, 0.8], 'gt_agg':'mean'},
           }

GT_AGGS = ['mean', 'median', 'mode']

CAT_FEATURES = ['gender_w','gender_n','gender_nw','single_char']

FEATURES = [
            'ngender_r',
            'ngender_n',

            'gender_n',
            'gender_w',
            'gender_nw',
            'single_char', 

            'mean_ngender_rn', 
            'mean_ngender_rw', 
            'mean_ngender_nw', 
            'mean_ngender_rnw', 
            ]

        # 'median_ngender_rnw',
        # 'gender_w_female', 'gender_w_male',
        # 'gender_n_female', 'gender_n_male',
        # 'gender_nw_female', 'gender_nw_male', 'gender_nw_neutral',

def get_all_feature_combinations():
        items = FEATURES
        all_combinations = []

        # Generate combinations of different sizes
        for r in range(1, len(items) + 1):
                for comb in combinations(items, r):
                        # Exclude combinations that contain both "item 8" and "item 10"
                        # if not ("ngender_n" in comb and "gender_n" in comb):
                        #         all_combinations.append(list(comb))
                        all_combinations.append(list(comb))

        # Print the number of combinations
        print("Total combinations of features:", len(all_combinations))

        # Print all combinations
        # for comb in all_combinations:
        #         print(comb)
                # 511 total (388 without the ngender_n, gender_n) (255 without gender_n)
        return all_combinations


FEATURE_NAMES = get_all_feature_combinations()


# FEATURE_NAMES = [FEATURES[0:1], # r
#                  FEATURES[1:2], # n

#                  FEATURES[2:3], # w(f, m)
#                  FEATURES[3:4], # n(f, m)
#                  FEATURES[4:5], # nw(f, m, n)

#                  FEATURES[5:6],  # mean_rn
#                  FEATURES[6:7],  # mean_rw
#                  FEATURES[7:8],  # mean_nw

#                  FEATURES[8:9],  # mean_rnw
#                  FEATURES[9:10], # median_rnw

#                  FEATURES[0:2], # r, n
#                  FEATURES[2:4], # w, n(f, m)
#                  FEATURES[2:5], # w, n, nw(f, m)
#                  FEATURES[3:5], # n, nw(f, m, n)
                
#                  FEATURES[5:7],  # mean_rn mean_rw
#                  FEATURES[6:8],  # mean_rw mean_nw
#                  FEATURES[5:6] + FEATURES[7:8],  # mean_rn mean_nw
#                  FEATURES[5:8],  # mean_rn mean_rw mean_nw
#                  FEATURES[5:9],  # mean_rn mean_rw mean_nw mean_rnw
#                  FEATURES[5:10], # mean_rn mean_rw mean_nw mean_rnw median_rnw

#                  FEATURES[0:3], # r n w(f,m)
#                  FEATURES[0:4], # r n w(f,m) n(f,m)
#                  FEATURES[0:5], # r n w(f,m) n(f,m) nw(f,m,n)    # redundant
#                  FEATURES[0:2] + FEATURES[3:4], # r n n(f,m)     # redundant
#                  FEATURES[0:2] + FEATURES[4:5], # r n nw(f,m,n)

#                  FEATURES[1:2] + FEATURES[3:4], # n n(f,m)   # redundant
#                  FEATURES[1:5], # n w(f,m) n(f,m) nw(f,m,n)  # redundant

#                  FEATURES[0:2] + FEATURES[3:4] + FEATURES[5:6], # r n n(f,m) mean_rn                                            # redundant
#                  FEATURES[0:2] + FEATURES[3:4] + FEATURES[5:6] + FEATURES[8:9], # r n n(f,m) mean_rn mean_rnw                   # redundant       
#                  FEATURES[0:2] + FEATURES[3:4] + FEATURES[5:6] + FEATURES[9:10], # r n n(f,m) mean_rn median_rnw                # redundant
#                  FEATURES[0:2] + FEATURES[3:4] + FEATURES[5:6] + FEATURES[8:10], # r n n(f,m) mean_rn mean_rnw median_rnw       # redundant

#                  FEATURES[0:1] + FEATURES[2:3] + FEATURES[6:7], # r w(f,m) mean_rw
#                  FEATURES[0:1] + FEATURES[2:3] + FEATURES[8:9], # r w(f,m) mean_rnw
#                  FEATURES[0:1] + FEATURES[2:3] + FEATURES[9:10], # r w(f,m) median_rnw
#                  FEATURES[0:1] + FEATURES[2:3] + FEATURES[6:10], # r w(f,m) mean_rw mean_rnw median_rnw
                 
#                  FEATURES
# ]

### INDEXES USING f,m,n for gender_w, gender_n and gender_nw
# FEATURE_NAMES = [FEATURES[0:1], # r
#                  FEATURES[1:2], # n
#                  FEATURES[2:4], # w_f w_m
#                  FEATURES[4:6], # n_f n_m
                 
#                  FEATURES[6:9], # nw_f nw_m nw_n

#                  FEATURES[9:10],  # mean_rn
#                  FEATURES[10:11], # mean_rw
#                  FEATURES[11:12], # mean_nw
#                  FEATURES[12:13], # mean_rnw
#                  FEATURES[13:14], # median_rnw

#                  FEATURES[0:2], # r, n
#                  FEATURES[2:6], # w_f w_m n_f n_m
#                  FEATURES[2:9], # w_f w_m n_f n_m nw_f nw_m nw_n
#                  FEATURES[4:9], # n_f n_m nw_f nw_m nw_n

#                  FEATURES[9:12], # mean_rn mean_rw mean_nw
#                  FEATURES[9:13], # mean_rn mean_rw mean_nw mean_rnw
#                  FEATURES[9:14], # mean_rn mean_rw mean_nw mean_rnw median_rnw

#                  FEATURES[0:6], # r n w_f w_m n_f n_m
#                  FEATURES[0:9], # r n w_f w_m n_f n_m nw_f nw_m nw_n
#                  FEATURES[0:4], # r n w_f w_m
#                  FEATURES[0:2] + FEATURES[4:6], # r n n_f n_m
#                  FEATURES[0:2] + FEATURES[6:9], # r n nw_f nw_m nw_n

#                  FEATURES[1:2] + FEATURES[4:6], # n n_f n_m
#                  FEATURES[1:9], # n w_f w_m n_f n_m nw_f nw_m nw_n

#                  FEATURES[0:2] + FEATURES[4:6] + FEATURES[9:10], # r n n_f n_m mean_rn
#                  FEATURES[0:2] + FEATURES[4:6] + FEATURES[9:10] + FEATURES[12:13], # r n n_f n_m mean_rn mean_rnw
#                  FEATURES[0:2] + FEATURES[4:6] + FEATURES[9:10] + FEATURES[13:14], # r n n_f n_m mean_rn median_rnw
#                  FEATURES[0:2] + FEATURES[4:6] + FEATURES[9:10] + FEATURES[12:14], # r n n_f n_m mean_rn mean_rnw median_rnw

#                  FEATURES[0:1] + FEATURES[2:4] + FEATURES[10:11], # r w_f w_m mean_rw
#                  FEATURES[0:1] + FEATURES[2:4] + FEATURES[12:13], # r w_f w_m mean_rnw
#                  FEATURES[0:1] + FEATURES[2:4] + FEATURES[13:14], # r w_f w_m median_rnw
#                  FEATURES[0:1] + FEATURES[2:4] + FEATURES[10:14], # r w_f w_m mean_rw mean_rnw median_rnw
                 
#                  FEATURES
# ]

# FEATURES = [
#             'ngender_r',
#             'ngender_n',
#             'gender_w_female', 'gender_w_male',
#             'gender_n_female', 'gender_n_male',

#             'mean_ngender_rn', 
#             'mean_ngender_rw', 
#             'mean_ngender_nw', 
#             'mean_ngender_rnw',

#             'median_ngender_rn', 
#             'median_ngender_rw', 
#             'median_ngender_nw', 
#             'median_ngender_rnw',

#             'mode_ngender_rn', 
#             'mode_ngender_rw', 
#             'mode_ngender_nw', 
#             'mode_ngender_rnw',
#             ]

# FEATURE_NAMES = [FEATURES[0:1], 
#                  FEATURES[1:2], 
#                  FEATURES[6:7],
#                  FEATURES[7:8],
#                  FEATURES[8:9],
#                  FEATURES[9:10],
#                  FEATURES[10:11],
#                  FEATURES[11:12],
#                  FEATURES[12:13],
#                  FEATURES[13:14],
#                  FEATURES[14:15],
#                  FEATURES[15:16],
#                  FEATURES[16:17],
#                  FEATURES[17:18],

#                  FEATURES[0:2], FEATURES[2:4], FEATURES[4:6],
#                  FEATURES[2:6],
#                  FEATURES[0:1] + FEATURES[2:4], FEATURES[0:1] + FEATURES[4:6],
#                  FEATURES[1:2] + FEATURES[2:4], FEATURES[1:2] + FEATURES[4:6],

#                  FEATURES[:4], FEATURES[:2] + FEATURES[4:6],
#                  FEATURES[6:9], FEATURES[6:10], 
#                  FEATURES[10:13], FEATURES[10:14], 
#                  FEATURES[14:17], FEATURES[14:18], 
#                  FEATURES[18:21], FEATURES[18:22], 

#                  FEATURES[6:7] + FEATURES[10:11]+ FEATURES[14:15],
#                  FEATURES[7:8] + FEATURES[11:12]+ FEATURES[15:16], 
#                  FEATURES[8:9] + FEATURES[12:13]+ FEATURES[16:17], 
#                  FEATURES[9:10] + FEATURES[13:14]+ FEATURES[17:18], 

#                  FEATURES[0:2] + FEATURES[6:18],
#                  FEATURES
# ]
