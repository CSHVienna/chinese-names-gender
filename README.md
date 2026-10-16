# Chinese Names Gender
**Perceived Gender Inference of Chinese Names**

This repository provides tools for analyzing Chinese given names. It includes methods for inferring gender based on linguistic and statistical features, and visualizing gender distributions.

## Repository Structure

Scripts are located in the `code/scripts` directory:

```bash
cd code/scripts
```

## Citation Extraction

Extract metadata and citation information for each paper (based on title):

```bash
python batch_paper_metadata.py
```

## Sampling

Prepares the train/test split of the ground-truth names used to train and evaluate the ML models (classification and regression). One split is generated per ground-truth aggregation (`mean`, `median`, `mode`) and random seed, holding out `--test_fct` (e.g., 20%) of names as the test set.

```bash
parallel -j 10 python batch_sampling.py \
    --gt_agg {1} \
    --test_fct 0.2  \
    --seed {2}  \
    --output_dir ../../results/v3/gender/predictions/comparison/  \
    ::: mean median mode ::: 1 2 3
```

## Gender Inference Methods

### Gender Classification (training)

Classifies names into three categories: **male**, **neutral**, and **female**, based on selected features and a thresholding strategy.

- `--seed`: random seed, which selects the train/test split from [Sampling](#sampling).
- `--features_id`: which feature combination to use. There are 10 features (`ngender_r`, `ngender_n`, `gender_n`, `gender_w`, `gender_nw`, `mean_ngender_rn`, `mean_ngender_rw`, `mean_ngender_nw`, `mean_ngender_rnw`, `single_char`). All combinations of 1, 2, …, up to all 10 features are enumerated (2¹⁰ − 1 = 1023), and this id (0–1022) picks one (see `FEATURE_NAMES` in `code/libs/setup.py`).
- `--strategy_id`: the discretization strategy that converts the numeric gender scores of the gold standard into gender labels. Each strategy sets the ground-truth aggregation (`mean`, `median`, `mode`) and the two cutoffs that separate male / neutral / female (see `STRATEGY` in `code/libs/setup.py`):

  | id | gt_agg | cuts |
  |----|--------|------|
  | 1 | mean | 0.33, 0.66 |
  | 2 | mean | 0.32, 0.57 |
  | 3 | mean | 0.24, 0.70 |
  | 4 | median | 0.30, 0.70 |
  | 5 | mode | 0.40, 0.60 |
  | 6 | mean | 0.40, 0.60 |
  | 7 | mean | 0.20, 0.80 |

```bash
parallel -j 50 python batch_comparison_classification.py \
  --seed {1} \
  --features_id {2} \
  --strategy_id {3} \
  --output_dir ../../results/v3/gender/predictions/comparison \
  ::: $(seq 1 3) ::: $(seq 0 1022) ::: $(seq 1 7)
```

### Gender Regression (training)

Assigns a continuous gender score:  
- **0.0** = mostly male  
- **1.0** = mostly female

- `--seed`: random seed, which selects the train/test split from [Sampling](#sampling).
- `--gt_agg`: how the annotators' gender scores in the gold standard dataset are aggregated into one target value per name (`mean`, `median`, or `mode`).
- `--features_id`: which feature combination to use (0–1022). As for classification, this indexes every combination of 1, 2, …, up to all 10 features.

```bash
parallel -j 50 python batch_comparison_regression.py \
    --seed {1} \
    --gt_agg {2} \
    --features_id {3} \
    --output_dir ../../results/v3/gender/predictions/comparison/ \
    :::  $(seq 1 3) ::: mean median mode ::: $(seq 0 1022)
```

### Model Selection and Evaluation

Reads all training results from the comparison directory, identifies the best model per random seed and the best overall (according to `--metric`), infers gender for all names with those models, and stores their predictions together with the overall evaluation metrics.

```bash
export PYTHONPATH=$PYTHONPATH:../

# classification (best by AUC)
python batch_eval_inference.py \
    --comparison_dir ../../results/v3/gender/predictions_ch/comparison \
    --input_fn ../../results/v3/gender/predictions_ch/CNKI_uniqueChineseAllGivenNames_gender.csv \
    --metric test_auc \
    --classification \
    --output_dir ../../results/v3/gender/predictions_ch/best_models \
    --resources_dir ../../resources \
    --cache_dir ../../cache

# regression (best by RMSE)
python batch_eval_inference.py \
    --comparison_dir ../../results/v3/gender/predictions_ch/comparison \
    --input_fn ../../results/v3/gender/predictions_ch/CNKI_uniqueChineseAllGivenNames_gender.csv \
    --metric test_rmse \
    --output_dir ../../results/v3/gender/predictions_ch/best_models \
    --resources_dir ../../resources \
    --cache_dir ../../cache
```

### Gender Inference

Predicts gender using a trained ML model and givenname features (from three tools: N, R, W).

```bash
python batch_gender_prediction.py \
    --input_fn /path/to/names.csv \
    --output_dir /path/to/results/ \
    --model_fn /path/to/model.cbm \
    --resources_dir ../../resources/ \
    --cache_dir ../../cache/
```

### LLM Gender Inference (baseline)

Infers gender of given names with an LLM (Claude or GPT) for comparison against our models. Supports Pinyin or Hanzi input (CSV with a `py_givenname` or `hz_givenname` column) and two modes: `categorical` (gender label) or `numeric` (gender score in 0–1). Names are sent in batches; each batch is saved as a chunk file, and already-queried names found in `--output_dir` are skipped, so interrupted runs can be resumed. A consolidated file is written at the end.

Requires the API key in `ANTHROPIC_API_KEY_LEEN` (Claude) or `OPENAI_API_KEY_LEEN` (GPT).

```bash
export PYTHONPATH=$PYTHONPATH:../

python batch_llm_gender_inference.py \
    --input_file /path/to/names.csv \
    --output_dir /path/to/results/ \
    --fn gender_results \
    --ext .csv \
    --mode categorical \
    --script pinyin \
    --llm claude
```


## Methods and Data Sources

- **W**: Chinese Tool [GenderGuesser](https://www.chinese-tools.com/tools/gender-guesser.html)
  - Labels: male, female  

- **R**: [ChineseNames R Package](https://psychbruce.github.io/ChineseNames/)  
  - NG: Name-gender score  
  - Formula:  
    ```  
    NG = (N_male – N_female) / (N_male + N_female)  
    ```  
  - NG ranges from –1 to 1:  
    - –1 = completely feminine (100% female)  
    -  0 = gender-neutral (50/50)  
    - +1 = completely masculine (100% male)  

- **N**: [NamSor API](https://namsor.app/api-documentation/#genderize-chinese-name-batch)  
  - `genderScale`: –1 (male) to +1 (female)  
  - `score`: Confidence score (not normalized)  
  - `probabilityCalibrated`: Calibrated probability from 0 to 1  

- **Groundtruth Labels**:  
  Five-point scale:  
  1. Definitely male  
  2. More likely male  
  3. Neutral  
  4. More likely female  
  5. Definitely female

**Note:** For analysis, all scores from external tools and ground truth labels are rescaled to a common range between 0 (male) and 1 (female).
