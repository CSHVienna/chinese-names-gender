# Chinese Names  
**Name Disambiguation, Gender Inference, and Visualization**

This repository provides tools for analyzing Chinese personal names. It includes methods for disambiguating names, inferring gender based on linguistic and statistical features, and visualizing gender distributions.

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

## Gender Inference Methods

### Rule-Based Gender Classification

Classifies names into three categories: **male**, **neutral**, and **female**, based on selected features and a thresholding strategy.

```bash
parallel -j 5 python batch_comparison_classification.py \
    --features_id {1} \
    --strategy_id {2} \
    --output_dir /path/to/results/ \
    ::: $(seq 0 254) ::: 1 2 3 4
```

### Rule-Based Gender Regression

Assigns a continuous gender score:  
- **0.0** = mostly male  
- **1.0** = mostly female

```bash
parallel -j 5 python batch_comparison_regression.py \
    --features_id {1} \
    --output_dir /path/to/results/ \
    ::: $(seq 0 254)
```

### Machine Learning-Based Gender Prediction

Predicts gender using a trained ML model and givenname features (from three tools: N, R, W).

```bash
python batch_gender_prediction.py \
    --input_fn /path/to/names.csv \
    --output_dir /path/to/results/ \
    --model_fn /path/to/model.cbm \
    --resources_dir ../../resources/ \
    --cache_dir ../../cache/
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