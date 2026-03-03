# Inferred Gender from Pinyin

Author: Liuhuaying yang

## Fields

**Basics**:
- `givenname`: Chinese given name in characters.
- `pop_CNKI`: a population count in CNKI data.
- `pop_NU`: a population estimate from ChineseNames R Package.
- `NLen`: number of characters in the given name.
- `english_one`: pinyin/english of the given name.

**Character-name related gender values**:
- `pred`: reg_mean from `/predictions`.
- `pred_label`: class_median from `/predictions`.

**Pinyin related gender values**:
- `unique_name_count`: number of unique character-based given names.
- `givennames`: unique character-based names share the same pinyin.
- `male_label_count`: counts how many `pred_label` are "male" for its pinyin.
- `female_label_count`: counts how many `pred_label` are "female" for its pinyin.
- `neutral_label_count`: counts how many `pred_label` are "neutral" for its pinyin.
- `pop_sum_CNKI`: total CNKI population at pinyin level (sum up `pop_CNKI` of character names share the same pinyin).
- `pop_sum_NU`: total NU population at pinyin level.
- `gender_weighted_pop_CNKI`: a population-weighted average `pred` value, using `pop_CNKI` as weights. `NaN` means the name is not in CNKI data, but can be used as a single-character name.  
- `gender_weighted_pop_NU`: a population-weighted average `pred` value, using `pop_NU` as weights.
- `gender_avg_all`: a simple (unweighted) average of `pred` value for all names.
- `gender_avg_CNKI`: a simple (unweighted) average of `pred` value for only CNKI names.



## Files

`uniqueChineseAllGivenNames_PYgender_reg_mean_894_class_mean_607_1.csv` (best models across all runs)

    Calculated gender values for unique give names, based on `/results/v3/gender/predictions_ch/best_models/rs2`:
    - `CNKI_uniqueChineseAllGivenNames_gender_reg_mean_894.csv`
    - `CNKI_uniqueChineseAllGivenNames_gender_class_mean_607_1.csv`


`uniqueChineseAllGivenNames_PYgender_reg_mean_224_class_meadian_251_4.csv` (best models for rs1)

    Calculated gender values for unique give names, based on `/results/v3/gender/predictions_ch/best_models/rs1`:
    - `CNKI_uniqueChineseAllGivenNames_gender_reg_mean_224.csv`
    - `CNKI_uniqueChineseAllGivenNames_gender_class_median_251_4.csv`

