# Gender from Pinyin

Author: Liuhuaying yang

## Unique Chinese given names
Calculated gender values for unique give names, based on `/results/v3/gender/predictions/`:
- `CNKI_uniqueChineseAllGivenNames_gender_reg_mean_224.csv`
- `CNKI_uniqueChineseAllGivenNames_gender_class_median_251_4.csv`
### Fields
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
- `major_label`: majority vote on `pred_label`, comparing `male_label_count`, `female_label_count`, and `neutral_label_count`, and assign the label with the highest count. If there is a tie, the label is set to `neutral`.
- `thres_label`: threshold-based labeling on `pred_label`, computing the proportion of `male_label_count`, `female_label_count`, and `neutral_label_count` and assign the label with the highest proportion only if it exceeds a predefined threshold (default = 0.6). Otherwise, the result is set to `neutral`.
- `label_score`: a continuous gender score calculated by mapping `male_label_count`, `female_label_count`, and `neutral_label_count` to numeric values (0.000001, 0.5, and 0.999999, respectively) and averaging them. Exact 0 and 1 values are avoided for numerical stability.
- `gender_weighted_pop_CNKI`: a population-weighted average `pred` value, using `pop_CNKI` as weights. `NaN` means the name is not in CNKI data, but can be used as a single-character name.  
- `gender_weighted_pop_NU`: a population-weighted average `pred` value, using `pop_NU` as weights.
- `gender_avg`: a simple (unweighted) average of `pred` value.


### Calculations
Let n_m, n_f, n_n be `male_label_count`, `female_label_count`, and `neutral_label_count`.

**Calculate `major_label`**
<pre>
major_label =
  argmax{n_m, n_f, n_n}, if the maximum is unique
  neutral, otherwise
</pre>
**Calculate `thres_label`**
<pre>
Let N = n_m + n_f + n_n
Let p_l = n_l / N

thres_label =
  argmax{p_m, p_f, p_n}, if max(p_l) >= 0.6
  neutral, otherwise
</pre>
**Calculate `label_score`**
<pre>
male = ε, neutral = 0.5, female = 1 − ε, where ε = 1e−6

label_score = (n_m * ε + n_n * 0.5 + n_f * (1 − ε)) / (n_m + n_f + n_n)
</pre>

