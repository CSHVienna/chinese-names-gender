# Notebooks

This folder contains Jupyter Notebooks used for data exploration, evaluation, and the analyses reported in the paper.  
Notebooks are numbered in the order they are typically run. Older, exploratory notebooks are kept in [`dev/`](dev/).

---

## `0_summary_names.ipynb`
**Author:** Lisette Espin  
**Description:** Descriptive statistics of the name datasets: the full list of Chinese names (Hanzi and Pinyin, `a-z_zip`), the unique given names found in CNKI (and their popularity), and the unique single-character names. Counts unique Hanzi vs. Pinyin forms, strips tones from Pinyin, and shows how many distinct characters map to the same Pinyin syllable (e.g., *yi*).  

---

## `1_chinese_gender_comparison.ipynb`
**Author:** Lisette Espin  
**Description:** Prepares the gender ground truth (annotations aggregated by mean, median, and mode) and compares it with the inferences of the three tools (W, R, N) and their combinations (RN, RW, NW, RNW). Includes ground-truth statistics, tool-vs-tool comparisons (categorical and numerical), tool-vs-ground-truth scatter plots, and the threshold strategies (cutoffs) used to convert numeric gender scores into categorical labels.  

---

## `2_evaluation_disambiguation.ipynb`
**Author:** Lisette Espin  
**Description:** Evaluates the manual annotations of the name disambiguation. Measures inter-annotator agreement (raw agreement, Krippendorff's alpha, and Cohen's kappa) for the first and second (final) annotation rounds.  

---

## `3_predictions_classification_ch.ipynb`
**Author:** Lisette Espin  
**Description:** Loads the best classification models (Hanzi given names) selected by `batch_eval_inference.py` and shows their results overall and per random seed.  

---

## `3_predictions_regression_ch.ipynb`
**Author:** Lisette Espin  
**Description:** Same as above for the regression models: loads the best models (by RMSE / R²) and shows their results overall and per random seed.  

---

## `4_predictions_py.ipynb`
**Author:** Lisette Espin  
**Description:** Gender inference for Pinyin given names, obtained by aggregating the Hanzi predictions of all characters that share the same Pinyin (e.g., simple mean, mean over CNKI names, population-weighted). Compares the aggregation strategies against the Hanzi predictions and the held-out ground truth, for both regression (scores) and classification (labels, confusion matrices), and analyzes ambiguity (entropy/disagreement) of Pinyin names.  

---

## `5_comparison_baselines.ipynb`
**Author:** Lisette Espin  
**Description:** Compares our models (Hanzi and Pinyin) with baselines: NamSor, LLMs (Claude and GPT, from `batch_llm_gender_inference.py`), and other Pinyin-based methods (e.g., gender-guesser, Karimi, Van Buskirk, Saeta). Evaluates gender scores and labels against the ground truth, and measures pairwise agreement between methods on APS authors with both Pinyin and Hanzi names.  
