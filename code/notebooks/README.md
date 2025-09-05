# Notebooks

This folder contains Jupyter Notebooks used for various _development_ tasks.  
Each file includes details on its function, purpose, and authorship.

---

## `_classification.ipynb`
**Author:** Lisette Espin
**Description:** Step-by-step classification of gender using a `CatBoostClassifier`. Includes data preprocessing (label encoding), grid search, and manual strategy assignment (cutoffs for forming categories from numeric gender values, e.g., 3 or 5 classes, and ground truth aggregation from annotations such as mean, median, or mode). Also shows validation results.  
**Creation Date:** [YYYY-MM-DD]
**Status:** *(Development)*  

---

## `_fixing_predictions_new_name.ipynb`
**Author:** Lisette Espin  
**Description:** Adds missing given names and removes given names not included in new disambiguation.
**Creation Date:** [YYYY-MM-DD]  
**Status:** *(Not needed anymore)*

---

## `_gender.ipynb`
**Author:** Lisette Espin 
**Description:** Standardizes the gender files based on given name files. 
**Creation Date:** [YYYY-MM-DD]  
**Status:** *(Not needed anymore)*
 

---

## `chinese_gender_comparison.ipynb`
**Author:** Lisette Espin
**Description:** Using the ground truth, compares inference results by the three tools. Also manually shows threshold cutoffs for converting numeric to categorical gender labels. 
**Creation Date:** [YYYY-MM-DD]  
**Status:** *(Development)*  


---

## `chinese_gender_tool.ipynb`
**Author:** Lisette Espin 
**Description:** Loads inference results from W tool.
**Creation Date:** [YYYY-MM-DD]  
**Status:** *(Development)*  


---

## `predictions_alldata.ipynb`
**Author:** Lisette Espin 
**Description:** Shows the best classification and regression results.
**Creation Date:** [YYYY-MM-DD]  
**Status:** *(Development)*  


---

## `predictions_classification.ipynb`
**Author:** Lisette Espin
**Description:** Loads all classification results and shows the best one.  
**Creation Date:** [YYYY-MM-DD]  
**Status:** *(Development)*  

---

## `predictions_regression.ipynb`
**Author:** Lisette Espin
**Description:** Loads all regression results and shows the best one.  
**Creation Date:** [YYYY-MM-DD]  
**Status:** *(Development)*  


---

## `summary_CNKI_crawls.ipynb`
**Author:** Lisette Espin
**Purpose:** Reads all chunks of data collection (publications, references, and citations) from web crawling CNKI.  
**Creation Date:** [YYYY-MM-DD]
**Status:** *(Not needed anymore)*

---

**Note:** Notebooks are marked as *(Not needed anymore)* or *(Development)*, indicating their status in the current workflow.