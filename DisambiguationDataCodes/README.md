# CNKI Name Disambiguation Pipeline

This repository provides a **complete pipeline** for processing CNKI bibliometric data, cleaning and translating metadata, and evaluating different **name disambiguation methods** (Our method, Sinatra, Waqas).

The pipeline is modular: each step is implemented both as a **Python script (.py)** for reusable logic and as a **Jupyter Notebook (.ipynb)** for interactive execution.

---

## 🚀 Workflow

### **Step 1. Get Authorship**
Notebook: `1_GetAuthorship.ipynb`  
Runs the core preprocessing pipeline:

1. **Detect language** of abstracts → add a `Language` column.  
   - Input: `CNKI_dataset_updated_withValidNames.csv`  
   - Output: `CNKI_dataset_updated_withValidNames_withPID_withLanguage.csv`  

2. **Translate non-Chinese texts** (titles, abstracts, keywords) into Chinese.  
   - Output: `CNKI_dataset_updated_withValidNames_withPID_FullChinese.csv`  

3. **Clean authorship data** (split authors, match departments).  
   - Output: `Authorship_ValidChinese.csv`  

4. **Translate metadata into English** (names, departments, titles, abstracts, keywords).  
   - Outputs:  
     - `Authorship_English.csv`
---

### **Step 2. Similarity Analysis**
Notebook: `2_SimilarityChineseEnglish.ipynb`  

- Compare **Chinese vs English abstracts** to measure alignment quality.  
- Compute similarity scores.  

---

### **Step 3. Chinese Name Disambiguation**
Notebook: `3_ChineseDisambiguation.ipynb`  

- Apply **different disambiguation methods** on Chinese-character names:  
  - use co-authors feature 
  - use co-authors, affiliation features  
  - use co-authors, affiliation and abstract features
  - use co-authors, affiliation, abstract and citations features 
- Save results in:  
  - `DisambiguationData/name_disambiguation_output/*.csv`

---

### **Step 4. English Name Disambiguation**
Notebook: `4_Englishdisambiguation.ipynb`  

- Apply the methods on **transliterated English (Pinyin) names**.  
- Save results in:  
  - `DisambiguationData/English_names_disambiguation/*.csv`

---

### **Step 5. Evaluation**
Notebook: `5_Estimation.ipynb`  

- Input: Annotated dataset `name_test_Liu_Mingrong_final.xlsx`  
- Compare predictions vs ground truth:  
  - Precision  
  - Recall  
  - F1 Score  
- Generate plots for Chinese & English evaluation:  
  - Output: `PLOTS/Comparison_Sinatra_Waqas.pdf`  

---

## 📊 Main Outputs

- **Processed Datasets**  
  - `CNKI_dataset_updated_withValidNames_withPID_withLanguage.csv`  
  - `CNKI_dataset_updated_withValidNames_withPID_FullChinese.csv`  
  - `Authorship_ValidChinese.csv`  
  - `Authorship_English.csv`  

- **Translation Files**  
  - `English/Paper_Translation.csv`  

- **Disambiguation Results**  
  - `name_disambiguation_output/*.csv` (Chinese results)  
  - `English_names_disambiguation/*.csv` (English results)  

- **Evaluation Plots**  
  - `PLOTS/Comparison_Sinatra_Waqas.pdf`  






