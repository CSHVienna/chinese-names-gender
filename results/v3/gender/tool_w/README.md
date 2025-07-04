# Data Tool W: Chinese-Tools

This repository contains datasets related to gender inference using the Chinese Name Gender Guesser from Chinese-Tools.com


## Updates

- **2025-07-04**: We use `CNKI_uniqueChineseGivenNames_gender.csv`


## Files

### 1. `CNKI_uniqueChineseGivenNames_gender.csv`

- **Source**: Curated after merging `CNKI_uniqueChineseGivenNames_g.csv` with `results/v3/names/chinese/CNKI_uniqueChineseGivenNames_CNKIpop.csv` (new disambiguation)
- **Author**: Lisette Espin
- **Date**: 2025-07-04  
- **Size**: 1.8 K  
- **Records**: 39,712  
- **Columns**:  
  - `givenname`
  - `pinyin_wacc`
  - `pinyin`
  - `english`
  - `gender`


### 2. `CNKI_uniqueChineseGivenNames_g.csv`

- **Source**: Google Drive (`CNKI_dataset_cleaned/v2/gender_chinese_tool/CNKI_uniqueChineseGivenNames_gender.csv`)
- **Author**: Lisette Espin
- **Date**: 2024-03-27
- **Size**: 2.0 M
- **Records**: 39,713
- **Columns**:  
  - `givenname`
  - `pinyin_wacc`
  - `pinyin`
  - `english`
  - `gender`


## Package Reference

Chinsese Tool GenderGuesser: [https://www.chinese-tools.com/tools/gender-guesser.html](https://www.chinese-tools.com/tools/gender-guesser.html)
- `male`, `female`