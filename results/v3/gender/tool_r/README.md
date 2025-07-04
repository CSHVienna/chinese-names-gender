# Data Tool R: ChineseNames R package

This repository contains datasets related to gender inference using the ChineseNames R package.


## Updates

- **2025-07-04**: We use `CNKI_uniqueChineseGivenNames_gender.csv`


## Files

### 1. `CNKI_uniqueChineseGivenNames_gender.csv`

- **Source**: Curated after checking `CNKI_uniqueChineseGivenNames_withGNU.csv` with `results/v3/names/chinese/CNKI_uniqueChineseGivenNames_CNKIpop.csv` (new disambiguation)
- **Author**: Lisette Espin
- **Date**: 2025-07-04  
- **Size**: 887 K  
- **Records**: 39,712  
- **Columns**:  
  - `givenname`
  - `NLen`
  - `NU`
  - `NG`


### 2. `CNKI_uniqueChineseGivenNames_withGNU.csv`

- **Source**: Google Drive (`CNKI_dataset_cleaned/v2/CNKI unique names/v2`)
- **Author**: Liuhuang Yang
- **Date**: 2025-02-26  
- **Size**: 965 K
- **Records**: 39,712
- **Columns**:  
  - `givenname`
  - `NLen`
  - `NU`
  - `NG`


## Package Reference

ChineseNames R package: [https://psychbruce.github.io/ChineseNames](https://psychbruce.github.io/ChineseNames)
- `NG`: name gender (difference in proportions of a character used by male vs. female)
- $NG = (Nmale – Nfemale) / (Nmale + Nfemale) \in [-1, 1]$
- NG ranges from $–1$ (completely feminine; 100% used by female) through $0$ (gender-neutral; half by female and half by male) to $1$ (completely masculine; 100% used by male).