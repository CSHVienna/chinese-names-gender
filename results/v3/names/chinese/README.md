# Chinese Name Datasets

This repository contains curated datasets related to Chinese names. These resources support tasks such as name analysis, gender inference, and linguistic research.

## Files

### 1. `character_dictionary_withEng_name.csv`

- **Records**: 8,108 single Chinese characters  
- **Description**: A comprehensive dictionary of single-character names, including pinyin (with and without tone), tones, meanings, and associated English names.  
- **Fields**:
  - `character`
  - `pinyin_hanziDB`
  - `pinyin_qxk`
  - `meaning`
  - `pinyin_hanziDB_qxk`
  - `pinyin_name`
  - `pinyin_givenname`
  - `pinyin_familyname`
  - `pinyin`
  - `tone`
  - `tone_name`
  - `tone_familyname`
  - `tone_givenname`
  - `pinyin_notone`
  - `eng`
  - `eng_name`
  - `eng_familyname`
  - `eng_givenname`

### 2. `CNKI_uniqueChineseFamilyNames.csv`

- **Records**: 885  
- **Description**: A list of unique Chinese family names extracted from CNKI data.

### 3. `CNKI_uniqueChineseGivenNames_CNKIpop.csv`

- **Records**: 39,712  
- **Description**: A list of unique Chinese given name characters along with their pinyin, English equivalents, and popularity based on CNKI data.  
- **Fields**:
  - `givenname`
  - `pinyin`
  - `english`
  - `CNKI_pop`

### 4. `CNKI_uniqueChineseNames.csv`

- **Records**: 81,197  
- **Description**: Full Chinese names, including family name and given name, parsed into separate fields.  
- **Fields**:
  - `fullname`
  - `familyname`
  - `givenname`

## Purpose

These datasets support linguistic, cultural, and computational studies involving Chinese personal names, with applications in NLP, gender inference, and cross-cultural naming analysis.

**Source:** Google Drive (`chinese names/CNKI_datasets/CNKI_dataset_cleaned/v2/CNKI unique names/v2`)

