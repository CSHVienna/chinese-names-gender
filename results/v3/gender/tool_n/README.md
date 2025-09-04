# Data Tool N: Namsor

This repository contains datasets related to gender inference using the Namsor API and comparative annotations.

## Updates

- **2025-08-12**: Lisette Espin updated the header of `CNKI_uniqueChineseGivenNames_gender.csv` (keeping `script` and `id` lower case, and changing `givename` to `givenname`)
- **2025-07-21**: Ana M. Jaramillo updated the `CNKI_uniqueChineseGivenNames_gender.csv` file with new inference from Namsor
- **2025-07-04**: We use `CNKI_uniqueChineseGivenNames_gender.csv`



## Files

### 1. `CNKI_uniqueChineseGivenNames_gender.csv`

- **Source**: Curated after merging `Dict_gender_all.pkl` and `results/v3/names/chinese/CNKI_uniqueChineseGivenNames_CNKIpop.csv` (new disambiguation)
- **Author**: Lisette Espin, Ana M. Jaramillo
- **Date**: 2025-07-04  
- **Size**: 4.7 MB  
- **Records**: 39,712  
- **Columns**:  
  - `givenname`
  - `script`
  - `id`
  - `name`
  - `likelyGender`
  - `genderScale`
  - `score`
  - `probabilityCalibrated`


### 2. `Dict_gender_all.pkl` (deprecated)

- **Source**: Email attachment  
- **Author**: Ana M. Jaramillo  
- **Date**: 2025-11-28  
- **Size**: 7.2 MB  
- **Records**: 40,303  
- **Columns**:  
  - `givenname`  
  - `script`  
  - `id`  
  - `name`  
  - `likelyGender`  
  - `genderScale`  
  - `score`  
  - `probabilityCalibrated`


### 3. `DDBB_final.csv` (deprecated)

- **Source**: Google Drive (`v2/gender_comparison`)
- **Author**: Ana M. Jaramillo  
- **Date**: 2024-04-02  
- **Size**: 2.5 MB  
- **Records**: 39,712  
- **Columns**:  
  - `givenname`  
  - `Gender_Lisette`  
  - `Gender_Liu`  
  - `Gender_Ana`  
  - `probabilityCalibrated_Namzor`  
  - `Same_gender_all`  
  - `Same_gender_Liu_Lisette`  
  - `Same_gender_Ana_Lisette`  
  - `Same_gender_Liu_Ana`
- **Note**: Not used because it does not have all values from Namsor.


## Namsor Reference

Namsor API: [https://gender-guesser.com/api/#genderize-chinese-full-name-batch]([https://namsor.app/api-documentation/#genderize-chinese-name-batch](https://gender-guesser.com/api/#genderize-chinese-full-name-batch))  
- `genderScale`: Gender scale ranging from -1 (male) to +1 (female).  
- `score`: Higher implies a more reliable result; not normalized.  
- `probabilityCalibrated`: Higher implies a more reliable result; ranges from 0 to 1.
