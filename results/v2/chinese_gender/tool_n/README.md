# Gender Data from NamSor

This folder contains gender prediction outputs from the *NamSor API*.

## Source

- Tool: [NamSor API](https://namsor.app/api-documentation/#genderize-chinese-name-batch)
- Output includes:
  - `genderScale`: from –1 (male) to +1 (female)
  - `score`: confidence level (not normalized)
  - `probabilityCalibrated`: probability from 0 (male) to 1 (female)

## Format

- All files are `.csv`
- Each file includes at least: `givenname` or `character`, and the three gender-related fields

## Notes

- `givenname`-based data was collected by @ana  
- `character`-based data was collected by @lisette-espin 
- Data is used for gender inference research
- **2024-12-17** — *lespin*: `CNKI_uniqueChineseGivenNames_gender.csv` was generated using `old/Dict_gender_all.pkl` and `regenerate_csv.py`
