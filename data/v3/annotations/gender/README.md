# Annotations (ground-truth)

## Sampling Criteria

### Single-character given names: 50 names 

* `False_female_r = 10` Not the same gender between the `R` and `N` tools, but inferred female by `R`
* `False_male_r = 10` Not the same gender between the `R` and `N` tools, but inferred male by `R`
* `True_female = 10` The same gender between the `R` and `N` tools, it is inferred `female`
* `True_male = 10` The same gender between the `R` and `N` tools, it is inferred `male`
* `Non_found_character = 10` Any of the tools did not find these characters/names

**NOTE:** Comparison only between `N` and `R` because `W` doesn't not work for single-char given names (always returns `male`)

### Two-character given names: 80 names

* Group of `female` given names that **do NOT** agree between the three tools (`R`, `N`, `W`), but with agreement between two tools represented by the letters in the end (e.g., nr = Namsor & R)

    - `False_male_rw = 10`
    - `False_male_nw = 10`
    - `False_male_nr = 10`

* Group of `male` given names that **do NOT** agree between the three tools (`R`, `N`, `W`), but with agreement between two tools represented by the letters in the end (e.g., nr = Namsor & R)

    - `False_female_rw = 10`
    - `False_female_nw = 10`
    - `False_female_nr = 10`

* Group of `female` and `male` given names that **do AGREE** between the three tools:

    - `True_female = 10`
    - `True_male = 10`

**NOTE:** This `80` given names are a subset of the `116` given names sampled in total. The `36` names removed were to balanced the remaining sample across `female` and `male` given names according to the three tools.

### Two-character given names: 116 names
@TODO @AJ

## Files 

### 1. `Chinese first names gender inference (Responses) - round 1.xlsx`

- **Description**: First round of survey responses (rows correspond to annotators, and columns to `givenname`)
- **Source**: [Google Form](https://docs.google.com/spreadsheets/d/1hmYyQawcp3eidKgrFCbzNUpacXfglAfXN4HBBXpSPn8/edit?usp=sharing)
- **Author**: Lisette Espin
- **Date**: 2025-05-09
- **Size**: 9.1 KB  
- **Records**: 11 x 100 
- **Note**: Sampled givennames by Ana Jaramillo, and data collection managed by Liu Yang and Mingrong She.

### 2. `Chinese first names gender inference (Responses) - round 2.xlsx`

- **Description**: Second round of survey responses (rows correspond to annotators, and columns to `givenname`)
- **Source**: [Google Form](https://docs.google.com/spreadsheets/d/1JdSROcfROEU35F-wZgOboW1HORT8FN1Ytr7PnoDO70I/edit?usp=sharing)
- **Author**: Lisette Espin
- **Date**: 2025-08-12 
- **Size**: 8.2 KB  
- **Records**: 11 x 48 
- **Note**: Sampled givennames by Ana Jaramillo, and data collection managed by Liu Yang, Mingrong She, and Lisette Espin.

### 3. `Chinese first names gender inference (Responses) - round 3.xlsx`

- **Description**: Third round of survey responses (rows correspond to annotators, and columns to `givenname`)
- **Source**: [Google Form](https://docs.google.com/spreadsheets/d/1UUOgSeLa3qKIpfUs9m9NpXhZRDf4qHCENZpQno4OXLw/edit?usp=sharing)
- **Author**: Lisette Espin
- **Date**: 2025-08-16 
- **Size**: 6.9 KB  
- **Records**: 11 x 18 
- **Note**: Sampled givennames by Ana Jaramillo, and data collection managed by Liu Yang, Mingrong She, and Lisette Espin.

### 4. `ground_truth.csv` 
- **Description**: Merge responses from all survey rounds. Rows correspond to all givennames used in the annotations, and columns the annotators plus two columns `single_char` (bool) and `round` (round of data collection: 1, 2, or 3). Source code in `prepare_ground_truth()` under `code/libs/tools_comparison.py`.
- **Author**: Lisette Espin
- **Date**: 2025-08-16
- **Size**: 7.4 KB
- **Records**: 130 x 14
- **Note**: This file is the final GT.

### 5. `Names_test_multiple_character_round1.csv` 
- **Description**: Names with multiple characters for the first round of surveys (rows correspond to `givenname`)
- **Author**: Ana Jaramillo
- **Date**: 2025-09-16
- **Size**: 797 bytes
- **Records**: 100 x 1
- **Note 1**: *See sampling criteria above*
- **Note 2**: Lisette Espin created the file based in the names in the responses from round 1.

### 6. `Names_test_multiple_character_round2.csv` 
- **Description**: Names with multiple characters for the second round of surveys (rows correspond to `givenname`)
- **Author**: Ana Jaramillo
- **Date**: 2025-07-24
- **Size**: 24 bytes
- **Records**: 2 x 1
- **Note**: *See sampling criteria above*

### 7. `Names_test_multiple_character_round3.csv` 
- **Description**: Names with multiple characters for the third round of surveys (rows correspond to `givenname`)
- **Author**: Ana Jaramillo
- **Date**: 2025-08-13
- **Size**: 136 bytes
- **Records**: 18 x 1
- **Note**: *See sampling criteria above*

### 8. `Names_test_unique_character_round2.csv`
- **Description**: Names with a unique character for the second round of surveys (rows correspond to `givenname`).
- **Author**: Ana Jaramillo
- **Date**: 2025-07-24
- **Size**: 204 bytes
- **Records**: 46 x 1
- **Note**: *See sampling criteria above*

### 9. `Names_to_keep_round1.csv`
- **Description**: Names that should remain from ``Chinese first names gender inference (Responses).''
- **Author**: Ana Jaramillo
- **Date**: 2025-07-24
- **Size**: 446 bytes
- **Records**: 64 x 1
- **Note 1**: *See sampling criteria above*
- **Note 2**: Lisette Espin renamed the file to the current version (previously called: `Remaining_names_first_survey.csv`), and added the four single character names included in round 1.