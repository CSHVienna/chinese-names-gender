# Files with names for the surveys

### 1. `Chinese first names gender inference (Responses).xlsx`

- **Description**: First round of survey responses (rows correspond to annotators, and columns to `givenname`)
- **Source**: [Google Form](https://docs.google.com/spreadsheets/d/1hmYyQawcp3eidKgrFCbzNUpacXfglAfXN4HBBXpSPn8/edit?usp=sharing)
- **Author**: Lisette Espin
- **Date**: 2025-05-09
- **Size**: 9.1 KB  
- **Records**: 11 x 100 
- **Note**: Sampled givennames by Ana Jaramillo, and data collection managed by Liu Yang and Mingrong She.

### 2. `Chinese first names gender inference (Responses) - round 2 (incomplete).xlsx`

- **Description**: Second round of survey responses (rows correspond to annotators, and columns to `givenname`)
- **Source**: [Google Form](https://docs.google.com/spreadsheets/d/1JdSROcfROEU35F-wZgOboW1HORT8FN1Ytr7PnoDO70I/edit?usp=sharing)
- **Author**: Lisette Espin
- **Date**: 2025-08-12 
- **Size**: 7.7 KB  
- **Records**: 8 x 48 
- **Note 1**: This dataset is incomplete. Waiting for 3 more responses (we should have 11 x 50)
- **Note 2**: Sampled givennames by Ana Jaramillo, and data collection managed by Liu Yang and Mingrong She.

### 3. `ground_truth.csv` 
- **Description**: Merge responses from all survey rounds. Rows correspond to all givennames used in the annotations, and columns the annotators plus two columns `single_char` (bool) and `source` (filename of the source file from Google forms). Source code in `prepare_ground_truth()` under `code/libs/tools_comparison.py`.
- **Author**: Lisette Espin
- **Date**: 2025-08-12
- **Size**: 148 x 13

### 4. `Names_test_multiple_character.csv` 
- **Description**: Names with multiple characters for the second round of surveys (rows correspond to `givenname`)
- **Author**: Ana Jaramillo
- **Date**: 2025-07-24
- **Size**: 2 x 1

### 5. Names_test_unique_character.csv`
- **Description**: Names with a unique character for the second round of surveys (rows correspond to `givenname`).
- **Author**: Ana Jaramillo
- **Date**: 2025-07-24
- **Size**: 46 x 1
