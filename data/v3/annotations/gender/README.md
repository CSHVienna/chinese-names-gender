# Files with names for the surveys

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
- **Note 1**: [EXPLAIN CRITERIA OF SELECTION @aj]
- **Note 2**: Lisette Espin created the file based in the names in the responses from round 1.

### 6. `Names_test_multiple_character_round2.csv` 
- **Description**: Names with multiple characters for the second round of surveys (rows correspond to `givenname`)
- **Author**: Ana Jaramillo
- **Date**: 2025-07-24
- **Size**: 24 bytes
- **Records**: 2 x 1
- **Note**: [EXPLAIN CRITERIA OF SELECTION @aj]

### 7. `Names_test_multiple_character_round3.csv` 
- **Description**: Names with multiple characters for the third round of surveys (rows correspond to `givenname`)
- **Author**: Ana Jaramillo
- **Date**: 2025-08-13
- **Size**: 136 bytes
- **Records**: 18 x 1
- **Note**: [EXPLAIN CRITERIA OF SELECTION @aj]

### 8. `Names_test_unique_character.csv`
- **Description**: Names with a unique character for the second round of surveys (rows correspond to `givenname`).
- **Author**: Ana Jaramillo
- **Date**: 2025-07-24
- **Size**: 204 bytes
- **Records**: 46 x 1
- **Note**: [EXPLAIN CRITERIA OF SELECTION @aj]

### 9. `Names_test_unique_character.csv`
- **Description**: Names with a unique character for the second round of surveys (rows correspond to `givenname`).
- **Author**: Ana Jaramillo
- **Date**: 2025-07-24
- **Size**: 446 bytes
- **Records**: 64 x 1
- **Note 1**: [EXPLAIN CRITERIA OF SELECTION @aj]
- **Note 2**: Lisette Espin renamed the file to the current version (previously called: `Remaining_names_first_survey.csv`), and added the four single character names included in round 1.