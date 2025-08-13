# Files with names for the surveys

- **Chinese first names gender inference (Responses).xlsx**: First round of surveys.
- **Names_test_multiple_character.csv**: Names with multiple characters for the second round of surveys (2 names).  
- **Names_test_multiple_character_2.csv**: Names with multiple characters for the third round of surveys (3 names).  
- **Names_test_unique_character.csv**: Names with a unique character for the second round of surveys (46 names).
- **Remaining_names_first_survey.csv**: Names that should remain from ``Chinese first names gender inference (Responses).'' (60 names).


In **Selecting a random sample.py** is the code used to select the random names for the surveys:
- In total, we chose 10 names per group:
- Names with one character: 50 names
-   False_female_r = 10 (Not the same gender between the r-tool and namsor, but inferred female by the r-tool)
-   False_male_r = 10 (Not the same gender between the r-tool and Namsor, but inferred male by the r-tool)
-   True_female = 10 (The same gender between the r-tool and Namsor, it is inferred female)
-   True_male = 10 (The same gender between the r-tool and Namsor, it is inferred male)
-   Non_found_character = 10 (Any of the tools did not find these characters/names)

- Names with multiple characters: 80 names
Group of female names that do NOT agree between the three tools, but with agreement between two tools represented by the letters in the end (e.g., nr=namzor-tool_r)
-  False_male_rw = 10
-  False_male_nw = 10
-  False_male_nr = 10


Group of male names that do NOT agree between the three tools, but with agreement between two tools represented by the letters in the end (e.g., nr=namzor-tool_r)
-  False_female_rw = 10
-  False_female_nw = 10
-  False_female_nr = 10

Group of female and male names that do AGREE between the three tools:
-  True_female = 10
- True_male = 10


