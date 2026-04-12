# Plan: Infer Perceived Gender of Chinese Hanzi (character) Given Names

## Context
We need to evaluate a gender inference algorithm by creating a ground-truth column of perceived gender for Chinese Hanzi character given names. The file `aps_chinese_given_names.csv` contains 32 names that need gender labels.

## Approach

### Step 1: Read the CSV file
- Read `/code/espinl/chinese_name_project/chinese-names/results/v3/gender/predictions_ch_others/chinese_given_names.csv`
- Parse the single column `givenname` (note: file also has an unnamed index column)

### Step 2: Add `ch_gender_gpt` column
- For each row, infer the perceived gender of the pinyin given name
- Use one of four labels:
  - **male** — name is clearly or mostly male
  - **female** — name is clearly or mostly female
  - **neutral** — name is commonly used for either sex

### Step 3: Write the updated CSV
- Output the file with both original columns plus the new `ch_gender_gpt` column
- Save as a **new file**: `chinese_given_names_gendered.csv` in the same directory of this instruction file.

### Gender inference logic
GPT will use knowledge of Chinese naming conventions to assign gender.

### Step 4: Verification
- Check the output CSV has the correct number of rows
- Verify the column was added correctly
- Show distribution of gender labels (counts of male/female/neutral/unknown)

## Implementation
Process the CSV using Python (pandas), iterating through each name and applying GPT's gender inference inline during the conversation. Since this is 32 names, GPT will process them in batches, writing the gender for each name based on Chinese naming knowledge.
