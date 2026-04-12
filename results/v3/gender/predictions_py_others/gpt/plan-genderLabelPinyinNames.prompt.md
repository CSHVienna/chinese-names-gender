## Plan: Gender Label Pinyin Names

Create a new CSV at `/code/espinl/chinese_name_project/chinese-names/results/v3/gender/predictions_py_others/gpt/aps_pinyin_given_names_gendered.csv` by reading `/code/espinl/chinese_name_project/chinese-names/results/v3/gender/predictions_py_others/aps_pinyin_given_names.csv`, preserving the existing rows/index structure, and adding a `py_gender_gpt` column inferred from Chinese pinyin given-name conventions. The work stays strictly inside the `results/v3/gender/predictions_py_others` area and follows the four-label scheme in the instruction file.

**Steps**
1. Inspect `/code/espinl/chinese_name_project/chinese-names/results/v3/gender/predictions_py_others/aps_pinyin_given_names.csv` and confirm the exact schema to preserve: unnamed index column plus `py_givenname`, with 632 data rows.
2. Implement a local processing script or notebook step inside `/code/espinl/chinese_name_project/chinese-names/results/v3/gender/predictions_py_others/gpt` that reads the CSV with pandas while preserving the existing index column format.
3. Define the allowed output labels exactly as `male`, `female`, `neutral`, and `unknown`, and encode the inference rule as knowledge-based classification of each pinyin given name.
4. Process every `py_givenname` value and assign `py_gender_gpt` using Chinese naming knowledge: clearly gendered names map to `male` or `female`, commonly cross-gender names map to `neutral`, and empty, malformed, or unresolvable names map to `unknown`.
5. Write the full result to `/code/espinl/chinese_name_project/chinese-names/results/v3/gender/predictions_py_others/gpt/aps_pinyin_given_names_gendered.csv` without overwriting the source file. This step depends on steps 2 to 4.
6. Verify the output by checking row count parity with the source, confirming the new column exists, and summarizing the counts of `male`, `female`, `neutral`, and `unknown`. This step depends on step 5.

**Relevant files**
- `/code/espinl/chinese_name_project/chinese-names/results/v3/gender/predictions_py_others/gpt/gpt_instructions.md` — task contract, label set, destination path, and verification expectations.
- `/code/espinl/chinese_name_project/chinese-names/results/v3/gender/predictions_py_others/aps_pinyin_given_names.csv` — source data with unnamed index column and `py_givenname` values.
- `/code/espinl/chinese_name_project/chinese-names/results/v3/gender/predictions_py_others/gpt/aps_pinyin_given_names_gendered.csv` — required output file to create.

**Verification**
1. Read the first rows of the output CSV and confirm the schema includes the original columns plus `py_gender_gpt`.
2. Confirm the output row count matches the source row count of 632.
3. Compute a frequency table for `py_gender_gpt` and confirm every value is one of the four allowed labels.
4. Spot-check ambiguous names such as `yu`, `jing`, `yi`, and `xin` to ensure they are treated consistently with the neutral/unknown rules in the instructions.

**Decisions**
- Included scope: only the files under `/code/espinl/chinese_name_project/chinese-names/results/v3/gender/predictions_py_others` that are explicitly required for this task.
- Excluded scope: no reads from other project folders, no reuse of code or outputs from sibling or legacy directories.
- Recommended implementation choice: use a small pandas script in the `gpt` folder so the transformation is reproducible and the verification can be rerun.
- Interpretation rule: when confidence is weak but the name is still a recognized cross-gender pinyin given name, prefer `neutral`; reserve `unknown` for blank, malformed, or genuinely unresolvable cases.

**Further Considerations**
1. If repeated runs are expected, keep the classification logic in a dedicated script in the `gpt` folder rather than performing manual one-off edits to the CSV.
2. If the user wants higher auditability later, add a separate rationale column in a follow-up task rather than expanding scope here.
