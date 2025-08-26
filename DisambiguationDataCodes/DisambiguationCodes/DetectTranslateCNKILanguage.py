import pandas as pd
from deep_translator import single_detection
from tqdm import tqdm
import langid
from deep_translator import GoogleTranslator
import csv

def language_detect(text, api_key):
    """
    Detect language of a given text using deep_translator.
    """
    return single_detection(text, api_key=api_key)

    
def detect_languages(input_file, output_file, api_key):
    """
    Main function to process dataset:
    1. Load dataset
    2. Assign paper IDs
    3. Detect language with deep_translator
    4. Refine non-Chinese results with langid
    5. Save final dataset with 'Language' column
    """
    # Step 1. Load Dataset
    df = pd.read_csv(input_file)
    df_with_pid = df.rename_axis("paper id").reset_index()

    # Step 2. Build dictionaries
    pid_abstracts = dict(zip(df_with_pid["paper id"], df_with_pid["Abstract"]))
    pid_language = {}

    # Step 3. Detect language (deep_translator)
    for pid in tqdm(pid_abstracts):
        if pid not in pid_language:
            text = pid_abstracts[pid]
            if text != "无":
                lang = language_detect(text, api_key)
                pid_language[pid] = lang
            else:
                pid_language[pid] = "zh"

    # Step 4. Refine with langid if not zh
    pid_new_language = {}
    for pid, lang in tqdm(pid_language.items()):
        if lang != "zh":
            language, _ = langid.classify(pid_abstracts[pid])
            pid_new_language[pid] = language
        else:
            pid_new_language[pid] = "zh"

    # Step 5. Save results
    df_with_pid["Language"] = df_with_pid["paper id"].map(pid_new_language)
    df_with_pid.to_csv(output_file, index=False)

    return df_with_pid

##convert English into Chinese
def translate_to_zh(text):
    """
    Translate given text into Simplified Chinese using GoogleTranslator.
    """
    return GoogleTranslator(source="auto", target="zh-CN").translate(text=text)

def translate_dataset(input_file, output_file):
    """
    Translate all non-Chinese titles, abstracts, and keywords into Chinese,
    and save the final dataset into a new CSV.
    """
    # === Step 1. Load dataset ===
    df = pd.read_csv(input_file)

    # === Step 2. Extract non-Chinese entries ===
    non_chinese_df = df[df["Language"] != "zh"]
    NonChinese_Titles = non_chinese_df["Title"].to_list()
    NonChinese_Abstracts = non_chinese_df["Abstract"].to_list()
    NonChinese_KeyWords = non_chinese_df["Keywords"].to_list()

    # === Step 3. Translate ===
    Abstract_to_Chinese = {}
    for abs_text in tqdm(NonChinese_Abstracts, desc="Translating abstracts"):
        if abs_text not in Abstract_to_Chinese:
            Abstract_to_Chinese[abs_text] = translate_to_zh(abs_text)

    Titles_to_Chinese = {}
    for title in tqdm(NonChinese_Titles, desc="Translating titles"):
        if title not in Titles_to_Chinese:
            Titles_to_Chinese[title] = translate_to_zh(title)

    KeyWords_to_Chinese = {}
    for kw in tqdm(NonChinese_KeyWords, desc="Translating keywords"):
        if kw not in KeyWords_to_Chinese:
            KeyWords_to_Chinese[kw] = translate_to_zh(kw)

    # === Step 4. Write results into a new file ===
    with open(input_file, "r", encoding="utf-8") as f:
        with open(output_file, "w", encoding="utf-8", newline="") as fw:
            csv_write = csv.writer(fw)
            csv_write.writerow(df.columns.to_list())

            for i, row in enumerate(csv.reader(f)):
                if i != 0:
                    if row[10] == "zh":
                        csv_write.writerow(row)
                    else:
                        csv_write.writerow([
                            row[0],
                            Titles_to_Chinese[row[1]],
                            row[2],
                            row[3],
                            row[4],
                            row[5],
                            row[6],
                            KeyWords_to_Chinese[row[7]],
                            Abstract_to_Chinese[row[8]],
                            row[9],
                            row[10],
                        ])

