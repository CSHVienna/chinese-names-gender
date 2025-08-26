import pandas as pd
from deep_translator import GoogleTranslator
import ast
import csv
from tqdm import tqdm


def translate_to_en(text):
    """
    Translate text into English using GoogleTranslator.
    """
    return GoogleTranslator(source="auto", target="en").translate(text=text)

def translate_text_list(data_list):
    """
    Translate a list of texts into English and return a mapping dict.
    """
    translations = {}
    for item in tqdm(data_list):
        if item not in translations:
            translations[item] = translate_to_en(item)
    return translations

def build_paper_translation(df_cn_file, df_lang_file):
    """
    Build a unified Paper_Translation.csv containing both
    CN->EN translations and EN->CN alignments.
    """
    # --- Step 1. Read datasets ---
    df_cn = pd.read_csv(df_cn_file)
    df_lang = pd.read_csv(df_lang_file)
    # df_cn = pd.read_csv("CNKI_dataset_updated_withValidNames_withPID_FullChinese.csv")
    # df_lang = pd.read_csv("CNKI_dataset_updated_withValidNames_withPID_withLanguage.csv")

    # --- Step 2. Extract EN originals with Chinese translations (from FullChinese file) ---
    non_chinese_en = df_lang[df_lang["Language"] == "en"][["paper id", "Title", "Abstract", "Keywords"]]
    non_chinese_en_cn = df_cn[df_cn["Language"] == "en"][["paper id", "Title", "Abstract", "Keywords"]]

    en_cn = non_chinese_en.merge(non_chinese_en_cn, on="paper id", suffixes=("_English", ""))
    en_cn = en_cn.drop("paper id", axis=1)

    # --- Step 3. Extract CN originals with EN translations (Google translate) ---
    non_english = df_lang[df_lang["Language"] != "en"]
    titles_to_en = translate_text_list(non_english["Title"].to_list())
    abstracts_to_en = translate_text_list(non_english["Abstract"].to_list())
    keywords_to_en = translate_text_list(non_english["Keywords"].to_list())

    cn_en_records = []
    for _, row in non_english.iterrows():
        cn_en_records.append([
            row["Title"],
            row["Abstract"],
            row["Keywords"],
            titles_to_en[row["Title"]],
            abstracts_to_en[row["Abstract"]],
            keywords_to_en[row["Keywords"]],
        ])

    cn_en = pd.DataFrame(cn_en_records, columns=[
        "Title", "Abstract", "Keywords",
        "Title_English", "Abstract_English", "Keywords_English"
    ])

    # --- Step 4. Combine ---
    df_combined = pd.concat([en_cn, cn_en])
    df_combined = df_combined.drop_duplicates()
    df_combined.to_csv("Paper_Translation.csv", index=False, encoding="utf-8")
    return df_combined

def create_authorship_english(inputfile, output_file, cn_names_file, df_cn_file,
                              df_lang_file):
    """
    Create an English version of the authorship dataset by replacing
    names, departments, titles, abstracts, and keywords with English translations.
    Parameters
    ----------
    inputfile : str
        Path to Authorship_ValidChinese.csv
    output_file : str
        Path to save Authorship_English.csv
    cn_names_file : str
        Path to Chinese->Pinyin name mapping file; CNKI_uniqueChineseNames_withPinyin
    df_cn_file : str
        Path to CNKI full Chinese dataset; CNKI_dataset_updated_withValidNames_withPID_FullChinese
    df_lang_file : str
        Path to CNKI dataset with language info; CNKI_dataset_updated_withValidNames_withPID_withLanguage
    """
    # --- Step A. Departments translation ---
    authorship = pd.read_csv(inputfile)
    authorship["department"] = authorship["department"].apply(ast.literal_eval)
    departments = authorship["department"].to_list()
    departments_cn = set([d for dep in departments for d in dep])
    depart_cn_en = translate_text_list(departments_cn)

    # --- Step B. Names translation (Chinese->Pinyin English) ---
    cn_names = pd.read_csv(cn_names_file)
    # cn_names = pd.read_csv("Pinyin/CNKI_uniqueChineseNames_withPinyin.csv")
    name_cn_en = dict(zip(cn_names["fullname"], cn_names["fullname_english"]))

    # --- Step C. Paper translations (now includes both EN-CN and CN-EN) ---
    paper_translations = build_paper_translation()
    title_cn_en = dict(zip(paper_translations["Title"], paper_translations["Title_English"]))
    abstract_cn_en = dict(zip(paper_translations["Abstract"], paper_translations["Abstract_English"]))
    keywords_cn_en = dict(zip(paper_translations["Keywords"], paper_translations["Keywords_English"]))

    # --- Step D. Create English authorship ---
    with open(inputfile, "r", encoding="utf-8") as f:
        with open(output_file, "w", encoding="utf-8", newline="") as fw:
            csv_write = csv.writer(fw)
            csv_head = ["paper id", "title", "author_name", "department", "coauthors", "publish_time", "abstract", "keywords", "journal"]
            csv_write.writerow(csv_head)

            for i, row in tqdm(enumerate(csv.reader(f))):
                if i == 0:  # skip header
                    continue

                title_en = title_cn_en.get(row[1], row[1])
                author_name_en = name_cn_en.get(row[2], row[2])

                # Departments
                department_en = []
                if row[3]:
                    row[3] = ast.literal_eval(row[3])
                    department_en = [depart_cn_en.get(dep, dep) for dep in row[3] if dep != ""]

                # Co-authors
                coauthors_en = []
                if row[4]:
                    row[4] = ast.literal_eval(row[4])
                    coauthors_en = [name_cn_en.get(name, name) for name in row[4]]

                abstract_en = abstract_cn_en.get(row[6], row[6])
                keywords_en = keywords_cn_en.get(row[7], row[7])

                csv_write.writerow([
                    row[0], title_en, author_name_en, department_en,
                    coauthors_en, row[5], abstract_en, keywords_en, row[8]
                ])




