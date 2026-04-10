import pandas as pd

from libs.utils import characternames_handler
from libs.utils import pinyinnames_handler
from libs.utils import names

def _resolve_gender(labels):
    """Resolve a list of gender labels to a single value. Treat '-' as unknown."""
    # Normalize: "-" -> "unknown", strip and lower
    normalized = [
        "unknown" if str(x).strip().lower() in ("-", "unknown", "") else str(x).strip().lower()
        for x in labels
    ]
    unique = set(normalized)
    if len(unique) == 1:
        return list(unique)[0]
    # Contradiction: both male and female present -> neutral
    if "male" in unique and "female" in unique:
        return "neutral"
    # Drop unknowns and pick the not-unknown
    not_unknown = [x for x in normalized if x != "unknown"]
    if len(not_unknown) == 0:
        # all are unknown
        return "unknown"
    
    if len(set(not_unknown)) == 1:
        # Only one not-unknown value (unknown tied with one known: female, male or neutral)
        return not_unknown[0]

    # Should not reach if logic is correct (we already handled male+female)
    return "neutral"


def clean_aps_names(df_aps):
    """
    Clean the APS names
    """

    # create new instance to modify
    data = df_aps.copy()
    
    # Aggregate: one row per id_author
    # if 'alternative_names' in data.columns:
    #     # alternative_names column exists only from openalex data (Saeta/Barolo)
    #     data.rename(columns={'alternative_names': 'alternative_names_original',
    #                          'longest_name': 'longest_name_original'}, inplace=True)
    #     data.loc[:, 'alternative_names'] = data.alternative_names_original.apply(lambda s: names.get_valid_names(s))
        
    #     def _merge_alternative_names(s):
    #         names = [
    #             name
    #             for val in s.dropna()
    #             for name in (val if isinstance(val, list) else [val])
    #             if isinstance(name, str)
    #         ]
    #         return list(set(names))

    #     def _merged_gender_list(s):
    #         return s.dropna().tolist()

    #     agg = (
    #         data
    #         .groupby("id_author", as_index=False)
    #         .agg(
    #             alternative_names=("alternative_names", _merge_alternative_names),
    #             genders=("gender", _merged_gender_list),
    #             genders_nq=("gender_nq", _merged_gender_list),
    #             gender_clean=("gender", lambda s: _resolve_gender(s)),
    #             gender_nq_clean=("gender_nq", lambda s: _resolve_gender(s)),
    #             longest_name=(
    #                 "alternative_names",
    #                 lambda s: max(
    #                     (
    #                         name
    #                         for val in s.dropna()
    #                         for name in (val if isinstance(val, list) else [val])
    #                         if isinstance(name, str) and '-' not in name
    #                     ),
    #                     key=len,
    #                     default="",
    #                 ),
    #             ),
    #         )
    #     )
    #     agg["n_alternative_names"] = agg.alternative_names.apply(len)
        
    # else:

    # alternative_names column does not exist in Karimi/vanBuskirk data
    agg = (
        data
        .groupby("id_author", as_index=False)
        .agg(
            alternative_names=("name", lambda s: s.dropna().unique().tolist()),
            n_alternative_names=("name", "size"),
            genders=("gender", lambda s: s.dropna().tolist()),
            genders_nq=("gender_nq", lambda s: s.dropna().tolist()),
            gender_clean=("gender", _resolve_gender),
            gender_nq_clean=("gender_nq", _resolve_gender),
            longest_name=("name", lambda s: max(s.dropna(), key=len)),
        )
    )

    # 1. lower-case
    agg.loc[:, 'longest_name_clean'] = agg.longest_name.str.lower()

    # 2. remove accent
    agg.loc[:, 'longest_name_clean'] = agg.longest_name_clean.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

    # remove parentheses (names written in both pinyin and chinese characters; keep pinyin)
    agg.loc[:, 'longest_name_clean'] = agg.longest_name_clean.apply(lambda x: x.replace('(', '').replace(')', '').replace(';','').strip())

    return agg

def detect_pinyin_and_chinese_names(df_aps_clean):
    """
    Detect the pinyin and chinese names
    """


    def _get_ch_py_givenname(row):

        py_givenname = pd.NA
        py_surname = pd.NA
        ch_givenname = pd.NA
        ch_surname = pd.NA
        py = pd.NA
        ch = pd.NA
        has_py_ch = False
        py_givenname_original = pd.NA

        # does it contain pinyin and chinese characters?
        might_be_py_ch = '(' in row.longest_name and ')' in row.longest_name

        if might_be_py_ch:

            # from data
            py = row.longest_name_clean.strip()
            first_py = py.split(' ')[0].strip()
            last_py = py.split(' ')[-1].strip()
            
            ch = row.longest_name.split('(')[-1].split(')')[0].replace(';','').strip()
            first_ch = ch[0].strip()
            last_ch = ch[-1].strip()
            tmp_ch = f"{first_ch} {last_ch}" # surname  givenname (in chinese chars)

            # from transliteration
            py_tmp_ch = characternames_handler.pinyinname([tmp_ch])[0] if characternames_handler.is_chinese_char(first_ch) and characternames_handler.is_chinese_char(last_ch) else pd.NA
            py_first_ch = py_tmp_ch.split(' ')[0] if isinstance(py_tmp_ch, str) else pd.NA # surname
            py_last_ch = py_tmp_ch.split(' ')[1] if isinstance(py_tmp_ch, str) else pd.NA # givenname
            
            # surname is always one character (or one pinyin) in Chinese names, so we can use the first and last character to check if they match with the pinyin name.

            if pd.notna(py_tmp_ch):

                if first_py == last_py and len(py.split(' ')) == 2:
                    # if the pinyin name has only two characters and they are the same, 
                    # then it is not clear which one is the firt and last name.
                    py_givenname = pd.NA
                    py_surname = pd.NA
                    ch_givenname = pd.NA
                    ch_surname = pd.NA
                    py_ch_givenname = pd.NA
                    has_py_ch = False

                elif first_py == py_first_ch:
                    # first part is the surname, then return the rest as givenname
                    py_givenname = ' '.join(py.split(' ')[1:]) #py.replace(first_py, '').strip()
                    py_surname = py.split(' ')[0] #first_py
                    ch_givenname = ch.replace(first_ch, '').strip()
                    ch_surname = first_ch
                    py_ch_givenname = characternames_handler.pinyin_givenname([ch.replace(first_ch,'')])[0]
                    has_py_ch = True

                elif last_py == py_last_ch:
                    # last part is the surname, then return the rest as givenname
                    py_givenname = ' '.join(py.split(' ')[:-1]) #py.replace(last_py, '').strip()
                    py_surname = py.split(' ')[-1] #last_py
                    ch_givenname = ch.replace(last_ch, '').strip()
                    ch_surname = last_ch
                    py_ch_givenname = characternames_handler.pinyin_givenname([ch.replace(last_ch,'')])[0]
                    has_py_ch = True

                elif last_py == py_first_ch:
                    # last part (in py) and first part (in ch) is the surname  --> return the opposite
                    py_givenname = ' '.join(py.split(' ')[:-1]) #py.replace(last_py, '').strip()
                    py_surname = py.split(' ')[-1] #last_py
                    ch_givenname = ch.replace(first_ch, '').strip()
                    ch_surname = first_ch
                    py_ch_givenname = characternames_handler.pinyin_givenname([ch.replace(first_ch,'')])[0]
                    has_py_ch = True

                elif first_py == py_last_ch:
                    # first part (in py) and last part (in ch) is the surname --> return the opposite
                    py_givenname = ' '.join(py.split(' ')[1:]) # py.replace(first_py, '').strip()
                    py_surname = py.split(" ")[0] #first_py
                    ch_givenname = ch.replace(last_ch, '').strip()
                    ch_surname = last_ch
                    py_ch_givenname = characternames_handler.pinyin_givenname([ch.replace(last_ch,'')])[0]
                    has_py_ch = True

                else:
                    py_givenname = pd.NA
                    py_surname = pd.NA
                    ch_givenname = pd.NA
                    ch_surname = pd.NA
                    py_ch_givenname = pd.NA
                    has_py_ch = False

                if pd.notna(py_givenname) and '-' in py_givenname:
                    py_givenname = pd.NA
                    py_surname = pd.NA
                    ch_givenname = pd.NA
                    ch_surname = pd.NA
                    py_ch_givenname = pd.NA
                    has_py_ch = False

                # original: not transliterated from chinese name
                py_givenname_original = py_givenname
                if pd.notna(py_ch_givenname):
                # if pd.notna(py_givenname) and '.' in py_givenname:
                    py_givenname = py_ch_givenname
               
        #print(f"py: {py}, py_givenname: {py_givenname}, py_surname: {py_surname} |  ch: {ch}, ch_givenname: {ch_givenname}, ch_surname: {ch_surname}")
        return py_givenname_original, py_givenname, py_surname, ch_givenname, ch_surname, has_py_ch
        
    df = df_aps_clean.copy()
    df.loc[:, 'might_be_py_ch'] = df.longest_name.apply(lambda x: '(' in x and ')' in x)
    index = df.query('might_be_py_ch == True').index

    tmp = df.loc[index,:].apply(lambda row: _get_ch_py_givenname(row), axis=1, result_type='expand')
    tmp.rename(columns={0: 'py_givenname_original', 
                        1: 'py_givenname', 
                        2: 'py_familyname', 
                        3: 'ch_givenname', 
                        4: 'ch_familyname', 
                        5: 'has_py_ch'}, inplace=True)

    return df.join(tmp, how='left').drop(columns=['might_be_py_ch'])
