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
    if 'alternative_names' in data.columns:
        data.rename(columns={'alternative_names': 'alternative_names_original',
                             'longest_name': 'longest_name_original'}, inplace=True)
        data.loc[:, 'alternative_names'] = data.alternative_names_original.apply(lambda s: names.get_valid_names(s))
        
        def _merge_alternative_names(s):
            names = [
                name
                for val in s.dropna()
                for name in (val if isinstance(val, list) else [val])
                if isinstance(name, str)
            ]
            return list(set(names))

        def _merged_gender_list(s):
            return s.dropna().tolist()

        agg = (
            data
            .groupby("id_author", as_index=False)
            .agg(
                alternative_names=("alternative_names", _merge_alternative_names),
                genders=("gender", _merged_gender_list),
                genders_nq=("gender_nq", _merged_gender_list),
                gender_clean=("gender", lambda s: _resolve_gender(s)),
                gender_nq_clean=("gender_nq", lambda s: _resolve_gender(s)),
                longest_name=(
                    "alternative_names",
                    lambda s: max(
                        (
                            name
                            for val in s.dropna()
                            for name in (val if isinstance(val, list) else [val])
                            if isinstance(name, str)
                        ),
                        key=len,
                        default="",
                    ),
                ),
            )
        )
        agg["n_alternative_names"] = agg.alternative_names.apply(len)
        
    else:
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

    # remove parentheses (names written in both pinyin and chinese characters)
    agg.loc[:, 'longest_name_clean'] = agg.longest_name_clean.apply(lambda x: x.replace('(', '').replace(')', '').replace(';','').strip())

    return agg

def detect_pinyin_and_chinese_names(df_aps_clean):
    """
    Detect the pinyin and chinese names
    """

    # 3. split givenname and familyname
    df_aps_clean.loc[:, 'givenname_clean'] = df_aps_clean.longest_name_clean.apply(lambda x: x.split(' ')[0])
    df_aps_clean.loc[:, 'familyname_clean'] = df_aps_clean.longest_name_clean.apply(lambda x: x.split(' ')[-1])

    # 4. evaluate if the given name (first name) is a pinyin given name
    df_aps_clean.loc[:, 'is_pinyin_givenname'] = df_aps_clean.givenname_clean.apply(lambda x: pinyinnames_handler.is_pinyin_givenname(x))

    # 5. evaluate if the family name is a pinyin family name
    df_aps_clean.loc[:, 'is_pinyin_familyname'] = df_aps_clean.familyname_clean.apply(lambda x: pinyinnames_handler.is_pinyin_familyname(x))

    # 6. evaluate if the name is a pinyin name (using both signals from given and family names)
    df_aps_clean.loc[:, 'is_pinyin'] = df_aps_clean.apply(lambda x: x.is_pinyin_givenname != False and x.is_pinyin_familyname and len(x.longest_name_clean.split(' ')) == 2, axis=1)

    # 7. evaluate if the name is a chinese name (using ch_name)
    df_aps_clean.loc[:, 'ch_name'] = df_aps_clean.loc[:, 'longest_name'].apply(lambda x: pd.NA if '(' not in x or ')' not in x else x.split('(')[-1].split(')')[0].replace(';','').strip())
    df_aps_clean.loc[:, "ch_splitname"] = df_aps_clean.apply(lambda row: characternames_handler.split_names(row.ch_name) if pd.notna(row.ch_name) else pd.NA, axis=1)
    df_aps_clean.loc[:, "ch_familyname"] = df_aps_clean.ch_splitname.apply(lambda x: x[0].split(' ')[0] if isinstance(x, list) and len(x) == 1 else pd.NA)
    df_aps_clean.loc[:, "ch_givenname"]  = df_aps_clean.ch_splitname.apply(lambda x: x[0].split(' ')[1] if isinstance(x, list) and len(x) == 1 else pd.NA)
    
    # 8. translate the chinese name to pinyin
    df_aps_clean.loc[:, 'ch_translated_name_in_pinyin'] = df_aps_clean.apply(lambda row: pd.NA if pd.isna(row.ch_splitname) else characternames_handler.pinyinname(row.ch_splitname)[0], axis=1)
    df_aps_clean.loc[:, 'ch_translated_familyname_in_pinyin'] = df_aps_clean.ch_translated_name_in_pinyin.apply(lambda x: x.split(' ')[0] if isinstance(x, str) else pd.NA)
    df_aps_clean.loc[:, 'ch_translated_givenname_in_pinyin'] = df_aps_clean.ch_translated_name_in_pinyin.apply(lambda x: x.split(' ')[1] if isinstance(x, str) else pd.NA)
    
    return df_aps_clean