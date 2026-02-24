import re

from libs.utils import pinyinnames_handler

def _has_chinese(s):
    """True if string contains at least one CJK character."""
    return bool(re.search(r"[\u4e00-\u9fff]", s))

def _extract_name_from_bio(s):
    """If string looks like a bio/description (e.g. 'Title Given Family is a Professor in Institute...'), extract just the name.
    Returns 'LATIN (CJK)' or 'LATIN'; otherwise returns s unchanged."""
    t = s.strip()
    if len(t) < 40 or (" is " not in t and " in " not in t):
        return s
    # Optional title, then 2–3 Western words, then optional CJK (2–4 chars); stop before " is " / " in " / comma / more CJK / end
    leading_title = r"(?:dr\.?|prof\.?|professor\s+)?\s*"
    western_name = r"([a-z]+(?:\s+[a-z]+){1,2})"
    optional_cjk = r"(?:\s+([\u4e00-\u9fff]{2,4}))?"
    lookahead = r"(?=\s+is\s|\s+in\s|,|\s*$|/|\s+[\u4e00-\u9fff])"
    m = re.match(r"^" + leading_title + western_name + optional_cjk + lookahead, t, re.IGNORECASE)
    if not m:
        return s
    name, cjk = m.group(1).strip(), m.group(2)
    if cjk and cjk.strip():
        return f"{name} ({cjk.strip()})"
    return name

def _normalize_dashes(s):
    """Replace all dash variants (Unicode hyphen, en-dash, em-dash, minus sign) with ASCII '-'."""
    return re.sub(r"[\-\‐\–\—\−]", "-", s)

def _is_likely_non_name(s):
    """Reject institutional, role, or email-like strings."""
    if not s or len(s.strip()) < 2:
        return True
    # Allow our formatted CJK-only name: (CJK) is valid
    m = re.match(r"^\s*\(\s*([^)]+)\s*\)\s*$", s.strip())
    if m and m.group(1).strip() and all("\u4e00" <= c <= "\u9fff" for c in m.group(1).strip()):
        return False
    lower = _normalize_dashes(s.lower())

    non_name_indicators = [
        r"^electronic\s+address",
        r"email\s*\(.*\)",
        r"@",
        r"twitter",
        r"\(former\s+",
        r"directeur\s+de\s+recherche",
        r"von\s+humboldt\s+fellow",
        r"university\s*\(",
        r"visiting\s+researcher",
        r"^\s*\([^)]*\)\s*$",  # entire string is one parenthetical (but CJK-only already allowed above)
        r"-->",
        r"\(dot\)|\(at\)",
        r"^\d+$",
        r"assistant\s+professor",
        r"associate\s+professor",
        r"president\s+of\s+(?:the\s+)?",
        r"senior\s+scientist\s+at\s+",
        r"research\s+officer\s+",
        r"research\s+fellow\s+",
        r"meaning\s+that\s+",
        r"given\s+by\s+myself",
        r"\)[a-z]",  # (X)word compound: no space after ); (Middle) Last has space so kept 
    ]
    for pat in non_name_indicators:
        if re.search(pat, lower, re.IGNORECASE):
            return True
    return False

def _remove_titles(s):
    """Strip academic/professional titles from the start (and after leading spaces)."""
    # Titles at start; M.A./B.A. only when followed by space/dot/end so we don't strip a given-name prefix
    title_pattern = (
        r"^(?:\s*(?:Prof\.?|PhD\.?|Dr\.?|MPhil\.?|Ph\.D\.?|M\.Sc\.?|B\.Sc\.?|"
        r"M\.?A\.?(?=[\s.]|$)|B\.?A\.?(?=[\s.]|$)|M\.?Eng\.?|Sr\.?|Jr\.?|Mr\.?|Mrs\.?|Ms\.?|Md\.?(?=[\s.]|$))\s*\.?\s*)+"
    )
    return re.sub(title_pattern, "", s, flags=re.IGNORECASE).strip()

def _paren_content_is_non_name(content):
    """True if content looks like a title, description, or not a person's name."""
    c = content.strip()
    if len(c) > 50:
        return True
    lower = c.lower()
    # Initials only: single letter (optional dot) or 2–3 all-uppercase letters (e.g. AB), not a 2-letter given name
    if re.match(r"^[A-Za-z]\.?$", c) or re.match(r"^[A-Z]{2,3}$", c):
        return True
    non_name = [
        r"^(?:native\s+)?alphabet$",
        r"years?\s+ago",
        r"male\s+\w+\s+name",
        r"female\s+\w+\s+name",
        r"directeur|director|researcher|fellow|prof\.?|dr\.?|phd",
        r"former\s+",
        r"university\s*\(|institute\s*\(",
    ]
    for pat in non_name:
        if re.search(pat, lower):
            return True
    return False

def _normalize_parens(s):
    """Remove parentheses (and content) only when: at start/end AND content is not Chinese; or in middle AND content is not a name (title/description). Otherwise keep parentheses."""
    s = s.strip()

    # 1. Leading parenthesis: remove only if content is NOT Chinese
    lead = re.match(r"^\s*\(\s*([^)]+)\s*\)(.*)$", s, re.DOTALL)
    if lead:
        content, rest = lead.group(1).strip(), lead.group(2).strip()
        if not _has_chinese(content):
            s = rest
        s = s.strip()

    # 2. Trailing parenthesis: ensure space before "(" if missing; remove only if content is NOT Chinese
    s = re.sub(r"([^\s])\s*\(\s*([^)]+)\s*\)\s*$", r"\1 (\2)", s)
    trailing = re.search(r"\s*\(\s*([^)]+)\s*\)\s*$", s)
    if trailing:
        content = trailing.group(1).strip()
        if not _has_chinese(content):
            s = s[: trailing.start()].strip()

    # 3. Middle parentheses: Chinese → keep "(CJK)"; non-name (titles, etc.) → remove "(...)"; name → drop parens, keep content
    def replace_middle(m):
        content = m.group(1).strip()
        if _has_chinese(content):
            return m.group(0)  # keep "(CJK)"
        if _paren_content_is_non_name(content):
            return " "  # remove "(non-name)" entirely
        return " " + content + " "  # middle name: remove parens, keep content

    s = re.sub(r"\(\s*([^)]+)\s*\)", replace_middle, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _format_chinese_only(s):
    """Fully Chinese name -> (CJK). E.g. CJK only or CJK (native alphabet) -> (CJK)."""
    if re.match(r"^\s*\([^)]*\)\s*$", s):
        inner = s.strip()[1:-1].strip()
        if inner and all("\u4e00" <= c <= "\u9fff" for c in inner):
            return s.strip()  # already (CJK)
    core = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()  # drop trailing (native alphabet) etc.
    core = re.sub(r"\s+", "", core)
    if core and all("\u4e00" <= c <= "\u9fff" for c in core):
        return f"({core})"
    return s

def _fix_reversed_chinese_western(s):
    """CJK(LATIN) -> LATIN (CJK). E.g. CJK(GIVEN FAMILY) -> GIVEN FAMILY (CJK). Western part keeps original casing."""
    t = s.strip()
    m = re.match(r"^([\u4e00-\u9fff]+)\s*\(\s*([A-Za-z\s\-]+)\s*\)\s*$", t)
    if m:
        ch, western = m.group(1).strip(), m.group(2).strip()
        return f"{western} ({ch})"
    return s

def _fix_chinese_then_latin(s):
    """CJK + space + LATIN -> LATIN (CJK)."""
    # Pattern: Chinese chars, then space(s), then Latin/pinyin (letters, spaces, hyphens)
    m = re.match(r"^([\u4e00-\u9fff]+)\s+([A-Za-z\s\-‐']+)$", s.strip())
    if m:
        ch, latin = m.group(1).strip(), m.group(2).strip()
        if latin:
            return f"{latin.lower()} ({ch})"
    return s

def _fix_alternating_western_chinese(s):
    """Alternating Western and CJK: 'W1 W2 (CJK1CJK2)' or 'W1 W2 W3 (CJK1CJK2)' when first Western part is multi-word."""
    t = s.strip()
    # Pattern: (one or more Western words) CJK (one Western word) CJK  -> W1 W2 CJK1 W3 CJK2 -> W1 W2 W3 (CJK1CJK2)
    m = re.match(
        r"^([a-z]+(?:\s+[a-z\-‐']+)*)\s+([\u4e00-\u9fff]+)\s+([a-z\-‐']+)\s+([\u4e00-\u9fff]+)\s*$",
        t,
        re.IGNORECASE,
    )
    if m:
        w1, ch1, w2, ch2 = m.group(1).strip(), m.group(2).strip(), m.group(3).strip(), m.group(4).strip()
        return f"{w1} {w2} ({ch1}{ch2})"
    # Pattern: single word, CJK, single word, CJK (original)
    m = re.match(
        r"^([A-Za-z\-‐']+)\s*([\u4e00-\u9fff]+)\s*([A-Za-z\-‐']+)\s*([\u4e00-\u9fff]+)\s*$",
        t,
    )
    if m:
        w1, ch1, w2, ch2 = m.group(1).strip(), m.group(2).strip(), m.group(3).strip(), m.group(4).strip()
        return f"{w1} {w2} ({ch1}{ch2})"
    return s

def _fix_western_order_two_words(s):
    """When string is 'W1 W2 (CJK)' with exactly 2 Western words: if W1 is pinyin family name, swap to 'W2 W1 (CJK)'."""
    m = re.match(r"^(\S+)\s+(\S+)\s+(\(\s*[^)]+\s*\))\s*$", s.strip())
    if not m:
        return s
    w1, w2, paren_part = m.group(1), m.group(2), m.group(3)
    if not _has_chinese(paren_part):
        return s
    try:
        gn = pinyinnames_handler.is_pinyin_givenname(w2) != False or len(w2) == 1 or (len(w2) > 2 and '-' in w2) or (len(w2) == 2 and '.' in w2)
        ln = pinyinnames_handler.is_pinyin_familyname(w1)
        if gn and ln:
            return f"{w2} {w1} {paren_part}"
    except Exception:
        pass
    return s

def _swap_if_first_word_pinyin_familyname(s):
    """When string has exactly 2 words: swap to W2 W1 only if first is pinyin family AND second is pinyin given."""
    m = re.match(r"^(\S+)\s+(\S+)\s*$", s.strip())
    if not m:
        return s
    w1, w2 = m.group(1), m.group(2)
    try:
        gn = pinyinnames_handler.is_pinyin_givenname(w2) != False or len(w2) == 1 or (len(w2) > 2 and '-' in w2) or (len(w2) == 2 and '.' in w2)
        ln = pinyinnames_handler.is_pinyin_familyname(w1)
        if gn and ln:
            return f"{w2} {w1}"
    except Exception:
        pass
    return s

# Hyphen/dash used as separator only when surrounded by spaces (keeps "s1-s2" as one name)
_HYPHEN_DASH = r"[\-\‐\–\—\−]"

def _is_initials_only(s):
    """True if segment is only one or more name initials (I., I. J.), not degree abbreviations (DEG, etc.)."""
    t = s.strip()
    if not t:
        return False
    # Don't merge degree abbreviations onto the name
    if re.match(r"^(?:Ph\.?D\.?|M\.?Sc\.?|B\.?Sc\.?|M\.?A\.?|B\.?A\.?|M\.?Phil\.?|M\.?Eng\.?)$", t, re.IGNORECASE):
        return False
    return bool(re.match(r"^([A-Za-z]\.?\s*)+$", t))

def _looks_like_western_name(content):
    """True if content looks like a Western name (Latin words, spaces, hyphens), not a note or initials."""
    c = content.strip()
    if len(c) < 4:
        return False
    return bool(re.match(r"^[A-Za-z\s\-]+$", c)) and len(c.split()) >= 1

def _strip_non_chinese_parens(s):
    """Remove only leading and trailing (...) with non-CJK content (e.g. trailing note). Do not strip when trailing content looks like a Western name (CJK(LATIN) case)."""
    s = s.strip()
    # Leading: ^\s*\(content\)...
    lead = re.match(r"^\s*\(\s*([^)]+)\s*\)(.*)$", s, re.DOTALL)
    if lead and not _has_chinese(lead.group(1)):
        s = lead.group(2).strip()
    # Trailing: ...\(content\)\s*$ — do not strip if content looks like Western name (leave for CJK(LATIN) handling)
    trailing = re.search(r"(.*)\s*\(\s*([^)]+)\s*\)\s*$", s)
    if trailing and not _has_chinese(trailing.group(2)):
        content = trailing.group(2).strip()
        if _looks_like_western_name(content):
            pass  # keep so _accept_name can do CJK(LATIN) -> LATIN (CJK)
        else:
            s = trailing.group(1).strip()
    return s

def _split_commas_outside_parens(s):
    """Split on comma only when outside parentheses, so degree lists in parens stay one piece."""
    parts = []
    depth = 0
    start = 0
    for i, c in enumerate(s):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(s[start:i].strip())
            start = i + 1
    parts.append(s[start:].strip())
    return parts

def _explode_compound(raw):
    """Split on ';', ', or ', ' or ', comma (outside parens), and space-hyphen-space. Keeps hyphenated given-names as one token.
    First strip leading/trailing (...) with non-CJK content; do not split on comma when what follows is only initials."""
    if not isinstance(raw, str):
        return []
    s = raw.strip()
    s = re.sub(r"\s+", " ", s)
    if not s:
        return []
    # Split on semicolon first so "Family (DEG); Role/Title" yields separate candidates
    by_semicolon = re.split(r"\s*;\s*", s)
    result = []
    for part in by_semicolon:
        part = _strip_non_chinese_parens(part.strip())
        if not part:
            continue
        # Split on ", or " then " or "
        parts = re.split(r",\s*or\s+", part, flags=re.IGNORECASE)
        out = []
        for p in parts:
            out.extend(re.split(r"\s+or\s+", p, flags=re.IGNORECASE))
        for p in out:
            segments = _split_commas_outside_parens(p)
            # Merge segments when the next is only initials (e.g. Family, I. or Family, I J -> one token)
            merged = []
            i = 0
            while i < len(segments):
                x = segments[i].strip()
                if not x:
                    i += 1
                    continue
                while i + 1 < len(segments) and _is_initials_only(segments[i + 1]):
                    x = x + " " + segments[i + 1].strip()
                    i += 1
                merged.append(x)
                i += 1
            for x in merged:
                if not x.strip():
                    continue
                # Split on hyphen/dash only when surrounded by spaces; keep in-name hyphens
                for piece in re.split(r"\s+" + _HYPHEN_DASH + r"\s+", x.strip()):
                    cleaned = re.sub(r"\s*" + _HYPHEN_DASH + r"+\s*$", "", piece.strip()).strip()
                    if cleaned:
                        result.append(cleaned)
    return result

def _accept_name(candidate):
    """Apply all normalizations and return final name if valid, else None.
    Conditions: fully CJK -> (CJK); CJK(LATIN) -> LATIN (CJK) with Western casing preserved."""
    raw = candidate.strip().strip("'\"")
    if not raw:
        return None
    s_dashes = _normalize_dashes(raw)
    # Apply CJK(LATIN) -> LATIN (CJK) before lowercasing so Western part keeps casing
    fixed_reversed = _fix_reversed_chinese_western(s_dashes)
    if fixed_reversed != s_dashes:
        if _is_likely_non_name(fixed_reversed.lower()):
            return None
        return _normalize_dashes(fixed_reversed)
    s = s_dashes.lower()
    if not s:
        return None
    s = _extract_name_from_bio(s)  # long bios -> just LATIN (CJK) or LATIN
    if _is_likely_non_name(s):
        return None
    s = _fix_chinese_then_latin(s)  # CJK + space + LATIN -> LATIN (CJK)
    s = _fix_alternating_western_chinese(s)  # W1 CJK1 W2 CJK2 -> W1 W2 (CJK1CJK2)
    s = _remove_titles(s)
    s = _normalize_parens(s)
    s = _swap_if_first_word_pinyin_familyname(s)
    s = _fix_western_order_two_words(s)
    s = _format_chinese_only(s)  # fully CJK -> (CJK)
    s = re.sub(r"\s+", " ", s).strip()
    if not s or _is_likely_non_name(s):
        return None
    return s

def get_valid_names(name_list):
    valid_names = set([])

    name_list = str(name_list).strip()
    if name_list == '':
        return None

    # Replace bare 'nan' in string so eval() does not raise (e.g. "['A', nan]" -> "['A', None]")
    name_list = re.sub(r"(\[|\,\s*)nan(\s*\]|\s*\,)", r"\1None\2", name_list, flags=re.IGNORECASE)

    try:
        name_list = eval(name_list)
    except Exception as e:
        print(e, name_list)
        return []

    if not isinstance(name_list, list):
        name_list = [name_list]

    # Drop nan, None, and other non-items; keep only strings
    def _valid_item(x):
        if x is None:
            return False
        if isinstance(x, float) and x != x:  # NaN
            return False
        return isinstance(x, str)

    name_list = [x for x in name_list if _valid_item(x)]

    for item in name_list:
        item = item.strip().strip("'\"")
        for raw in _explode_compound(item):
            cleaned = _accept_name(raw)
            if cleaned:
                valid_names.add(cleaned)

    return list(valid_names)