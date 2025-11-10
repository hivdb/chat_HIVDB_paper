from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Tuple

SPECIAL_NO = {
    "none",
    "not reported",
    "not applicable",
    "not available",
    "not provided",
    "na",
    "n/a",
    "not stated",
    "nr",
    "no data",
    "0",
    "zero",
}
YES_SYNONYMS = {"yes", "y", "true", "reported", "present"}
NO_SYNONYMS = {"no", "false", "not", "absent"}
NEGATION_PHRASES = {
    "no",
    "none",
    "not reported",
    "not applicable",
    "not available",
    "not provided",
    "not specified",
    "not described",
    "not mentioned",
    "zero",
    "none reported",
    "no samples",
    "no sequencing",
    "no individuals",
    "no patient",
    "not performed",
    "not done",
}
LAB_ONLY_PHRASES = {
    "cell culture",
    "culture supernatant",
    "tissue culture",
    "plasmid",
    "molecular clone",
    "cloned virus",
    "engineered",
    "recombinant",
    "in vitro",
    "lab derived",
    "laboratory derived",
    "virus stock",
}
NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
SCALE_WORDS = {"hundred": 100, "thousand": 1000}
LIST_PARTIAL_THRESHOLD = 0.6
LIST_PARTIAL_MIN_TOKENS = 4
BOOLEAN_POSITIVE_CUES = {
    "virological failure": {"virological failure", "failure", "mutation", "resistance", "patient", "n155h"},
}
TEXT_SYNONYMS = {
    "ngs": "next generation sequencing",
    "next-generation sequencing": "next generation sequencing",
    "us": "united states",
    "u.s.": "united states",
    "usa": "united states",
    "united states of america": "united states",
    "korea": "south korea",
    "republic of korea": "south korea",
    "korean": "south korea",
    "miseq": "illumina sequencing",
    "illumina miseq": "illumina sequencing",
    "sanger": "sanger sequencing",
}
ADDITIONAL_LIST_SYNONYMS = {
    "illumina sequencing": {"illumina", "miseq", "next generation sequencing", "nextera"},
    "sanger sequencing": {"sanger"},
    "south korea": {"korea"},
    "united states": {"usa", "u.s.", "us", "america", "san francisco", "united states of america"},
}
ARV_SYNONYMS = {
    "tfv": "tenofovir",
    "tdf": "tenofovir",
    "tenofovir disoproxil fumarate": "tenofovir",
    "taf": "tenofovir alafenamide",
    "ftc": "emtricitabine",
    "3tc": "lamivudine",
    "lamivudine (3tc)": "lamivudine",
    "azt": "zidovudine",
    "abc": "abacavir",
    "efv": "efavirenz",
    "efavirenz (efv)": "efavirenz",
    "nvp": "nevirapine",
    "dtg": "dolutegravir",
    "ral": "raltegravir",
    "bik": "bictegravir",
    "bic": "bictegravir",
    "lpv": "lopinavir",
    "rtv": "ritonavir",
    "etv": "etravirine",
    "etr": "etravirine",
    "rpv": "rilpivirine",
    "rilpivirine (rpv)": "rilpivirine",
    "cab": "cabotegravir",
    "cabotegravir (cab)": "cabotegravir",
    "d4t": "stavudine",
    "mvc": "maraviroc",
    "evg": "elvitegravir",
    "elvitegravir (evg)": "elvitegravir",
    "efv, nvp": "efavirenz | nevirapine",
}
GENE_SYNONYMS = {
    "integrase": "in",
    "in": "in",
    "reverse transcriptase": "rt",
    "rt": "rt",
    "protease": "pr",
    "pr": "pr",
    "capsid": "ca",
    "ca": "ca",
    "full genome": "full genome",
    "near full length genome": "nflg",
    "nflg": "nflg",
}
GENE_GROUP_EXPANSIONS = {"pol": {"in", "pr", "rt"}}

LIST_DELIM = re.compile(r",|;|/|\band\b|\bor\b")
NON_ALPHANUM = re.compile(r"[^a-z0-9\s]")
LEADING_YES_NO = re.compile(r"^(yes|no)\b")
YEAR_REGEX = re.compile(r"(?:19|20)\d{2}")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def token_synonym(token: str) -> str:
    token = token.lower().strip()
    for table in (TEXT_SYNONYMS, ARV_SYNONYMS, GENE_SYNONYMS):
        token = table.get(token, token)
    return token


def normalize_list(text: str) -> str:
    parts = [token_synonym(part.strip()) for part in LIST_DELIM.split(text) if part.strip()]
    if not parts:
        return text
    cleaned = [" ".join(part.split()) for part in parts]
    cleaned = [NON_ALPHANUM.sub(" ", part).strip() for part in cleaned]
    cleaned = sorted({part for part in cleaned if part})
    return " | ".join(cleaned) if len(cleaned) > 1 else (cleaned[0] if cleaned else text)


def normalize_year_range(text: str) -> str | None:
    if not text:
        return None
    cleaned = text.strip().lower().replace("–", "-").replace("—", "-")
    cleaned = re.sub(r"\s*(?:to|through|thru|and)\s*", "-", cleaned)
    compact = re.sub(r"\s+", "", cleaned)
    match = re.fullmatch(r"(\d{4})-(\d{2,4})", compact)
    if match:
        start, end = match.groups()
        if len(end) == 2:
            end = start[:2] + end
        return f"{start}-{end}"
    years = YEAR_REGEX.findall(cleaned)
    return f"{years[0]}-{years[1]}" if len(years) >= 2 else None


def canonicalize_answer(text: str | float | None, *, convert_special_no: bool) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    raw = str(text).strip()
    if not raw:
        return ""
    lowered = raw.lower()
    if lowered.startswith("answer:"):
        lowered = lowered.split("answer:", 1)[1].strip()
    leading = LEADING_YES_NO.match(lowered)
    if leading:
        return leading.group(1)
    if lowered in YES_SYNONYMS:
        return "yes"
    if lowered in NO_SYNONYMS:
        return "no"
    if convert_special_no:
        if lowered in SPECIAL_NO:
            return "no"
        if any(lowered.startswith(f"{special} ") for special in SPECIAL_NO):
            return "no"
    normalized_range = normalize_year_range(raw)
    if normalized_range:
        return normalized_range
    if lowered.isdigit():
        return "no" if convert_special_no and lowered == "0" else str(int(lowered))
    if any(char.isdigit() for char in lowered) and " or " in lowered:
        numbers = [token for token in re.split(r"[^\d]+", lowered) if token.isdigit()]
        if numbers:
            return " or ".join(sorted(numbers))
    normalized = normalize_list(lowered)
    if "|" in normalized:
        sanitized = " | ".join(token.strip() for token in normalized.split("|"))
    else:
        sanitized = " ".join(NON_ALPHANUM.sub(" ", normalized).split())
    if convert_special_no and sanitized in SPECIAL_NO:
        return "no"
    if sanitized in YES_SYNONYMS:
        return "yes"
    if sanitized in NO_SYNONYMS:
        return "no"
    arv_terms = detect_terms(raw, ARV_SYNONYMS)
    if arv_terms:
        return " | ".join(sorted(arv_terms))
    gene_terms = detect_terms(sanitized, GENE_SYNONYMS, min_len=3)
    if gene_terms:
        expanded: set[str] = set()
        for term in gene_terms:
            expanded.update(GENE_GROUP_EXPANSIONS.get(term, {term}))
        return " | ".join(sorted(expanded))
    return sanitized


def detect_terms(text: str, synonyms: dict[str, str], min_len: int = 3) -> List[str]:
    lowered = text.lower()
    found: set[str] = {
        canonical
        for key, canonical in synonyms.items()
        if len(key) >= min_len and re.search(rf"\b{re.escape(key)}\b", lowered)
    }
    return sorted(found)


def human_tokens(ref_norm: str) -> List[str]:
    return [token.strip() for token in ref_norm.split("|") if token.strip()]


def list_match_stats(ref_norm: str, pred_norm: str, pred_raw: str) -> tuple[int, int]:
    tokens = human_tokens(ref_norm)
    if not tokens:
        return 0, 0
    combined = " ".join(str(part) for part in (pred_norm or "", pred_raw or "") if str(part))
    combined = NON_ALPHANUM.sub(" ", combined.lower())
    matches = 0
    for token in tokens:
        normalized = NON_ALPHANUM.sub(" ", token.lower()).strip()
        if not normalized:
            continue
        matched = False
        synonyms = {normalized}
        if token in ARV_SYNONYMS:
            synonyms.add(ARV_SYNONYMS[token])
        if token in GENE_SYNONYMS:
            synonyms.add(GENE_SYNONYMS[token])
        if token in TEXT_SYNONYMS:
            synonyms.add(TEXT_SYNONYMS[token])
        for key, value in TEXT_SYNONYMS.items():
            if value == token:
                synonyms.add(key)
        if token in ADDITIONAL_LIST_SYNONYMS:
            synonyms.update(ADDITIONAL_LIST_SYNONYMS[token])
        for syn in synonyms:
            if syn and syn in combined:
                matched = True
                break
        if matched:
            matches += 1
    return matches, len(tokens)


def contains_negation(pred_raw: str) -> bool:
    normalized = NON_ALPHANUM.sub(" ", str(pred_raw).lower()).strip()
    return any(phrase in normalized for phrase in NEGATION_PHRASES)


def lab_only_context(pred_raw: str) -> bool:
    normalized = NON_ALPHANUM.sub(" ", str(pred_raw).lower()).strip()
    return any(phrase in normalized for phrase in LAB_ONLY_PHRASES)


def extract_numbers(text: str | float | int | None) -> list[int]:
    if text is None:
        return []
    normalized = str(text)
    numbers: list[int] = []
    for match in re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", normalized):
        cleaned = match.replace(",", "")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        numbers.append(int(value) if value.is_integer() else int(round(value)))
    tokens = re.findall(r"[a-z]+", normalized.lower())
    current = 0
    active = False
    for token in tokens:
        if token in NUMBER_WORDS:
            current += NUMBER_WORDS[token]
            active = True
        elif token in SCALE_WORDS:
            current = max(1, current) * SCALE_WORDS[token]
            active = True
        else:
            if active and current > 0:
                numbers.append(current)
            current = 0
            active = False
    if active and current > 0:
        numbers.append(current)
    return numbers


def numeric_match(ref_norm: str, ref_raw: str, pred_norm: str, pred_raw: str) -> bool:
    ref_numbers = set(extract_numbers(ref_norm)) | set(extract_numbers(ref_raw))
    pred_numbers = set(extract_numbers(pred_norm)) | set(extract_numbers(pred_raw))
    return bool(ref_numbers and pred_numbers and ref_numbers & pred_numbers)


def year_tokens(ref_norm: str) -> list[str]:
    normalized_range = normalize_year_range(ref_norm)
    if normalized_range:
        start, end = normalized_range.split("-")
        return [start] if start == end else [start, end]
    return YEAR_REGEX.findall(ref_norm)


def mentions_year(years: list[str], pred_norm: str, pred_raw: str) -> bool:
    if not years:
        return False
    combined = f"{pred_norm or ''} {pred_raw or ''}".lower()
    return any(year in combined for year in years)


def boolean_positive_context(question_text: str, pred_raw: str) -> bool:
    question = (question_text or "").lower()
    text = (pred_raw or "").lower()
    for cue, markers in BOOLEAN_POSITIVE_CUES.items():
        if cue in question and any(marker in text for marker in markers):
            return True
    return False


def is_empty_token(value: str) -> bool:
    return value in ("", "no")


def is_empty_number(value: str) -> bool:
    return value in ("", "no", "0")


def human_answer_counts(question_type: str, pred_norm: str, ref_norm: str, *, question_text: str, ref_raw: str, pred_raw: str) -> tuple[dict[str, int], bool]:
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    qtype = (question_type or "").lower()

    def mark(label: str) -> None:
        counts[label] += 1

    if qtype == "boolean":
        ref_positive = ref_norm == "yes"
        if ref_positive and boolean_positive_context(question_text, pred_raw):
            mark("tp")
            return counts, True
        pred_positive = pred_norm == "yes"
        if ref_positive:
            mark("tp" if pred_positive else "fn")
        else:
            mark("fp" if pred_positive else "tn")
        return counts, ref_positive == pred_positive

    if qtype == "list":
        ref_non_empty = not is_empty_token(ref_norm)
        matches, total = list_match_stats(ref_norm, pred_norm, pred_raw)
        match_ratio = matches / total if total else 0.0
        if ref_non_empty:
            exact = compare_lists(pred_norm, ref_norm)
            partial = total >= LIST_PARTIAL_MIN_TOKENS and match_ratio >= LIST_PARTIAL_THRESHOLD
            if exact or partial or (total and matches == total) or mentions_year(year_tokens(ref_norm), pred_norm, pred_raw):
                mark("tp")
                return counts, True
            mark("fn")
            return counts, False
        if not is_empty_token(pred_norm):
            if contains_negation(pred_raw) or lab_only_context(pred_raw):
                mark("tn")
                return counts, True
            mark("fp")
            return counts, False
        mark("tn")
        return counts, True

    if qtype == "number":
        ref_non_zero = not is_empty_number(ref_norm)
        pred_non_zero = not is_empty_number(pred_norm)
        if ref_non_zero:
            if pred_non_zero and (compare_lists(pred_norm, ref_norm) or numeric_match(ref_norm, ref_raw, pred_norm, pred_raw)):
                mark("tp")
                return counts, True
            mark("fn")
            return counts, False
        if contains_negation(pred_raw):
            mark("tn")
            return counts, True
        if pred_non_zero:
            mark("fp")
            return counts, False
        mark("tn")
        return counts, True

    both_empty = is_empty_token(ref_norm) and (is_empty_token(pred_norm) or contains_negation(pred_raw))
    if both_empty:
        mark("tn")
        return counts, True
    if compare_lists(pred_norm, ref_norm):
        mark("tp")
        return counts, True
    if is_empty_token(ref_norm):
        mark("fp")
        return counts, False
    mark("fn")
    return counts, False


def compare_lists(pred_norm: str, ref_norm: str) -> bool:
    if pred_norm == ref_norm:
        return True
    if " or " in ref_norm:
        options = {opt.strip() for opt in ref_norm.split("or")}
        if pred_norm.strip() in options:
            return True
    if " or " in pred_norm:
        options = {opt.strip() for opt in pred_norm.split("or")}
        if ref_norm.strip() in options:
            return True
    pred_set = {tok.strip() for tok in pred_norm.split("|") if tok.strip()}
    ref_set = {tok.strip() for tok in ref_norm.split("|") if tok.strip()}
    if pred_set and ref_set:
        if pred_set == ref_set or ref_set.issubset(pred_set):
            return True
    return False
