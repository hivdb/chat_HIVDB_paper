from __future__ import annotations

import math
import re
from typing import List, Tuple

from .constants import (
    ADDITIONAL_LIST_SYNONYMS,
    ARV_SYNONYMS,
    EMBEDDED_MAP,
    GENE_GROUP_EXPANSIONS,
    GENE_SYNONYMS,
    LAB_ONLY_PHRASES,
    LEADING_YES_NO,
    LIST_DELIM,
    LIST_PARTIAL_THRESHOLD,
    LOCAL_NUMBER_WORDS,
    LOCAL_SCALES,
    MATCH_SCENARIOS,
    NEGATIVE_PHRASES,
    NEGATIVE_TOKENS,
    NON_ALPHANUM,
    TEXT_SYNONYMS,
    YES_SYNONYMS,
    YEAR_REGEX,
)

# ---------------------------------------------------------------------------
# Canonicalization helpers
# ---------------------------------------------------------------------------

def canonicalize_answer(text: str | float | None) -> str:
    raw = _clean_answer_text(text)
    if not raw:
        return ""
    lowered = raw.lower()
    for val in (_canonical_boolean(lowered), _canonical_numeric(lowered, raw), _canonical_list(lowered, raw)):
        if val is not None:
            return val
    return ""


def _is_negative(text: str) -> bool:
    """Return True if text represents a negative answer."""
    if not text:
        return False
    lowered = text.strip().lower()
    return (
        lowered in NEGATIVE_TOKENS
        or any(lowered.startswith(f"{token} ") for token in NEGATIVE_TOKENS)
        or any(phrase in lowered for phrase in NEGATIVE_PHRASES)
    )


def _clean_answer_text(value: str | float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if text.lower().startswith("answer:"):
        text = text[len("answer:") :].strip()
    return text


def _canonical_boolean(lowered: str) -> str | None:
    match = LEADING_YES_NO.match(lowered)
    if match:
        return match.group(1)
    if lowered in YES_SYNONYMS:
        return "yes"
    if _is_negative(lowered):
        return "no"
    return None


def _canonical_numeric(lowered: str, raw: str) -> str | None:
    normalized_range = normalize_year_range(raw)
    if normalized_range:
        return normalized_range
    if lowered.isdigit():
        if lowered == "0":
            return "no"
        return str(int(lowered))
    if any(char.isdigit() for char in lowered) and " or " in lowered:
        numbers = sorted({token for token in re.split(r"[^\d]+", lowered) if token.isdigit()})
        if numbers:
            return " or ".join(numbers)
    years = YEAR_REGEX.findall(raw)
    if len(years) == 1:
        return years[0]
    return None


def _canonical_list(lowered: str, raw: str) -> str:
    tokens = _list_tokens(lowered)
    raw_lower = raw.lower()
    embedded_tokens: list[str] = []
    for phrase, replacement in EMBEDDED_MAP.items():
        if _contains_phrase(raw_lower, phrase) and replacement:
            for part in replacement.split("|"):
                embedded_tokens.extend(_list_tokens(part))

    tokens_to_process = tokens + embedded_tokens if embedded_tokens else tokens
    canonical_tokens: list[str] = []
    for token in tokens_to_process:
        if not token:
            continue
        if _is_negative(token):
            return "no"
        if token in YES_SYNONYMS:
            return "yes"
        canonical_tokens.extend(_expand_token(token))

    canonical_tokens = [" ".join(token.split()) for token in canonical_tokens if token]
    if not canonical_tokens:
        return ""
    unique_tokens = sorted(dict.fromkeys(canonical_tokens))
    return " | ".join(unique_tokens)


_expansion_cache: dict[tuple[str, bool], list[str]] = {}


def _expand_token(token: str, for_match: bool = False) -> List[str]:
    if not token:
        return []
    cache_key = (token, for_match)
    cached = _expansion_cache.get(cache_key)
    if cached is not None:
        return cached
    base = " ".join(NON_ALPHANUM.sub(" ", token).split())
    base = re.sub(r"^primarily\s+", "", base).replace("primarily ", "")
    tokens: list[str] = []
    lowered = token.lower()
    if " from " in lowered:
        tail = lowered.split(" from ", 1)[1]
        for part in re.split(r",|;| and ", tail):
            cleaned = " ".join(NON_ALPHANUM.sub(" ", part).split())
            cleaned = re.sub(r"^primarily\s+", "", cleaned).replace("primarily ", "")
            if cleaned:
                tokens.append(cleaned)
    for key, value in TEXT_SYNONYMS.items():
        if not value:
            continue
        if " " in key or len(key) > 3:
            if re.search(rf"\b{re.escape(key)}\b", lowered):
                tokens.append(value)
        else:
            # For short keys (<=3 chars), require exact match to avoid noisy hits.
            if lowered == key:
                tokens.append(value)
    arv = ARV_SYNONYMS.get(token)
    if arv:
        tokens.extend(part.strip() for part in arv.split("|") if part.strip())
    gene = GENE_SYNONYMS.get(token)
    if gene:
        expansions = GENE_GROUP_EXPANSIONS.get(gene)
        if expansions:
            tokens.extend([gene, *sorted(expansions)])
        else:
            tokens.append(gene)
    if for_match:
        tokens.extend(ADDITIONAL_LIST_SYNONYMS.get(token, set()))
    if not tokens:
        tokens.append(base)
    normalized_tokens = []
    for t in tokens:
        cleaned = NON_ALPHANUM.sub(" ", t.lower()).strip()
        normalized_tokens.append(cleaned)
        stripped = _normalize_ritonavir_suffix(cleaned)
        if stripped and stripped != cleaned:
            normalized_tokens.append(stripped)
    result = [t for t in normalized_tokens if t]
    _expansion_cache[cache_key] = result
    return result


def _list_tokens(text: str) -> list[str]:
    if not text:
        return []
    normalized = normalize_list(text)
    if "|" in normalized:
        return [part.strip() for part in normalized.split("|") if part.strip()]
    return [" ".join(NON_ALPHANUM.sub(" ", normalized).split())] if normalized.strip() else []


def _contains_phrase(text: str, phrase: str) -> bool:
    """Check if `phrase` occurs in `text` as a stand-alone token."""
    if not phrase or not text:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


# ---------------------------------------------------------------------------
# Normalization utilities
# ---------------------------------------------------------------------------

def normalize_list(value: str) -> str:
    tokens = [token.strip() for token in LIST_DELIM.split(value) if token.strip()]
    if not tokens:
        return value
    normalized = [" ".join(NON_ALPHANUM.sub(" ", token.lower()).split()) for token in tokens]
    normalized = sorted({token for token in normalized if token})
    return " | ".join(normalized)


def normalize_year_range(text: str) -> str | None:
    if not text:
        return None
    cleaned = re.sub(r"[^\w\s-]", " ", text.lower())
    cleaned = cleaned.replace("–", "-").replace("—", "-")
    cleaned = re.sub(r"\s*(?:to|through|thru|and|–|—|,|-)\s*", "-", cleaned)
    compact = re.sub(r"\s+", "", cleaned)
    match = re.fullmatch(r"(\d{4})-(\d{2,4})", compact)
    if match:
        start, end = match.groups()
        if len(end) == 2:
            end = start[:2] + end
        return f"{start}-{end}"
    years = YEAR_REGEX.findall(cleaned)
    if len(years) >= 2:
        years = sorted(years)
        return f"{years[0]}-{years[-1]}"
    if len(years) == 1:
        return years[0]
    return None


def human_tokens(ref_norm: str) -> List[str]:
    return [token.strip() for token in ref_norm.split("|") if token.strip()]


def _expand_pol(tokens: Iterable[str]) -> List[str]:
    tokens_set = {t.strip() for t in tokens if t.strip()}
    if "pol" in tokens_set:
        tokens_set.update({"pr", "rt", "in"})
        tokens_set.discard("pol")
    return sorted(tokens_set)


def list_match_stats(ref_norm: str, pred_norm: str, pred_raw: str) -> Tuple[int, int]:
    tokens = _expand_pol(human_tokens(ref_norm))
    if not tokens:
        return 0, 0
    haystack = NON_ALPHANUM.sub(" ", f"{pred_norm or ''} {pred_raw or ''}".lower())
    matches = sum(1 for token in tokens if _token_matches(token, haystack))
    return matches, len(tokens)


def _token_matches(token: str, haystack: str) -> bool:
    # Handle year tokens against ranges like 2010-2020
    if re.fullmatch(r"\d{4}", token):
        year_val = int(token)
        for start, end in re.findall(r"(?:(19|20)\d{2})\s*-\s*(?:(19|20)\d{2})", haystack):
            start_val, end_val = int(start), int(end)
            if start_val <= year_val <= end_val:
                return True
    for variant in _expand_token(token, for_match=True):
        if variant and variant in haystack:
            return True
    return False


def _normalize_ritonavir_suffix(text: str) -> str:
    """Strip optional ritonavir suffixes from drug names."""
    if not text:
        return text
    # Remove /r, /ritonavir, or comma-separated ritonavir
    normalized = re.sub(r'\s*/\s*r(?:itonavir)?\s*$', '', text, flags=re.IGNORECASE)
    normalized = re.sub(r'\s*,\s*r(?:itonavir)?\s*$', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\s+ritonavir\s*$', '', normalized, flags=re.IGNORECASE)
    return normalized.strip()


def contains_negation(pred_raw: str) -> bool:
    normalized = NON_ALPHANUM.sub(" ", str(pred_raw).lower()).strip()
    return _is_negative(normalized)


def lab_only_context(pred_raw: str) -> bool:
    normalized = NON_ALPHANUM.sub(" ", str(pred_raw).lower()).strip()
    return any(phrase in normalized for phrase in LAB_ONLY_PHRASES)


def _words_to_number(words: list[str]) -> int | None:
    total = 0
    current = 0
    for token in words:
        if token in LOCAL_NUMBER_WORDS:
            current += LOCAL_NUMBER_WORDS[token]
        elif token in LOCAL_SCALES:
            current = max(1, current) * LOCAL_SCALES[token]
        else:
            return None
    total += current
    return total if total else None


def extract_numbers(text: str | float | int | None) -> List[int]:
    if text is None:
        return []
    normalized = str(text)
    numbers: List[int] = []
    for match in re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", normalized):
        cleaned = match.replace(",", "")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        numbers.append(int(value) if value.is_integer() else int(round(value)))
    tokens = re.findall(r"[a-z]+", normalized.lower())
    buffer: list[str] = []
    for token in tokens:
        if token in LOCAL_NUMBER_WORDS or token in LOCAL_SCALES:
            buffer.append(token)
            continue
        if buffer:
            value = _words_to_number(buffer)
            if value is not None:
                numbers.append(value)
            buffer = []
    if buffer:
        value = _words_to_number(buffer)
        if value is not None:
            numbers.append(value)
    return numbers


def numeric_match(ref_norm: str, ref_raw: str, pred_norm: str, pred_raw: str) -> bool:
    ref_numbers = set(extract_numbers(ref_norm)) | set(extract_numbers(ref_raw))
    pred_numbers = set(extract_numbers(pred_norm)) | set(extract_numbers(pred_raw))
    return bool(ref_numbers and pred_numbers and ref_numbers & pred_numbers)


def year_tokens(ref_norm: str) -> List[str]:
    normalized_range = normalize_year_range(ref_norm)
    if normalized_range:
        start, end = normalized_range.split("-")
        return [start] if start == end else [start, end]
    return YEAR_REGEX.findall(ref_norm)


def is_empty_token(value: str, allow_zero: bool = True) -> bool:
    return value == "" or _is_negative(value) or (not allow_zero and value == "0")


def compare_lists(pred_norm: str, ref_norm: str) -> bool:
    """
    Check if prediction exactly matches reference (100% = 100%).
    Both sets must be identical after normalization.
    """
    if pred_norm == ref_norm:
        return True

    # Handle "or" alternatives: "10 or 20" accepts either value
    if " or " in ref_norm:
        options = {opt.strip() for opt in ref_norm.split("or")}
        if pred_norm.strip() in options:
            return True
    if " or " in pred_norm:
        options = {opt.strip() for opt in pred_norm.split("or")}
        if ref_norm.strip() in options:
            return True

    # Convert to sets and compare
    pred_set = set(_expand_pol(pred_norm.split("|")))
    ref_set = set(_expand_pol(ref_norm.split("|")))

    if not pred_set or not ref_set:
        return False

    # Exact match: both sets must be identical
    if pred_set == ref_set:
        return True

    # Allow treating "pol" as either {pr, rt, in} or {pr, rt}
    diff_prd = pred_set - ref_set
    diff_ref = ref_set - pred_set
    if (diff_prd == {"in"} and not diff_ref) or (diff_ref == {"in"} and not diff_prd):
        return True
    return False


def _list_partial_match(pred_norm: str, ref_norm: str, pred_raw: str) -> bool:
    matches, total = list_match_stats(ref_norm, pred_norm, pred_raw)
    if total:
        ratio = matches / total
        if ratio >= LIST_PARTIAL_THRESHOLD:
            return True
    pred_set = {tok.strip() for tok in pred_norm.split("|") if tok.strip()}
    ref_set = {tok.strip() for tok in ref_norm.split("|") if tok.strip()}
    if ref_set:
        overlap = len(pred_set & ref_set) / len(ref_set)
        if overlap >= LIST_PARTIAL_THRESHOLD:
            return True
    return False


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "figure"


def match_scenario_label(allow_partial: bool) -> str:
    return MATCH_SCENARIOS[bool(allow_partial)]


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------

def human_answer_counts(
    question_type: str,
    pred_norm: str,
    ref_norm: str,
    *,
    question_text: str,
    ref_raw: str,
    pred_raw: str,
    allow_partial_list: bool = False,
) -> Tuple[dict[str, int], bool]:
    qtype = (question_type or "").lower()
    handler = QUESTION_HANDLERS.get(qtype)
    if handler is None and allow_partial_list:
        handler = _score_list
    if handler is None:
        handler = _score_generic
    return handler(
        pred_norm,
        ref_norm,
        question_text=question_text,
        ref_raw=ref_raw,
        pred_raw=pred_raw,
        allow_partial=allow_partial_list,
    )


def _new_counts() -> dict[str, int]:
    return {"tp": 0, "tn": 0, "fp": 0, "fn": 0}


def _finalize(counts: dict[str, int], label: str, is_correct: bool) -> Tuple[dict[str, int], bool]:
    counts[label] += 1
    return counts, is_correct


def _score_boolean(
    pred_norm: str,
    ref_norm: str,
    *,
    question_text: str,
    pred_raw: str,
    **_: str,
) -> Tuple[dict[str, int], bool]:
    counts = _new_counts()
    ref_positive = ref_norm == "yes"
    pred_positive = pred_norm == "yes"
    if ref_positive:
        return _finalize(counts, "tp" if pred_positive else "fn", pred_positive)
    return _finalize(counts, "fp" if pred_positive else "tn", not pred_positive)


def _score_list(
    pred_norm: str,
    ref_norm: str,
    *,
    pred_raw: str,
    allow_partial: bool = False,
    **_: str,
) -> Tuple[dict[str, int], bool]:
    counts = _new_counts()
    ref_empty = is_empty_token(ref_norm)

    if not ref_empty:
        if compare_lists(pred_norm, ref_norm):
            return _finalize(counts, "tp", True)
        if allow_partial and _list_partial_match(pred_norm, ref_norm, pred_raw):
            return _finalize(counts, "tp", True)
        return _finalize(counts, "fn", False)

    # Reference is empty
    if is_empty_token(pred_norm) or contains_negation(pred_raw) or lab_only_context(pred_raw):
        return _finalize(counts, "tn", True)
    return _finalize(counts, "fp", False)


def _score_number(
    pred_norm: str,
    ref_norm: str,
    *,
    pred_raw: str,
    ref_raw: str,
    **_: str,
) -> Tuple[dict[str, int], bool]:
    counts = _new_counts()
    ref_non_zero = not is_empty_token(ref_norm, allow_zero=False)
    pred_non_zero = not is_empty_token(pred_norm, allow_zero=False)

    if ref_non_zero:
        correct = pred_non_zero and (
            compare_lists(pred_norm, ref_norm) or numeric_match(ref_norm, ref_raw, pred_norm, pred_raw)
        )
        return _finalize(counts, "tp" if correct else "fn", correct)

    if contains_negation(pred_raw):
        return _finalize(counts, "tn", True)
    if pred_non_zero:
        return _finalize(counts, "fp", False)
    return _finalize(counts, "tn", True)


def _score_generic(
    pred_norm: str,
    ref_norm: str,
    *,
    pred_raw: str,
    **_: str,
) -> Tuple[dict[str, int], bool]:
    counts = _new_counts()
    if is_empty_token(ref_norm):
        if is_empty_token(pred_norm) or contains_negation(pred_raw):
            return _finalize(counts, "tn", True)
        return _finalize(counts, "fp", False)
    if compare_lists(pred_norm, ref_norm):
        return _finalize(counts, "tp", True)
    return _finalize(counts, "fn", False)


QUESTION_HANDLERS = {
    "boolean": _score_boolean,
    "list": _score_list,
    "number": _score_number,
}
