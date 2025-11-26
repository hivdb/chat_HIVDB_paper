from __future__ import annotations

import math
import re
from typing import List, Tuple

from .constants import (
    ADDITIONAL_LIST_SYNONYMS,
    ARV_SYNONYMS,
    BOOLEAN_POSITIVE_CUES,
    GENE_GROUP_EXPANSIONS,
    GENE_SYNONYMS,
    LAB_ONLY_PHRASES,
    LEADING_YES_NO,
    LIST_DELIM,
    LIST_PARTIAL_THRESHOLD,
    MATCH_SCENARIOS,
    NEGATION_PHRASES,
    NO_SYNONYMS,
    NON_ALPHANUM,
    NUMBER_WORDS,
    SCALE_WORDS,
    SPECIAL_NO,
    SPECIAL_NO_INDICATORS,
    TEXT_SYNONYMS,
    YES_SYNONYMS,
    YEAR_REGEX,
)

NEGATION_PATTERNS = [
    re.compile(rf"\b{re.escape(phrase)}\b")
    for phrase in NEGATION_PHRASES
]

FULL_GENOME_TOKENS = {
    "full genome",
    "full length genome",
    "flg",
    "near full length genome",
    "near-full-length genome",
    "near full length proviruses",
    "near-full-length proviruses",
    "nflg",
}

# ---------------------------------------------------------------------------
# Canonicalization helpers
# ---------------------------------------------------------------------------

def canonicalize_answer(text: str | float | None, *, convert_special_no: bool) -> str:
    raw = _clean_answer_text(text)
    if not raw:
        return ""
    lowered = raw.lower()

    boolean = _canonical_boolean(lowered, convert_special_no)
    if boolean is not None:
        return boolean

    numeric = _canonical_numeric(lowered, raw, convert_special_no)
    if numeric is not None:
        return numeric

    return _canonical_list(lowered, raw, convert_special_no)


def _clean_answer_text(value: str | float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.lower().startswith("answer:"):
        text = text[len("answer:") :].strip()
    return text


def _canonical_boolean(lowered: str, convert_special_no: bool) -> str | None:
    match = LEADING_YES_NO.match(lowered)
    if match:
        return match.group(1)
    if lowered in YES_SYNONYMS:
        return "yes"
    if lowered in NO_SYNONYMS:
        return "no"
    if convert_special_no and (
        lowered in SPECIAL_NO
        or any(lowered.startswith(f"{token} ") for token in SPECIAL_NO)
    ):
        return "no"
    if convert_special_no and any(phrase in lowered for phrase in SPECIAL_NO_INDICATORS):
        return "no"
    return None


def _canonical_numeric(lowered: str, raw: str, convert_special_no: bool) -> str | None:
    normalized_range = normalize_year_range(raw)
    if normalized_range:
        return normalized_range
    if lowered.isdigit():
        if convert_special_no and lowered == "0":
            return "no"
        return str(int(lowered))
    if any(char.isdigit() for char in lowered) and " or " in lowered:
        numbers = sorted({token for token in re.split(r"[^\d]+", lowered) if token.isdigit()})
        if numbers:
            return " or ".join(numbers)
    return None


def _canonical_list(lowered: str, raw: str, convert_special_no: bool) -> str:
    normalized = normalize_list(lowered)
    if "|" in normalized:
        tokens = [part.strip() for part in normalized.split("|") if part.strip()]
    else:
        tokens = [" ".join(NON_ALPHANUM.sub(" ", normalized).split())] if normalized.strip() else []

    canonical_tokens: list[str] = []
    for token in tokens:
        if not token:
            continue
        # Normalize ritonavir suffix before looking up synonyms
        token = _normalize_ritonavir_suffix(token)
        token = TEXT_SYNONYMS.get(token, token)
        if convert_special_no and token in SPECIAL_NO:
            return "no"
        if token in YES_SYNONYMS:
            return "yes"
        if token in NO_SYNONYMS:
            return "no"
        canonical_tokens.extend(_canonical_list_token(token))

    canonical_tokens = [" ".join(token.split()) for token in canonical_tokens if token]
    if not canonical_tokens:
        return ""
    unique_tokens = sorted(dict.fromkeys(canonical_tokens))
    return " | ".join(unique_tokens)


def _canonical_list_token(token: str) -> List[str]:
    if not token:
        return []
    if token in ARV_SYNONYMS:
        return [part.strip() for part in ARV_SYNONYMS[token].split("|") if part.strip()]
    gene = GENE_SYNONYMS.get(token)
    if gene:
        expansions = GENE_GROUP_EXPANSIONS.get(gene, {gene})
        return sorted(expansions)
    return [token]


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


def human_tokens(ref_norm: str) -> List[str]:
    return [token.strip() for token in ref_norm.split("|") if token.strip()]


def list_match_stats(ref_norm: str, pred_norm: str, pred_raw: str) -> Tuple[int, int]:
    tokens = human_tokens(ref_norm)
    if not tokens:
        return 0, 0
    haystack = NON_ALPHANUM.sub(" ", f"{pred_norm or ''} {pred_raw or ''}".lower())
    matches = sum(1 for token in tokens if _token_matches(token, haystack))
    return matches, len(tokens)


def _token_matches(token: str, haystack: str) -> bool:
    for variant in _expand_token_synonyms(token):
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


def _expand_token_synonyms(token: str) -> set[str]:
    normalized = NON_ALPHANUM.sub(" ", token.lower()).strip()
    variants = {normalized}

    # Add variant without ritonavir suffix
    without_rtv = _normalize_ritonavir_suffix(normalized)
    if without_rtv != normalized:
        variants.add(without_rtv)

    key = token.lower()
    for mapping in (ARV_SYNONYMS, GENE_SYNONYMS, TEXT_SYNONYMS):
        if key in mapping:
            variants.add(mapping[key])
    for raw, canonical in TEXT_SYNONYMS.items():
        if canonical == token:
            variants.add(raw)
    variants.update(ADDITIONAL_LIST_SYNONYMS.get(token, set()))
    variants = {NON_ALPHANUM.sub(" ", variant.lower()).strip() for variant in variants if variant}

    # Also add ritonavir-stripped versions of all variants
    expanded_variants = set(variants)
    for variant in variants:
        stripped = _normalize_ritonavir_suffix(variant)
        if stripped and stripped != variant:
            expanded_variants.add(stripped)

    return {variant for variant in expanded_variants if variant}


def contains_negation(pred_raw: str) -> bool:
    normalized = NON_ALPHANUM.sub(" ", str(pred_raw).lower()).strip()
    return any(pattern.search(normalized) for pattern in NEGATION_PATTERNS)


def lab_only_context(pred_raw: str) -> bool:
    normalized = NON_ALPHANUM.sub(" ", str(pred_raw).lower()).strip()
    return any(phrase in normalized for phrase in LAB_ONLY_PHRASES)


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


def year_tokens(ref_norm: str) -> List[str]:
    normalized_range = normalize_year_range(ref_norm)
    if normalized_range:
        start, end = normalized_range.split("-")
        return [start] if start == end else [start, end]
    return YEAR_REGEX.findall(ref_norm)


def mentions_year(years: List[str], pred_norm: str, pred_raw: str) -> bool:
    if not years:
        return False
    combined = f"{pred_norm or ''} {pred_raw or ''}".lower()
    return any(year in combined for year in years)


def boolean_positive_context(question_text: str, pred_raw: str) -> bool:
    question = str(question_text or "").lower()
    text = str(pred_raw or "").lower()
    for cue, markers in BOOLEAN_POSITIVE_CUES.items():
        if cue in question and any(marker in text for marker in markers):
            return True
    return False


def is_empty_token(value: str) -> bool:
    return value in ("", "no")


def is_empty_number(value: str) -> bool:
    return value in ("", "no", "0")


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
    raw_ref_set = {tok.strip() for tok in ref_norm.split("|") if tok.strip()}
    if not pred_set or not raw_ref_set:
        return False
    core_ref = raw_ref_set - FULL_GENOME_TOKENS
    ref_set = core_ref if core_ref else raw_ref_set
    if pred_set == ref_set or ref_set.issubset(pred_set):
        return True
    if _matches_pol_group(ref_set, pred_set):
        return True
    if _full_genome_equivalent(raw_ref_set, pred_set, pred_norm):
        return True
    return False


def _matches_pol_group(ref_set: set[str], pred_set: set[str]) -> bool:
    """Treat any Pol reference as satisfied by at least two component genes or 'pol'."""
    pol_components = GENE_GROUP_EXPANSIONS.get("pol", set())
    if not pol_components:
        return False
    # Reference counts as Pol-style if it only contains Pol tokens (components or Pol itself)
    ref_tokens = {token for token in ref_set if token}
    ref_is_pol = ref_tokens.issubset(pol_components | {"pol"})
    ref_requires_pol = ref_is_pol and ("pol" in ref_tokens or len(ref_tokens) >= 3)
    if not ref_is_pol:
        return False
    normalized_pred = {token for token in pred_set if token}
    pred_components = normalized_pred & pol_components
    if "pol" in normalized_pred:
        return True
    if ref_requires_pol and len(pred_components) >= 2:
        return True
    if not ref_requires_pol and pred_components.issuperset(ref_tokens):
        return True
    return False


def _full_genome_equivalent(ref_set: set[str], pred_set: set[str], pred_norm: str) -> bool:
    if not (ref_set & FULL_GENOME_TOKENS):
        return False
    if pred_set & FULL_GENOME_TOKENS:
        return True
    lowered_pred = pred_norm.lower()
    has_pol = "pol" in lowered_pred
    has_env = "env" in lowered_pred or "gp120" in lowered_pred
    if has_pol and has_env:
        return True
    if {"pol", "env"}.issubset(pred_set) or {"pol", "gp120"}.issubset(pred_set):
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
    handler = QUESTION_HANDLERS.get(qtype, _score_generic)
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
    if ref_positive and boolean_positive_context(question_text, pred_raw):
        return _finalize(counts, "tp", True)
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
    matches, total = list_match_stats(ref_norm, pred_norm, pred_raw)
    match_ratio = matches / total if total else 0.0
    full_match = compare_lists(pred_norm, ref_norm) or (total and matches == total)
    partial_credit = (
        allow_partial
        and not ref_empty
        and total > 0
        and match_ratio >= LIST_PARTIAL_THRESHOLD
    )
    has_year_reference = False

    if not ref_empty:
        correct = full_match or partial_credit or has_year_reference
        return _finalize(counts, "tp" if correct else "fn", correct)

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
    ref_non_zero = not is_empty_number(ref_norm)
    pred_non_zero = not is_empty_number(pred_norm)

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
