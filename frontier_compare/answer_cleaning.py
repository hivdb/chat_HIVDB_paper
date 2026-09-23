"""Post-processing that separates a model's actual answer from explanatory scaffolding.

Models sometimes wrap a correct answer in commentary: a preamble ("The paper reports that: ..."),
a trailing note ("...; the paper does not say which"), or a parenthetical aside ("(paper states
either Sanger or NGS)"). The paper's scorer reads some of that commentary as hedging or negation
and marks the row wrong, even though the answer itself matches.

Cleaning is NEVER destructive at scoring time: `04_evaluate.py` scores the raw answer first and
only falls back to the cleaned form, so a rule can rescue a row but never break one. Every rule
that fires is recorded per row (`<model> cleaning_rule`) so the effect stays auditable, and the
same rules are applied to every model, including the cached GPT-4o comparators.

Rules (each returns the answer unchanged when it does not apply):
  strip_preamble              "The paper reports that: X"            -> "X"
  drop_commentary_parenthesis "X (paper states either X or Y)"       -> "X"
  drop_trailing_commentary    "X; the paper does not specify which"  -> "X"
"""

from __future__ import annotations

import re

# A parenthetical is commentary when it is wordy or hedges, rather than an abbreviation like "(3TC)"
_HEDGE = r"\b(not|no|only|unclear|unspecified|unknown|prior|other|either|assumed|presumed|paper|study|reported|stated|detailed|separately)\b"
_PREAMBLE = re.compile(
    r"^(?:the paper|this paper|the study|this study|the authors|it)\b[^:]{0,80}:\s*", re.I
)
_PAREN = re.compile(r"\s*\(([^()]*)\)")
_TRAILING = re.compile(
    rf"[;.]\s+(?:[A-Za-z][^;.]*{_HEDGE}[^;.]*)\.?\s*$", re.I
)


def _strip_preamble(text: str) -> str:
    return _PREAMBLE.sub("", text)


def _drop_commentary_parenthesis(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        inner = match.group(1)
        wordy = len(inner.split()) >= 4
        hedging = re.search(_HEDGE, inner, re.I) is not None
        return "" if (wordy or hedging) else match.group(0)

    return _PAREN.sub(replace, text)


def _drop_trailing_commentary(text: str) -> str:
    return _TRAILING.sub("", text)


RULES = [
    ("strip_preamble", _strip_preamble),
    ("drop_commentary_parenthesis", _drop_commentary_parenthesis),
    ("drop_trailing_commentary", _drop_trailing_commentary),
]


def clean_answer(text: str | float | None) -> tuple[str, str]:
    """Return (cleaned answer, '+'-joined names of the rules that changed it)."""
    if text is None or not isinstance(text, str):
        return "", ""
    cleaned, applied = text.strip(), []
    for name, rule in RULES:
        candidate = rule(cleaned)
        candidate = re.sub(r"\s{2,}", " ", candidate).strip(" ;,.")
        if candidate and candidate != cleaned:
            cleaned, _ = candidate, applied.append(name)
    return (cleaned, "+".join(applied)) if applied else (text.strip(), "")
