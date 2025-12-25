import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


INPUT_CSV = Path("csv/llama-3.1-8B-FT-PV1.csv")
S4_TABLE = Path("csv/S4Table.xlsx")
OUTPUT_CSV = INPUT_CSV.with_name(f"{INPUT_CSV.stem}_parsed.csv")

CANONICAL_QUESTIONS = [
    "Does the paper report HIV sequences from patient samples?",
    "Does the paper report in vitro drug susceptibility data?",
    "Were sequences from the paper made publicly available?",
    "What were the GenBank accession numbers for sequenced HIV isolates?",
    "How many individuals had samples obtained for HIV sequencing?",
    "From which countries were the sequenced samples obtained?",
    "From what years were the sequenced samples obtained?",
    "Were samples cloned prior to sequencing?",
    "Which HIV genes were reported to have been sequenced?",
    "What method was used for sequencing?",
    "What type of samples were sequenced?",
    "Were any sequences obtained from individuals with virological failure on a treatment regimen?",
    "Were the patients in the study in a clinical trial?",
    "Does the paper report HIV sequences from individuals who had previously received ARV drugs?",
    "Which drug classes were received by individuals in the study before sample sequencing?",
    "Which drugs were received by individuals in the study before sample sequencing?",
]


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("```", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_question(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"[?]+$", "", text)
    text = text.replace("–", "-")
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


CANONICAL_NORMALIZED = {normalize_question(q): q for q in CANONICAL_QUESTIONS}


def build_qid_lookup() -> Dict[str, int]:
    s4 = pd.read_excel(S4_TABLE, dtype={"PMID": str})
    lookup: Dict[str, int] = {}
    for _, row in s4[["Question", "QID"]].drop_duplicates().iterrows():
        norm = normalize_question(row["Question"])
        if norm not in lookup:
            lookup[norm] = int(row["QID"])

    missing = [
        q for q in CANONICAL_QUESTIONS if normalize_question(q) not in lookup
    ]
    if missing:
        raise ValueError(f"Missing QID mapping for questions: {missing}")
    return lookup


QUESTION_PREFIX_RE = re.compile(
    r"^\s*(?:#\s*)?(?:question\s*:|question\s+\d+\s*[-–]\s*|q\s*\d+\s*[-–]\s*|question\s+\d+\s*|q\s*\d+\s*)",
    re.IGNORECASE,
)


def strip_question_prefix(text: str) -> str:
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = QUESTION_PREFIX_RE.sub("", text)
    return text.strip()


def detect_qid(text: str) -> Optional[int]:
    match = re.search(r"\b(?:q|question)\s*(\d{1,2})", text, re.IGNORECASE)
    if match:
        qid = int(match.group(1))
        if 1 <= qid <= 16:
            return qid
    return None


def find_matching_question(line: str, next_line: str) -> Tuple[Optional[str], Optional[int]]:
    candidates = [line, f"{line} {next_line}".strip()]
    for cand in candidates:
        cand_core = strip_question_prefix(cand)
        cand_norm = normalize_question(cand_core)
        qid_hint = detect_qid(cand)
        for canon_norm, canon_question in CANONICAL_NORMALIZED.items():
            if canon_norm in cand_norm or cand_norm in canon_norm:
                return canon_question, qid_hint
    return None, None


def extract_answers(text: str) -> Dict[str, Tuple[str, Optional[int]]]:
    lines = [clean_text(line) for line in (text or "").splitlines() if clean_text(line)]
    found: Dict[str, Tuple[str, Optional[int]]] = {}

    for idx, line in enumerate(lines):
        next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
        match_question, qid_hint = find_matching_question(line, next_line)
        if not match_question:
            continue
        norm = normalize_question(match_question)
        if norm in found:
            continue

        answer = ""
        for j in range(idx, len(lines)):
            if re.search(r"^answer\s*:", lines[j], re.IGNORECASE):
                answer = re.sub(r"^answer\s*:\s*", "", lines[j], flags=re.IGNORECASE)
                break

        found[norm] = (answer, qid_hint)

    return found


def main():
    df = pd.read_csv(INPUT_CSV, dtype={"PMID": str})
    qid_lookup = build_qid_lookup()

    output_rows = []
    for _, row in df.iterrows():
        pmid = str(row["PMID"])
        parsed = extract_answers(row.get("FT Answer", ""))

        for question in CANONICAL_QUESTIONS:
            norm = normalize_question(question)
            answer, qid_hint = parsed.get(norm, ("", None))
            qid = qid_hint if qid_hint is not None else qid_lookup.get(norm)

            output_rows.append(
                {
                    "PMID": pmid,
                    "QID": qid,
                    "Question": question,
                    "Answer": clean_text(answer),
                }
            )

    output = pd.DataFrame(output_rows, columns=["PMID", "QID", "Question", "Answer"])
    output.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(output)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
