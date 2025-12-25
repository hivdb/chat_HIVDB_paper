#!/usr/bin/env python3
"""
Convert a DOCX file to Markdown while preserving decimal list numbering
such as 1.1, 1.2, etc.
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": NS_W}

_QUESTION_HEADER_RE = re.compile(r"^\*\*Question\s+(\d+):\s*\*\*(.+)$")
_QUESTION_GUIDELINES = '''## For each question:

Step 1: get the question, store as "question".
Step 2: extract two or three sentences from the "paper content" that can be used to answer the question, separate them using '.', store as 'evidence'.
Step 3: provide the rationale about how you found the answer from the content in details, store as 'rationale'.
Step 4: answer the question, store as 'answer'.
Step 5: format your answer in the format:

"""
Question: <question>

Evidence: <evidence>

Rationale: <rationale>

Answer: <answer>
"""

Make sure you answer all the questions.
'''


def qn(tag: str) -> str:
    return f"{{{NS_W}}}{tag}"


def escape_markdown(text: str) -> str:
    """Escape Markdown control characters while preserving newlines."""
    text = text.replace("\t", "    ")
    # Escape backslash first to avoid double escaping.
    text = text.replace("\\", "\\\\")
    specials = r"`{}[]()#+!|>"
    text = re.sub(f"([{re.escape(specials)}])", r"\\\1", text)
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    return text


def to_roman(number: int) -> str:
    """Convert an integer to uppercase Roman numerals."""
    numerals = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    result = []
    remainder = number
    for value, numeral in numerals:
        while remainder >= value:
            result.append(numeral)
            remainder -= value
    return "".join(result)


def format_number(value: int, num_fmt: str) -> str:
    if num_fmt in {"decimal", "decimalZero"}:
        return str(value)
    if num_fmt == "lowerLetter":
        return number_to_letters(value).lower()
    if num_fmt == "upperLetter":
        return number_to_letters(value).upper()
    if num_fmt == "lowerRoman":
        return to_roman(value).lower()
    if num_fmt == "upperRoman":
        return to_roman(value)
    # Fallback to decimal representation.
    return str(value)


def number_to_letters(number: int) -> str:
    """Convert 1 -> A, 27 -> AA, etc."""
    result = []
    n = number
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        result.append(chr(ord("A") + remainder))
    return "".join(reversed(result)) or "A"


def fix_first_three_question_headers(lines: list[str]) -> None:
    """Ensure the first three question headers keep the entire line bold."""
    fixed = set()
    for idx, line in enumerate(lines):
        match = _QUESTION_HEADER_RE.match(line)
        if not match:
            continue
        question_number = int(match.group(1))
        if question_number not in {1, 2, 3} or question_number in fixed:
            continue
        remainder = match.group(2).lstrip()
        if not remainder:
            continue
        lines[idx] = f"**Question {question_number}: {remainder}**"
        fixed.add(question_number)


def append_question_guidelines(content: str) -> str:
    clean_content = content.rstrip()
    return f"{clean_content}\n\n{_QUESTION_GUIDELINES.strip()}\n"


def merge_runs(runs):
    merged = []
    for text, bold, italic in runs:
        if not text:
            continue
        if merged and merged[-1][1:] == (bold, italic):
            merged[-1] = (merged[-1][0] + text, bold, italic)
        else:
            merged.append((text, bold, italic))
    return merged


def run_text(run) -> str:
    pieces = []
    for child in run:
        if child.tag == qn("t"):
            pieces.append(child.text or "")
        elif child.tag == qn("tab"):
            pieces.append("\t")
        elif child.tag == qn("br"):
            pieces.append("\n")
    return "".join(pieces)


def runs_with_formatting(paragraph):
    for run in paragraph.findall(".//w:r", NS):
        text = run_text(run)
        if not text:
            continue
        rpr = run.find("w:rPr", NS)
        bold = bool(rpr is not None and rpr.find("w:b", NS) is not None)
        italic = bool(rpr is not None and rpr.find("w:i", NS) is not None)
        yield text, bold, italic


def render_runs(paragraph) -> str:
    merged = merge_runs(list(runs_with_formatting(paragraph)))
    parts = []
    for text, bold, italic in merged:
        escaped = escape_markdown(text)
        if bold and italic:
            parts.append(f"***{escaped}***")
        elif bold:
            parts.append(f"**{escaped}**")
        elif italic:
            parts.append(f"*{escaped}*")
        else:
            parts.append(escaped)
    return "".join(parts).strip()


def load_numbering(docx_path: Path):
    num_to_abstract = {}
    level_defs = {}

    with zipfile.ZipFile(docx_path) as archive:
        try:
            numbering_xml = archive.read("word/numbering.xml")
        except KeyError:
            return num_to_abstract, level_defs

    root = ET.fromstring(numbering_xml)

    for abstract in root.findall("w:abstractNum", NS):
        abstract_id = abstract.attrib.get(qn("abstractNumId"))
        for lvl in abstract.findall("w:lvl", NS):
            ilvl = int(lvl.attrib.get(qn("ilvl"), "0"))
            num_fmt_el = lvl.find("w:numFmt", NS)
            lvl_text_el = lvl.find("w:lvlText", NS)
            start_el = lvl.find("w:start", NS)
            suff_el = lvl.find("w:suff", NS)
            level_defs[(abstract_id, ilvl)] = {
                "numFmt": (
                    num_fmt_el.attrib.get(qn("val"))
                    if num_fmt_el is not None
                    else "decimal"
                ),
                "lvlText": (
                    lvl_text_el.attrib.get(qn("val"))
                    if lvl_text_el is not None
                    else None
                ),
                "start": (
                    int(start_el.attrib.get(qn("val")))
                    if start_el is not None
                    else 1
                ),
                "suffix": (
                    suff_el.attrib.get(qn("val"))
                    if suff_el is not None
                    else "space"
                ),
            }

    for num in root.findall("w:num", NS):
        num_id = num.attrib.get(qn("numId"))
        abstract_id_el = num.find("w:abstractNumId", NS)
        if abstract_id_el is None:
            continue
        abstract_id = abstract_id_el.attrib.get(qn("val"))
        num_to_abstract[num_id] = abstract_id

    return num_to_abstract, level_defs


def compute_label(numbers, abstract_id, ilvl, level_defs):
    fmt = level_defs.get((abstract_id, ilvl), {})
    num_fmt = fmt.get("numFmt", "decimal")
    if num_fmt == "bullet":
        return None  # caller handles bullets

    formatted_numbers = []
    for idx, value in enumerate(numbers):
        info = level_defs.get((abstract_id, idx), {})
        formatted_numbers.append(format_number(value, info.get("numFmt", "decimal")))

    lvl_text = fmt.get("lvlText")
    if lvl_text:
        label = lvl_text
        for idx, formatted in enumerate(formatted_numbers, start=1):
            label = label.replace(f"%{idx}", formatted)
        label = label.replace("%", "").strip()
    else:
        label = ".".join(formatted_numbers)

    if not label:
        label = ".".join(formatted_numbers)

    return label


def convert(docx_path: Path, md_path: Path):
    num_to_abstract, level_defs = load_numbering(docx_path)
    list_state = {}  # numId -> {"levels": [], "abstract": str}
    lines = []
    previous_type = "blank"

    with zipfile.ZipFile(docx_path) as archive:
        document_xml = archive.read("word/document.xml")
    root = ET.fromstring(document_xml)
    body = root.find("w:body", NS)

    for element in body:
        if element.tag == qn("p"):
            ppr = element.find("w:pPr", NS)
            num_pr = None
            num_id = None
            ilvl = None
            if ppr is not None:
                num_pr = ppr.find("w:numPr", NS)
            if num_pr is not None:
                num_id_el = num_pr.find("w:numId", NS)
                ilvl_el = num_pr.find("w:ilvl", NS)
                if num_id_el is not None:
                    num_id = num_id_el.attrib.get(qn("val"))
                if ilvl_el is not None:
                    ilvl = int(ilvl_el.attrib.get(qn("val"), "0"))

            paragraph_text = render_runs(element)

            if num_id is None or ilvl is None:
                if paragraph_text:
                    if previous_type == "list" and lines and lines[-1] != "":
                        lines.append("")
                    lines.append(paragraph_text)
                    previous_type = "text"
                else:
                    if previous_type != "blank":
                        lines.append("")
                    previous_type = "blank"
                continue

            abstract_id = num_to_abstract.get(num_id)
            level_info = level_defs.get((abstract_id, ilvl), {})
            is_bullet = level_info.get("numFmt") == "bullet"

            state = list_state.setdefault(
                num_id, {"levels": [], "abstract": abstract_id}
            )
            if state["abstract"] != abstract_id:
                state["levels"] = []
                state["abstract"] = abstract_id

            while len(state["levels"]) <= ilvl:
                start = level_defs.get((abstract_id, len(state["levels"])), {}).get(
                    "start", 1
                )
                state["levels"].append(start - 1)

            state["levels"][ilvl] += 1
            for deeper in range(ilvl + 1, len(state["levels"])):
                start = level_defs.get((abstract_id, deeper), {}).get("start", 1)
                state["levels"][deeper] = start - 1

            numbers = state["levels"][: ilvl + 1]

            indent = "    " * ilvl
            if is_bullet:
                if paragraph_text:
                    line = f"{indent}- {paragraph_text}"
                else:
                    line = f"{indent}-"
            else:
                label = compute_label(numbers, abstract_id, ilvl, level_defs)
                suffix = ""
                suff_val = level_info.get("suffix", "space")
                if suff_val == "space":
                    suffix = " "
                elif suff_val == "tab":
                    suffix = "\t"
                if not label:
                    label = ".".join(str(n) for n in numbers)
                if paragraph_text:
                    line = f"{indent}{label}{suffix}{paragraph_text}"
                else:
                    line = f"{indent}{label}"

            if previous_type not in {"list", "blank"} and lines and lines[-1] != "":
                lines.append("")

            lines.append(line.rstrip())
            previous_type = "list"
        elif element.tag == qn("sectPr"):
            continue
        else:
            # Unsupported element types (tables, etc.) are skipped with a blank line.
            if previous_type != "blank":
                lines.append("")
            previous_type = "blank"

    fix_first_three_question_headers(lines)
    content = "\n".join(lines)
    content = append_question_guidelines(content)
    md_path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DOCX to Markdown preserving decimal numbering."
    )
    parser.add_argument("docx", type=Path, help="Input DOCX file")
    parser.add_argument(
        "markdown", type=Path, nargs="?", help="Output Markdown file path"
    )
    args = parser.parse_args()

    docx_path = args.docx
    md_path = args.markdown or docx_path.with_suffix(".md")

    if not docx_path.exists():
        print(f"Input file {docx_path} does not exist.", file=sys.stderr)
        return 1

    convert(docx_path, md_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
