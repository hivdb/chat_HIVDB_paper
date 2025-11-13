from __future__ import annotations

import re

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

# Styling helpers for evaluation plots
MODEL_BASE_COLORS = {
    "GPT-5": "#1f77b4",
    "GPT-4o": "#ff7f0e",
    "Llama3.1-70B": "#2ca02c",
    "Llama3.1-8B": "#d62728",
}
VARIANT_TINTS = {
    "base": 0.0,
    "FT": 0.2,
    "AP": 0.55,   # noticeably lighter shade for AP variants (Before/After)
    "BM25": 0.8,
}
