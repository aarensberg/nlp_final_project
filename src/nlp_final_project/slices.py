"""Text-derived slices used by the EDA and the audit.

LEDGAR ships no metadata (no contract type, no party, no date per clause), so
every "kind of contract / party" is inferred from the clause wording with
transparent keyword rules. They are proxies: good enough to surface gaps,
not ground truth. Each rule is listed here so a reviewer can check it.
"""

from __future__ import annotations

import re

import numpy as np

# Contract family, detected from the vocabulary of the parties / instrument.
CONTRACT_FAMILIES: dict[str, list[str]] = {
    "employment": [r"\bemployee", r"\bexecutive\b", r"\bemployment\b", r"\bsalary\b",
                   r"\bseverance\b", r"\bemployer\b", r"\bbonus\b"],
    "credit_finance": [r"\bborrower", r"\blender", r"\bloans?\b", r"\bcredit agreement\b",
                       r"\badministrative agent\b", r"\bindenture\b", r"\bdebenture\b",
                       r"\bnoteholder", r"\bcollateral\b", r"\bguarant"],
    "real_estate": [r"\blandlord\b", r"\btenant\b", r"\blease\b", r"\bpremises\b", r"\blessee\b",
                    r"\blessor\b"],
    "m&a_securities": [r"\bmerger\b", r"\bpurchaser\b", r"\bstockholder", r"\bshareholder",
                       r"\bacquir", r"\bunderwriter", r"\bsubscriber\b", r"\bwarrant\b"],
    "commercial_ip": [r"\blicensor\b", r"\blicensee\b", r"\bsupplier\b", r"\bcustomer\b",
                      r"\bdistributor\b", r"\bsoftware\b", r"\bintellectual property\b",
                      r"\bvendor\b", r"\bservices?\b"],
}
_FAMILY_RE = {k: [re.compile(p, re.I) for p in v] for k, v in CONTRACT_FAMILIES.items()}


def contract_family(text: str) -> str:
    """Family with the most keyword hits; 'generic' if none (boilerplate clauses)."""
    scores = {k: sum(bool(r.search(text)) for r in rs) for k, rs in _FAMILY_RE.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "generic"


_MALE = re.compile(r"\b(he|him|his|himself)\b", re.I)
_FEMALE = re.compile(r"\b(she|her|hers|herself)\b", re.I)


def pronoun_group(text: str) -> str:
    m, f = bool(_MALE.search(text)), bool(_FEMALE.search(text))
    if m and f:
        return "both (he/she)"
    return "male" if m else "female" if f else "none"


_US_STATES = (
    "alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|florida|georgia|"
    "hawaii|idaho|illinois|indiana|iowa|kansas|kentucky|louisiana|maine|maryland|massachusetts|"
    "michigan|minnesota|mississippi|missouri|montana|nebraska|nevada|new hampshire|new jersey|"
    "new mexico|new york|north carolina|north dakota|ohio|oklahoma|oregon|pennsylvania|"
    "rhode island|south carolina|south dakota|tennessee|texas|utah|vermont|virginia|washington|"
    "west virginia|wisconsin|wyoming"
)
_US_RE = re.compile(rf"\b(state of )?({_US_STATES})\b", re.I)
_NON_US_RE = re.compile(
    r"\b(england|wales|united kingdom|france|germany|netherlands|luxembourg|ireland|switzerland|"
    r"canada|ontario|british columbia|cayman|bermuda|hong kong|singapore|israel|china|japan|"
    r"european union|belgium|italy|spain)\b", re.I)


def jurisdiction(text: str) -> str:
    us, non_us = bool(_US_RE.search(text)), bool(_NON_US_RE.search(text))
    if non_us:
        return "non-US mentioned"
    return "US state mentioned" if us else "none"


def length_bucket(text: str) -> str:
    n = len(text.split())
    if n <= 30:
        return "a) <=30 words"
    if n <= 80:
        return "b) 31-80"
    if n <= 200:
        return "c) 81-200"
    return "d) >200 words"


SLICERS = {
    "contract_family": contract_family,
    "pronouns": pronoun_group,
    "jurisdiction": jurisdiction,
    "length": length_bucket,
}


def compute_slices(texts: list[str]) -> dict[str, np.ndarray]:
    return {name: np.array([fn(t) for t in texts]) for name, fn in SLICERS.items()}
