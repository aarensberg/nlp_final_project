"""Counterfactual edits used by the audit.

Each edit should leave the clause TYPE unchanged. Several of them do change the
legal CONTENT (a different jurisdiction or currency has different legal
effects): the audit only measures type stability, not legal equivalence.

Rules are deliberately conservative: clauses where an edit would likely produce
ungrammatical or incoherent text are excluded (`applies` returns False) rather
than edited badly. Each rule is a regex heuristic, not a parser; the audit saves
random before/after samples for manual review.
"""

from __future__ import annotations

import re

import numpy as np

from .config import SEED

# ------------------------------------------------------------------ gender ----
_SIMPLE = {"he": "she", "she": "he", "him": "her", "himself": "herself", "herself": "himself",
           "hers": "his", "mr.": "ms.", "ms.": "mr.", "mrs.": "mr."}
_GENDER_RE = re.compile(r"\b(he|she|him|himself|herself|hers|his|her|mr\.|ms\.|mrs\.)(?=\W|$)", re.I)
# Words after "her" that mean it is an object pronoun ("pay her within"), not a possessive.
_AFTER_OBJECT = {"to", "and", "or", "in", "on", "at", "by", "for", "from", "with", "of", "as", "that",
                 "under", "upon", "within", "if", "until", "unless", "prior", "pursuant", "any", "the",
                 "a", "an", "such", "all", "no", "such", "hereunder", "thereunder"}


def _keep_case(src: str, dst: str) -> str:
    return dst.capitalize() if src[:1].isupper() else dst


def _gender_sub(m: re.Match) -> str:
    w = m.group(0)
    lw = w.lower()
    if lw in _SIMPLE:
        return _keep_case(w, _SIMPLE[lw])
    rest = m.string[m.end():]
    nxt = re.match(r"\s*([A-Za-z]+|[^\sA-Za-z])", rest)
    nxt_word = nxt.group(1).lower() if nxt else ""
    if lw == "his":  # possessive determiner ("his salary") or pronoun ("is his")
        return _keep_case(w, "her" if nxt_word.isalpha() and nxt_word not in _AFTER_OBJECT else "hers")
    # lw == "her": possessive ("her salary") -> his ; object ("pay her within") -> him
    if not nxt_word.isalpha() or nxt_word in _AFTER_OBJECT:
        return _keep_case(w, "him")
    return _keep_case(w, "his")


def gender_swap(t: str) -> str:
    return _GENDER_RE.sub(_gender_sub, t)


def has_gender(t: str) -> bool:
    return bool(_GENDER_RE.search(t))


# ------------------------------------------------------------ jurisdiction ----
# Unambiguous state names only (no Washington, Georgia, Virginia...).
_STATES = r"delaware|new york|california|texas|nevada|illinois|massachusetts|new jersey|florida|" \
          r"pennsylvania|ohio|maryland|colorado|minnesota|north carolina"
_JUR_RE = re.compile(rf"\b(the )?(state of )?({_STATES})\b", re.I)
# Contexts where a country name would be incoherent -> clause excluded.
_JUR_BLOCK = re.compile(rf"\b({_STATES})\s+(county|city|district|general corporation|statutes?|code|"
                        rf"department|secretary)|district of ({_STATES})|({_STATES}),\s+({_STATES})\b"
                        rf"|county of|borough of", re.I)
# "a Delaware corporation" -> adjective form.
_ADJ_NEXT = re.compile(r"\s+(corporation|limited|company|statutory|general partnership|partnership|"
                       r"llc|l\.l\.c\.|law|laws|courts?|resident)\b", re.I)
_EU = [("England and Wales", "English"), ("Germany", "German"), ("France", "French"),
       ("the Netherlands", "Dutch"), ("Ireland", "Irish")]


def eu_jurisdiction(t: str) -> str:
    """US state -> one European jurisdiction per clause (rotated across clauses)."""
    country, adj = _EU[sum(map(ord, t)) % len(_EU)]

    def sub(m: re.Match) -> str:
        if _ADJ_NEXT.match(m.string[m.end():]) and not m.group(2):
            return ("the " if m.group(1) else "") + adj
        return country
    return _JUR_RE.sub(sub, t)


def has_us_state(t: str) -> bool:
    return bool(_JUR_RE.search(t)) and not _JUR_BLOCK.search(t)


# ---------------------------------------------------------------- currency ----
_USD_PHRASE = re.compile(r"\b(U\.S\.|US|United States)\s+[Dd]ollars\b")


def euro(t: str) -> str:
    t = _USD_PHRASE.sub("euros", t)
    t = re.sub(r"(?<![A-Za-z])(?:US)?\$\s?(\d)", r"EUR \1", t)
    return re.sub(r"\bDollars\b", "Euros", re.sub(r"\bdollars\b", "euros", t))


def has_dollars(t: str) -> bool:
    return bool(re.search(r"\$\s?\d|\bdollars\b", t, re.I))


# ---------------------------------------------------------------- spelling ----
# Whole-word pairs only; "judgment" is kept (also the British legal spelling).
_BRIT_WORDS = {
    "organization": "organisation", "organizations": "organisations", "organized": "organised",
    "authorize": "authorise", "authorized": "authorised", "authorizes": "authorises",
    "authorization": "authorisation", "recognize": "recognise", "recognized": "recognised",
    "favor": "favour", "favorable": "favourable", "favorably": "favourably", "labor": "labour",
    "defense": "defence", "defenses": "defences", "center": "centre", "centers": "centres",
    "canceled": "cancelled", "cancellation": "cancellation", "fulfill": "fulfil",
    "fulfillment": "fulfilment", "analyze": "analyse", "utilize": "utilise", "utilized": "utilised",
    "behalf": "behalf", "license": "licence", "licenses": "licences",
}
_BRIT_RE = re.compile(r"\b(" + "|".join(k for k, v in _BRIT_WORDS.items() if k != v) + r")\b", re.I)


def british(t: str) -> str:
    def sub(m: re.Match) -> str:
        w = m.group(0)
        b = _BRIT_WORDS[w.lower()]
        if w.isupper():
            return b.upper()
        return b.capitalize() if w[0].isupper() else b
    # "license" stays a verb in British English: only the noun after an article/possessive changes.
    t = re.sub(r"\b(to|may|shall|will|not)\s+license\b", lambda m: m.group(0).replace("license", "\x00"), t)
    return _BRIT_RE.sub(sub, t).replace("\x00", "license")


def has_american_spelling(t: str) -> bool:
    return british(t) != t


# ------------------------------------------------------------- party names ----
def eu_party_names(t: str) -> str:
    """Generic US party designations -> fictitious European company names."""
    t = re.sub(r"\bthe Company\b", "Lumen Technik GmbH", t)
    t = re.sub(r"\bthe Borrower\b", "Rivière Énergie SAS", t)
    return re.sub(r",? Inc\.", " B.V.", t)


def has_party_names(t: str) -> bool:
    return eu_party_names(t) != t


# ------------------------------------------------------------------- noise ----
def typos(t: str, rate: float = 0.03, seed: int = SEED) -> str:
    """Drop ~3% of letters (OCR / scanning noise)."""
    rng = np.random.default_rng(seed + len(t))
    return "".join(c for c in t if not (c.isalpha() and rng.random() < rate))


PERTURBATIONS = {
    "gender swap (he<->she)": (gender_swap, has_gender,
                               "type should not change; content unchanged apart from gender"),
    "US state -> European jurisdiction": (eu_jurisdiction, has_us_state,
                                          "type should not change; LEGAL EFFECT DOES change"),
    "$ -> EUR": (euro, has_dollars, "type should not change; economic/legal content changes"),
    "British spelling": (british, has_american_spelling, "type and content should not change"),
    "European party names": (eu_party_names, has_party_names,
                             "type should not change; parties change"),
    "3% letter drops (OCR noise)": (typos, lambda t: True, "type should not change; text degraded"),
}
