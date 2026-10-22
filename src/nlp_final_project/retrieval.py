"""Clause-bank retrieval: dense sentence-transformer index + BM25 + hybrid fusion.

- Clause bank  = LEDGAR train split, de-duplicated (normalised text).
- Dense        = BAAI/bge-small-en-v1.5 embeddings, cosine similarity (exact search
                 with a normalised matrix product; 60k x 384 fits in memory, swap in
                 FAISS for millions of clauses).
- Lexical      = BM25 (exact wording matters in law: "thirty (30) days").
- Hybrid       = reciprocal-rank fusion of the two.
- Constraints  = a "N-day" period in the question is extracted and enforced as a filter,
                 because embeddings are bad at numbers.

Limits of the period filter (`day_periods`): it checks that a clause MENTIONS a
duration of N days, not that this duration plays the requested legal role (a
"30-day notice period" query also matches "payment within 30 days"). Business
and calendar days are not distinguished, and number words outside a short list
("thirty-one days") are missed unless digits are also written.

The index is stored with the fingerprint of everything that produced it
(encoder + revision, sequence length, clause bank). A mismatch triggers a rebuild.

Run: uv run nlp-final-project index        # build + evaluate
     uv run nlp-final-project search "show me termination clauses with a 30-day notice period"
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from .config import ARTIFACTS, EMBEDDING_MODEL, EMBEDDING_REVISION, REPORTS, SEED, get_device, save_json
from .data import label_names, split_arrays
from .fingerprint import sha256_file, sha256_texts

INDEX_DIR = ARTIFACTS / "clause_index"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge convention
_TOKEN = re.compile(r"[a-z0-9]+")
_NUM_WORDS = {"five": 5, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30, "forty-five": 45,
              "sixty": 60, "ninety": 90, "one hundred twenty": 120, "one hundred eighty": 180}


def bm25_tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class SparseBM25:
    """Okapi BM25 as one sparse matrix: score(q) = W @ q_counts.

    Pre-computing the per-(document, term) BM25 weight makes a query a single
    sparse product (~ms on 60k clauses) instead of a Python loop over documents.
    """

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.vec = CountVectorizer(tokenizer=bm25_tokens, lowercase=False, token_pattern=None)
        tf = self.vec.fit_transform(texts).tocsr().astype(np.float32)
        n_docs = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
        dl = np.asarray(tf.sum(1)).ravel()
        norm = k1 * (1 - b + b * dl / dl.mean())
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        data = tf.data * (k1 + 1) / (tf.data + norm[rows]) * idf[tf.indices]
        self.W = sparse.csr_matrix((data, tf.indices, tf.indptr), shape=tf.shape)

    def get_scores(self, query: str) -> np.ndarray:
        q = self.vec.transform([query])
        return np.asarray((self.W @ q.T).todense()).ravel()


def norm_hash(t: str) -> str:
    return hashlib.md5(re.sub(r"\s+", " ", t.lower()).strip().encode()).hexdigest()


def day_periods(text: str) -> set[int]:
    """All 'N day(s)' durations mentioned in a text.

    Handles '30 days', '30-day', 'thirty (30) days', 'thirty days', '5 business days',
    '10 calendar days'. Returns the numbers only: the role of the duration (notice,
    payment, cure...) and the business/calendar distinction are NOT captured.
    """
    t = text.lower()
    found = {int(n) for n in re.findall(r"\(?(\d{1,3})\)?[\s-]*(?:calendar |business )?days?\b", t)}
    for w, n in _NUM_WORDS.items():
        if re.search(rf"\b{w}[\s-]*(?:calendar |business )?days?\b", t):
            found.add(n)
    return found


def encoder():
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION, device=str(get_device()))
    m.max_seq_length = MAX_SEQ_LENGTH
    return m


MAX_SEQ_LENGTH = 256


def encoder_context() -> dict:
    """Numeric context of embeddings computed now (informational for the stored bank).

    Policy: the bank embeddings are a stored artifact, like model weights. They are
    REUSED across backends (re-encoding 59k clauses on CPU would take on the order of
    an hour); a mismatch is reported on stderr. Query embeddings are computed in the
    current context, so scores can differ slightly from the reported evaluation.
    """
    import sentence_transformers
    import torch

    return {"backend": get_device().type, "precision": "float32",
            "torch": torch.__version__, "sentence_transformers": sentence_transformers.__version__}


def index_spec() -> dict:
    """What the index depends on. Any change invalidates the stored index."""
    X, y = split_arrays("train")
    return {"encoder": EMBEDDING_MODEL, "encoder_revision": EMBEDDING_REVISION,
            "max_seq_length": MAX_SEQ_LENGTH, "normalize_embeddings": True,
            "bank": "LEDGAR train split, exact duplicates removed (lower-cased, whitespace-normalised; "
                    "first occurrence kept)",
            "train_sha256": sha256_texts(X, y)}


@dataclass
class Hit:
    rank: int
    score: float
    label: str
    text: str


class ClauseIndex:
    def __init__(self, texts: list[str], labels: np.ndarray, emb: np.ndarray):
        self.texts, self.labels, self.emb = texts, labels, emb
        self.names = label_names()
        self._model = None

    # ---------------------------------------------------------- build ----
    @classmethod
    def build(cls) -> "ClauseIndex":
        X, y = split_arrays("train")
        seen, keep = set(), []
        for i, t in enumerate(X):
            h = norm_hash(t)
            if h not in seen:
                seen.add(h)
                keep.append(i)
        texts, labels = [X[i] for i in keep], y[keep]
        t0 = time.time()
        emb = encoder().encode(texts, batch_size=64, normalize_embeddings=True,
                               show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        np.save(INDEX_DIR / "embeddings.npy", emb)
        np.save(INDEX_DIR / "labels.npy", labels)
        (INDEX_DIR / "texts.json").write_text(json.dumps(texts))
        save_json({"spec": index_spec(), "built_with": encoder_context(),
                   "n_clauses": len(texts), "dim": int(emb.shape[1]),
                   "embeddings_sha256": sha256_file(INDEX_DIR / "embeddings.npy"),
                   "encode_seconds": time.time() - t0}, INDEX_DIR / "meta.json")
        return cls(texts, labels, emb)

    @classmethod
    def load(cls, verify_files: bool = True) -> "ClauseIndex":
        meta_path = INDEX_DIR / "meta.json"
        if not (INDEX_DIR / "embeddings.npy").exists() or not meta_path.exists():
            return cls.build()
        meta = json.loads(meta_path.read_text())
        if meta.get("spec") != index_spec():
            print("[index] stored index was built with a different encoder/config/bank; rebuilding",
                  file=sys.stderr)
            return cls.build()
        if verify_files and sha256_file(INDEX_DIR / "embeddings.npy") != meta.get("embeddings_sha256"):
            print("[index] embeddings file does not match its recorded fingerprint; rebuilding",
                  file=sys.stderr)
            return cls.build()
        built, now = meta.get("built_with"), encoder_context()
        if built is None:
            print("[index] build context not recorded", file=sys.stderr)
        elif any(built.get(k) != now[k] for k in now):
            print(f"[index] bank embeddings built with {built}; queries now encoded with {now} "
                  "(reused by policy; scores may differ slightly)", file=sys.stderr)
        return cls(json.loads((INDEX_DIR / "texts.json").read_text()),
                   np.load(INDEX_DIR / "labels.npy"), np.load(INDEX_DIR / "embeddings.npy"))

    @cached_property
    def bm25(self) -> SparseBM25:
        return SparseBM25(self.texts)

    @cached_property
    def hashes(self) -> np.ndarray:
        return np.array([norm_hash(t) for t in self.texts])

    @cached_property
    def periods(self) -> list[set[int]]:
        return [day_periods(t) for t in self.texts]

    @property
    def model(self):
        if self._model is None:
            self._model = encoder()
        return self._model

    # --------------------------------------------------------- scoring ----
    def embed(self, queries: list[str], is_question: bool) -> np.ndarray:
        qs = [QUERY_PREFIX + q for q in queries] if is_question else queries
        return self.model.encode(qs, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)

    def dense_scores(self, qemb: np.ndarray) -> np.ndarray:
        return qemb @ self.emb.T

    def bm25_scores(self, query: str) -> np.ndarray:
        return self.bm25.get_scores(query)

    # ---------------------------------------------------- ranking rules ----
    # 1. Filters (period, label) define the ELIGIBLE set first.
    # 2. Each signal ranks only eligible documents with an informative score:
    #    dense = every eligible document; BM25 = eligible documents with score > 0
    #    (a zero BM25 score shares no term with the query and carries no order).
    # 3. RRF fuses the top `depth` of each ranking. A document ranked by neither
    #    signal has no fused score and is NEVER returned: results are not padded.
    # 4. Ties are broken by bank position (lower index first) - deterministic.
    @staticmethod
    def ranking(scores: np.ndarray, mask: np.ndarray, depth: int | None = None) -> np.ndarray:
        """Indices of `mask` documents sorted by score desc, ties by index asc."""
        idx = np.flatnonzero(mask)
        order = idx[np.lexsort((idx, -scores[idx]))]
        return order if depth is None else order[:depth]

    @staticmethod
    def rrf(rankings: list[np.ndarray], n: int, k: int = 60) -> np.ndarray:
        """Reciprocal-rank fusion: sum of 1/(k + rank) over the rankings a document is in."""
        fused = np.zeros(n, dtype=np.float64)
        for r in rankings:
            fused[r] += 1.0 / (k + np.arange(1, len(r) + 1))
        return fused

    def scored(self, mode: str, dense: np.ndarray | None, lex: np.ndarray | None,
               eligible: np.ndarray | None = None, depth: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """(scores, ranked) for one retrieval mode; `ranked` marks documents with an informative score."""
        n = len(self.texts)
        eligible = np.ones(n, dtype=bool) if eligible is None else eligible
        if mode == "dense":
            return dense.astype(np.float64), eligible.copy()
        if mode == "bm25":
            return lex.astype(np.float64), eligible & (lex > 0)
        fused = self.rrf([self.ranking(dense, eligible, depth), self.ranking(lex, eligible & (lex > 0), depth)], n)
        return fused, fused > 0

    def top(self, scores: np.ndarray, ranked: np.ndarray, k: int, exclude_hash: str | None = None) -> list[int]:
        out = []
        for i in self.ranking(scores, ranked):
            if exclude_hash and self.hashes[i] == exclude_hash:
                continue
            out.append(int(i))
            if len(out) == k:
                break
        return out

    def eligible_mask(self, query: str, enforce_periods: bool, is_question: bool,
                      period_match: str = "any", label_filter: set[int] | None = None) -> tuple[np.ndarray, dict]:
        """Filters. Several durations in one query: 'any' (default) keeps clauses mentioning at
        least one of them; 'all' keeps clauses mentioning every one of them."""
        if period_match not in ("any", "all"):
            raise ValueError("period_match must be 'any' or 'all'")
        eligible = np.ones(len(self.texts), dtype=bool)
        info = {"requested_days": None, "period_match": None}
        if enforce_periods and is_question:
            wanted = day_periods(query)
            if wanted:
                test = (lambda p: bool(wanted & p)) if period_match == "any" else (lambda p: wanted <= p)
                eligible &= np.array([test(p) for p in self.periods])
                info = {"requested_days": sorted(wanted), "period_match": period_match}
        if label_filter is not None:
            eligible &= np.isin(self.labels, list(label_filter))
        info["n_eligible"] = int(eligible.sum())
        return eligible, info

    def search_detailed(self, query: str, k: int = 10, mode: str = "hybrid", is_question: bool = True,
                        exclude_hash: str | None = None, enforce_periods: bool = True,
                        label_filter: set[int] | None = None, period_match: str = "any",
                        depth: int = 200) -> dict:
        if k < 1:
            raise ValueError("k must be >= 1")
        if mode not in ("dense", "bm25", "hybrid"):
            raise ValueError(f"unknown mode {mode!r}")
        eligible, info = self.eligible_mask(query, enforce_periods, is_question, period_match, label_filter)
        dense = self.dense_scores(self.embed([query], is_question)[0]) if mode in ("dense", "hybrid") else None
        lex = self.bm25_scores(query) if mode in ("bm25", "hybrid") else None
        scores, ranked = self.scored(mode, dense, lex, eligible, depth)
        ids = self.top(scores, ranked, k, exclude_hash)
        hits = [Hit(r + 1, float(scores[i]), self.names[self.labels[i]], self.texts[i]) for r, i in enumerate(ids)]
        return {"hits": hits, "n_returned": len(hits), "n_ranked": int(ranked.sum()), **info}

    def search(self, query: str, k: int = 10, mode: str = "hybrid", is_question: bool = True,
               exclude_hash: str | None = None, enforce_periods: bool = True,
               label_filter: set[int] | None = None, period_match: str = "any", depth: int = 200) -> list[Hit]:
        """Top-k hits; may return FEWER than k (never padded with unranked documents)."""
        return self.search_detailed(query, k, mode, is_question, exclude_hash, enforce_periods,
                                    label_filter, period_match, depth)["hits"]


# ------------------------------------------------------- evaluation ----
# 40 natural-language questions written by the authors BEFORE any retrieval result
# was seen, with the LEDGAR labels accepted as relevant (near-synonym labels are
# accepted together, e.g. Governing Laws / Applicable Laws). This is a sanity check
# of the demo use case, not an independent legal benchmark: the questions and the
# relevance criteria come from the same people who built the system. They must not
# be edited after looking at results.
QUESTIONS: list[tuple[str, list[str]]] = [
    ("Which law governs this agreement?", ["Governing Laws", "Applicable Laws"]),
    ("show me termination clauses with a 30-day notice period", ["Terminations"]),
    ("termination for convenience with ninety days written notice", ["Terminations"]),
    ("Which courts have jurisdiction over disputes?", ["Submission To Jurisdiction", "Consent To Jurisdiction", "Jurisdictions", "Venues"]),
    ("Do the parties waive their right to a jury trial?", ["Waiver Of Jury Trials"]),
    ("disputes resolved by binding arbitration", ["Arbitration"]),
    ("obligation to keep information confidential and not disclose it", ["Confidentiality"]),
    ("who indemnifies whom for losses and claims", ["Indemnifications", "Indemnity"]),
    ("if a provision is invalid the rest of the agreement remains in force", ["Severability"]),
    ("this agreement is the entire agreement and supersedes prior understandings", ["Entire Agreements", "Integration"]),
    ("the agreement may be signed in counterparts and by electronic signature", ["Counterparts"]),
    ("how must notices be delivered to the parties", ["Notices"]),
    ("can a party assign the contract without consent", ["Assignments", "Successors", "Assigns", "Binding Effects"]),
    ("amendments must be in writing and signed by both parties", ["Amendments", "Modifications"]),
    ("failure to enforce a right is not a waiver", ["No Waivers", "Waivers"]),
    ("which obligations survive termination of the agreement", ["Survival"]),
    ("the employee's annual base salary", ["Base Salary"]),
    ("paid vacation days for the executive", ["Vacations"]),
    ("what happens if the executive dies during the term", ["Death"]),
    ("termination of employment due to disability", ["Disability"]),
    ("vesting schedule of stock options or restricted shares", ["Vesting"]),
    ("change in control triggers", ["Change In Control"]),
    ("the employee shall not make disparaging statements about the company", ["Non-Disparagement"]),
    ("taxes withheld from payments", ["Tax Withholdings", "Withholdings", "Taxes"]),
    ("how the borrower may use the proceeds of the loans", ["Use Of Proceeds"]),
    ("compliance with anti-bribery and FCPA laws", ["Anti-Corruption Laws"]),
    ("OFAC sanctions and sanctioned persons", ["Sanctions"]),
    ("no broker or finder fees are payable", ["Brokers"]),
    ("each party bears its own costs and expenses", ["Expenses", "Costs", "Fees"]),
    ("the company must maintain insurance coverage", ["Insurances"]),
    ("no public announcement without the other party's consent", ["Publicity"]),
    ("the party is entitled to specific performance and injunctive relief", ["Specific Performance", "Remedies"]),
    ("ownership of intellectual property and inventions", ["Intellectual Property"]),
    ("no pending litigation or proceedings against the company", ["Litigations"]),
    ("the company is solvent and able to pay its debts", ["Solvency"]),
    ("the parties shall execute further documents as reasonably requested", ["Further Assurances"]),
    ("headings are for convenience only and do not affect interpretation", ["Headings"]),
    ("employee releases all claims against the company", ["Releases"]),
    ("notice of termination of at least 60 days", ["Terminations"]),
    ("payment due within 30 days of invoice", ["Payments", "Fees"]),
]


def _sample_queries(n: int = 1000) -> tuple[list[str], np.ndarray]:
    """Stratified-ish sample of test clauses: up to n/100 per class so tail classes count."""
    X, y = split_arrays("test")
    rng = np.random.default_rng(SEED)
    per = max(1, n // len(np.unique(y)))
    idx = []
    for c in np.unique(y):
        pool = np.where(y == c)[0]
        idx += rng.choice(pool, min(per, len(pool)), replace=False).tolist()
    return [X[i] for i in idx], y[idx]


LABEL_APPROXIMATION = (
    "Relevance is approximated by the LEDGAR label: a hit is 'relevant' iff its label is accepted. "
    "This ignores whether the clause actually answers the question (e.g. the requested notice "
    "period) and penalises correct clauses filed under a near-synonym label not listed.")


def _precision_at(rel: np.ndarray, k: int) -> float:
    """Fixed denominator k: missing hits count as non-relevant."""
    return float(rel[:k].sum() / k)


def evaluate(index: ClauseIndex) -> dict:
    names = index.names
    out: dict = {"bank_size": len(index.texts), "embedding_model": EMBEDDING_MODEL,
                 "ranking_rules": "filters first; dense ranks all eligible docs, BM25 only docs with "
                                  "score > 0; RRF(k=60, depth=200); unranked docs never returned; "
                                  "ties broken by bank index",
                 "label_approximation": LABEL_APPROXIMATION}

    # 1) clause -> similar clauses (relevance = same clause type), duplicates excluded.
    qs, qy = _sample_queries()
    qemb = index.embed(qs, is_question=False)
    res = {m: {"p@1": [], "p@5": [], "p@10": [], "mrr@10": [], "n_returned": []}
           for m in ("dense", "bm25", "hybrid")}
    per_label_p10 = {m: {} for m in res}
    for q, y, e in zip(qs, qy, qemb):
        dense = index.dense_scores(e)
        lex = index.bm25_scores(q)
        h = norm_hash(q)
        for mode in res:
            scores, ranked = index.scored(mode, dense, lex)
            top = index.top(scores, ranked, 10, exclude_hash=h)
            rel = index.labels[top] == y
            res[mode]["p@1"].append(_precision_at(rel, 1))
            res[mode]["p@5"].append(_precision_at(rel, 5))
            res[mode]["p@10"].append(_precision_at(rel, 10))
            res[mode]["n_returned"].append(len(top))
            first = np.where(rel)[0]
            res[mode]["mrr@10"].append(1 / (first[0] + 1) if len(first) else 0.0)
            per_label_p10[mode].setdefault(names[y], []).append(_precision_at(rel, 10))
    out["clause_to_clause"] = {
        m: {k: float(np.mean(v)) for k, v in d.items()} | {"n_queries": len(qs)} for m, d in res.items()
    }
    worst = sorted(((k, float(np.mean(v))) for k, v in per_label_p10["hybrid"].items()), key=lambda kv: kv[1])
    out["clause_to_clause_worst_labels_hybrid_p@10"] = worst[:10]
    out["clause_to_clause_protocol"] = (
        "Queries: test clauses sampled per class (up to 10 per class -> 981 queries; classes with "
        "fewer test clauses contribute all of them). Bank: de-duplicated train split. A retrieved "
        "clause is relevant iff it has the same LEDGAR label as the query. Exact duplicates of the "
        "query (normalised text) are removed from the results before scoring. p@k has a fixed "
        "denominator k.")

    # 2) lawyer questions -> clause types (+ numeric day-period constraint).
    qres, per_question = [], []
    for mode in ("dense", "bm25", "hybrid"):
        for enforce in ((False, True) if mode == "hybrid" else (False,)):
            name = mode + (" + period filter" if enforce else "")
            agg = {"p@1": [], "precision_at_10": [], "precision_among_returned": [], "n_returned": []}
            c_at10, c_among = [], []
            for q, ok_labels in QUESTIONS:
                d = index.search_detailed(q, 10, mode=mode, is_question=True, enforce_periods=enforce)
                hits = d["hits"]
                rel = np.array([h.label in ok_labels for h in hits], dtype=bool)
                agg["p@1"].append(_precision_at(rel, 1))
                agg["precision_at_10"].append(_precision_at(rel, 10))
                agg["precision_among_returned"].append(float(rel.mean()) if len(rel) else None)
                agg["n_returned"].append(len(hits))
                want = day_periods(q)
                mention = np.array([bool(want & day_periods(h.text)) for h in hits], dtype=bool) if want else None
                if want:
                    c_at10.append(_precision_at(mention, 10))
                    c_among.append(float(mention.mean()) if len(mention) else None)
                per_question.append({
                    "mode": name, "question": q, "accepted_labels": ok_labels,
                    "n_returned": len(hits), "n_eligible": d["n_eligible"], "n_ranked": d["n_ranked"],
                    "p@1": agg["p@1"][-1], "precision_at_10": agg["precision_at_10"][-1],
                    "precision_among_returned": agg["precision_among_returned"][-1],
                    "requested_days": sorted(want) if want else None,
                    "period_match": d["period_match"],
                    "mention_rate_at_10": (c_at10[-1] if want else None),
                    "hits": [{"rank": h.rank, "label": h.label, "relevant_by_label": bool(r),
                              "mentions_requested_days": (bool(mention[j]) if want else None),
                              "text": h.text[:240]} for j, (h, r) in enumerate(zip(hits, rel))],
                })
            among = [x for x in agg["precision_among_returned"] if x is not None]
            qres.append({
                "mode": name, "p@1": float(np.mean(agg["p@1"])),
                "precision_at_10": float(np.mean(agg["precision_at_10"])),
                "precision_among_returned": float(np.mean(among)) if among else None,
                "mean_n_returned": float(np.mean(agg["n_returned"])),
                "n_questions_with_fewer_than_10": int(sum(n < 10 for n in agg["n_returned"])),
                "duration_mention_rate_at_10": float(np.mean(c_at10)),
                "duration_mention_rate_among_returned": float(np.mean([x for x in c_among if x is not None])),
                "n_constraint_questions": len(c_at10)})
    out["questions"] = qres
    out["n_questions"] = len(QUESTIONS)
    out["questions_protocol"] = (
        "40 hand-written questions, fixed before any result was seen and unchanged since (so a "
        "re-run is a non-regression check, not an independent validation). Top-10 per mode. "
        "precision_at_10 divides by 10 (missing hits count as misses); precision_among_returned "
        "divides by the number of hits returned. duration_mention_rate = share of hits that MENTION "
        "the requested number of days, whatever its legal role: it does not validate a notice "
        "period. " + LABEL_APPROXIMATION)
    save_json({"protocol": out["questions_protocol"], "results": per_question},
              REPORTS / "retrieval_questions.json")

    demo = QUESTIONS[1][0]
    out["example"] = {"query": demo, "hits": [
        {"label": h.label, "text": h.text[:300]} for h in index.search(demo, 5)]}
    save_json(out, REPORTS / "retrieval_metrics.json")
    return out


def cli_index():
    t0 = time.time()
    idx = ClauseIndex.load()
    print(f"index ready ({len(idx.texts)} clauses) in {time.time() - t0:.0f}s; evaluating...")
    print(json.dumps(evaluate(idx), indent=2)[:3000])


def cli_search(argv=None):
    import argparse

    p = argparse.ArgumentParser(prog="nlp-final-project search",
                                description="Search the clause bank (LEDGAR train clauses).")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5, help="maximum number of results (1-100)")
    p.add_argument("--mode", default="hybrid", choices=["dense", "bm25", "hybrid"])
    p.add_argument("--clause", action="store_true", help="query is a pasted clause, not a question")
    p.add_argument("--period-match", choices=["any", "all"], default="any",
                   help="query with several durations: keep clauses mentioning any / all of them")
    p.add_argument("--no-period-filter", action="store_true", help="do not filter on 'N days'")
    a = p.parse_args(argv)
    if not 1 <= a.k <= 100:
        p.error("-k must be between 1 and 100")
    if not a.query.strip():
        p.error("empty query")
    d = ClauseIndex.load().search_detailed(a.query, a.k, a.mode, is_question=not a.clause,
                                           enforce_periods=not a.no_period_filter, period_match=a.period_match)
    if d["requested_days"]:
        print(f"(period filter: clauses mentioning {d['period_match']} of {d['requested_days']} day(s); "
              f"{d['n_eligible']} eligible. Mentioning a duration does not mean it is the requested "
              f"notice period: the role of the duration is not checked)\n")
    if not d["hits"]:
        print("no clause in the bank matches this query" + (" and its period filter" if d["requested_days"] else ""))
    elif d["n_returned"] < a.k:
        print(f"(only {d['n_returned']} ranked result(s); unranked clauses are not shown)\n")
    for h in d["hits"]:
        print(f"[{h.rank}] {h.label}  (score {h.score:.4f})\n    {h.text[:400]}\n")


if __name__ == "__main__":
    cli_search(sys.argv[1:])
