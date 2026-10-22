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
        save_json({"spec": index_spec(), "n_clauses": len(texts), "dim": int(emb.shape[1]),
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
            print("[index] stored index was built with a different encoder/config/bank; rebuilding")
            return cls.build()
        if verify_files and sha256_file(INDEX_DIR / "embeddings.npy") != meta.get("embeddings_sha256"):
            print("[index] embeddings file does not match its recorded fingerprint; rebuilding")
            return cls.build()
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

    @staticmethod
    def rrf(*score_arrays: np.ndarray, k: int = 60, depth: int = 200) -> np.ndarray:
        """Reciprocal-rank fusion over the top `depth` of each ranking."""
        fused = np.zeros_like(score_arrays[0], dtype=np.float64)
        for s in score_arrays:
            top = np.argpartition(-s, depth)[:depth]
            top = top[np.argsort(-s[top])]
            fused[top] += 1.0 / (k + np.arange(1, depth + 1))
        return fused

    def search(self, query: str, k: int = 10, mode: str = "hybrid", is_question: bool = True,
               exclude_hash: str | None = None, enforce_periods: bool = True,
               label_filter: set[int] | None = None) -> list[Hit]:
        if mode in ("dense", "hybrid"):
            dense = self.dense_scores(self.embed([query], is_question)[0])
        if mode in ("bm25", "hybrid"):
            lex = self.bm25_scores(query)
        scores = dense if mode == "dense" else lex if mode == "bm25" else self.rrf(dense, lex)
        scores = scores.astype(np.float64).copy()

        if enforce_periods and is_question:
            wanted = day_periods(query)
            if wanted:
                ok = np.array([bool(wanted & p) for p in self.periods])
                scores[~ok] = -np.inf
        if label_filter is not None:
            scores[~np.isin(self.labels, list(label_filter))] = -np.inf

        order = np.argsort(-scores)
        hits = []
        for i in order:
            if not np.isfinite(scores[i]):
                break
            if exclude_hash and self.hashes[i] == exclude_hash:
                continue
            hits.append(Hit(len(hits) + 1, float(scores[i]), self.names[self.labels[i]], self.texts[i]))
            if len(hits) == k:
                break
        return hits


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


def evaluate(index: ClauseIndex) -> dict:
    names = index.names
    out: dict = {"bank_size": len(index.texts), "embedding_model": EMBEDDING_MODEL}

    # 1) clause -> similar clauses (relevance = same clause type), duplicates excluded.
    qs, qy = _sample_queries()
    qemb = index.embed(qs, is_question=False)
    res = {m: {"p@1": [], "p@5": [], "p@10": [], "mrr@10": []} for m in ("dense", "bm25", "hybrid")}
    per_label_p10 = {m: {} for m in res}
    for q, y, e in zip(qs, qy, qemb):
        dense = index.dense_scores(e)
        lex = index.bm25_scores(q)
        h = norm_hash(q)
        for mode, s in (("dense", dense), ("bm25", lex), ("hybrid", index.rrf(dense, lex))):
            top = np.argpartition(-s, 11)[:11]
            top = top[np.argsort(-s[top])]
            top = [i for i in top if index.hashes[i] != h][:10]
            rel = index.labels[top] == y
            res[mode]["p@1"].append(rel[0])
            res[mode]["p@5"].append(rel[:5].mean())
            res[mode]["p@10"].append(rel.mean())
            first = np.where(rel)[0]
            res[mode]["mrr@10"].append(1 / (first[0] + 1) if len(first) else 0.0)
            per_label_p10[mode].setdefault(names[y], []).append(rel.mean())
    out["clause_to_clause"] = {
        m: {k: float(np.mean(v)) for k, v in d.items()} | {"n_queries": len(qs)} for m, d in res.items()
    }
    worst = sorted(((k, float(np.mean(v))) for k, v in per_label_p10["hybrid"].items()), key=lambda kv: kv[1])
    out["clause_to_clause_worst_labels_hybrid_p@10"] = worst[:10]

    out["clause_to_clause_protocol"] = (
        "Queries: test clauses sampled per class (up to 10 per class -> 981 queries; classes with "
        "fewer test clauses contribute all of them). Bank: de-duplicated train split. A retrieved "
        "clause is relevant iff it has the same LEDGAR label as the query. Exact duplicates of the "
        "query (normalised text) are removed from the results before scoring.")

    # 2) lawyer questions -> clause types (+ day-period constraint satisfaction).
    qres, per_question = [], []
    for mode in ("dense", "bm25", "hybrid"):
        for enforce in ((False, True) if mode == "hybrid" else (False,)):
            name = mode + (" + period filter" if enforce else "")
            p10, p1, period_ok = [], [], []
            for q, ok_labels in QUESTIONS:
                hits = index.search(q, 10, mode=mode, is_question=True, enforce_periods=enforce)
                rel = [h.label in ok_labels for h in hits]
                p10.append(np.mean(rel) if rel else 0.0)
                p1.append(float(rel[0]) if rel else 0.0)
                want = day_periods(q)
                pm = [bool(want & day_periods(h.text)) for h in hits] if want else None
                if want:
                    period_ok.append(np.mean(pm) if pm else 0.0)
                per_question.append({
                    "mode": name, "question": q, "accepted_labels": ok_labels, "n_hits": len(hits),
                    "p@1": p1[-1], "p@10": p10[-1],
                    "requested_days": sorted(want) if want else None,
                    "hits": [{"rank": h.rank, "label": h.label, "relevant": r,
                              "mentions_requested_days": (pm[j] if pm else None),
                              "text": h.text[:240]} for j, (h, r) in enumerate(zip(hits, rel))],
                })
            qres.append({"mode": name, "p@1": float(np.mean(p1)), "p@10": float(np.mean(p10)),
                         "period_match@10 (constraint questions)": float(np.mean(period_ok)),
                         "n_constraint_questions": len(period_ok)})
    out["questions"] = qres
    out["n_questions"] = len(QUESTIONS)
    out["questions_protocol"] = (
        "40 hand-written questions (fixed before results), hybrid/dense/BM25 top-10. Relevant iff the "
        "hit's LEDGAR label is in the accepted list for that question. p@10 is computed over the hits "
        "returned (fewer than 10 if the period filter leaves fewer). period_match@10 = share of hits "
        "that MENTION the requested number of days, whatever its legal role.")
    save_json({"protocol": out["questions_protocol"], "results": per_question},
              REPORTS / "retrieval_questions.json")

    # Show the headline example in the report.
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

    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--mode", default="hybrid", choices=["dense", "bm25", "hybrid"])
    p.add_argument("--clause", action="store_true", help="query is a pasted clause, not a question")
    a = p.parse_args(argv)
    hits = ClauseIndex.load().search(a.query, a.k, a.mode, is_question=not a.clause)
    wanted = day_periods(a.query) if not a.clause else set()
    if wanted:
        print(f"(period filter: only clauses mentioning {sorted(wanted)} day(s); the role of the "
              f"duration is not checked)\n")
    if not hits:
        print("no clause in the bank matches this query" + (" and its period filter" if wanted else ""))
    for h in hits:
        print(f"[{h.rank}] {h.label}  (score {h.score:.3f})\n    {h.text[:400]}\n")


if __name__ == "__main__":
    cli_search(sys.argv[1:])
