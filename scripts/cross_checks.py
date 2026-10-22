"""Cross-consistency checks between reports, artifacts and the manifest.

    uv run python scripts/cross_checks.py [--manifest]

Prints one line per check (OK / FAIL) and exits 1 if any check fails.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

from nlp_final_project.config import ARTIFACTS, REPORTS, ROOT
from nlp_final_project.data import label_names, split_arrays
from nlp_final_project.fingerprint import sha256_file

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"{'OK  ' if ok else 'FAIL'} {name}" + (f" - {detail}" if detail else ""))


ev = json.loads((REPORTS / "eval_metrics.json").read_text())
cfg = json.loads((REPORTS / "inference_config.json").read_text())
names = label_names()

rows = list(csv.DictReader(open(REPORTS / "per_class_test.csv")))
check("per_class_test.csv has 100 labels", len(rows) == 100 and {r["label"] for r in rows} == set(names))
check("sum of test supports = 10 000", sum(int(r["test_support"]) for r in rows) == 10_000)
for v in ("ensemble", "bert", "tfidf"):
    csv_macro = np.mean([float(r[f"{v}_f1"]) for r in rows])
    js = ev["variants"][v]["test"]["macro_f1"]
    check(f"{v}: mean of CSV F1 = JSON macro-F1 (4-decimal rounding)", abs(csv_macro - js) <= 5e-5,
          f"{csv_macro:.6f} vs {js:.6f}")

check("inference config fitted on validation only", cfg["created_from"].startswith("validation")
      and cfg["data"]["validation"]["split"] == "validation")
check("thresholds come from the validation details",
      all(cfg["abstention"]["thresholds"][v] == cfg["abstention"]["threshold_details_on_validation"][v]["threshold"]
          for v in ("ensemble", "bert", "tfidf")))
check("ensemble weight = argmax of the validation grid (first maximum)",
      cfg["ensemble"]["weight_bert"] == max(cfg["ensemble"]["val_macro_f1_by_weight"].items(),
                                             key=lambda kv: kv[1])[0] or
      float(cfg["ensemble"]["weight_bert"]) == float(max(cfg["ensemble"]["val_macro_f1_by_weight"].items(),
                                                        key=lambda kv: kv[1])[0]))

bert_cfg = json.loads((ARTIFACTS / "classifier" / "config.json").read_text())
import joblib  # noqa: E402
clf = joblib.load(ARTIFACTS / "baseline.joblib")["classifier"]
check("same label order: dataset = inference config = BERT id2label = TF-IDF classes_",
      cfg["labels"] == names == [bert_cfg["id2label"][str(i)] for i in range(100)]
      and np.array_equal(clf.classes_, np.arange(100)))

# One macro-F1 convention in current reports.
from sklearn.metrics import f1_score  # noqa: E402
from nlp_final_project.inference import Predictor, split_logits  # noqa: E402
P = Predictor()
lb, yt = split_logits("test", "bert")
lt, _ = split_logits("test", "tfidf")
pr = P.probs_from_logits(lb, lt)
lbv, yv = split_logits("validation", "bert")
ltv, _ = split_logits("validation", "tfidf")
prv = P.probs_from_logits(lbv, ltv)
ok = True
for v in ("ensemble", "bert", "tfidf"):
    for split, y, p in (("test", yt, pr[v]), ("validation", yv, prv[v])):
        sk = f1_score(y, p.argmax(1), labels=range(100), average="macro", zero_division=0)
        ok &= abs(ev["variants"][v][split]["macro_f1"] - sk) < 1e-12
check("eval_metrics macro-F1 (test + validation) = sklearn zero_division=0 on range(100)", ok)
check("eval_metrics declares the standard definition", "zero_division=0" in ev["macro_f1_definition"])
check("inference config declares the standard definition", "zero_division=0" in cfg["macro_f1_definition"])
check("all 100 labels present in test truth (point scores do not depend on the convention)",
      len(np.unique(yt)) == 100)
check("test split has 10 000 clauses", len(split_arrays("test")[1]) == 10_000)

if "--manifest" in sys.argv:
    m = json.loads((REPORTS / "run_manifest.json").read_text())
    bad = [r["path"] for r in m["reports"] + list(m["artifacts"].values())
           if r.get("present", True) and sha256_file(ROOT / r["path"]) != r["sha256"]]
    check("manifest hashes match the files on disk", not bad, ", ".join(bad))

n_fail = sum(not ok for _, ok in results)
print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
sys.exit(1 if n_fail else 0)
