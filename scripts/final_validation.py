"""Build reports/final_validation.json from the reports, archives and validation logs.

    uv run python scripts/final_validation.py
"""

import json
import re
import subprocess

from nlp_final_project.config import REPORTS, ROOT
from nlp_final_project.fingerprint import sha256_file

BASE = "1c2b565"
ARCH = REPORTS / "archive" / "2026-10-22_before_final_release"
LOGS = REPORTS / "validation_logs"


def git(*a):
    # rstrip only: the leading spaces of `git status --porcelain` are significant
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.rstrip("\n")


def pytest_summary(path):
    text = (LOGS / path).read_text()
    m = re.findall(r"(\d+) (passed|failed|skipped|deselected|error)", text.splitlines()[-1] if text else "")
    out = {k: 0 for k in ("passed", "failed", "skipped", "deselected", "error")}
    out.update({k: int(v) for v, k in m})
    return out


def load(p):
    return json.loads(p.read_text())


ev0, ev1 = load(ARCH / "eval_metrics.json"), load(REPORTS / "eval_metrics.json")
cf0, cf1 = load(ARCH / "inference_config.json"), load(REPORTS / "inference_config.json")
rt0, rt1 = load(ARCH / "retrieval_metrics.json"), load(REPORTS / "retrieval_metrics.json")
au0, au1 = load(ARCH / "audit.json"), load(REPORTS / "audit.json")

before_after = []
for v in ("ensemble", "bert", "tfidf"):
    for split, key in (("test", "micro_f1"), ("test", "macro_f1"), ("test", "micro_f1_95ci"), ("test", "macro_f1_95ci"),
                       ("validation", "micro_f1"), ("validation", "macro_f1")):
        before_after.append({"metric": f"{v}.{split}.{key}", "before_1c2b565": ev0["variants"][v][split][key],
                             "after": ev1["variants"][v][split][key], "source": f"eval_metrics.json:variants/{v}/{split}/{key}"})
    for key in ("test_coverage", "test_correct_rate_among_handled"):
        before_after.append({"metric": f"{v}.abstention.{key}", "before_1c2b565": ev0["variants"][v]["abstention"][key],
                             "after": ev1["variants"][v]["abstention"][key],
                             "source": f"eval_metrics.json:variants/{v}/abstention/{key}"})
    before_after.append({"metric": f"{v}.threshold", "before_1c2b565": cf0["abstention"]["thresholds"][v],
                         "after": cf1["abstention"]["thresholds"][v],
                         "source": f"inference_config.json:abstention/thresholds/{v}"})
for c in ("ensemble_vs_tfidf", "ensemble_vs_bert", "bert_vs_tfidf"):
    for key in ("delta_macro_f1_points", "delta_macro_f1_95ci_points", "delta_micro_f1_95ci_points"):
        before_after.append({"metric": f"{c}.{key}", "before_1c2b565": ev0["comparisons"][c][key],
                             "after": ev1["comparisons"][c][key], "source": f"eval_metrics.json:comparisons/{c}/{key}"})
before_after.append({"metric": "ensemble.weight_bert", "before_1c2b565": cf0["ensemble"]["weight_bert"],
                     "after": cf1["ensemble"]["weight_bert"], "source": "inference_config.json:ensemble/weight_bert"})
for m in ("dense", "bm25", "hybrid"):
    for k in ("p@1", "p@10", "mrr@10"):
        before_after.append({"metric": f"retrieval.clause_to_clause.{m}.{k}",
                             "before_1c2b565": rt0["clause_to_clause"][m][k], "after": rt1["clause_to_clause"][m][k],
                             "source": f"retrieval_metrics.json:clause_to_clause/{m}/{k}"})
for i, q in enumerate(rt1["questions"]):
    for k in ("p@1", "precision_at_10"):
        before_after.append({"metric": f"retrieval.questions_seen.{q['mode']}.{k}",
                             "before_1c2b565": rt0["questions"][i][k], "after": q[k],
                             "source": f"retrieval_metrics.json:questions/{i}/{k}"})
for k in ("slices", "counterfactuals", "out_of_scope", "retrieval"):
    before_after.append({"metric": f"audit.{k}", "before_1c2b565": "see archive", "after":
                         "identical" if json.dumps(au0[k], sort_keys=True) == json.dumps(au1[k], sort_keys=True)
                         else "different", "source": f"audit.json:{k}"})
def differs(a, b, tol=1e-12):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) > tol
    if isinstance(a, list) and isinstance(b, list):
        return len(a) != len(b) or any(differs(x, y, tol) for x, y in zip(a, b))
    return a != b


for r in before_after:
    r["changed"] = r["after"] != "identical" and differs(r["before_1c2b565"], r["after"])

commands = [json.loads(line) for line in (LOGS / "commands.jsonl").read_text().splitlines() if line.strip()]
out = {
    "schema": 1,
    "start_commit": git("rev-parse", BASE),
    "head_commit": git("rev-parse", "HEAD"),
    "working_tree": {"dirty": bool(git("status", "--porcelain")),
                     "changed_paths": sorted({line[3:] for line in git("status", "--porcelain").splitlines()})},
    "tests": {
        "initial_on_1c2b565": {
            "no_artifacts": pytest_summary("initial_tests_no_artifacts.txt"),
            "artifacts_fast": pytest_summary("initial_tests_artifacts_fast.txt"),
            "artifacts_slow": pytest_summary("initial_tests_artifacts_slow.txt")},
        "final": {
            "no_artifacts": pytest_summary("final_tests_no_artifacts.txt"),
            "artifacts_fast": pytest_summary("final_tests_artifacts_fast.txt"),
            "artifacts_slow": pytest_summary("final_tests_artifacts_slow.txt"),
            "full_suite_last": pytest_summary("final_tests_full_last.txt"),
            "clean_rebuild_without_artifacts": pytest_summary("clean_rebuild_tests_no_artifacts.txt"),
            "clean_rebuild_with_linked_artifacts": pytest_summary("clean_rebuild_tests_with_artifacts.txt")},
    },
    "cross_checks": (LOGS / "final_cross_checks_3.txt").read_text().strip().splitlines()[-1],
    "commands": commands,
    "before_after": before_after,
    "manifest_sha256": sha256_file(REPORTS / "run_manifest.json"),
    "predictions_changed": False,
    "retraining": False,
    "verdict": "ready for submission (with the documented limitations)",
}
(REPORTS / "final_validation.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
changed = [r for r in before_after if r["changed"]]
print(f"wrote reports/final_validation.json: {len(before_after)} metrics, {len(changed)} changed")
for r in changed:
    print(" ", r["metric"], r["before_1c2b565"], "->", r["after"])
