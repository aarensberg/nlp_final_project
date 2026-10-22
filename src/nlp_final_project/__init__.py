"""Contract-review NLP core: clause classification, clause-bank retrieval, audit."""

import sys

USAGE = """usage: nlp-final-project <command> [options]

  eda        explore LEDGAR (imbalance, lengths, leakage, slices)
  baseline   train the TF-IDF + logistic regression baseline (~5 min, CPU)
  train      fine-tune Legal-BERT (~2.5 h on an Apple M3; see `train --help`)
  evaluate   fit inference parameters on validation, evaluate 3 variants on test
  classify   predict the type of a clause:  classify "This Agreement shall be governed by ..."
  index      build (if needed) + evaluate the clause-bank retrieval index
  search     query the clause bank:  search "termination with 30 days notice"
  audit      bias and failure-mode audit (needs `evaluate` first)
  all        eda, evaluate, index, audit  (never trains; needs existing models)
"""


def main() -> None:
    import json

    cmd, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("help", [])
    if cmd == "eda":
        from .eda import run
        print(json.dumps(run(), indent=2)[:3000])
    elif cmd == "baseline":
        from .baseline import run
        print(json.dumps(run(), indent=2))
    elif cmd == "train":
        from .train import cli
        cli(args)
    elif cmd == "evaluate":
        from .evaluate import run
        out = run()
        print(json.dumps({v: r["test"] for v, r in out["variants"].items()}, indent=2))
    elif cmd == "classify":
        from .inference import cli_classify
        cli_classify(args)
    elif cmd == "index":
        from .retrieval import cli_index
        cli_index()
    elif cmd == "search":
        from .retrieval import cli_search
        cli_search(args)
    elif cmd == "audit":
        from .audit import run
        run()
        print("audit written to reports/audit.json and reports/figures/")
    elif cmd == "all":
        from .config import ARTIFACTS
        missing = [p for p in (ARTIFACTS / "classifier" / "model.safetensors", ARTIFACTS / "baseline.joblib")
                   if not p.exists()]
        if missing:
            sys.exit("missing models: " + ", ".join(map(str, missing))
                     + "\nrun `baseline` and `train` explicitly first (training takes hours).")
        from . import audit, eda, evaluate, retrieval
        eda.run(); evaluate.run(); retrieval.cli_index(); audit.run()  # noqa: E702
    else:
        print(USAGE)
