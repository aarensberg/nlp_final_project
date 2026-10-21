"""Contract-review NLP core: clause classification, clause-bank retrieval, audit."""

import sys

USAGE = """usage: nlp-final-project <command> [options]

  eda        explore LEDGAR (imbalance, lengths, leakage, slices)
  baseline   TF-IDF + logistic regression
  train      fine-tune Legal-BERT (see `train --help`)
  evaluate   held-out test evaluation, calibration, abstention
  index      build + evaluate the clause-bank retrieval index
  search     query the clause bank: search "termination with 30 days notice"
  audit      bias and failure-mode audit
  all        eda, baseline, train, evaluate, index, audit in order
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
        print(json.dumps(run(), indent=2)[:3000])
    elif cmd == "index":
        from .retrieval import cli_index
        cli_index()
    elif cmd == "search":
        from .retrieval import cli_search
        cli_search(args)
    elif cmd == "audit":
        from .audit import run
        print(json.dumps(run(), indent=2)[:3000])
    elif cmd == "all":
        from . import audit, baseline, eda, evaluate, retrieval, train
        eda.run(); baseline.run(); train.cli(args); evaluate.run()  # noqa: E702
        retrieval.cli_index(); audit.run()  # noqa: E702
    else:
        print(USAGE)
