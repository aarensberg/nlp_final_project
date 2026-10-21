# ROLE

You are an AI engineer at a European legal-tech start-up: build, evaluate and audit the NLP core of a contract-review assistant that recognises the type of every clause, retrieves similar clauses from a clause bank, and documents where it can (and cannot) be trusted.

# DATA

```python
from datasets import load_dataset
import numpy as np

ledgar = load_dataset("coastalcph/lex_glue", "ledgar")
names = ledgar["train"].features["label"].names
print(ledgar)

counts = np.bincount(ledgar["train"]["label"], minlength=len(names))
print(names[counts.argmax()], counts.max(), "|", names[counts.argmin()], counts.min())
texts = list(ledgar["train"]["text"])
```

# CONTEXT

Law firms and in-house legal teams spend thousands of hours reading contracts during due diligence (mergers, financings, supplier audits). A junior lawyer reads every clause, tags its type (governing law, termination, indemnification, confidentiality…), looks for precedents in the firm's clause bank, and flags unusual wording for a senior partner.

Your start-up sells an assistant that speeds this up. Its NLP core must:

1. classify every clause into one of the clause types used by the review checklist ;
2. retrieve the most similar clauses from a clause bank, given a pasted clause or a lawyer's question ("show me termination clauses with a 30-day notice period") ;
3. come with evidence that it works (overall, per clause type, and across the kinds of contracts and parties it will meet) and with a clear statement of what it must not be used for.

European clients will not deploy a black box: they will ask for reproducible results, a documented evaluation and an audit of failure modes. You are the engineer in charge of this core, from the first look at the data to the defence of your choices in front of the CTO (the jury).

# YOUR MISSION

- Fine-tune a pretrained language model (BERT / CamemBERT) on a contract-clause classification dataset.
- Implement the full pipeline: tokenisation, training loop, evaluation on a held-out set.
- Build a sentence-transformer document-retrieval index.
- Produce a bias audit.
