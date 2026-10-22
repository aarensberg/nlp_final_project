# Architecture

## Classification : des données à l'inférence

```mermaid
flowchart LR
    D[(LEDGAR révision figée<br/>train 60k · val 10k · test 10k)] --> T1[train.py<br/>Legal-BERT, 2 époques, 256 tokens]
    D --> T2[baseline.py<br/>TF-IDF 1-2 grammes + rég. logistique]
    T1 --> W[[artifacts/classifier/*<br/>poids, tokenizer]]
    T2 --> J[[artifacts/baseline.joblib]]
    W --> LV[logits validation]
    J --> LV
    LV --> FIT[fit_inference_config<br/>températures · poids 0,2 · seuils<br/>VALIDATION uniquement]
    FIT --> CFG[[artifacts/inference_config.json<br/>copie : reports/inference_config.json]]
    W --> LT[logits test]
    J --> LT
    CFG --> EV[evaluate.py<br/>3 variantes, bootstrap, calibration, abstention]
    LT --> EV
    EV --> R1[(reports/eval_metrics.json<br/>reports/per_class_test.csv)]
    CFG --> P[Predictor<br/>refuse un artefact dont l'empreinte a changé]
    W --> P
    J --> P
    P --> CLI[classify<br/>type · top-k · confiance · ABSTAIN]
    P --> AU[audit.py<br/>groupes · contrefactuels · hors périmètre]
    AU --> R2[(reports/audit.json)]
```

## Recherche : de la requête aux résultats

```mermaid
flowchart LR
    Q[question ou clause collée] --> F{filtre « N jours »<br/>any / all}
    F -->|clauses admissibles| DN[dense : bge-small-en-v1.5<br/>cosinus sur 59 080 clauses]
    F -->|clauses admissibles| BM[BM25 vectorisé<br/>score > 0 uniquement]
    DN --> RRF[fusion RRF k=60<br/>profondeur 200]
    BM --> RRF
    RRF --> TOP[top-k<br/>égalités départagées par la position<br/>jamais complété avec des clauses non classées]
    IDX[[artifacts/clause_index/<br/>embeddings · labels · texts · meta avec SHA-256]] --> DN
    IDX --> BM
    TOP --> OUT[résultats + avertissement :<br/>mentionner N jours ≠ préavis de N jours]
```

## Ce qui est versionné, ce qui ne l'est pas

| Versionné (Git) | Exclu de Git (`artifacts/`, reconstruit par les commandes) |
|---|---|
| code `src/`, tests `tests/`, scripts `scripts/` | poids Legal-BERT (418 Mo), checkpoint (1,2 Go) |
| `reports/*.json`, `*.csv`, `*.md`, figures, archives, `validation_logs/` | `baseline.joblib` (139 Mo) |
| `reports/inference_config.json` (copie) | logits en cache (`*_logits.npz`), avec leur empreinte |
| `reports/run_manifest.json` (empreintes de tous les artefacts) | index de recherche (128 Mo) |
| `presentation/` | journaux d'exécution (`*.log`) |
