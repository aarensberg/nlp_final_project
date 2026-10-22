"""Fine-tune a pretrained BERT on LEDGAR with an explicit PyTorch training loop.

Pipeline: tokenise (truncate, no padding) -> length-bucketed batches with
dynamic padding -> AdamW + linear warm-up/decay -> evaluate on validation after
each epoch -> keep the checkpoint with the best macro-F1 (early stopping).

Run:  uv run nlp-final-project train                  # full run
      uv run nlp-final-project train --resume         # continue after an interruption
      uv run nlp-final-project train --max-train 2000 --epochs 1   # smoke test

Progress is written to artifacts/progress.json; a full checkpoint (weights,
optimiser, scheduler, position in the epoch, RNG states, configuration) to
artifacts/classifier_checkpoint/ every `ckpt_every` steps, so an interrupted run
loses at most a few minutes.

Resuming is refused when the configuration differs from the checkpoint's. In
particular `--resume --epochs 3` cannot extend a finished 2-epoch run: its
learning-rate schedule was built for 2 epochs and has already decayed to zero.
Extending training is a new experiment with its own schedule.

Determinism: data order and RNG states are restored exactly, but kernels on the
Apple GPU (MPS) and bfloat16 autocast are not guaranteed bit-reproducible, so a
resumed run is equivalent in procedure, not proven numerically identical.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .config import (ARTIFACTS, CLASSIFIER_MODEL, CLASSIFIER_REVISION, REPORTS, SEED, get_device, save_json,
                     set_seed)
from .data import label_names, split_arrays
from .metrics import core_metrics


@dataclass
class TrainConfig:
    model_name: str = CLASSIFIER_MODEL
    model_revision: str = CLASSIFIER_REVISION
    max_length: int = 256  # covers ~88% of clauses; 512 covers ~98% but doubles cost (see EDA)
    batch_size: int = 16
    eval_batch_size: int = 64
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    epochs: int = 2  # ~1h20 per epoch on an Apple M3; see README for the trade-off
    grad_clip: float = 1.0
    patience: int = 1  # epochs without macro-F1 gain before stopping
    amp: bool = True  # bfloat16 autocast (halves activation memory on the Apple GPU)
    seed: int = SEED
    max_train: int | None = None  # subsample for quick experiments
    max_val: int | None = None  # subsample validation (smoke tests only)
    output_dir: str = str(ARTIFACTS / "classifier")
    ckpt_every: int = 300  # steps between resumable checkpoints
    resume: bool = False


# ---------------------------------------------------------------- data ----
class ClauseDataset(Dataset):
    def __init__(self, encodings: list[list[int]], labels: np.ndarray):
        self.enc, self.labels = encodings, labels

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, i):
        return self.enc[i], int(self.labels[i])


def tokenize(tokenizer, texts: list[str], max_length: int) -> list[list[int]]:
    return tokenizer(texts, truncation=True, max_length=max_length)["input_ids"]


def make_collate(pad_id: int, multiple: int = 64):
    """Dynamic padding: pad each batch to its longest clause, rounded up to a multiple
    of 64. Rounding keeps the number of distinct tensor shapes small, otherwise the
    Apple-GPU allocator caches one buffer per shape and memory blows up."""
    def collate(batch):
        ids, labels = zip(*batch)
        width = -(-max(len(x) for x in ids) // multiple) * multiple
        input_ids = torch.full((len(ids), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(ids), width), dtype=torch.long)
        for i, x in enumerate(ids):
            input_ids[i, : len(x)] = torch.tensor(x)
            mask[i, : len(x)] = 1
        return {"input_ids": input_ids, "attention_mask": mask,
                "labels": torch.tensor(labels, dtype=torch.long)}
    return collate


def bucketed_batches(lengths: list[int], batch_size: int, seed: int, pool: int = 50):
    """Shuffle, sort inside pools of `pool` batches by length, then shuffle batches.

    Keeps randomness while cutting padding (clause lengths range 5-2000 tokens).
    """
    rng = random.Random(seed)
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    batches = []
    chunk = batch_size * pool
    for s in range(0, len(idx), chunk):
        part = sorted(idx[s : s + chunk], key=lambda i: lengths[i])
        batches += [part[j : j + batch_size] for j in range(0, len(part), batch_size)]
    rng.shuffle(batches)
    return batches


def sorted_batches(lengths: list[int], batch_size: int):
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    return [order[j : j + batch_size] for j in range(0, len(order), batch_size)]


# ------------------------------------------------------------ inference ----
@torch.no_grad()
def predict_logits(model, encodings, pad_id, device, batch_size=64, desc="predict") -> np.ndarray:
    """Logits in the original order (batches are length-sorted for speed)."""
    model.eval()
    ds = ClauseDataset(encodings, np.zeros(len(encodings), dtype=int))
    batches = sorted_batches([len(e) for e in encodings], batch_size)
    loader = DataLoader(ds, batch_sampler=batches, collate_fn=make_collate(pad_id))
    out = np.zeros((len(encodings), model.config.num_labels), dtype=np.float32)
    for b_idx, batch in zip(batches, tqdm(loader, desc=desc, leave=False)):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type != "cpu"):
            logits = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device)).logits
        out[b_idx] = logits.float().cpu().numpy()
    _free(device)
    return out


def _free(device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()


# ---------------------------------------------------------------- train ----
def _has_safetensors(name: str, revision: str | None = None) -> bool:
    """Legal-BERT only ships pytorch_model.bin. Without this check, transformers
    tries to fetch a safetensors conversion from a Hub pull request, which can hang."""
    if Path(name).is_dir():
        return any(Path(name).glob("*.safetensors"))
    try:
        from huggingface_hub import list_repo_files

        return any(f.endswith(".safetensors") for f in list_repo_files(name, revision=revision))
    except Exception:
        return False


# Settings that define the optimisation trajectory; they must match to resume.
RESUME_KEYS = ("model_name", "model_revision", "max_length", "batch_size", "lr", "weight_decay",
               "warmup_ratio", "epochs", "grad_clip", "amp", "seed", "max_train", "max_val")


class ResumeError(RuntimeError):
    pass


def check_resume_compatible(saved_cfg: dict | None, cfg: "TrainConfig") -> None:
    if saved_cfg is None:
        raise ResumeError("checkpoint has no saved configuration (written by an older version); "
                          "its compatibility cannot be verified, so it is not resumed")
    current = asdict(cfg)
    diff = {k: (saved_cfg.get(k), current[k]) for k in RESUME_KEYS if saved_cfg.get(k) != current[k]}
    if diff:
        lines = ", ".join(f"{k}: checkpoint={a!r} now={b!r}" for k, (a, b) in diff.items())
        hint = (" Changing `epochs` would not extend training: the LR schedule of the checkpoint "
                "was built for its own number of epochs." if "epochs" in diff else "")
        raise ResumeError(f"configuration differs from the checkpoint ({lines}).{hint}")


def rng_state() -> dict:
    s = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        s["mps"] = torch.mps.get_rng_state()
    return s


def set_rng_state(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])
    if "mps" in s and torch.backends.mps.is_available():
        torch.mps.set_rng_state(s["mps"])


def train(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    device = get_device()
    names = label_names()

    Xtr, ytr = split_arrays("train")
    Xva, yva = split_arrays("validation")
    if cfg.max_train:
        rng = np.random.default_rng(cfg.seed)
        keep = rng.choice(len(Xtr), cfg.max_train, replace=False)
        Xtr, ytr = [Xtr[i] for i in keep], ytr[keep]
    if cfg.max_val:
        Xva, yva = Xva[: cfg.max_val], yva[: cfg.max_val]

    tok = AutoTokenizer.from_pretrained(cfg.model_name, revision=cfg.model_revision)
    enc_tr = tokenize(tok, Xtr, cfg.max_length)
    enc_va = tokenize(tok, Xva, cfg.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, revision=cfg.model_revision, num_labels=len(names),
        use_safetensors=_has_safetensors(cfg.model_name, cfg.model_revision),
        id2label=dict(enumerate(names)), label2id={n: i for i, n in enumerate(names)},
    ).to(device)

    # No weight decay on biases and LayerNorm weights (standard BERT recipe).
    no_decay = ("bias", "LayerNorm.weight", "layernorm", "norm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    optim = torch.optim.AdamW(groups, lr=cfg.lr)
    steps_per_epoch = math.ceil(len(enc_tr) / cfg.batch_size)
    total = steps_per_epoch * cfg.epochs
    sched = get_linear_schedule_with_warmup(optim, int(cfg.warmup_ratio * total), total)
    collate = make_collate(tok.pad_token_id)
    lengths = [len(e) for e in enc_tr]
    ds_tr = ClauseDataset(enc_tr, ytr)

    log = {"config": asdict(cfg), "device": str(device), "epochs": []}
    st = {"epoch": 0, "step_in_epoch": 0, "step": 0, "best": -1.0, "bad_epochs": 0,
          "running": 0.0, "seen": 0, "elapsed": 0.0}
    ckpt = Path(cfg.output_dir + "_checkpoint") / "state.pt"
    default_run = Path(cfg.output_dir) == ARTIFACTS / "classifier"
    progress_path = ARTIFACTS / "progress.json" if default_run else Path(cfg.output_dir + "_progress.json")
    if cfg.resume and ckpt.exists():
        saved = torch.load(ckpt, map_location="cpu", weights_only=False)
        check_resume_compatible(saved.get("config"), cfg)
        model.load_state_dict(saved["model"])
        optim.load_state_dict(saved["optim"])
        sched.load_state_dict(saved["sched"])
        st, log["epochs"] = saved["state"], saved["epochs"]
        if "rng" in saved:
            set_rng_state(saved["rng"])
        print(f"resumed at epoch {st['epoch'] + 1}, step {st['step']}/{total}")

    def save_checkpoint():
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        st["elapsed"] = base + time.time() - t_start
        tmp = ckpt.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "sched": sched.state_dict(),
                    "state": st, "epochs": log["epochs"], "config": asdict(cfg), "rng": rng_state()}, tmp)
        tmp.replace(ckpt)  # atomic: an interruption never leaves a corrupt checkpoint

    def report(phase: str):
        el = base + time.time() - t_start
        done = st["step"] - step0  # rate measured on this session only
        eta = (total - st["step"]) * (time.time() - t_start) / done / 60 if done else None
        save_json({"phase": phase, "epoch": min(st["epoch"] + 1, cfg.epochs), "epochs": cfg.epochs, "step": st["step"],
                   "total_steps": total, "percent": round(100 * st["step"] / total, 1),
                   "train_loss": st["running"] / max(st["seen"], 1),
                   "elapsed_min": round(el / 60, 1), "eta_min": round(eta, 1) if eta else None,
                   "val_history": log["epochs"]}, progress_path)

    step0, base, t_start = st["step"], st["elapsed"], time.time()
    for epoch in range(st["epoch"], cfg.epochs):
        st["epoch"] = epoch
        model.train()
        batches = bucketed_batches(lengths, cfg.batch_size, cfg.seed + epoch)[st["step_in_epoch"]:]
        loader = DataLoader(ds_tr, batch_sampler=batches, collate_fn=collate)
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{cfg.epochs}", initial=st["step_in_epoch"],
                    total=steps_per_epoch)
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.amp and device.type != "cpu"):
                out = model(**batch)  # cross-entropy computed by the HF head
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            st["running"] += out.loss.item()
            st["seen"] += 1
            st["step"] += 1
            st["step_in_epoch"] += 1
            if st["step"] % 200 == 0:
                _free(device)
            if st["step"] % 50 == 0:
                pbar.set_postfix(loss=f"{st['running'] / st['seen']:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")
                report("training")
            if st["step"] % cfg.ckpt_every == 0:
                save_checkpoint()

        report("validating")
        logits = predict_logits(model, enc_va, tok.pad_token_id, device, cfg.eval_batch_size, "validate")
        m = core_metrics(yva, logits.argmax(1))
        m.update(epoch=epoch + 1, train_loss=st["running"] / max(st["seen"], 1),
                 elapsed_min=(base + time.time() - t_start) / 60)
        log["epochs"].append(m)
        print(json.dumps(m))
        st.update(step_in_epoch=0, running=0.0, seen=0, epoch=epoch + 1)

        improved = m["macro_f1"] > st["best"]
        if improved:
            st["best"], st["bad_epochs"] = m["macro_f1"], 0
            model.save_pretrained(cfg.output_dir)
            tok.save_pretrained(cfg.output_dir)
            save_json(asdict(cfg), Path(cfg.output_dir) / "training_config.json")
        else:
            st["bad_epochs"] += 1
        save_checkpoint()
        report("epoch done")
        if st["bad_epochs"] > cfg.patience:
            print("early stopping")
            break

    log["best_epoch"] = int(np.argmax([e["macro_f1"] for e in log["epochs"]])) + 1
    log["best_val_macro_f1"] = st["best"]
    log["train_minutes"] = (base + time.time() - t_start) / 60
    report("finished")
    # Only the default run writes the committed report; smoke tests stay in their own folder.
    save_json(log, REPORTS / "train_log.json" if default_run else Path(cfg.output_dir + "_train_log.json"))
    return log


def cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f, v in asdict(TrainConfig()).items():
        if isinstance(v, bool):
            p.add_argument(f"--{f.replace('_', '-')}", action=argparse.BooleanOptionalAction, default=v)
            continue
        typ = int if f in ("max_train", "max_val") else type(v) if v is not None else str
        p.add_argument(f"--{f.replace('_', '-')}", type=typ, default=v)
    args = p.parse_args(argv)
    return train(TrainConfig(**vars(args)))


if __name__ == "__main__":
    cli()
