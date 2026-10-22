"""Resume = continuous? Checked on CPU with a tiny local BERT (dropout on), no download.

This proves the procedure (checkpoint content, RNG restoration, data order) on CPU.
It says nothing about bit-identity on the Apple GPU (MPS) or with bfloat16.
"""

import random

import numpy as np
import pytest
import torch

from nlp_final_project.train import TrainConfig, fit

DEVICE = torch.device("cpu")


def _tiny_model():
    from transformers import BertConfig, BertForSequenceClassification
    cfg = BertConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=32, max_position_embeddings=64, num_labels=3,
                     hidden_dropout_prob=0.3, attention_probs_dropout_prob=0.3)
    return BertForSequenceClassification(cfg)


def _data(seed=0):
    rng = np.random.default_rng(seed)
    enc = [list(rng.integers(1, 50, rng.integers(3, 13))) for _ in range(40)]
    return enc[:28], rng.integers(0, 3, 28), enc[28:], rng.integers(0, 3, 12)


def _cfg(out, resume=False):
    # 28 clauses / batch 4 = 7 steps per epoch, 2 epochs = 14 steps, checkpoint every 3 steps.
    return TrainConfig(model_name="tiny-local", model_revision="none", epochs=2, batch_size=4,
                       eval_batch_size=8, ckpt_every=3, amp=False, seed=0, patience=5,
                       warmup_ratio=0.1, max_length=16, output_dir=str(out), resume=resume)


def _run(out, stop=None, resume=False, init_seed=0):
    torch.manual_seed(init_seed); np.random.seed(init_seed); random.seed(init_seed)  # noqa: E702
    model = _tiny_model()
    etr, ytr, eva, yva = _data()
    log = fit(_cfg(out, resume), model, etr, ytr, eva, yva, pad_id=0, device=DEVICE,
              stop_after_steps=stop)
    return model, log


def _final_state(out):
    return torch.load(str(out) + "_checkpoint/state.pt", map_location="cpu", weights_only=False)


def _assert_same(a_model, a_out, b_model, b_out):
    for (n, p), (_, q) in zip(a_model.state_dict().items(), b_model.state_dict().items()):
        assert torch.equal(p, q), f"parameter {n} differs"
    A, B = _final_state(a_out), _final_state(b_out)
    for ga, gb in zip(A["optim"]["state"].values(), B["optim"]["state"].values()):
        for k in ga:
            assert torch.equal(torch.as_tensor(ga[k]), torch.as_tensor(gb[k])), f"optimizer {k} differs"
    assert A["sched"] == B["sched"]
    for k in ("epoch", "step", "step_in_epoch", "best", "bad_epochs"):
        assert A["state"][k] == B["state"][k], k
    assert A["epochs"] == [] or [e["macro_f1"] for e in A["epochs"]] == [e["macro_f1"] for e in B["epochs"]]
    assert torch.equal(A["rng"]["torch"], B["rng"]["torch"])
    assert np.array_equal(A["rng"]["numpy"][1], B["rng"]["numpy"][1])
    assert A["rng"]["python"] == B["rng"]["python"]


@pytest.mark.parametrize("stop, where", [
    (5, "mid-epoch: last checkpoint at step 3, steps 4-5 are lost and redone"),
    (7, "last step of epoch 1, before its validation (checkpoint at step 6)"),
    (8, "first step of epoch 2 (resumes from the end-of-epoch-1 checkpoint)"),
])
def test_interrupted_and_resumed_equals_continuous(tmp_path, stop, where):
    cont_model, cont_log = _run(tmp_path / "continuous")
    _, part = _run(tmp_path / "resumed", stop=stop)
    assert part["interrupted_at_step"] == stop
    # A new "process": different global seeds, so only the checkpoint can restore the state.
    res_model, res_log = _run(tmp_path / "resumed", resume=True, init_seed=123)
    assert len(res_log["epochs"]) == 2
    _assert_same(cont_model, tmp_path / "continuous", res_model, tmp_path / "resumed")


def test_dropout_is_active_so_the_test_is_meaningful(tmp_path):
    torch.manual_seed(0)
    m = _tiny_model().train()
    x = torch.tensor([[1, 2, 3, 4]])
    assert not torch.equal(m(input_ids=x).logits, m(input_ids=x).logits)
