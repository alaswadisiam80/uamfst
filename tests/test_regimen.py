from __future__ import annotations

import torch

from tests.conftest import make_model
from uamfst.fining.adjustments import CompositeObjective
from uamfst.maturation.bottling import load_checkpoint, save_checkpoint
from uamfst.maturation.inoculation import set_seed
from uamfst.maturation.racking import build_optimizer


def test_overfits_single_batch(batch) -> None:
    set_seed(0)
    model = make_model(dropout=0.0)
    objective = torch.nn.functional.cross_entropy
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-3)
    model.train()
    first = None
    last = 0.0
    for step in range(150):
        bundle = model(batch, passes=1)
        loss = objective(bundle.logits, batch.labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 0:
            first = float(loss.detach())
        last = float(loss.detach())
    assert first is not None
    assert last < first * 0.5


def test_frozen_backbone_and_gradient_flow(batch) -> None:
    set_seed(0)
    model = make_model(dropout=0.0)
    objective = CompositeObjective()
    assert sum(p.requires_grad for p in model.pool.backbone.parameters()) == 0
    bundle = model(batch, passes=1)
    loss = objective(bundle.logits, bundle.concentration, batch.labels)
    loss.total.backward()
    for _name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    assert model.projection.linear.weight.grad is not None
    assert float(model.sngp.beta.weight.grad.abs().sum()) > 0.0
    assert float(model.dirichlet.net[0].weight.grad.abs().sum()) > 0.0


def test_seed_determinism(batch) -> None:
    set_seed(7)
    first = make_model(dropout=0.1)
    set_seed(7)
    second = make_model(dropout=0.1)
    set_seed(0)
    out_a = first(batch, passes=3)
    set_seed(0)
    out_b = second(batch, passes=3)
    assert torch.allclose(out_a.logits, out_b.logits, atol=1e-5)


def test_checkpoint_round_trip(tmp_path, batch) -> None:
    set_seed(0)
    model = make_model(dropout=0.0)
    optimizer = build_optimizer(model)
    bundle = model(batch, passes=1)
    torch.nn.functional.cross_entropy(bundle.logits, batch.labels).backward()
    optimizer.step()
    path = tmp_path / "model.ckpt"
    save_checkpoint(path, model, optimizer, epoch=3, seed=99)
    assert path.exists()
    restored = make_model(dropout=0.0)
    payload = load_checkpoint(path, restored)
    assert payload["seed"] == 99
    assert payload["epoch"] == 3
    for (_, original), (_, copy) in zip(
        model.state_dict().items(), restored.state_dict().items(), strict=True
    ):
        assert torch.allclose(original, copy)
