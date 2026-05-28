from __future__ import annotations

import torch

from uamfst.grafting.scions import PrototypeClassifier, build_prototypes


def test_uncertainty_weighting_favours_confident_supports() -> None:
    features = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    labels = torch.tensor([0, 0])
    confident = build_prototypes(features, torch.tensor([0.01, 5.0]), labels, num_classes=1)
    uniform = build_prototypes(features, torch.tensor([1.0, 1.0]), labels, num_classes=1)
    assert float(confident[0, 0]) > float(uniform[0, 0])


def test_nearest_prototype_assigns_closest_class() -> None:
    classifier = PrototypeClassifier(temperature=0.2)
    prototypes = torch.tensor([[0.0, 0.0], [5.0, 5.0]])
    query = torch.tensor([[0.1, 0.1], [4.8, 5.2]])
    logits = classifier(query, prototypes)
    assert int(logits[0].argmax()) == 0
    assert int(logits[1].argmax()) == 1


def test_prototype_temperature_is_learnable() -> None:
    classifier = PrototypeClassifier(temperature=0.07)
    assert classifier.log_temperature.requires_grad
    assert abs(float(classifier.temperature.detach()) - 0.07) < 1e-5
