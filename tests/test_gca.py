from __future__ import annotations

import unittest

import torch

from fedmuse.algorithms.fedmuse_train import global_local_contrastive_loss


class GlobalLocalContrastiveLossTests(unittest.TestCase):
    def test_loss_is_small_for_perfect_alignment(self):
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        labels = torch.tensor([0, 1], dtype=torch.long)
        global_prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

        loss = global_local_contrastive_loss(
            image_features=features,
            labels=labels,
            global_prototypes=global_prototypes,
            temperature=0.1,
        )
        self.assertLess(float(loss.item()), 0.01)

    def test_missing_positive_prototype_is_ignored(self):
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        labels = torch.tensor([0, 1], dtype=torch.long)
        global_prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        mask = torch.tensor([True, False], dtype=torch.bool)

        loss = global_local_contrastive_loss(
            image_features=features,
            labels=labels,
            global_prototypes=global_prototypes,
            temperature=0.1,
            prototype_mask=mask,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(loss.item()), 0.01)

    def test_all_invalid_targets_return_zero(self):
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        labels = torch.tensor([0, 1], dtype=torch.long)
        global_prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        mask = torch.tensor([False, False], dtype=torch.bool)

        loss = global_local_contrastive_loss(
            image_features=features,
            labels=labels,
            global_prototypes=global_prototypes,
            temperature=0.1,
            prototype_mask=mask,
        )
        self.assertTrue(torch.allclose(loss, torch.tensor(0.0)))


if __name__ == "__main__":
    unittest.main()
