from __future__ import annotations

import unittest

import torch

from fedmuse.algorithms.fedmuse_train import (
    adain_feature_transform,
    fuse_expert_weights,
    orthogonal_disentanglement_loss,
    uncertainty_gate,
)


class FedMUSECoreTests(unittest.TestCase):
    def test_orthogonal_loss_is_zero_for_orthogonal_prompts(self):
        global_prompt = torch.tensor(
            [
                [[1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        local_prompt = torch.tensor(
            [
                [[0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0]],
            ],
            dtype=torch.float32,
        )
        loss = orthogonal_disentanglement_loss(global_prompt, local_prompt)
        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)

    def test_adain_feature_transform_output_shape(self):
        feat = torch.randn(5, 8, dtype=torch.float32)
        mu = torch.randn(3, 8, dtype=torch.float32)
        sigma = torch.rand(3, 8, dtype=torch.float32) + 0.1
        out = adain_feature_transform(feat, mu, sigma)
        self.assertEqual(tuple(out.shape), (5, 3, 8))

    def test_uncertainty_gate_increases_with_entropy(self):
        low = uncertainty_gate(torch.tensor([0.1]), beta=1.0, gamma=0.0)
        high = uncertainty_gate(torch.tensor([3.0]), beta=1.0, gamma=0.0)
        self.assertGreater(float(high.item()), float(low.item()))

    def test_fuse_expert_weights_is_probability(self):
        d = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32)
        router_logits = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float32)
        w = fuse_expert_weights(d, router_logits=router_logits)
        self.assertTrue(torch.all(w >= 0))
        self.assertAlmostEqual(float(w.sum().item()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
