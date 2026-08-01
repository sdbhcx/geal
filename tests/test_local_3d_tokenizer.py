import unittest
from pathlib import Path

import torch

from model.local_3d_tokenizer import (
    Local3DTokenizer,
    TokenFusion,
    TokenToPointInterpolator,
    validate_local_tokenizer_checkpoint,
)


class Local3DTokenizerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_tokenizer_returns_fixed_tokens_and_explicit_mapping(self):
        tokenizer = Local3DTokenizer(
            token_dim=16,
            num_tokens=8,
            neighbor_k=6,
            hidden_dim=12,
            pooling="max",
        )
        xyz = torch.randn(2, 32, 3)

        tokens, metadata = tokenizer(xyz)

        self.assertEqual(tokens.shape, (2, 16, 8))
        self.assertEqual(metadata["center_idx"].shape, (2, 8))
        self.assertEqual(metadata["centers"].shape, (2, 8, 3))
        self.assertEqual(metadata["neighbor_idx"].shape, (2, 8, 6))
        self.assertEqual(metadata["relative_xyz"].shape, (2, 8, 6, 3))
        self.assertTrue(torch.all(metadata["center_idx"] >= 0))
        self.assertTrue(torch.all(metadata["center_idx"] < xyz.shape[1]))
        self.assertTrue(torch.all(metadata["neighbor_idx"] >= 0))
        self.assertTrue(torch.all(metadata["neighbor_idx"] < xyz.shape[1]))

    def test_neighbor_count_is_clamped_to_available_points(self):
        tokenizer = Local3DTokenizer(
            token_dim=8,
            num_tokens=9,
            neighbor_k=16,
            hidden_dim=8,
        )
        xyz = torch.randn(1, 5, 3)

        tokens, metadata = tokenizer(xyz)

        self.assertEqual(tokens.shape, (1, 8, 5))
        self.assertEqual(metadata["neighbor_idx"].shape, (1, 5, 5))

    def test_relative_geometry_is_translation_invariant(self):
        tokenizer = Local3DTokenizer(
            token_dim=12,
            num_tokens=6,
            neighbor_k=5,
            hidden_dim=10,
        )
        tokenizer.eval()
        xyz = torch.randn(1, 20, 3)
        offset = torch.tensor([[[3.5, -2.0, 7.25]]])

        torch.manual_seed(123)
        tokens_a, metadata_a = tokenizer(xyz)
        torch.manual_seed(123)
        tokens_b, metadata_b = tokenizer(xyz + offset)

        torch.testing.assert_close(tokens_a, tokens_b, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            metadata_a["relative_xyz"], metadata_b["relative_xyz"], atol=1e-6, rtol=1e-6
        )

    def test_interpolator_preserves_features_at_token_centers(self):
        interpolator = TokenToPointInterpolator(interpolate_k=1)
        centers = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
        token_features = torch.tensor([[[1.0, 4.0], [2.0, 8.0]]])

        point_features, metadata = interpolator(centers, centers, token_features)

        torch.testing.assert_close(point_features, token_features)
        self.assertEqual(metadata["point_to_token_idx"].shape, (1, 2, 1))
        self.assertEqual(metadata["point_to_token_weight"].shape, (1, 2, 1))

    def test_zero_initialized_fusion_starts_as_identity_and_backpropagates(self):
        fusion = TokenFusion(channels=8, mode="gated_residual", init_value=0.0)
        pointnet_features = torch.randn(2, 8, 11, requires_grad=True)
        token_point_features = torch.randn(2, 8, 11, requires_grad=True)

        output = fusion(pointnet_features, token_point_features)

        torch.testing.assert_close(output, pointnet_features)
        output.square().mean().backward()
        self.assertIsNotNone(fusion.residual_scale.grad)
        self.assertGreater(float(fusion.residual_scale.grad.abs()), 0.0)

    def test_checkpoint_validation_rejects_missing_tokenizer_weights(self):
        with self.assertRaisesRegex(RuntimeError, "local tokenizer"):
            validate_local_tokenizer_checkpoint({"point_encoder.weight": torch.ones(1)})

        validate_local_tokenizer_checkpoint(
            {
                "local_tokenizer.local_encoder.0.weight": torch.ones(1),
                "token_fusion.residual_scale": torch.ones(1),
            }
        )

    def test_branch3d_source_integrates_tokenizer_before_decoder(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "model" / "branch_3d.py").read_text(encoding="utf-8")
        self.assertIn('cfg.get("local_tokenizer", {})', source)
        self.assertIn("self.local_tokenizer", source)
        self.assertIn("self.token_to_point", source)
        self.assertIn("self.token_fusion", source)
        fusion_pos = source.index("self.token_fusion(")
        decoder_pos = source.index("self.decoder(")
        self.assertLess(fusion_pos, decoder_pos)


class V1ConfigurationTest(unittest.TestCase):
    def test_v1_configs_share_tokenizer_structure(self):
        import yaml

        root = Path(__file__).resolve().parents[1]
        paths = [
            root / "config" / "train_stage2_v1_tokenizer.yaml",
            root / "config" / "evaluation_v1_tokenizer.yaml",
            root / "config" / "evaluation_corrupt_v1_tokenizer.yaml",
        ]
        configs = []
        for path in paths:
            with path.open("r", encoding="utf-8") as handle:
                configs.append(yaml.safe_load(handle))

        tokenizer_configs = [cfg["model_3d"]["local_tokenizer"] for cfg in configs]
        self.assertTrue(all(config["enabled"] for config in tokenizer_configs))
        self.assertTrue(all(config == tokenizer_configs[0] for config in tokenizer_configs[1:]))
        self.assertTrue(
            all(cfg["model_3d"]["llm_dim"] == configs[0]["model_3d"]["llm_dim"] for cfg in configs)
        )
        self.assertFalse(configs[0]["train"]["use_iam_adm"])
        self.assertFalse(configs[0]["train"]["use_new_losses"])


if __name__ == "__main__":
    unittest.main()
