"""
Self-contained Unit Test Runner for FedUA-Net.
Executes all unit tests using Python's standard unittest test loader.
Requires zero external dependencies.
"""

import sys
import unittest
import torch
import numpy as np

# Add repo root to path
sys.path.insert(0, ".")

from src.models.cbam import ChannelAttention, SpatialAttention, CBAM
from src.models.fedua_model import SharedBackbone, FedUANetClientModel, CentralizedGlobalModel
from src.federated.aggregator import aggregate_weights, compute_aggregation_weights
from src.uncertainty.temperature_scaling import TemperatureScaling
from src.uncertainty.conformal import (
    compute_nonconformity_scores,
    calibrate_conformal_quantile,
    generate_prediction_sets,
    evaluate_conformal_coverage,
)
from src.uncertainty.metrics import (
    compute_expected_calibration_error,
    compute_multiclass_brier_score,
    compute_risk_coverage_curve,
)


class TestCBAM(unittest.TestCase):
    def test_channel_attention(self):
        x = torch.randn(2, 64, 14, 14)
        ca = ChannelAttention(in_planes=64, ratio=16)
        out = ca(x)
        self.assertEqual(out.shape, (2, 64, 1, 1))
        self.assertTrue(torch.all(out >= 0.0) and torch.all(out <= 1.0))

    def test_spatial_attention(self):
        x = torch.randn(2, 64, 14, 14)
        sa = SpatialAttention(kernel_size=7)
        out = sa(x)
        self.assertEqual(out.shape, (2, 1, 14, 14))
        self.assertTrue(torch.all(out >= 0.0) and torch.all(out <= 1.0))

    def test_cbam_forward(self):
        x = torch.randn(4, 1280, 7, 7)
        cbam = CBAM(in_planes=1280, ratio=16, kernel_size=7)
        out = cbam(x)
        self.assertEqual(out.shape, x.shape)


class TestModels(unittest.TestCase):
    def test_shared_backbone(self):
        backbone = SharedBackbone(backbone_name="efficientnet_v2_s", pretrained=False, attention_type="cbam", embed_dim=512)
        x = torch.randn(2, 3, 224, 224)
        emb = backbone(x)
        self.assertEqual(emb.shape, (2, 512))

    def test_client_model(self):
        backbone = SharedBackbone(backbone_name="efficientnet_v2_s", pretrained=False, attention_type="cbam", embed_dim=512)
        model = FedUANetClientModel(backbone=backbone, num_classes=4, embed_dim=512)
        x = torch.randn(2, 3, 224, 224)
        logits = model(x)
        self.assertEqual(logits.shape, (2, 4))
        loss = logits.sum()
        loss.backward()
        self.assertIsNotNone(model.head.weight.grad)


class TestAggregation(unittest.TestCase):
    def test_uniform_weights(self):
        sample_sizes = [4855, 546, 14815]
        weights = compute_aggregation_weights(sample_sizes, agg_weight_type="uniform")
        self.assertEqual(len(weights), 3)
        self.assertTrue(all(abs(w - 1/3) < 1e-6 for w in weights))

    def test_sample_size_weights(self):
        sample_sizes = [100, 200, 700]
        weights = compute_aggregation_weights(sample_sizes, agg_weight_type="sample_size")
        self.assertAlmostEqual(weights[0], 0.1)
        self.assertAlmostEqual(weights[1], 0.2)
        self.assertAlmostEqual(weights[2], 0.7)


class TestUncertaintyAndConformal(unittest.TestCase):
    def test_temperature_scaling(self):
        ts = TemperatureScaling(init_temperature=1.2)
        self.assertAlmostEqual(ts.temp_value, 1.2, places=3)
        logits = torch.randn(50, 4) * 3.0
        labels = torch.randint(0, 4, (50,))
        t = ts.fit(logits, labels, max_iter=10)
        self.assertGreater(t, 0.0)

    def test_conformal_aps(self):
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(4), size=100)
        labels = np.random.randint(0, 4, size=100)
        scores = compute_nonconformity_scores(probs, labels)
        self.assertEqual(len(scores), 100)
        q_hat = calibrate_conformal_quantile(scores[:50], alpha=0.10)
        self.assertTrue(0.0 <= q_hat <= 1.0)
        psets, _ = generate_prediction_sets(probs[50:], q_hat)
        cov, size = evaluate_conformal_coverage(psets, labels[50:])
        self.assertTrue(0.0 <= cov <= 1.0)
        self.assertTrue(1.0 <= size <= 4.0)

    def test_metrics(self):
        probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])
        labels = np.array([0, 0, 1, 1])
        ece = compute_expected_calibration_error(probs, labels)
        self.assertTrue(0.0 <= ece <= 1.0)
        brier = compute_multiclass_brier_score(probs, labels, num_classes=2)
        self.assertTrue(0.0 <= brier <= 1.0)


class TestTransforms(unittest.TestCase):
    def test_speckle_noise(self):
        import fedua_net as m
        sn = m.SpeckleNoise(sigma=0.08, p=1.0)
        x = torch.full((3, 64, 64), 0.5, dtype=torch.float32)
        out = sn(x)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.all(out >= 0.0) and torch.all(out <= 1.0))
        self.assertFalse(torch.allclose(out, x))

    def test_ultrasound_transforms(self):
        import fedua_net as m
        tf_us = m.train_transforms(ultrasound=True)
        tf_std = m.train_transforms(ultrasound=False)
        img = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
        out_us = tf_us(img)
        out_std = tf_std(img)
        self.assertEqual(out_us.shape, (3, m.cfg.IMG_SIZE, m.cfg.IMG_SIZE))
        self.assertEqual(out_std.shape, (3, m.cfg.IMG_SIZE, m.cfg.IMG_SIZE))


def main():
    print("=" * 70)
    print("Running FedUA-Net Unit Test Suite")
    print("=" * 70)
    suite = unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestCBAM))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestModels))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestAggregation))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestUncertaintyAndConformal))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestTransforms))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()

