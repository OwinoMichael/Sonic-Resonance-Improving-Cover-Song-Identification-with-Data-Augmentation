import random
import unittest

import numpy as np
import torch

from src.model import Model

import logging


class TestModel(unittest.TestCase):
    """Test for model with both Triplet and ArcFace loss."""

    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.device = "cpu"
        
        # Base hyperparameters
        self.hp_base = {
            "device": self.device,
            "batch_size": 32,
            "learning_rate": 0.001,
            "adam_b1": 0.8,
            "adam_b2": 0.99,
            "input_dim": 96,
            "embed_dim": 128,
            "seed": 1234,  # Added missing seed
            "encoder": {
                "output_dims": 128,
                "num_blocks": 6,
                "attention_dim": 256,
            },
            "foc": {
                "output_dims": 3000,
                "gamma": 2,
                "weight": 1.0,
            },
            "center": {
                "weight": 0.001,
            },
        }
        
        # Triplet configuration
        self.hp_triplet = {**self.hp_base}
        self.hp_triplet["triplet"] = {
            "margin": 0.3,
            "weight": 0.1,
        }
        
        # ArcFace configuration
        self.hp_arcface = {**self.hp_base}
        self.hp_arcface["arcface"] = {
            "s": 30.0,
            "m": 0.50,
            "weight": 0.1,
        }
        
        self.input_tensor = torch.zeros((32, 1125, 96))

    @staticmethod
    def make_deterministic(seed=1234) -> None:
        """Set the randomness with a given seed."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    @torch.no_grad()
    def test_shape_triplet(self):
        """Test model output shape with Triplet loss."""
        model = Model(self.hp_triplet)
        output_tensor = model(self.input_tensor)
        expected_shape = (32, 128)
        self.assertEqual(expected_shape, output_tensor.shape)

    @torch.no_grad()
    def test_shape_arcface(self):
        """Test model output shape with ArcFace loss."""
        model = Model(self.hp_arcface)
        output_tensor = model(self.input_tensor)
        expected_shape = (32, 128)
        self.assertEqual(expected_shape, output_tensor.shape)

    @torch.no_grad()
    @unittest.skipUnless(torch.cuda.is_available(), "No GPU was detected")
    def test_device_moving_cuda_triplet(self):
        """Test Triplet model can be moved from cpu to gpu."""
        model = Model(self.hp_triplet)
        inputs = torch.randn((32, 1125, 96))
        
        # Test on CPU
        self.make_deterministic()
        outputs_cpu = model(inputs)
        
        # Move to GPU and test
        model_on_gpu = model.to("cuda")
        self.make_deterministic()
        outputs_gpu = model_on_gpu(inputs.to("cuda"))
        
        # Move back to CPU and test
        model_back_on_cpu = model_on_gpu.cpu()
        self.make_deterministic()
        outputs_back_on_cpu = model_back_on_cpu(inputs)

        torch.testing.assert_close(outputs_cpu, outputs_gpu.cpu(), rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(outputs_cpu, outputs_back_on_cpu, rtol=1e-3, atol=1e-3)

    @torch.no_grad()
    @unittest.skipUnless(torch.cuda.is_available(), "No GPU was detected")
    def test_device_moving_cuda_arcface(self):
        """Test ArcFace model can be moved from cpu to gpu."""
        model = Model(self.hp_arcface)
        inputs = torch.randn((32, 1125, 96))
        
        # Test on CPU
        self.make_deterministic()
        outputs_cpu = model(inputs)
        
        # Move to GPU and test
        model_on_gpu = model.to("cuda")
        self.make_deterministic()
        outputs_gpu = model_on_gpu(inputs.to("cuda"))
        
        # Move back to CPU and test
        model_back_on_cpu = model_on_gpu.cpu()
        self.make_deterministic()
        outputs_back_on_cpu = model_back_on_cpu(inputs)

        torch.testing.assert_close(outputs_cpu, outputs_gpu.cpu(), rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(outputs_cpu, outputs_back_on_cpu, rtol=1e-3, atol=1e-3)

    @torch.no_grad()
    @unittest.skipUnless(torch.backends.mps.is_available(), "No MPS was detected")
    def test_device_moving_mps_triplet(self):
        """Test Triplet model can be moved from cpu to mps."""
        model = Model(self.hp_triplet)
        model_on_mps = model.to("mps")
        model_back_on_cpu = model_on_mps.cpu()

        inputs = torch.randn((32, 1125, 96))

        self.make_deterministic()
        outputs_cpu = model(inputs)
        self.make_deterministic()
        outputs_mps = model_on_mps(inputs.to("mps"))
        self.make_deterministic()
        outputs_back_on_cpu = model_back_on_cpu(inputs)

        torch.testing.assert_close(outputs_cpu, outputs_mps.cpu())
        torch.testing.assert_close(outputs_cpu, outputs_back_on_cpu)

    @torch.no_grad()
    @unittest.skipUnless(torch.backends.mps.is_available(), "No MPS was detected")
    def test_device_moving_mps_arcface(self):
        """Test ArcFace model can be moved from cpu to mps."""
        model = Model(self.hp_arcface)
        model_on_mps = model.to("mps")
        model_back_on_cpu = model_on_mps.cpu()

        inputs = torch.randn((32, 1125, 96))

        self.make_deterministic()
        outputs_cpu = model(inputs)
        self.make_deterministic()
        outputs_mps = model_on_mps(inputs.to("mps"))
        self.make_deterministic()
        outputs_back_on_cpu = model_back_on_cpu(inputs)

        torch.testing.assert_close(outputs_cpu, outputs_mps.cpu())
        torch.testing.assert_close(outputs_cpu, outputs_back_on_cpu)

    def test_all_parameters_updated_triplet(self):
        """Ensure no dead subgraph with Triplet loss."""
        model = Model(self.hp_triplet)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            self.hp_triplet["learning_rate"],
            betas=[self.hp_triplet["adam_b1"], self.hp_triplet["adam_b2"]],
        )
        
        inputs = torch.randn((32, 1125, 96))
        labels = torch.zeros((32), dtype=torch.long)
        total_loss, losses = model.compute_loss(inputs, labels)
        total_loss.backward()
        optimizer.step()

        for param_name, param in model.named_parameters():
            if param.requires_grad:
                with self.subTest(name=param_name):
                    self.assertIsNotNone(param.grad)
                    self.assertNotEqual(0.0, torch.sum(param.grad**2))

    def test_all_parameters_updated_arcface(self):
        """Ensure no dead subgraph with ArcFace loss."""
        model = Model(self.hp_arcface)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            self.hp_arcface["learning_rate"],
            betas=[self.hp_arcface["adam_b1"], self.hp_arcface["adam_b2"]],
        )
        
        inputs = torch.randn((32, 1125, 96))
        labels = torch.zeros((32), dtype=torch.long)
        total_loss, losses = model.compute_loss(inputs, labels)
        total_loss.backward()
        optimizer.step()

        for param_name, param in model.named_parameters():
            if param.requires_grad:
                with self.subTest(name=param_name):
                    self.assertIsNotNone(param.grad)
                    self.assertNotEqual(0.0, torch.sum(param.grad**2))

    def test_loss_dict_triplet(self):
        """Test that loss dict contains correct keys for Triplet."""
        model = Model(self.hp_triplet)
        inputs = torch.randn((32, 1125, 96))
        labels = torch.zeros((32), dtype=torch.long)
        
        total_loss, losses = model.compute_loss(inputs, labels)
        
        self.assertIn("foc_loss", losses)
        self.assertIn("tri_loss", losses)
        self.assertNotIn("arcface_loss", losses)

    def test_loss_dict_arcface(self):
        """Test that loss dict contains correct keys for ArcFace."""
        model = Model(self.hp_arcface)
        inputs = torch.randn((32, 1125, 96))
        labels = torch.zeros((32), dtype=torch.long)
        
        total_loss, losses = model.compute_loss(inputs, labels)
        
        self.assertIn("foc_loss", losses)
        self.assertIn("arcface_loss", losses)
        self.assertNotIn("tri_loss", losses)