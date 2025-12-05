import logging
import unittest

import torch
from torch.utils.tensorboard import SummaryWriter

from src.model import Model
from src.trainer import Trainer


class TestTrainer(unittest.TestCase):
    """Test for trainer code with both Triplet and ArcFace."""

    def setUp(self):
        logging.disable(logging.CRITICAL)
        
        # Base hyperparameters
        self.hp_base = {
            "batch_size": 32,
            "seed": 1234,  # Added missing seed
            "covers80": {
                "query_path": "data/covers80/full.txt",
                "ref_path": "data/covers80/full.txt",
                "every_n_epoch_to_test": 1,
            },
            "train_path": "data/covers80/train.txt",
            "test_path": "data/covers80/test.txt",
            "chunk_frame": [1125, 900, 675],
            "chunk_s": 135,
            "mode": "random",
            "learning_rate": 0.001,
            "mean_size": 3,
            "num_workers": 1,
            "m_per_class": 8,
            "adam_b1": 0.8,
            "adam_b2": 0.99,
            "lr_decay": 0.99,
            "min_lr": 0.0001,
            "input_dim": 96,
            "embed_dim": 128,
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
                "weight": 0,
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
        
        self.log_path = "/tmp/cover_hunter_logs"
        self.checkpoint_dir = "/tmp/cover_hunter_logs"
        self.model_dir = "/tmp/cover_hunter_logs"
        self.sw = SummaryWriter(self.log_path)

    def _test_train(self, hp, device, max_epochs):
        hp["device"] = device
        trainer = Trainer(
            hp,
            Model,
            device,
            self.log_path,
            self.checkpoint_dir,
            self.model_dir,
            only_eval=False,
            first_eval=False,
        )
        trainer.summary_writer = self.sw
        trainer.configure_optimizer()
        trainer.configure_scheduler()
        trainer.train(max_epochs=max_epochs)

    @unittest.skipUnless(torch.cuda.is_available(), "No GPU was detected")
    def test_train_cuda_triplet(self):
        """Test training with Triplet loss on CUDA."""
        self._test_train(self.hp_triplet, "cuda", 1)

    @unittest.skipUnless(torch.cuda.is_available(), "No GPU was detected")
    def test_train_cuda_arcface(self):
        """Test training with ArcFace loss on CUDA."""
        self._test_train(self.hp_arcface, "cuda", 1)

    @unittest.skipUnless(torch.backends.mps.is_available(), "No MPS was detected")
    def test_train_mps_triplet(self):
        """Test training with Triplet loss on MPS."""
        self._test_train(self.hp_triplet, "mps", 1)

    @unittest.skipUnless(torch.backends.mps.is_available(), "No MPS was detected")
    def test_train_mps_arcface(self):
        """Test training with ArcFace loss on MPS."""
        self._test_train(self.hp_arcface, "mps", 1)

    def _test_logging(self, hp, device, expected_loss_keys):
        with unittest.mock.patch.object(self.sw, "add_scalar") as add_scalar_mock:
            self._test_train(hp, device, 1)

        expected_calls = [
            unittest.mock.call("csi/lr", unittest.mock.ANY, 1),
            unittest.mock.call("csi/foc_loss", unittest.mock.ANY, 1),
            unittest.mock.call("csi/total", unittest.mock.ANY, 1),
            unittest.mock.call("csi_test/foc_loss", unittest.mock.ANY, 0),
            unittest.mock.call("mAP/covers80", unittest.mock.ANY, 0),
            unittest.mock.call("hit_rate/covers80", unittest.mock.ANY, 0),
        ]
        
        # Add specific loss type calls
        for loss_key in expected_loss_keys:
            expected_calls.append(unittest.mock.call(f"csi/{loss_key}", unittest.mock.ANY, 1))
            expected_calls.append(unittest.mock.call(f"csi_test/{loss_key}", unittest.mock.ANY, 0))
        
        add_scalar_mock.assert_has_calls(expected_calls)

    @unittest.skipUnless(torch.cuda.is_available(), "No GPU was detected")
    def test_logging_cuda_triplet(self):
        """Test logging with Triplet loss on CUDA."""
        self._test_logging(self.hp_triplet, "cuda", ["tri_loss"])

    @unittest.skipUnless(torch.cuda.is_available(), "No GPU was detected")
    def test_logging_cuda_arcface(self):
        """Test logging with ArcFace loss on CUDA."""
        self._test_logging(self.hp_arcface, "cuda", ["arcface_loss"])

    @unittest.skipUnless(torch.backends.mps.is_available(), "No MPS was detected")
    def test_logging_mps_triplet(self):
        """Test logging with Triplet loss on MPS."""
        self._test_logging(self.hp_triplet, "mps", ["tri_loss"])

    @unittest.skipUnless(torch.backends.mps.is_available(), "No MPS was detected")
    def test_logging_mps_arcface(self):
        """Test logging with ArcFace loss on MPS."""
        self._test_logging(self.hp_arcface, "mps", ["arcface_loss"])


if __name__ == "__main__":
    unittest.main()