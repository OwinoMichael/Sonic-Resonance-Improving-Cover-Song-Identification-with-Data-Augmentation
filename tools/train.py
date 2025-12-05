#!/usr/bin/env python3

import argparse
import os
import sys
import torch
import torch.multiprocessing as mp

from src.trainer import Trainer

from src.model import Model
from src.utils import create_logger, get_hparams_as_string, load_hparams


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Train: python3 -m tools.train model_dir",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("model_dir")
    parser.add_argument(
        "--first_eval",
        default=False,
        action="store_true",
        help="Set for run eval first before train",
    )
    parser.add_argument(
        "--only_eval",
        default=False,
        action="store_true",
        help="Set for run eval first before train",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="give more debug log",
    )
    parser.add_argument(
        "--runid",
        default="",
        action="store",
        help="put TensorBoard logs in this subfolder of ../logs/",
    )
    # NEW: Add loss-type flag
    parser.add_argument(
        "--loss-type",
        default=None,
        choices=["triplet", "arcface"],
        help="Override loss type: triplet or arcface (overrides config file)",
    )
    
    args = parser.parse_args()
    model_dir = args.model_dir
    first_eval = args.first_eval
    only_eval = args.only_eval
    run_id = args.runid
    loss_type = args.loss_type
    first_eval = True if only_eval else first_eval

    logger = create_logger()
    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))

    # NEW: Override loss type if specified via command line
    if loss_type is not None:
        logger.info(f"Overriding loss type to: {loss_type.upper()}")
        
        if loss_type == "triplet":
            # Remove arcface, add triplet
            if "arcface" in hp:
                del hp["arcface"]
            if "triplet" not in hp:
                hp["triplet"] = {
                    "margin": 0.3,
                    "weight": 0.1
                }
            logger.info("Using TRIPLET LOSS (margin=%.2f, weight=%.2f)", 
                       hp["triplet"]["margin"], hp["triplet"]["weight"])
        
        elif loss_type == "arcface":
            # Remove triplet, add arcface
            if "triplet" in hp:
                del hp["triplet"]
            if "arcface" not in hp:
                hp["arcface"] = {
                    "s": 30.0,
                    "m": 0.50,
                    "weight": 0.1
                }
            logger.info("Using ARCFACE LOSS (s=%.1f, m=%.2f, weight=%.2f)",
                       hp["arcface"]["s"], hp["arcface"]["m"], hp["arcface"]["weight"])
    else:
        # Determine from config
        has_triplet = "triplet" in hp
        has_arcface = "arcface" in hp
        
        if has_triplet and has_arcface:
            logger.info("Using BOTH Triplet and ArcFace losses")
        elif has_triplet:
            logger.info("Using TRIPLET LOSS")
        elif has_arcface:
            logger.info("Using ARCFACE LOSS")
        else:
            logger.error("Config must specify at least 'triplet' or 'arcface' loss!")
            sys.exit(1)

    match hp["device"]:  # noqa requires python 3.10
        case "mps":
            if not torch.backends.mps.is_available():
                logger.error(
                    "You requested 'mps' device in your hyperparameters"
                    "but you are not running on an Apple M-series chip or "
                    "have not compiled PyTorch for MPS support."
                )
                sys.exit()
            device = torch.device("mps")
            # set multiprocessing method because 'fork'
            # has significant performance boost on MPS vs. default 'spawn'
            mp.set_start_method("fork")
        case "cuda":
            if not torch.cuda.is_available():
                logger.error(
                    "You requested 'cuda' device in your hyperparameters"
                    "but you do not have a CUDA-compatible GPU available."
                )
                sys.exit()
            device = torch.device("cuda")
        case _:
            logger.error(
                "You set device: %s"
                " in your hyperparameters but that is not a valid option or is an untested option.",
                hp["device"],
            )
            sys.exit()

    logger.info("%s", get_hparams_as_string(hp))

    torch.manual_seed(hp["seed"])

    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_path = os.path.join(model_dir, "logs", run_id)
    os.makedirs(log_path, exist_ok=True)

    t = Trainer(
        hp,
        Model,
        device,
        log_path,
        checkpoint_dir,
        model_dir,
        only_eval,
        first_eval,
    )

    t.configure_optimizer()
    t.load_model(advanced=True)  # Use advanced=True to handle loss type switches
    
    # CRITICAL: When using advanced mode, reset learning rate BEFORE configuring scheduler
    # This ensures optimizer has 'initial_lr' set properly
    if loss_type is not None:
        logger.info("Resetting optimizer for new loss type")
        t.reset_learning_rate()
    
    t.configure_scheduler()
    
    # If not switching loss types, reset LR normally
    if loss_type is None:
        t.reset_learning_rate()
    
    t.train(max_epochs=100)


if __name__ == "__main__":
    _main()