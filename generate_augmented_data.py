#!/usr/bin/env python3
"""
Generate augmented versions of training data for improved generalization.

Usage:
    python generate_augmented_data.py data/covers80/
    
This will create augmented versions with:
- Pitch shifts: ±1, ±2 semitones
- Tempo changes: ±5%, ±10%
- Volume variations
- Then compute CQT features for all augmented versions

Saves to: data/covers80/aug_wav/ and data/covers80/aug_cqt_feat/
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import logging
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import line_to_dict, dict_to_line
from src.dataset import SignalAug
from src.cqt import compute_cqt_with_librosa, shorter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_augmented_audio(input_wav, output_dir, aug_config):
    """
    Generate augmented versions of an audio file
    
    Returns: List of (aug_name, aug_path) tuples
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original audio
    signal, sr = librosa.load(input_wav, sr=16000, mono=True)
    base_name = Path(input_wav).stem
    
    augmented_files = []
    signal_aug = SignalAug(aug_config)
    
    # Define augmentation variants
    variants = [
        ("pitch_down2", {"pitch": {"prob": 1.0, "shift": [0.891]}}),
        ("pitch_down1", {"pitch": {"prob": 1.0, "shift": [0.944]}}),
        ("pitch_up1", {"pitch": {"prob": 1.0, "shift": [1.059]}}),
        ("pitch_up2", {"pitch": {"prob": 1.0, "shift": [1.122]}}),
        ("tempo_down10", {"tempo": {"prob": 1.0, "coef": ["0.9"]}}),
        ("tempo_down5", {"tempo": {"prob": 1.0, "coef": ["0.95"]}}),
        ("tempo_up5", {"tempo": {"prob": 1.0, "coef": ["1.05"]}}),
        ("tempo_up10", {"tempo": {"prob": 1.0, "coef": ["1.1"]}}),
    ]
    
    for aug_name, aug_params in variants:
        # Apply specific augmentation
        aug_hp = {**aug_config, **aug_params, "seed": 1234}
        aug_signal = SignalAug(aug_hp).augmentation(signal.copy())
        
        # Save augmented audio
        output_path = output_dir / f"{aug_name}-{base_name}.wav"
        import soundfile as sf
        sf.write(str(output_path), aug_signal, sr)
        
        augmented_files.append((aug_name, str(output_path)))
    
    return augmented_files


def compute_cqt_for_file(wav_path, cqt_output_dir, hp):
    """Compute and save CQT features"""
    cqt_output_dir = Path(cqt_output_dir)
    cqt_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load audio
    signal, sr = librosa.load(wav_path, sr=16000, mono=True)
    
    # Compute CQT
    cqt_feat = compute_cqt_with_librosa(signal=signal, sr=sr)
    
    # Apply mean pooling if specified
    if hp.get("mean_size", 1) > 1:
        cqt_feat = shorter(cqt_feat, hp["mean_size"])
    
    # Save CQT
    base_name = Path(wav_path).stem
    cqt_path = cqt_output_dir / f"{base_name}.cqt.npy"
    np.save(str(cqt_path), cqt_feat)
    
    return str(cqt_path), cqt_feat.shape[0]


def update_data_list(original_list_path, output_list_path, 
                     aug_wav_dir, aug_cqt_dir, hp):
    """
    Read original data list and create augmented version list
    """
    with open(original_list_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    augmented_lines = []
    
    logger.info(f"Processing {len(lines)} original files...")
    
    for line in tqdm(lines):
        data = line_to_dict(line)
        
        # Generate augmented audio files
        augmented_files = generate_augmented_audio(
            data['wav'], 
            aug_wav_dir,
            hp.get('audio_augmentation', {})
        )
        
        # Compute CQT for each augmented version
        for aug_name, aug_wav_path in augmented_files:
            cqt_path, feat_len = compute_cqt_for_file(
                aug_wav_path, 
                aug_cqt_dir, 
                hp
            )
            
            # Create new data entry
            aug_data = {
                'perf': f"{aug_name}-{data['perf']}",
                'wav': aug_wav_path,
                'dur_s': data['dur_s'],  # Approximate
                'work': data['work'],
                'version': data['version'],
                'feat': cqt_path,
                'feat_len': feat_len,
                'work_id': data['work_id']
            }
            
            augmented_lines.append(dict_to_line(aug_data))
    
    # Write augmented data list
    with open(output_list_path, 'w') as f:
        for line in augmented_lines:
            f.write(line + '\n')
    
    logger.info(f"Created {len(augmented_lines)} augmented samples")
    logger.info(f"Saved to: {output_list_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate augmented training data'
    )
    parser.add_argument(
        'data_dir',
        help='Data directory (e.g., data/covers80/)'
    )
    parser.add_argument(
        '--config',
        default='training/covers80/config/hparams.yaml',
        help='Config file with augmentation parameters'
    )
    parser.add_argument(
        '--only-train',
        action='store_true',
        help='Only augment training data (not val/test)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be generated without actually creating files'
    )
    parser.add_argument(
        '--test-one',
        action='store_true',
        help='Test augmentation on just one file to verify it works'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Load config
    with open(args.config) as f:
        hp = yaml.safe_load(f)
    
    logger.info("="*60)
    logger.info("GENERATING AUGMENTED TRAINING DATA")
    logger.info("="*60)
    
    # SAFETY: Use separate directories that won't conflict
    aug_wav_dir = data_dir / 'aug_wav'
    aug_cqt_dir = data_dir / 'aug_cqt_feat'
    
    # SAFETY: Check what already exists
    logger.info("\n📁 Output directories:")
    logger.info(f"   Audio: {aug_wav_dir}")
    logger.info(f"   CQT:   {aug_cqt_dir}")
    
    if aug_wav_dir.exists():
        existing_wavs = len(list(aug_wav_dir.glob('*.wav')))
        logger.warning(f"   ⚠️  {aug_wav_dir} already exists with {existing_wavs} files")
    
    if aug_cqt_dir.exists():
        existing_cqts = len(list(aug_cqt_dir.glob('*.npy')))
        logger.warning(f"   ⚠️  {aug_cqt_dir} already exists with {existing_cqts} files")
    
    # Process training data
    train_list = data_dir / 'train.txt'
    aug_train_list = data_dir / 'train_augmented.txt'
    
    if not train_list.exists():
        logger.error(f"❌ Training list not found: {train_list}")
        return
    
    # Count original samples
    with open(train_list) as f:
        original_count = len([l for l in f if l.strip()])
    
    logger.info(f"\n📊 Original training samples: {original_count}")
    logger.info(f"📊 Will generate: {original_count * 8} augmented samples")
    logger.info(f"📊 Total after augmentation: {original_count * 9} samples")
    
    # SAFETY: Dry run option
    if args.dry_run:
        logger.info("\n🔍 DRY RUN - No files will be created")
        logger.info("   Remove --dry-run flag to actually generate files")
        return
    
    # SAFETY: Test mode - process just one file
    if args.test_one:
        logger.info("\n🧪 TEST MODE - Processing one file only")
        with open(train_list) as f:
            test_line = f.readline().strip()
        
        test_data = line_to_dict(test_line)
        logger.info(f"   Testing with: {test_data['perf']}")
        
        # Generate augmentations for one file
        try:
            aug_files = generate_augmented_audio(
                test_data['wav'],
                aug_wav_dir,
                hp.get('audio_augmentation', {})
            )
            logger.info(f"   ✅ Generated {len(aug_files)} audio variants")
            
            # Test CQT computation
            test_cqt_path, test_feat_len = compute_cqt_for_file(
                aug_files[0][1],
                aug_cqt_dir,
                hp
            )
            logger.info(f"   ✅ CQT computation works (length: {test_feat_len})")
            logger.info(f"\n✅ TEST PASSED! Augmentation pipeline works correctly.")
            logger.info(f"   Run without --test-one to process all files")
            
        except Exception as e:
            logger.error(f"\n❌ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
        
        return
    
    # SAFETY: Confirmation prompt
    logger.info("\n⚠️  This will create many files. Continue? [y/N]: ")
    response = input().strip().lower()
    if response != 'y':
        logger.info("❌ Cancelled by user")
        return
    
    if train_list.exists():
        logger.info("\n🚀 Processing training data...")
        update_data_list(
            train_list, 
            aug_train_list,
            aug_wav_dir,
            aug_cqt_dir,
            hp
        )
    
    logger.info("\n" + "="*60)
    logger.info("✅ COMPLETE!")
    logger.info("="*60)
    logger.info(f"\n📁 Generated files:")
    logger.info(f"   Audio: {aug_wav_dir}")
    logger.info(f"   CQT:   {aug_cqt_dir}")
    logger.info(f"   List:  {aug_train_list}")
    
    logger.info("\n📝 To use augmented data:")
    logger.info("   1. Backup your current hparams.yaml")
    logger.info("   2. Change train_path to:")
    logger.info(f'      train_path: "data/covers80/train_augmented.txt"')
    logger.info("\n📝 To revert to original data:")
    logger.info('      train_path: "data/covers80/train.txt"')
    logger.info("\n⚠️  Keep both train.txt and train_augmented.txt so you can switch!")


if __name__ == '__main__':
    main()