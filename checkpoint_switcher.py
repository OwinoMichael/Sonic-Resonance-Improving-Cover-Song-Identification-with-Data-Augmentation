#!/usr/bin/env python3
"""
Checkpoint Switcher for ArcFace ↔ Triplet Loss Training
Run this cell to switch between different loss function checkpoints
"""

import os
import shutil
from pathlib import Path

# Configuration
BASE_CHECKPOINT_DIR = "egs/covers80/checkpoints"
ARCFACE_BACKUP_DIR = "egs/covers80/checkpoints_arcface"
TRIPLET_BACKUP_DIR = "egs/covers80/checkpoints_triplet"

def get_latest_epoch(checkpoint_dir):
    """Get the latest epoch number from checkpoint directory"""
    if not os.path.exists(checkpoint_dir):
        return 0
    
    epochs = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith("g_") and f.endswith(""):
            try:
                epoch = int(f.split("_")[1])
                epochs.append(epoch)
            except:
                pass
    return max(epochs) if epochs else 0

def backup_current_checkpoints(source_dir, backup_dir, label):
    """Backup current checkpoints"""
    if not os.path.exists(source_dir):
        print(f"⚠️  No checkpoints found in {source_dir}")
        return False
    
    # Create backup directory if it doesn't exist
    os.makedirs(backup_dir, exist_ok=True)
    
    # Get list of checkpoint files
    files = [f for f in os.listdir(source_dir) if f.startswith(("g_", "do_"))]
    
    if not files:
        print(f"⚠️  No checkpoint files found in {source_dir}")
        return False
    
    # Copy files to backup
    for f in files:
        src = os.path.join(source_dir, f)
        dst = os.path.join(backup_dir, f)
        shutil.copy2(src, dst)
    
    latest_epoch = get_latest_epoch(source_dir)
    print(f"✓ Backed up {label} checkpoints (latest: epoch {latest_epoch})")
    print(f"  → {backup_dir}")
    return True

def restore_checkpoints(backup_dir, target_dir, label):
    """Restore checkpoints from backup"""
    if not os.path.exists(backup_dir):
        print(f"⚠️  No backup found at {backup_dir}")
        return False
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Clear target directory
    for f in os.listdir(target_dir):
        if f.startswith(("g_", "do_")):
            os.remove(os.path.join(target_dir, f))
    
    # Copy files from backup
    files = [f for f in os.listdir(backup_dir) if f.startswith(("g_", "do_"))]
    for f in files:
        src = os.path.join(backup_dir, f)
        dst = os.path.join(target_dir, f)
        shutil.copy2(src, dst)
    
    latest_epoch = get_latest_epoch(target_dir)
    print(f"✓ Restored {label} checkpoints (latest: epoch {latest_epoch})")
    print(f"  ← {backup_dir}")
    return True

def switch_to_mode(mode):
    """Switch to specified training mode"""
    print(f"\n{'='*60}")
    print(f"SWITCHING TO: {mode.upper()}")
    print(f"{'='*60}\n")
    
    if mode.lower() == "arcface":
        # Backup current (assume it's triplet)
        backup_current_checkpoints(BASE_CHECKPOINT_DIR, TRIPLET_BACKUP_DIR, "Triplet")
        # Restore ArcFace
        restore_checkpoints(ARCFACE_BACKUP_DIR, BASE_CHECKPOINT_DIR, "ArcFace")
        
    elif mode.lower() == "triplet":
        # Backup current (assume it's arcface)
        backup_current_checkpoints(BASE_CHECKPOINT_DIR, ARCFACE_BACKUP_DIR, "ArcFace")
        # Restore Triplet
        restore_checkpoints(TRIPLET_BACKUP_DIR, BASE_CHECKPOINT_DIR, "Triplet")
    
    else:
        print(f"❌ Unknown mode: {mode}")
        return
    
    print(f"\n{'='*60}")
    print(f"✓ Ready to train with {mode.upper()}")
    print(f"{'='*60}\n")

def show_status():
    """Show current checkpoint status"""
    print(f"\n{'='*60}")
    print("CHECKPOINT STATUS")
    print(f"{'='*60}\n")
    
    # Current checkpoints
    current_epoch = get_latest_epoch(BASE_CHECKPOINT_DIR)
    print(f"📁 Current ({BASE_CHECKPOINT_DIR}):")
    print(f"   Latest epoch: {current_epoch if current_epoch > 0 else 'None'}\n")
    
    # ArcFace backup
    arcface_epoch = get_latest_epoch(ARCFACE_BACKUP_DIR)
    print(f"📁 ArcFace Backup ({ARCFACE_BACKUP_DIR}):")
    print(f"   Latest epoch: {arcface_epoch if arcface_epoch > 0 else 'None'}\n")
    
    # Triplet backup
    triplet_epoch = get_latest_epoch(TRIPLET_BACKUP_DIR)
    print(f"📁 Triplet Backup ({TRIPLET_BACKUP_DIR}):")
    print(f"   Latest epoch: {triplet_epoch if triplet_epoch > 0 else 'None'}\n")
    
    print(f"{'='*60}\n")