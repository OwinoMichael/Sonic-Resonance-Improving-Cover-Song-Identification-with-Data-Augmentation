#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot a histogram of a CoverHunterMPS dataset displaying on the horizontal axis the 
count of works, and on the vertical axis the count of performances per work. 
The goal is to understand the distribution of data between works with few 
performances and works with many performances.

Expects a dataset file prepared for extract_csi_features with JSON lines containing
"work" and "perf" fields.

Created on 2025-06-03
@author: alanngnet and Claude Sonnet 4

"""

import json
import argparse
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


def load_dataset_work_counts(dataset_path):
    """Load dataset and count performances per work."""
    work_counts = Counter()
    
    with open(dataset_path, 'r') as file:
        for line in file:
            try:
                record = json.loads(line.strip())
                work_id = record['work']
                work_counts[work_id] += 1
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Skipping invalid line: {e}")
                continue
    
    return work_counts


def plot_histogram(work_counts, dataset_name="Dataset"):
    """Create histogram plot matching SHS100k style."""
    # Create a histogram
    plt.figure(figsize=(12, 6))
    plt.hist(
        list(work_counts.values()), 
        bins=np.arange(0.5, max(work_counts.values()) + 1.5, 1),  # Explicit bin edges
        rwidth=0.8
    )

    plt.xlabel("Number of Performances per Work")
    plt.ylabel("Count of Works")
    plt.title(f"Distribution of Performances per Work in {dataset_name}")
    plt.xscale("log")

    # Set custom ticks on the x-axis
    plt.xticks(
        [1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 30, 40, 50, 100, 200, 500],
        [str(x) for x in [1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 30, 40, 50, 100, 200, 500]],
    )

    # Show plot
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Generate histogram of performances per work for CoverHunterMPS dataset"
    )
    parser.add_argument("dataset_path", help="Path to dataset file (JSON lines format)")
    parser.add_argument("--name", default="Dataset", help="Dataset name for plot title")
    
    args = parser.parse_args()
    
    print(f"Loading dataset from {args.dataset_path}...")
    work_counts = load_dataset_work_counts(args.dataset_path)
    
    print(f"Found {len(work_counts)} unique works")
    print(f"Total performances: {sum(work_counts.values())}")
    print(f"Performance counts range: {min(work_counts.values())} - {max(work_counts.values())}")
    
    plot_histogram(work_counts, args.name)


if __name__ == "__main__":
    main()
