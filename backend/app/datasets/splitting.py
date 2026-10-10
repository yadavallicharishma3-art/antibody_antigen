"""Cluster-aware dataset splitting and biological leakage prevention."""

import csv
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd

from .models import DatasetSample, SplitMetrics

DEFAULT_SEED = 42


class ClusterAwareSplitter:
    """Partitions dataset samples into train, validation, and test splits without cluster leakage."""

    def __init__(
        self,
        val_fraction: float = 0.20,
        seed: int = DEFAULT_SEED,
        cluster_col: str = "ab_ag_cluster",
    ):
        self.val_fraction = val_fraction
        self.seed = seed
        self.cluster_col = cluster_col

    def split_samples(
        self,
        samples: List[DatasetSample],
    ) -> Tuple[List[DatasetSample], List[DatasetSample], List[DatasetSample], SplitMetrics]:
        """Perform cluster-aware deterministic train / validation / test partitioning.

        Strategy:
        1. If SAbDab metadata already defines ab_ag_split ('train' vs 'test'), preserve the
           official test partition (which is strictly separated by ab_ag_cluster).
        2. Sub-divide the training partition into 'train' and 'validation' using cluster-grouped
           splitting on ab_ag_cluster so that no cluster is shared between train and validation.
        3. If no pre-existing split exists, partition all clusters into train / val / test (70/15/15).

        Returns:
            Tuple of (train_samples, val_samples, test_samples, split_metrics).
        """
        if not samples:
            empty_metrics = SplitMetrics(
                total_samples=0, train_count=0, val_count=0, test_count=0,
                train_pct=0.0, val_pct=0.0, test_pct=0.0,
                pdb_overlap_train_val=0, pdb_overlap_train_test=0, pdb_overlap_val_test=0,
                ab_cluster_overlap_train_val=0, ab_cluster_overlap_train_test=0, ab_cluster_overlap_val_test=0,
                ag_cluster_overlap_train_val=0, ag_cluster_overlap_train_test=0, ag_cluster_overlap_val_test=0,
                ab_ag_cluster_overlap_train_val=0, ab_ag_cluster_overlap_train_test=0, ab_ag_cluster_overlap_val_test=0,
            )
            return [], [], [], empty_metrics

        has_sabdab_test = any(s.split.lower() == "test" for s in samples)

        train_pool: List[DatasetSample] = []
        test_samples: List[DatasetSample] = []

        if has_sabdab_test:
            for s in samples:
                if s.split.lower() == "test":
                    test_samples.append(s)
                else:
                    train_pool.append(s)
        else:
            # Full 3-way cluster partition (70% train, 15% val, 15% test)
            train_pool = samples

        # Group train_pool by cluster_col
        cluster_to_samples: Dict[str, List[DatasetSample]] = {}
        for s in train_pool:
            raw_val = getattr(s, self.cluster_col, "")
            c_val = str(raw_val).strip() if raw_val is not None and str(raw_val).strip() not in {"", "nan"} else s.pdb_id
            cluster_to_samples.setdefault(c_val, []).append(s)

        unique_clusters = sorted(list(cluster_to_samples.keys()))
        rng = random.Random(self.seed)
        shuffled_clusters = list(unique_clusters)
        rng.shuffle(shuffled_clusters)

        train_samples: List[DatasetSample] = []
        val_samples: List[DatasetSample] = []

        if has_sabdab_test:
            target_val_count = max(1, int(len(train_pool) * self.val_fraction))
            curr_val_count = 0

            for c_val in shuffled_clusters:
                c_samples = cluster_to_samples[c_val]
                if curr_val_count + len(c_samples) <= target_val_count or curr_val_count == 0:
                    val_samples.extend(c_samples)
                    curr_val_count += len(c_samples)
                else:
                    train_samples.extend(c_samples)
        else:
            # 3-way split: 70% train, 15% val, 15% test
            total = len(samples)
            target_test = max(1, int(total * 0.15))
            target_val = max(1, int(total * 0.15))

            curr_test, curr_val = 0, 0
            for c_val in shuffled_clusters:
                c_samples = cluster_to_samples[c_val]
                if curr_test + len(c_samples) <= target_test or curr_test == 0:
                    test_samples.extend(c_samples)
                    curr_test += len(c_samples)
                elif curr_val + len(c_samples) <= target_val:
                    val_samples.extend(c_samples)
                    curr_val += len(c_samples)
                else:
                    train_samples.extend(c_samples)

        # Update split field on samples
        for s in train_samples:
            s.split = "train"
        for s in val_samples:
            s.split = "validation"
        for s in test_samples:
            s.split = "test"

        metrics = self._calculate_metrics(train_samples, val_samples, test_samples)
        return train_samples, val_samples, test_samples, metrics

    def export_splits(
        self,
        train_samples: List[DatasetSample],
        val_samples: List[DatasetSample],
        test_samples: List[DatasetSample],
        output_dir: Path,
    ) -> Dict[str, Path]:
        """Save train.csv, validation.csv, and test.csv deterministically."""
        output_dir.mkdir(parents=True, exist_ok=True)
        split_map = {
            "train": (train_samples, output_dir / "train.csv"),
            "validation": (val_samples, output_dir / "validation.csv"),
            "test": (test_samples, output_dir / "test.csv"),
        }

        paths = {}
        for name, (smpls, path) in split_map.items():
            if smpls:
                keys = list(smpls[0].to_dict().keys())
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    for s in sorted(smpls, key=lambda x: x.sample_id):
                        writer.writerow(s.to_dict())
            else:
                path.touch()
            paths[name] = path

        return paths

    def _calculate_metrics(
        self,
        train: List[DatasetSample],
        val: List[DatasetSample],
        test: List[DatasetSample],
    ) -> SplitMetrics:
        """Calculate overlap diagnostics between splits."""
        total = len(train) + len(val) + len(test)

        def _get_set(smpls: List[DatasetSample], attr: str) -> Set[str]:
            return {str(getattr(s, attr)) for s in smpls if getattr(s, attr, None)}

        pdbs_train = _get_set(train, "pdb_id")
        pdbs_val = _get_set(val, "pdb_id")
        pdbs_test = _get_set(test, "pdb_id")

        ab_train = _get_set(train, "ab_cluster")
        ab_val = _get_set(val, "ab_cluster")
        ab_test = _get_set(test, "ab_cluster")

        ag_train = _get_set(train, "ag_cluster")
        ag_val = _get_set(val, "ag_cluster")
        ag_test = _get_set(test, "ag_cluster")

        abag_train = _get_set(train, "ab_ag_cluster")
        abag_val = _get_set(val, "ab_ag_cluster")
        abag_test = _get_set(test, "ab_ag_cluster")

        return SplitMetrics(
            total_samples=total,
            train_count=len(train),
            val_count=len(val),
            test_count=len(test),
            train_pct=(len(train) / total * 100) if total else 0.0,
            val_pct=(len(val) / total * 100) if total else 0.0,
            test_pct=(len(test) / total * 100) if total else 0.0,
            pdb_overlap_train_val=len(pdbs_train & pdbs_val),
            pdb_overlap_train_test=len(pdbs_train & pdbs_test),
            pdb_overlap_val_test=len(pdbs_val & pdbs_test),
            ab_cluster_overlap_train_val=len(ab_train & ab_val),
            ab_cluster_overlap_train_test=len(ab_train & ab_test),
            ab_cluster_overlap_val_test=len(ab_val & ab_test),
            ag_cluster_overlap_train_val=len(ag_train & ag_val),
            ag_cluster_overlap_train_test=len(ag_train & ag_test),
            ag_cluster_overlap_val_test=len(ag_val & ag_test),
            ab_ag_cluster_overlap_train_val=len(abag_train & abag_val),
            ab_ag_cluster_overlap_train_test=len(abag_train & abag_test),
            ab_ag_cluster_overlap_val_test=len(abag_val & abag_test),
        )
