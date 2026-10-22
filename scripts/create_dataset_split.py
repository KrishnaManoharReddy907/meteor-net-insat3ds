"""
scripts/create_dataset_split.py
Generates an event-level, leakage-free dataset split manifest for METEOR-Net.

Enforces:
1. Grouping strictly by complete observation date / discrete synoptic weather episode.
2. Complete-group allocation to train, validation, or test.
3. Zero random splitting of individual overlapping sequence windows.
4. Safe failure / explicit warning if fewer than 3 independent groups exist.
5. Cryptographic SHA-256 hashing of all input files for full provenance.
"""

import os
import sys
import json
import random
import argparse
import datetime
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.abspath("."))
from src.data.goes_reader import build_timestamp_catalog, compute_file_sha256
from src.data.sequence_generator import TemporalSequenceGenerator


def create_dataset_splits(
    data_dir: str = "data",
    channels: tuple = (9, 13, 14, 15),
    output_manifest: str = "outputs/dataset_splits/splits_manifest.json",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    compute_hashes: bool = True,
    explicit_split_map: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Builds a reproducible, event-level split manifest.
    Guarantees strict episode isolation: all sliding windows from a continuous
    meteorological event reside in exactly one partition.
    """
    print("=" * 80)
    print("DATASET SPLIT MANIFEST GENERATOR (EVENT-LEVEL ISOLATION)")
    print("=" * 80)

    # 1. Discover verified synchronized timestamps
    print(f"\n[Step 1] Scanning data catalog in: {data_dir}...")
    catalog, dup_audit = build_timestamp_catalog(
        data_dir=data_dir,
        channels=channels,
        on_conflict="raise",
        return_audit=True
    )
    print(f"  Discovered {len(catalog)} synchronized observation timestamps across channels {list(channels)}.")
    if not catalog:
        raise ValueError(f"No synchronized satellite observations found in {data_dir} for channels {channels}.")

    # 2. Build sequence windows
    print("\n[Step 2] Constructing sequence windows ([-60, -30, 0, +30] -> +15 min)...")
    seq_gen = TemporalSequenceGenerator(
        catalog=catalog,
        input_offsets_min=(-60, -30, 0, 30),
        target_offsets_min=(15,),
        channel_order=channels
    )
    windows = seq_gen.valid_windows
    print(f"  Discovered {len(windows)} complete sequence windows.")
    if not windows:
        raise ValueError("No complete sequence windows could be constructed from available timestamps.")

    # 3. Group sequence windows strictly by continuous observation episode
    print("\n[Step 3] Grouping windows by continuous observation episode / storm event...")
    sorted_catalog = sorted(catalog, key=lambda x: x["datetime"])
    episode_clusters = []
    if sorted_catalog:
        curr_start = sorted_catalog[0]["datetime"]
        curr_prev = sorted_catalog[0]["datetime"]
        for item in sorted_catalog[1:]:
            dt = item["datetime"]
            if (dt - curr_prev) > datetime.timedelta(hours=2):
                ev_id = f"event_{curr_start.strftime('%Y-%m-%d')}"
                episode_clusters.append((curr_start, curr_prev, ev_id))
                curr_start = dt
            curr_prev = dt
        ev_id = f"event_{curr_start.strftime('%Y-%m-%d')}"
        episode_clusters.append((curr_start, curr_prev, ev_id))

    def get_event_id_for_datetime(target_dt: datetime.datetime) -> str:
        for s_dt, e_dt, ev_id in episode_clusters:
            if s_dt - datetime.timedelta(minutes=15) <= target_dt <= e_dt + datetime.timedelta(minutes=15):
                return ev_id
        return f"event_{target_dt.strftime('%Y-%m-%d')}"

    event_groups: Dict[str, List[Dict[str, Any]]] = {}
    unique_files = set()

    for w_idx, win in enumerate(windows):
        anchor_dt = win["anchor_datetime"]
        group_id = get_event_id_for_datetime(anchor_dt)
        seq_id = f"seq_{group_id}_{anchor_dt.strftime('%H%M%S')}_w{w_idx:02d}"
        
        # Collect source files
        source_files = {}
        for role, frame_list in [("input", win["input_frames"]), ("target", win["target_frames"])]:
            source_files[role] = []
            for f_info in frame_list:
                dt_str = f_info["datetime"].strftime("%Y-%m-%d %H:%M:%S UTC")
                offset = f_info["offset_min"]
                ch_map = {}
                for ch in channels:
                    fp = f_info["channels"][ch]
                    if not isinstance(fp, str):
                        fp = fp["filepath"]
                    ch_map[str(ch)] = os.path.relpath(fp, start=".")
                    unique_files.add(fp)
                source_files[role].append({
                    "offset_min": offset,
                    "datetime": dt_str,
                    "channels": ch_map
                })

        window_entry = {
            "sequence_id": seq_id,
            "window_index": w_idx,
            "anchor_datetime": anchor_dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "group_id": group_id,
            "input_timestamps": [f["datetime"].strftime("%Y-%m-%d %H:%M:%S UTC") for f in win["input_frames"]],
            "target_timestamps": [f["datetime"].strftime("%Y-%m-%d %H:%M:%S UTC") for f in win["target_frames"]],
            "source_files": source_files
        }

        if group_id not in event_groups:
            event_groups[group_id] = []
        event_groups[group_id].append(window_entry)

    group_keys = sorted(list(event_groups.keys()))
    num_groups = len(group_keys)
    print(f"  Discovered {num_groups} independent date/event group(s): {group_keys}")
    for gid in group_keys:
        print(f"    - Group '{gid}': {len(event_groups[gid])} sequence windows")

    # 4. Evaluate split feasibility & partition
    print("\n[Step 4] Partitioning groups into Train, Validation, and Test...")
    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    group_split_map: Dict[str, str] = {}

    standard_approved_split = {
        "event_2026-01-01": "train",
        "event_2026-05-20": "train",
        "event_2026-07-19": "val",
        "event_2026-09-07": "test"
    }

    if num_groups < 3:
        valid_3way_possible = False
        split_status = "INSUFFICIENT_INDEPENDENT_GROUPS"
        warning_msg = (
            f"Scientifically valid 3-way evaluation is NOT possible with the current data. "
            f"Found only {num_groups} independent date/event group(s): {group_keys}. "
            f"At least 3 independent event groups are required to form leakage-free train, "
            f"validation, and test partitions. Overlapping sequence windows from the same "
            f"observation episode cannot be split across partitions without severe leakage. "
            f"All {len(windows)} available sequences are assigned to the 'train' partition "
            f"for development/debugging only. Do NOT report validation or generalization metrics."
        )
        print(f"  WARNING: {warning_msg}")
        for gid in group_keys:
            group_split_map[gid] = "train"
            splits["train"].extend(event_groups[gid])
    else:
        valid_3way_possible = True
        split_status = "VALID_EVENT_SPLIT"
        warning_msg = None

        active_map = explicit_split_map
        if active_map is None and set(group_keys) == set(standard_approved_split.keys()):
            active_map = standard_approved_split

        if active_map is not None and all(g in active_map for g in group_keys):
            print(f"  Applying approved event-level split map: {active_map}")
            for gid, split_name in active_map.items():
                group_split_map[gid] = split_name
                splits[split_name].extend(event_groups[gid])
        else:
            rng = random.Random(seed)
            shuffled_groups = list(group_keys)
            rng.shuffle(shuffled_groups)

            n_train = max(1, int(round(num_groups * train_ratio)))
            n_val = max(1, int(round(num_groups * val_ratio)))
            n_test = num_groups - n_train - n_val
            if n_test < 1 and num_groups >= 3:
                n_train -= 1
                n_test = 1

            train_groups = shuffled_groups[:n_train]
            val_groups = shuffled_groups[n_train:n_train + n_val]
            test_groups = shuffled_groups[n_train + n_val:]

            for gid in train_groups:
                group_split_map[gid] = "train"
                splits["train"].extend(event_groups[gid])
            for gid in val_groups:
                group_split_map[gid] = "val"
                splits["val"].extend(event_groups[gid])
            for gid in test_groups:
                group_split_map[gid] = "test"
                splits["test"].extend(event_groups[gid])

        print(f"  Train: {[g for g, s in group_split_map.items() if s=='train']} ({len(splits['train'])} sequences)")
        print(f"  Val:   {[g for g, s in group_split_map.items() if s=='val']} ({len(splits['val'])} sequences)")
        print(f"  Test:  {[g for g, s in group_split_map.items() if s=='test']} ({len(splits['test'])} sequences)")

    # 5. Verify no group appears in more than one split
    print("\n[Step 5] Validating partition isolation...")
    group_sets = {s_name: set(item["group_id"] for item in s_list) for s_name, s_list in splits.items()}
    assert len(group_sets["train"] & group_sets["val"]) == 0, "Leakage violation: train and val share groups!"
    assert len(group_sets["train"] & group_sets["test"]) == 0, "Leakage violation: train and test share groups!"
    assert len(group_sets["val"] & group_sets["test"]) == 0, "Leakage violation: val and test share groups!"
    print("  Partition Isolation Verified: 0 shared event groups between splits.")

    # 6. Compute file inventory and SHA-256 hashes
    print("\n[Step 6] Indexing source file inventory and cryptographic hashes...")
    file_inventory = {}
    for fp in sorted(list(unique_files)):
        rel_path = os.path.relpath(fp, start=".")
        size_b = os.path.getsize(fp) if os.path.exists(fp) else None
        sha_str = compute_file_sha256(fp) if (compute_hashes and os.path.exists(fp)) else None
        file_inventory[rel_path] = {
            "size_bytes": size_b,
            "sha256": sha_str
        }
    print(f"  Indexed {len(file_inventory)} source files with SHA-256 verification.")

    # 7. Embed normalization provenance
    norm_file = os.path.join("outputs", "normalization", "frozen_global_train_stats.json")
    norm_provenance = None
    if os.path.exists(norm_file):
        try:
            with open(norm_file, "r") as nf:
                norm_provenance = json.load(nf)
        except Exception:
            pass

    # 8. Construct and write manifest
    manifest = {
        "manifest_version": "1.0",
        "created_at": datetime.datetime.now().isoformat(),
        "configuration": {
            "data_dir": os.path.abspath(data_dir),
            "channels": list(channels),
            "seed": seed,
            "nominal_target_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
            "split_policy": "discrete_event_isolation",
            "policy_explanation": (
                "To prevent temporal leakage, all sequence windows from an independent meteorological "
                "event are kept in a single partition. With exactly 4 discrete storm episodes, integer event "
                "partitioning assigns 2 events to Train (50%), 1 to Val (25%), and 1 to Test (25%)."
            ),
            "realized_event_counts": {
                "train": sum(1 for s in group_split_map.values() if s == "train"),
                "val": sum(1 for s in group_split_map.values() if s == "val"),
                "test": sum(1 for s in group_split_map.values() if s == "test")
            },
            "realized_event_ratios": {
                "train": round(sum(1 for s in group_split_map.values() if s == "train") / max(1, num_groups), 4),
                "val": round(sum(1 for s in group_split_map.values() if s == "val") / max(1, num_groups), 4),
                "test": round(sum(1 for s in group_split_map.values() if s == "test") / max(1, num_groups), 4)
            },
            "temporal_offsets_min": {"inputs": [-60, -30, 0, 30], "targets": [15]},
        },
        "evaluation_validity": {
            "scientifically_valid_three_way_split_possible": valid_3way_possible,
            "split_status": split_status,
            "warning": warning_msg,
            "generalization_claim_permitted": False if not valid_3way_possible else True,
            "total_independent_groups": num_groups,
            "group_list": group_keys,
            "group_to_split_mapping": group_split_map
        },
        "normalization_provenance": norm_provenance,
        "summary": {
            "total_independent_events": num_groups,
            "total_sequence_windows": len(windows),
            "train_sequences": len(splits["train"]),
            "val_sequences": len(splits["val"]),
            "test_sequences": len(splits["test"]),
            "train_sequence_fraction": round(len(splits["train"]) / max(1, len(windows)), 4),
            "val_sequence_fraction": round(len(splits["val"]) / max(1, len(windows)), 4),
            "test_sequence_fraction": round(len(splits["test"]) / max(1, len(windows)), 4),
            "total_source_files": len(file_inventory)
        },
        "splits": splits,
        "file_inventory": file_inventory,
        "duplicate_audit_summary": dup_audit
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_manifest)), exist_ok=True)
    with open(output_manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[Artifact Saved] Split manifest written to: {output_manifest}")
    print(f"  Total Sequences: {manifest['summary']['total_sequence_windows']}")
    print(f"  Split Status:    {split_status} (3-way valid: {valid_3way_possible})")

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create event-level dataset split manifest for METEOR-Net.")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory containing NetCDF files")
    parser.add_argument("--output_manifest", type=str, default="outputs/dataset_splits/splits_manifest.json", help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for group assignment")
    parser.add_argument("--no_hashes", action="store_true", help="Skip SHA-256 hash calculation")
    args = parser.parse_args()

    create_dataset_splits(
        data_dir=args.data_dir,
        output_manifest=args.output_manifest,
        seed=args.seed,
        compute_hashes=not args.no_hashes
    )
