"""
Reviewer-friendly experiment execution CLI for FedUA-Net.
Supports individual method execution, multi-seed sweeps, data scarcity, ablations, and fast smoke tests.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_command(cmd: str):
    print(f"\n[RUNNING] {cmd}")
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print(f"[ERROR] Command failed with exit code {ret.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="FedUA-Net: Multi-Task Cross-Silo Federated Learning Benchmark CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--method", type=str, default="fedua",
                        choices=["fedua", "fedavg", "fedbn", "fedprox", "fedbabu", "ditto", "local_only", "centralized"],
                        help="Federated optimization strategy to run.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="List of random seeds (e.g. 0 1 2).")
    parser.add_argument("--all", action="store_true", help="Run all 8 benchmark methods across all 3 seeds.")
    parser.add_argument("--smoke", action="store_true", help="Fast smoke test with 1 round / 1 epoch.")
    parser.add_argument("--rounds", type=int, default=12, help="Number of federated communication rounds.")
    parser.add_argument("--local_epochs", type=int, default=1, help="Local training epochs per round.")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW.")
    parser.add_argument("--agg_weight_type", type=str, default="uniform", choices=["uniform", "sample_size"],
                        help="Server aggregation weighting strategy.")
    parser.add_argument("--hospital_b_subset_size", type=int, default=None,
                        help="Subsample Hospital B training data for data scarcity experiments (e.g. 200 or 100).")
    parser.add_argument("--output_dir", type=str, default="results/raw", help="Directory to save raw outputs and reports.")
    parser.add_argument("--data_root", type=str, default="Dataset", help="Root directory containing medical datasets.")
    parser.add_argument("--generate_tables", action="store_true", help="Auto-compile summary tables after completion.")
    parser.add_argument("--generate_figures", action="store_true", help="Auto-compile figures after completion.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.smoke:
        print("\n" + "=" * 70)
        print("EXECUTING FEDUA-NET SMOKE TEST")
        print("=" * 70)
        smoke_cmd = f"python fedua_net.py --smoke --rounds 1 --agg_weight_type {args.agg_weight_type}"
        success = run_command(smoke_cmd)
        if success:
            print("\n[SUCCESS] Smoke test completed successfully.")
        return

    seeds = args.seeds if args.seeds is not None else [args.seed]

    if args.all:
        print("\n" + "=" * 70)
        print(f"RUNNING ALL 8 BENCHMARK STRATEGIES ACROSS SEEDS {seeds}")
        print("=" * 70)
        seeds_str = " ".join(str(s) for s in seeds)
        scarcity_arg = f"--hospital_b_subset_size {args.hospital_b_subset_size}" if args.hospital_b_subset_size else ""
        cmd = (
            f"python experiment.py --run_all_baselines --rounds {args.rounds} "
            f"--local_epochs {args.local_epochs} --batch_size {args.batch_size} "
            f"--lr {args.lr} --agg_weight_type {args.agg_weight_type} "
            f"--seeds {seeds_str} --output_dir {args.output_dir} {scarcity_arg}"
        )
        run_command(cmd)
    else:
        for seed in seeds:
            print("\n" + "=" * 70)
            print(f"RUNNING STRATEGY: {args.method} (Seed: {seed})")
            print("=" * 70)
            scarcity_arg = f"--hospital_b_subset_size {args.hospital_b_subset_size}" if args.hospital_b_subset_size else ""
            cmd = (
                f"python experiment.py --strategy {args.method} --rounds {args.rounds} "
                f"--local_epochs {args.local_epochs} --batch_size {args.batch_size} "
                f"--lr {args.lr} --agg_weight_type {args.agg_weight_type} "
                f"--seed {seed} --output_dir {args.output_dir} {scarcity_arg}"
            )
            run_command(cmd)

    if args.generate_tables:
        run_command(f"python scripts/generate_tables.py --results_dir {args.output_dir}")

    if args.generate_figures:
        run_command(f"python scripts/generate_figures.py --results_dir {args.output_dir}")


if __name__ == "__main__":
    main()
