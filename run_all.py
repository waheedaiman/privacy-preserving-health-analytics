#!/usr/bin/env python3
# rebuild every number and picture in this project, in order.

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_SEEDS = [0, 1, 2]

# these run once. the cohort is the same for every seed.
SETUP = [("generate_data.py", "synthetic cohorts, and one split per seed")]

# these run once per seed. each writes into runs/seed<n>/.
PER_SEED = [
    ("train_baseline.py", [], "non-private targets: overfit and tuned"),
    ("train_dp.py", [], "DP federated targets across the epsilon sweep"),
    ("lira.py", ["--arms", "overfit", "tuned"], "LiRA shadow models"),
    ("mia_attack.py", [], "attack suite and privacy audit"),
]


def run(script, args):
    result = subprocess.run([sys.executable, str(ROOT / "src" / script), *args], cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"\n{script} failed with exit code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(
        description="rebuild every number and picture in this project, in order."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--fast", action="store_true", help="skip the LiRA stage")
    parser.add_argument("--shadows", type=int, default=32,
                        help="shadow models per arm per seed")
    args = parser.parse_args()

    seeds = args.seeds
    started = time.time()

    print(f"\n{'=' * 72}\nsetup: {SETUP[0][1]}\n{'=' * 72}")
    run("generate_data.py", ["--seeds", *[str(s) for s in seeds]])

    stages = [s for s in PER_SEED if not (args.fast and s[0] == "lira.py")]

    for seed in seeds:
        print(f"\n{'=' * 72}\nseed {seed}\n{'=' * 72}")
        for script, extra, description in stages:
            step = time.time()
            print(f"\n-- {script}: {description}")

            options = ["--seed", str(seed), *extra]
            if script == "lira.py":
                options += ["--shadows", str(args.shadows)]
            run(script, options)

            print(f"-- {script} finished in {time.time() - step:.1f}s")

    print(f"\n{'=' * 72}\naggregate: mean and spread over {len(seeds)} runs\n{'=' * 72}")
    run("mia_attack.py", ["--aggregate", *[str(s) for s in seeds]])
    run("build_deck.py", [])

    print(f"\nall stages complete in {time.time() - started:.0f}s "
          f"over {len(seeds)} seeds. figures are in results/.")


if __name__ == "__main__":
    main()
