import argparse
import json
from collections import defaultdict

import numpy as np


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="audit_results/guidance_effect.json",
        help="Output from audit_guidance_effect.py",
    )

    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[1e-8, 1e-7, 1e-6, 1e-5],
    )

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs = data["runs"]

    print("=" * 78)
    print("TIE-AWARE PROXY GUIDANCE AUDIT")
    print("=" * 78)
    print("Runs:", len(runs))
    print()

    # ----------------------------------------------------------
    # Basic split: proxy-guided == first-candidate or not
    # ----------------------------------------------------------

    same_runs = [
        r for r in runs
        if r["proxy_vs_first_exact"]
    ]

    changed_runs = [
        r for r in runs
        if not r["proxy_vs_first_exact"]
    ]

    print("=== BASIC ===")
    print(
        f"Proxy == First candidate: "
        f"{len(same_runs)}/{len(runs)} "
        f"({len(same_runs)/len(runs):.1%})"
    )

    print(
        f"Proxy != First candidate: "
        f"{len(changed_runs)}/{len(runs)} "
        f"({len(changed_runs)/len(runs):.1%})"
    )

    print()

    # ----------------------------------------------------------
    # Reward spread overall
    # ----------------------------------------------------------

    mean_spreads = np.array(
        [
            r["mean_candidate_reward_spread"]
            for r in runs
        ],
        dtype=float,
    )

    max_spreads = np.array(
        [
            r["max_candidate_reward_spread"]
            for r in runs
        ],
        dtype=float,
    )

    print("=== REWARD SPREAD ===")
    print(
        "Mean of mean reward spread:",
        f"{mean_spreads.mean():.12e}",
    )
    print(
        "Median of mean reward spread:",
        f"{np.median(mean_spreads):.12e}",
    )
    print(
        "Maximum observed reward spread:",
        f"{max_spreads.max():.12e}",
    )

    print()

    # ----------------------------------------------------------
    # Tie analysis under different epsilon values
    # ----------------------------------------------------------

    print("=" * 78)
    print("EPSILON / TIE ANALYSIS")
    print("=" * 78)

    for eps in args.thresholds:

        # If maximum spread across ALL 50 diffusion steps
        # in one run is <= eps, then no step in that run
        # ever had candidate discrimination larger than eps.
        fully_tied_runs = [
            r for r in runs
            if r["max_candidate_reward_spread"] <= eps
        ]

        fully_tied_changed = [
            r for r in changed_runs
            if r["max_candidate_reward_spread"] <= eps
        ]

        genuinely_discriminated = [
            r for r in runs
            if r["max_candidate_reward_spread"] > eps
        ]

        genuinely_discriminated_changed = [
            r for r in changed_runs
            if r["max_candidate_reward_spread"] > eps
        ]

        print()
        print(f"--- epsilon = {eps:.1e} ---")

        print(
            "Runs with ALL diffusion steps tied:",
            f"{len(fully_tied_runs)}/{len(runs)} "
            f"({len(fully_tied_runs)/len(runs):.1%})",
        )

        print(
            "Runs with any reward discrimination > eps:",
            f"{len(genuinely_discriminated)}/{len(runs)} "
            f"({len(genuinely_discriminated)/len(runs):.1%})",
        )

        print(
            "Among Proxy != First runs:",
        )

        print(
            "  changed despite being entirely within tie tolerance:",
            f"{len(fully_tied_changed)}/{len(changed_runs)}"
            if changed_runs else "N/A",
        )

        print(
            "  changed with at least one > eps discrimination:",
            f"{len(genuinely_discriminated_changed)}/{len(changed_runs)}"
            if changed_runs else "N/A",
        )

    # ----------------------------------------------------------
    # Examine the 11% differing runs specifically
    # ----------------------------------------------------------

    print()
    print("=" * 78)
    print("PROXY != FIRST-CANDIDATE RUNS")
    print("=" * 78)

    if not changed_runs:
        print("No differing runs.")
    else:

        changed_max_spreads = np.array(
            [
                r["max_candidate_reward_spread"]
                for r in changed_runs
            ],
            dtype=float,
        )

        changed_mean_spreads = np.array(
            [
                r["mean_candidate_reward_spread"]
                for r in changed_runs
            ],
            dtype=float,
        )

        changed_hamming = np.array(
            [
                r["proxy_vs_first_hamming"]
                for r in changed_runs
            ],
            dtype=float,
        )

        print(
            "Number of changed runs:",
            len(changed_runs),
        )

        print(
            "Mean reward spread in changed runs:",
            f"{changed_mean_spreads.mean():.12e}",
        )

        print(
            "Max reward spread among changed runs:",
            f"{changed_max_spreads.max():.12e}",
        )

        print(
            "Median max reward spread among changed runs:",
            f"{np.median(changed_max_spreads):.12e}",
        )

        print(
            "Mean normalized Hamming when changed:",
            f"{changed_hamming.mean():.6f}",
        )

        print()

        print(
            f"{'q':>4} "
            f"{'seed':>8} "
            f"{'max_spread':>16} "
            f"{'mean_spread':>16} "
            f"{'hamming':>10} "
            f"{'P_edges':>8} "
            f"{'F_edges':>8}"
        )

        for r in sorted(
            changed_runs,
            key=lambda x: x["max_candidate_reward_spread"],
            reverse=True,
        ):

            print(
                f"{r['question_index']:4d} "
                f"{r['seed']:8d} "
                f"{r['max_candidate_reward_spread']:16.9e} "
                f"{r['mean_candidate_reward_spread']:16.9e} "
                f"{r['proxy_vs_first_hamming']:10.4f} "
                f"{r['proxy_raw_edges']:8d} "
                f"{r['first_raw_edges']:8d}"
            )

    # ----------------------------------------------------------
    # Edge count comparison
    # ----------------------------------------------------------

    p_edges = np.array(
        [r["proxy_raw_edges"] for r in runs],
        dtype=float,
    )

    f_edges = np.array(
        [r["first_raw_edges"] for r in runs],
        dtype=float,
    )

    p_exec = np.array(
        [r["proxy_executed_edges"] for r in runs],
        dtype=float,
    )

    f_exec = np.array(
        [r["first_executed_edges"] for r in runs],
        dtype=float,
    )

    print()
    print("=" * 78)
    print("EDGE-COUNT EFFECT")
    print("=" * 78)

    print(
        "Proxy raw avg:",
        round(float(p_edges.mean()), 4),
    )

    print(
        "First raw avg:",
        round(float(f_edges.mean()), 4),
    )

    print(
        "Mean absolute raw-edge difference:",
        round(
            float(
                np.abs(p_edges - f_edges).mean()
            ),
            4,
        ),
    )

    print()

    print(
        "Proxy executed avg:",
        round(float(p_exec.mean()), 4),
    )

    print(
        "First executed avg:",
        round(float(f_exec.mean()), 4),
    )

    print(
        "Mean absolute executed-edge difference:",
        round(
            float(
                np.abs(p_exec - f_exec).mean()
            ),
            4,
        ),
    )

    # ----------------------------------------------------------
    # Final diagnosis
    # ----------------------------------------------------------

    print()
    print("=" * 78)
    print("AUTOMATIC DIAGNOSIS")
    print("=" * 78)

    eps = 1e-6

    meaningful_runs = sum(
        r["max_candidate_reward_spread"] > eps
        for r in runs
    )

    changed_meaningfully = sum(
        (
            not r["proxy_vs_first_exact"]
            and
            r["max_candidate_reward_spread"] > eps
        )
        for r in runs
    )

    print(
        f"Using epsilon={eps:.1e}:"
    )

    print(
        "Runs with any meaningful candidate discrimination:",
        meaningful_runs,
        "/",
        len(runs),
    )

    print(
        "Runs where Proxy changed output AND had meaningful discrimination:",
        changed_meaningfully,
        "/",
        len(runs),
    )

    if meaningful_runs == 0:

        print()
        print(
            "RESULT: Under epsilon=1e-6, "
            "the Proxy never meaningfully distinguished candidates."
        )

        print(
            "Any Proxy-vs-control trajectory differences therefore occurred "
            "under numerical near-ties."
        )

    else:

        print()
        print(
            "RESULT: Some runs contained reward differences larger than "
            "the tolerance. These should be inspected separately."
        )


if __name__ == "__main__":
    main()