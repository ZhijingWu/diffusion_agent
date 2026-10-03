import argparse
import json
import math
from collections import Counter


def topo_key(A):
    return tuple(tuple(int(x) for x in row) for row in A)


def exact_mcnemar_p(b, c):
    """
    Exact two-sided McNemar test using Binomial(n=b+c, p=0.5).

    b = proxy correct, first wrong
    c = proxy wrong, first correct
    """
    n = b + c

    if n == 0:
        return 1.0

    k = min(b, c)

    tail = sum(
        math.comb(n, i)
        for i in range(k + 1)
    ) / (2 ** n)

    return min(1.0, 2.0 * tail)


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--proxy",
        required=True,
    )

    p.add_argument(
        "--first",
        required=True,
    )

    args = p.parse_args()

    proxy = json.load(
        open(args.proxy, encoding="utf-8")
    )

    first = json.load(
        open(args.first, encoding="utf-8")
    )

    assert len(proxy) == len(first)

    proxy_by_idx = {
        x["Index"]: x for x in proxy
    }

    first_by_idx = {
        x["Index"]: x for x in first
    }

    indices = sorted(
        set(proxy_by_idx)
        & set(first_by_idx)
    )

    same_raw = 0
    same_exec = 0

    # McNemar:
    # b = proxy correct, first wrong
    # c = proxy wrong, first correct
    b = 0
    c = 0
    both_correct = 0
    both_wrong = 0

    same_exec_b = 0
    same_exec_c = 0

    diff_exec_b = 0
    diff_exec_c = 0

    same_exec_discordant = []
    diff_exec_discordant = []

    hamming_raw = []
    hamming_exec = []

    for idx in indices:

        P = proxy_by_idx[idx]
        F = first_by_idx[idx]

        p_raw = P["Generated_Topology"]
        f_raw = F["Generated_Topology"]

        p_exec = P["Executed_Topology"]
        f_exec = F["Executed_Topology"]

        raw_same = topo_key(p_raw) == topo_key(f_raw)
        exec_same = topo_key(p_exec) == topo_key(f_exec)

        same_raw += int(raw_same)
        same_exec += int(exec_same)

        raw_diff = sum(
            int(p_raw[i][j] != f_raw[i][j])
            for i in range(len(p_raw))
            for j in range(len(p_raw))
        )

        exec_diff = sum(
            int(p_exec[i][j] != f_exec[i][j])
            for i in range(len(p_exec))
            for j in range(len(p_exec))
        )

        hamming_raw.append(raw_diff)
        hamming_exec.append(exec_diff)

        ps = bool(P["Solved"])
        fs = bool(F["Solved"])

        if ps and fs:
            both_correct += 1

        elif (not ps) and (not fs):
            both_wrong += 1

        elif ps and (not fs):

            b += 1

            if exec_same:
                same_exec_b += 1
                same_exec_discordant.append(
                    (idx, "proxy_only")
                )
            else:
                diff_exec_b += 1
                diff_exec_discordant.append(
                    (idx, "proxy_only")
                )

        elif (not ps) and fs:

            c += 1

            if exec_same:
                same_exec_c += 1
                same_exec_discordant.append(
                    (idx, "first_only")
                )
            else:
                diff_exec_c += 1
                diff_exec_discordant.append(
                    (idx, "first_only")
                )

    n = len(indices)

    proxy_acc = sum(
        proxy_by_idx[i]["Solved"]
        for i in indices
    ) / n

    first_acc = sum(
        first_by_idx[i]["Solved"]
        for i in indices
    ) / n

    print("=" * 72)
    print("PAIRED PROXY vs FIRST DOWNSTREAM ANALYSIS")
    print("=" * 72)

    print("records:", n)
    print()

    print("=== ACCURACY ===")
    print("proxy:", proxy_acc)
    print("first:", first_acc)
    print("difference:", proxy_acc - first_acc)

    print()
    print("=== TOPOLOGY AGREEMENT ===")

    print(
        f"same raw topology: "
        f"{same_raw}/{n} "
        f"({same_raw/n:.1%})"
    )

    print(
        f"same executed topology: "
        f"{same_exec}/{n} "
        f"({same_exec/n:.1%})"
    )

    print(
        "mean raw Hamming:",
        sum(hamming_raw) / n
    )

    print(
        "mean executed Hamming:",
        sum(hamming_exec) / n
    )

    print()
    print("=== PAIRED OUTCOMES ===")

    print(
        "both correct:",
        both_correct
    )

    print(
        "both wrong:",
        both_wrong
    )

    print(
        "proxy correct / first wrong:",
        b
    )

    print(
        "proxy wrong / first correct:",
        c
    )

    print(
        "exact McNemar p-value:",
        exact_mcnemar_p(b, c)
    )

    print()
    print("=== WHEN EXECUTED TOPOLOGY IS IDENTICAL ===")

    print(
        "proxy-only wins:",
        same_exec_b
    )

    print(
        "first-only wins:",
        same_exec_c
    )

    print(
        "discordant outcomes:",
        same_exec_b + same_exec_c
    )

    print()
    print("=== WHEN EXECUTED TOPOLOGY DIFFERS ===")

    print(
        "proxy-only wins:",
        diff_exec_b
    )

    print(
        "first-only wins:",
        diff_exec_c
    )

    print(
        "discordant outcomes:",
        diff_exec_b + diff_exec_c
    )

    print()
    print("=== IDENTICAL-TOPOLOGY DISCORDANT CASES ===")

    for idx, winner in same_exec_discordant:
        print(idx, winner)

    print()
    print("=== DIFFERENT-TOPOLOGY DISCORDANT CASES ===")

    for idx, winner in diff_exec_discordant:
        print(idx, winner)


if __name__ == "__main__":
    main()
