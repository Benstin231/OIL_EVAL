"""
select_hard.py
==============
Pick the N hardest problems from question.jsonl and write them to a subset
file that run_eval.py can consume.

Ranking strategy
----------------
1. difficulty == "hard" problems always come first.
2. Ties (and "unknown" / AtCoder problems when --include-atcoder is set) are
   ranked by:  len(tests) * 3 + len(prompt)   (more tests & longer problem
   statement ≈ harder for small models).

Why hard problems?
   Small models (Gemma 4) typically fail in one of three instructive ways:
     a) Right idea, wrong complexity  →  brute-force passes public tests but
        TLEs on the large private ones.
     b) Missed edge cases  →  passes most tests, fails a corner case.
     c) Needs a non-obvious algorithm  →  entirely wrong approach.
   These failure modes make the refine loop *actually have to work*.

Usage
-----
    # 5 hard LeetCode-only problems (default)
    python select_hard.py

    # 8 hard problems, include AtCoder stdin problems too
    python select_hard.py --n 8 --include-atcoder

    # Custom source / output
    python select_hard.py --src path/to/question.jsonl --out my_subset.jsonl
"""

import argparse
import json
import sys

from lcb_loader import load_problems

DEFAULT_SRC = r"question.jsonl"
DEFAULT_OUT = "hard_subset.jsonl"
DEFAULT_N   = 5


def rank_key(p: dict) -> tuple:
    """Lower tuple  →  picked first (we sort ascending then take head)."""
    diff_order = {"hard": 0, "unknown": 1, "medium": 2, "easy": 3}
    d = diff_order.get(p["difficulty"], 2)
    # Within same difficulty: most tests and longest prompt come first
    tiebreak = -(len(p["tests"]) * 3 + len(p["prompt"]))
    return (d, tiebreak)


def select(problems: list, n: int, include_atcoder: bool) -> list:
    candidates = problems
    if not include_atcoder:
        candidates = [p for p in problems if p["platform"] == "leetcode"]

    candidates.sort(key=rank_key)

    chosen = candidates[:n]
    if len(chosen) < n:
        print(
            f"  [warn] only {len(chosen)} candidates available "
            f"(requested {n}); returning all of them.",
            file=sys.stderr,
        )
    return chosen


def write_subset(problems: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for p in problems:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {len(problems)} problems -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select hard problems for eval")
    parser.add_argument("--src", default=DEFAULT_SRC,
                        help="Source question.jsonl path")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="Output subset JSONL path")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help="Number of problems to select")
    parser.add_argument("--include-atcoder", action="store_true",
                        help="Include AtCoder (stdin) problems in the pool")
    args = parser.parse_args()

    problems = load_problems(args.src)
    chosen   = select(problems, args.n, args.include_atcoder)

    print(f"\nSelected {len(chosen)} problems:")
    for p in chosen:
        print(
            f"  [{p['difficulty']:7s}] {p['title']:<55} "
            f"tests={len(p['tests']):2d}  platform={p['platform']}"
        )

    write_subset(chosen, args.out)


if __name__ == "__main__":
    main()
