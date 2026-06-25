"""
lcb_loader.py
=============
Loads LiveBench LCB_generation/question.jsonl and normalises every row into
one flat dict.  All the schema quirks are handled here so the rest of the
pipeline never has to care.

Schema quirks (confirmed against the actual file):
  - difficulty / starter_code / metadata live inside original_json, NOT at
    the top level.
  - original_json.metadata is a JSON string; func_name is inside it.
  - AtCoder problems have no difficulty label  →  normalised to "unknown".
  - public_test_cases  is a JSON string.
  - private_test_cases is base64( zlib( pickle( list ) ) ).
  - Each test: {"input": str, "output": str, "testtype": "stdin"|"functional"}
  - Functional input:  one positional arg per line (ast.literal_eval each).
  - Functional output: JSON-encoded value (json.loads handles true/false).

Normalised shape returned by load_problems():
{
    "question_id":  str,
    "title":        str,          # human-readable slug
    "platform":     str,          # "leetcode" | "atcoder"
    "prompt":       str,          # full problem statement
    "difficulty":   str,          # "easy"|"medium"|"hard"|"unknown"
    "task":         str,          # always "LCB_generation" for this file
    "starter_code": str,          # empty for stdin (AtCoder) problems
    "fn_name":      str | None,   # function name for functional problems
    "tests":        list[dict],   # public + private, see shape below
    "solution":     str,          # usually empty in this dump
    "partial":      str,          # code_completion prefix, usually empty
}

Each test dict:
{
    "input":    str,   # raw arg string (functional) or raw stdin (stdin)
    "output":   str,   # raw expected value / stdout
    "testtype": str,   # "functional" | "stdin"
}
"""

import ast
import base64
import json
import pickle
import zlib


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _decode_private(raw: str) -> list:
    """base64 → zlib → pickle → (optionally json.loads) → list of test dicts."""
    blob = zlib.decompress(base64.b64decode(raw.encode("utf-8")))
    obj = pickle.loads(blob)
    if isinstance(obj, str):
        obj = json.loads(obj)
    return obj


def _decode_tests(raw) -> list:
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        return _decode_private(raw)
    except Exception as e:
        raise ValueError(f"Cannot decode test cases: {e}") from e


def _parse_original_json(q: dict) -> dict:
    oj = q.get("original_json", {})
    if isinstance(oj, str):
        try:
            oj = json.loads(oj)
        except Exception:
            oj = {}
    return oj


def _parse_metadata(oj: dict) -> dict:
    raw = oj.get("metadata", "")
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _extract_prompt(q: dict, oj: dict) -> str:
    turns = q.get("turns")
    if isinstance(turns, list) and turns:
        return "\n".join(str(t) for t in turns)
    return oj.get("question_content") or q.get("question_content") or ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_problems(path: str) -> list:
    """
    Read *path* (a LiveBench LCB_generation question.jsonl) and return a list
    of normalised problem dicts.
    """
    problems = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                q = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  [warn] skipping malformed line {line_no}: {exc}")
                continue

            oj = _parse_original_json(q)
            meta = _parse_metadata(oj)

            difficulty = (oj.get("difficulty") or "unknown").lower()
            fn_name = meta.get("func_name") or meta.get("fn_name") or None
            platform = oj.get("platform", "unknown")

            public = _decode_tests(q.get("public_test_cases"))
            private_raw = q.get("private_test_cases", "")
            private = _decode_private(private_raw) if private_raw else []

            problems.append({
                "question_id":  str(q.get("question_id", f"line{line_no}")),
                "title":        oj.get("question_title") or q.get("question_title", "(untitled)"),
                "platform":     platform,
                "prompt":       _extract_prompt(q, oj),
                "difficulty":   difficulty,
                "task":         q.get("task", "LCB_generation"),
                "starter_code": oj.get("starter_code") or "",
                "fn_name":      fn_name,
                "tests":        list(public) + list(private),
                "solution":     q.get("solution") or "",
                "partial":      q.get("partial_solution") or "",
            })

    return problems


# ---------------------------------------------------------------------------
# Quick schema inspection (run directly to confirm the file looks right)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from collections import Counter

    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else r"livebench\data\live_bench\coding\LCB_generation\question.jsonl"
    )

    problems = load_problems(path)
    print(f"\nLoaded {len(problems)} problems from {path}\n")

    by_diff = Counter(p["difficulty"] for p in problems)
    by_plat = Counter(p["platform"] for p in problems)
    by_type = Counter(
        t["testtype"]
        for p in problems
        for t in p["tests"]
    )

    print("By difficulty:", dict(by_diff))
    print("By platform:  ", dict(by_plat))
    print("By test type: ", dict(by_type))

    # Show one example per difficulty bucket
    print()
    seen = set()
    for p in problems:
        d = p["difficulty"]
        if d not in seen:
            seen.add(d)
            n_tests = len(p["tests"])
            print(
                f"  [{d:7s}] {p['title']:<55} "
                f"tests={n_tests:2d}  fn={p['fn_name']}  platform={p['platform']}"
            )
