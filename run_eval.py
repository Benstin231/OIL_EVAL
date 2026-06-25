"""
run_eval.py
===========
LiveBench-style harness for stress-testing a code-generation pipeline against a
small set of HARD problems.

For each problem it:
  1. builds a prompt (the LiveBench `turns` text already contains the full
     statement + starter code + "enclose code in ```python" instructions),
  2. calls generate(prompt) -> a complete Python program,
  3. extracts the code from the model's reply,
  4. runs that code against EVERY public + private test in a subprocess with a
     per-test timeout,
  5. scores it: a problem is SOLVED only if every single test passes
     (this is the LiveBench rule).

Two execution modes, decided per test by `testtype`:
  - "stdin"      : feed `input` to the program's stdin, compare stdout
                   (whitespace-normalised).
  - "functional" : wrap the generated `Solution` class with a driver that parses
                   one literal positional arg per line, calls
                   Solution().<fn_name>(*args), and compares the return value.

Output: per-problem PASS/FAIL with the first failing test, an overall score,
and full detail written to results.json.

Stdlib only, except the optional `openai` package used by the default
generate() hook. Runs on Windows.

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
  # 1. pick the hard subset (see select_hard.py)
  python select_hard.py --n 5
  python select_hard.py --n 8 --include-atcoder

  # 2. point the default hook at your Gemma 4 (OpenAI-compatible) server
  set MODEL_API_BASE=http://localhost:8000/v1     (Windows: use `set`)
  set MODEL_NAME=gemma-4
  set MODEL_API_KEY=sk-anything                    (often unused locally)

  # 3. run the eval
  python run_eval.py                               # defaults to hard_subset.jsonl
  python run_eval.py --subset hard_subset.jsonl --timeout 10
  python run_eval.py --max-tests 5                 # cap tests/problem for a quick smoke run

The single pluggable hook is generate() below — swap it for a direct call into
the students' orchestrator (see the comment there).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# THE ONE PLUGGABLE HOOK
# ---------------------------------------------------------------------------
def generate(prompt: str) -> str:
    """
    Turn a problem prompt into a complete Python program (as a string).

    DEFAULT: call an OpenAI-compatible chat endpoint configured via env vars:
        MODEL_API_BASE   e.g. http://localhost:8000/v1   (your Gemma 4 server)
        MODEL_NAME       e.g. gemma-4
        MODEL_API_KEY    e.g. sk-anything  (many local servers ignore this)

    ---------------------------------------------------------------------------
    TO PLUG IN THE STUDENTS' ORCHESTRATOR INSTEAD, replace this whole body with:

        import orchestrator
        return orchestrator.solve(prompt)        # must return a Python program

    That is the entire integration point. The orchestrator's iterative refine
    loop lives behind solve(); this harness just measures whether the final
    program passes the tests.
    ---------------------------------------------------------------------------

    NOTE on litellm: if you route through litellm instead of the openai client,
    do NOT use a `gemma/` or `google/` model prefix (not valid providers). Use
    `openai/<name>` with api_base set, or `ollama/<name>`.
    """
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit(
            "The default generate() hook needs the `openai` package.\n"
            "  pip install openai\n"
            "...or replace generate() with a direct orchestrator.solve(prompt) call."
        )

    base = os.environ.get("MODEL_API_BASE", "http://localhost:8000/v1")
    model = os.environ.get("MODEL_NAME", "gemma-4")
    key = os.environ.get("MODEL_API_KEY", "not-needed")

    client = OpenAI(base_url=base, api_key=key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

def extract_code(text: str) -> str:
    """Pull the program out of a model reply. Prefer a ```python fenced block;
    if several, take the longest; fall back to the raw text."""
    blocks = _FENCE.findall(text or "")
    if blocks:
        return max(blocks, key=len).strip()
    return (text or "").strip()


# ---------------------------------------------------------------------------
# Functional driver template
#   {CODE} = generated solution, {FN} = repr of the method name
# The driver reads the test input from stdin and the expected value from the
# file path given as argv[1], then prints a one-line JSON result envelope.
# ---------------------------------------------------------------------------
_FUNC_DRIVER = '''\
import sys, ast, json, re, math
from typing import *
from collections import *

# ===== generated solution =====
{CODE}
# ===== end generated solution =====

_PREFIX = re.compile(r"^[A-Za-z_]\\w*\\s*=(?!=)\\s*")  # strip leading "name ="

def _read_args(raw):
    args = []
    for line in raw.split("\\n"):
        s = line.strip()
        if not s:
            continue
        s = _PREFIX.sub("", s)
        args.append(ast.literal_eval(s))
    return args

def _norm(v):
    # tuples<->lists, so [1,2]==(1,2); recurse into containers
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    return v

def _equal(got, expected_raw):
    er = expected_raw.strip()
    for parse in (ast.literal_eval, json.loads):
        try:
            exp = parse(er)
            if got == exp or _norm(got) == _norm(exp):
                return True
        except Exception:
            pass
    return str(got).strip() == er

def _main():
    raw_in = sys.stdin.read()
    with open(sys.argv[1], encoding="utf-8") as fh:
        expected = fh.read()
    args = _read_args(raw_in)
    result = getattr(Solution(), {FN})(*args)
    ok = _equal(result, expected)
    sys.stdout.write("<<<LB_RESULT>>>" + json.dumps({{"ok": ok, "got": repr(result)}}))

_main()
'''


# ---------------------------------------------------------------------------
# Running a single test
# ---------------------------------------------------------------------------
def _normalize_ws(s: str) -> list:
    return s.split()

def run_stdin_test(py: str, prog_path: str, test: dict, timeout: float) -> dict:
    """Run a stdin-style test. Returns {ok, type, got}."""
    try:
        proc = subprocess.run(
            [py, prog_path],
            input=test["input"],
            capture_output=True, text=True, encoding="utf-8",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "type": "TIMEOUT", "got": ""}
    if proc.returncode != 0:
        return {"ok": False, "type": "RUNTIME_ERROR",
                "got": (proc.stderr or "").strip()[-500:]}
    ok = _normalize_ws(proc.stdout) == _normalize_ws(test["output"])
    return {"ok": ok, "type": "OK" if ok else "WRONG_ANSWER",
            "got": proc.stdout.strip()[:500]}

def run_functional_test(py: str, driver_path: str, test: dict,
                        exp_path: str, timeout: float) -> dict:
    """Run a functional-style test. Expected value is read from exp_path."""
    try:
        with open(exp_path, "w", encoding="utf-8") as fh:
            fh.write(test["output"])
        proc = subprocess.run(
            [py, driver_path, exp_path],
            input=test["input"],
            capture_output=True, text=True, encoding="utf-8",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "type": "TIMEOUT", "got": ""}
    if proc.returncode != 0:
        return {"ok": False, "type": "RUNTIME_ERROR",
                "got": (proc.stderr or "").strip()[-500:]}
    marker = "<<<LB_RESULT>>>"
    idx = proc.stdout.rfind(marker)
    if idx == -1:
        return {"ok": False, "type": "RUNTIME_ERROR",
                "got": (proc.stderr or proc.stdout).strip()[-500:]}
    try:
        payload = json.loads(proc.stdout[idx + len(marker):])
    except Exception:
        return {"ok": False, "type": "RUNTIME_ERROR", "got": "bad driver output"}
    return {"ok": payload["ok"],
            "type": "OK" if payload["ok"] else "WRONG_ANSWER",
            "got": str(payload.get("got", ""))[:500]}


# ---------------------------------------------------------------------------
# Evaluate one problem
# ---------------------------------------------------------------------------
def evaluate_problem(p: dict, code: str, timeout: float, max_tests: int) -> dict:
    py = sys.executable
    tests = p["tests"]
    if max_tests > 0:
        tests = tests[:max_tests]

    is_functional = bool(p.get("fn_name")) and any(
        t.get("testtype") == "functional" for t in tests
    )

    passed = 0
    first_failure = None
    with tempfile.TemporaryDirectory() as tmp:
        if is_functional:
            prog_path = os.path.join(tmp, "driver.py")
            with open(prog_path, "w", encoding="utf-8") as fh:
                fh.write(_FUNC_DRIVER.format(CODE=code, FN=repr(p["fn_name"])))
            exp_path = os.path.join(tmp, "expected.txt")
        else:
            prog_path = os.path.join(tmp, "sol.py")
            with open(prog_path, "w", encoding="utf-8") as fh:
                fh.write(code)

        for i, t in enumerate(tests):
            ttype = t.get("testtype", "functional" if is_functional else "stdin")
            if ttype == "functional" and is_functional:
                r = run_functional_test(py, prog_path, t, exp_path, timeout)
            else:
                r = run_stdin_test(py, prog_path, t, timeout)

            if r["ok"]:
                passed += 1
            elif first_failure is None:
                first_failure = {
                    "index": i,
                    "type": r["type"],
                    "input": t["input"][:300],
                    "expected": t["output"][:300],
                    "got": r["got"],
                }

    return {
        "solved": (first_failure is None and len(tests) > 0),
        "num_tests": len(tests),
        "tests_passed": passed,
        "first_failure": first_failure,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_subset(path: str) -> list:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="LiveBench-style code eval harness")
    ap.add_argument("--subset", default="hard_subset.jsonl",
                    help="JSONL of normalised problems (from select_hard.py)")
    ap.add_argument("--out", default="results.json", help="Detailed results file")
    ap.add_argument("--timeout", type=float, default=10.0,
                    help="Per-test timeout in seconds")
    ap.add_argument("--max-tests", type=int, default=0,
                    help="Cap tests per problem (0 = all). Use for quick smoke runs.")
    args = ap.parse_args()

    if not os.path.exists(args.subset):
        sys.exit(f"Subset file not found: {args.subset}\n"
                 f"Run:  python select_hard.py --n 5")

    problems = load_subset(args.subset)
    print(f"Loaded {len(problems)} problems from {args.subset}\n")

    results = []
    solved = 0
    for n, p in enumerate(problems, 1):
        title = p.get("title", "(untitled)")
        diff = p.get("difficulty", "?")
        prompt = p["prompt"]

        try:
            reply = generate(prompt)
        except Exception as exc:
            print(f"[{n}/{len(problems)}] ERROR  generate() failed: {exc}")
            results.append({**_meta(p), "solved": False, "error": str(exc)})
            continue

        code = extract_code(reply)
        ev = evaluate_problem(p, code, args.timeout, args.max_tests)
        if ev["solved"]:
            solved += 1

        status = "PASS" if ev["solved"] else "FAIL"
        line = (f"[{n}/{len(problems)}] {status}  {diff:6s}  {title:<48} "
                f"({ev['tests_passed']}/{ev['num_tests']})")
        if not ev["solved"] and ev["first_failure"]:
            ff = ev["first_failure"]
            line += f"  test#{ff['index']} {ff['type']}"
        print(line)
        if not ev["solved"] and ev["first_failure"]:
            ff = ev["first_failure"]
            print(f"        input:    {ff['input']!r}")
            print(f"        expected: {ff['expected']!r}")
            print(f"        got:      {ff['got']!r}")

        results.append({**_meta(p), **ev, "code": code, "reply_len": len(reply)})

    total = len(problems)
    score = solved / total if total else 0.0
    print(f"\n==== SCORE: {solved}/{total} solved  ({score:.0%}) ====")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({
            "summary": {
                "total": total, "solved": solved, "score": score,
                "model": os.environ.get("MODEL_NAME", "(default)"),
                "timeout": args.timeout, "max_tests": args.max_tests,
            },
            "results": results,
        }, fh, indent=2, ensure_ascii=False)
    print(f"Wrote detail -> {args.out}")


def _meta(p: dict) -> dict:
    return {
        "question_id": p.get("question_id"),
        "title": p.get("title"),
        "platform": p.get("platform"),
        "difficulty": p.get("difficulty"),
        "fn_name": p.get("fn_name"),
    }


if __name__ == "__main__":
    main()
