"""
run_eval.py
===========
LiveBench-style harness for stress-testing a code-generation pipeline against a
small set of HARD problems.

For each problem it:
  1. builds a prompt (the LiveBench `turns` text already contains the full
     statement + starter code + "enclose code in ```python" instructions),
  2. calls orchestrator.plan() -> strategy-specific analysis (or None if single),
  3. calls orchestrator.code() -> a complete Python program,
  4. extracts the code from the model's reply,
  5. runs that code against EVERY public + private test in a subprocess with a
     per-test timeout,
  6. scores it: a problem is SOLVED only if every single test passes
     (this is the LiveBench rule).

Two execution modes, decided per test by `testtype`:
  - "stdin"      : feed `input` to the program's stdin, compare stdout
                   (whitespace-normalised).
  - "functional" : wrap the generated `Solution` class with a driver that parses
                   one literal positional arg per line, calls
                   Solution().<fn_name>(*args), and compares the return value.

If a problem fails, the harness retries by feeding the previous code and the
first failing test back to orchestrator.code() and asking for a fix, up to
`--max-retries` times (stops early as soon as a problem passes). On retry,
only the coder step runs again — the planning/debate stage is not repeated.

Output: per-problem PASS/FAIL with the first failing test, an overall score,
and full detail written to results.json.

Dependencies: python-dotenv (for .env loading), openai (used by orchestrator.py
for all model calls). Runs on Windows.

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
  # 1. pick the hard subset (see select_hard.py)
  python select_hard.py --n 5
  python select_hard.py --n 8 --include-atcoder

  # 2. configure orchestrator.py and provider credentials in .env
  STRATEGY=single                          # or "analyze-then-code" or "debate"
  SINGLE_MODEL=deepseek/deepseek-v4-flash
  DEEPSEEK_API_KEY=...
  DEEPSEEK_BASE_URL=https://api.deepseek.com

  # 3. run the eval
  python run_eval.py                               # defaults to hard_subset.jsonl
  python run_eval.py --subset hard_subset.jsonl --timeout 10
  python run_eval.py --strategy analyze-then-code  # override STRATEGY env var
  python run_eval.py --max-tests 5                 # cap tests/problem for a quick smoke run
  python run_eval.py --max-retries 3                # retry failing problems up to 3 times

See .env.example for all available providers and role-model configuration options.
The integration point is orchestrator.py's plan()/code() functions.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import orchestrator

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


def solve_problem(p: dict, strategy: str, timeout: float, max_tests: int, max_retries: int) -> tuple:
    """Run one problem end-to-end: plan() once, then code()+evaluate up to
    max_retries+1 attempts. Returns (result, log_entry).
    result matches the existing results.json per-problem shape.
    log_entry is the full JSON-loggable trace (plan/debate events + every attempt)."""
    original_prompt = p["prompt"]
    log_entry = {**_meta(p), "plan_events": [], "debate_events": [], "attempts": [], "solved": False}

    try:
        plan_text, events = orchestrator.plan(strategy, original_prompt)
    except Exception as exc:
        log_entry["error"] = str(exc)
        return {**_meta(p), "solved": False, "error": str(exc), "attempts": 0}, log_entry

    if strategy == "debate":
        log_entry["debate_events"] = events
    else:
        log_entry["plan_events"] = events

    prev_code = None
    failure = None
    ev = None
    attempt_result = None
    attempt = 0
    for attempt in range(1, max_retries + 2):  # 1 initial + N retries
        try:
            attempt_result = orchestrator.code(strategy, original_prompt, plan_text, prev_code, failure)
        except Exception as exc:
            log_entry["error"] = str(exc)
            return {**_meta(p), "solved": False, "error": str(exc), "attempts": attempt}, log_entry

        code_text = attempt_result["code"]
        ev = evaluate_problem(p, code_text, timeout, max_tests)
        log_entry["attempts"].append({
            "attempt": attempt,
            "model": attempt_result["model"],
            "prompt": attempt_result["prompt"],
            "reply": attempt_result["reply"],
            "code": code_text,
            "duration_s": attempt_result["duration_s"],
            "test_result": ev,
        })
        prev_code = code_text
        if ev["solved"] or attempt == max_retries + 1:
            break
        failure = ev["first_failure"]

    log_entry["solved"] = bool(ev and ev["solved"])
    result = {
        **_meta(p), **ev, "code": prev_code,
        "reply_len": len(attempt_result["reply"]), "attempts": attempt,
    }
    return result, log_entry


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
    ap.add_argument("--max-retries", type=int, default=2,
                    help="Retries per problem after a failed attempt (0 = no retries).")
    ap.add_argument("--strategy", default=None,
                    help="single | analyze-then-code | debate "
                         "(default: STRATEGY env var, else 'single')")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    strategy = args.strategy or os.environ.get("STRATEGY", "single")
    try:
        orchestrator.validate_config(strategy)
    except orchestrator.ConfigError as exc:
        sys.exit(str(exc))

    if not os.path.exists(args.subset):
        sys.exit(f"Subset file not found: {args.subset}\n"
                 f"Run:  python select_hard.py --n 5")

    problems = load_subset(args.subset)
    print(f"Loaded {len(problems)} problems from {args.subset}  [strategy={strategy}]\n")

    results = []
    log_problems = []
    solved = 0
    for n, p in enumerate(problems, 1):
        title = p.get("title", "(untitled)")
        diff = p.get("difficulty", "?")

        result, log_entry = solve_problem(p, strategy, args.timeout, args.max_tests, args.max_retries)
        log_problems.append(log_entry)
        results.append(result)

        if "error" in result:
            print(f"[{n}/{len(problems)}] ERROR  {title}: {result['error']}")
            continue

        if result["solved"]:
            solved += 1

        status = "PASS" if result["solved"] else "FAIL"
        line = (f"[{n}/{len(problems)}] {status}  {diff:6s}  {title:<48} "
                f"({result['tests_passed']}/{result['num_tests']})  attempts={result['attempts']}")
        if not result["solved"] and result["first_failure"]:
            ff = result["first_failure"]
            line += f"  test#{ff['index']} {ff['type']}"
        print(line)
        if not result["solved"] and result["first_failure"]:
            ff = result["first_failure"]
            print(f"        input:    {ff['input']!r}")
            print(f"        expected: {ff['expected']!r}")
            print(f"        got:      {ff['got']!r}")

    total = len(problems)
    score = solved / total if total else 0.0
    print(f"\n==== SCORE: {solved}/{total} solved  ({score:.0%}) ====")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({
            "summary": {
                "total": total, "solved": solved, "score": score,
                "strategy": strategy, "models": orchestrator.active_role_models(strategy),
                "timeout": args.timeout, "max_tests": args.max_tests,
                "max_retries": args.max_retries,
            },
            "results": results,
        }, fh, indent=2, ensure_ascii=False)
    print(f"Wrote detail -> {args.out}")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("logs", f"run_{run_id}.json")
    run_log = {
        "run_id": run_id,
        "config": {
            "strategy": strategy, "models": orchestrator.active_role_models(strategy),
            "subset": args.subset, "max_retries": args.max_retries,
        },
        "problems": log_problems,
    }
    orchestrator.write_run_log(log_path, run_log)
    print(f"Wrote detailed log -> {log_path}")


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
