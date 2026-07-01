import json
import os

import orchestrator


def test_write_run_log_creates_parent_dir_and_valid_json(tmp_path):
    path = os.path.join(str(tmp_path), "logs", "run_20260701_000000.json")
    run_log = {"run_id": "20260701_000000", "problems": [{"question_id": "1", "solved": True}]}

    orchestrator.write_run_log(path, run_log)

    assert os.path.exists(path)
    with open(path, encoding="utf-8") as fh:
        loaded = json.load(fh)
    assert loaded == run_log
