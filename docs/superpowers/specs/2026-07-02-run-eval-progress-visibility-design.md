# run_eval.py 執行可視性強化設計

日期：2026-07-02

## 背景 / 問題

`run_eval.py` 執行時：

1. 開場只印出 `Loaded N problems from ... [strategy=xxx]`，看不到目前 strategy 底下實際用了哪些模型（`SINGLE_MODEL` / `ARCHITECT_MODEL`+`CODER_MODEL` / `DEBATER1_MODEL`+`DEBATER2_MODEL`+`JUDGE_MODEL`+`CODER_MODEL`）。
2. 每一題要跑完「plan（可能是 debate 的 5 次模型呼叫）→ 每次 attempt 的 code + 測試」才會印一行 PASS/FAIL 總結。中間完全沒有輸出。由於每次模型呼叫都是同步阻塞的網路請求（可能要數十秒），使用者無法分辨「還在跑」還是「卡住了」。

## 目標

- 執行開始時列出目前 strategy 使用的所有角色模型。
- 執行過程中，在每個會呼叫模型的步驟前後印出階段性訊息（角色、模型、完成耗時），讓使用者能即時看到目前跑到哪一步。

## 非目標

- 不做背景心跳執行緒 / spinner（使用者選擇「階段性訊息」而非心跳）。
- 不列印每一筆測試案例的即時進度（測試在本地 subprocess 執行，速度快，只需在整個 attempt 測完後印通過數）。
- 不引入 callback/依賴注入等新抽象；沿用現有 `orchestrator.py` 直接 `print()` 的風格（與現有 `_call_with_retry` 的重試訊息一致）。

## 設計

### 1. 啟動時顯示模型（`run_eval.py`）

在 `main()` 印出 `Loaded N problems...` 之後，呼叫既有的 `orchestrator.active_role_models(strategy)`，將角色名稱轉換為對應 env 變數名稱（`role.upper() + "_MODEL"`，與 `orchestrator._ROLE_ENV` 的既有命名規則一致）並逐行印出：

```
Loaded 5 problems from hard_subset.jsonl  [strategy=debate]

Models:
  DEBATER1_MODEL = deepseek/deepseek-v4-flash
  DEBATER2_MODEL = qwen/qwen3-max
  JUDGE_MODEL     = gemma/gemma-3-27b
  CODER_MODEL     = deepseek/deepseek-v4-flash
```

### 2. 即時階段訊息

**`run_eval.py`**：在呼叫 `solve_problem()` 之前，先印出「開始處理第幾題」的標頭行（目前是全部跑完才印），例如：

```
[2/5] hard    count-the-number-of-good-partitions
```

原本跑完後的 `[n/N] PASS/FAIL ...` 總結行維持不變，仍在 `solve_problem()` 回傳後印出。

**`orchestrator.py`**：在每個會觸發 `_chat()`（即模型呼叫）的地方，呼叫前後各印一行，帶上角色、模型、耗時，並用 `flush=True` 確保在阻塞呼叫期間訊息仍即時顯示（避免 Windows/管線緩衝延遲輸出）：

- `code()`：印 `coding attempt N/M (role_model)...` 與完成後 `coding attempt N/M done (Xs), running tests...`
- `_run_analyze_then_code_plan()`：印 `planning (architect, role_model)...` 與完成後 `planning done (Xs)`
- `_run_debate_plan()`：對 5 次模型呼叫（round1 debater1/debater2、round2 debater1/debater2、judge）各印開始/完成一行

`run_eval.py` 在收到每個 attempt 的測試結果後，印一行通過數摘要（若非最後一次嘗試會標註「retrying」）。

範例（debate strategy，第 2/5 題）：

```
[2/5] hard    count-the-number-of-good-partitions
        -> debate round 1: debater1 (deepseek/deepseek-v4-flash)...
        -> debate round 1: debater1 done (8.2s)
        -> debate round 1: debater2 (qwen/qwen3-max)...
        -> debate round 1: debater2 done (7.5s)
        -> debate round 2: debater1...
        -> debate round 2: debater1 done (6.1s)
        -> debate round 2: debater2...
        -> debate round 2: debater2 done (5.9s)
        -> judge synthesizing...
        -> judge done (4.3s)
        -> coding attempt 1/3 (deepseek/deepseek-v4-flash)...
        -> coding attempt 1/3 done (11.2s), running tests...
        -> attempt 1: 9/15 tests passed, retrying...
        -> coding attempt 2/3...
        -> coding attempt 2/3 done (10.8s), running tests...
[2/5] PASS  hard    count-the-number-of-good-partitions              (15/15)  attempts=2
```

## 資料流 / 錯誤處理

- 不新增資料結構；`log_entry` / `results.json` 內容不變，這些都只是額外的 stdout 側寫。
- 若某次模型呼叫拋出例外，現有的 `solve_problem()` try/except 仍會捕捉並記錄 `error`；新增的「開始」列印會讓使用者清楚看到是卡在哪一步失敗，不需額外錯誤處理邏輯。

## 測試考量

檢查過 `tests/` 目錄，沒有測試斷言 stdout 內容（`capsys` 未被使用），因此新增列印不會破壞既有測試。既有的 `test_orchestrator_code.py`、`test_run_eval_solve_problem.py`、`test_run_eval_main_integration.py` 應維持通過。
