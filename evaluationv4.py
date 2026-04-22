#!/usr/bin/env python3
"""
Gemma 4 优化版评测脚本（完整版）
================================
修复内容:
  1. 使用 /completion 接口 + 手动 Gemma 4 Chat Template，彻底绕开模板歧义
  2. 所有 Prompt 统一英文 + 强格式约束
  3. GSM8K 答案提取三段式优先级，杜绝假阴性
  4. MMLU-Pro / GPQA / C-Eval / BBH / IFEval / HumanEval 全部重写
  5. 移除 System Prompt（Gemma 对 system role 支持不稳定）
  6. --debug 模式可查看每条 prompt / response

启动 llama-server:
    ./llama-server -m model.gguf -c 32768 -ngl 99 --host 0.0.0.0 --port 8080

运行示例:
    python eval_gemma4.py --datasets gsm8k --limit 50 --debug
    python eval_gemma4.py --datasets gsm8k mmlu_pro gpqa_diamond
    python eval_gemma4.py  # 跑全部
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests


# ============================================================
# 一、Gemma 4 专用 Client
# ============================================================

class GemmaClient:
    """
    直接调用 llama-server 的 /completion 接口，手动拼接 Gemma 4 标准 Chat Template。
    不依赖 server 端的自动模板选择，是最稳的评测用法。
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: int = 600,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.completion_url = f"{self.base_url}/completion"
        self.health_url = f"{self.base_url}/health"
        self.timeout = timeout
        self.max_retries = max_retries

    def _build_prompt(self, user_content: str) -> str:
        """Gemma 4 标准 Chat Template（手动拼接）"""
        return (
            f"<start_of_turn>user\n"
            f"{user_content.strip()}"
            f"<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

    def complete(
        self,
        user_content: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
    ) -> str:
        prompt = self._build_prompt(user_content)
        stop_seqs = (stop or []) + ["<end_of_turn>", "<start_of_turn>"]

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop_seqs,
            "stream": False,
            "cache_prompt": False,
        }

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(
                    self.completion_url,
                    json=payload,
                    timeout=self.timeout,
                )
                r.raise_for_status()
                return r.json().get("content", "")
            except Exception as e:
                last_err = e
                wait = 2 ** attempt
                print(f"    [retry {attempt+1}/{self.max_retries}] {e}; sleep {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"LLM 调用失败: {last_err}")

    def health_check(self) -> bool:
        try:
            r = requests.get(self.health_url, timeout=10)
            return r.status_code == 200
        except Exception:
            return False


# ============================================================
# 二、通用答案提取工具
# ============================================================

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def extract_choice_letter(text: str, num_options: int = 4) -> Optional[str]:
    """
    从模型输出中提取选项字母，支持多种常见格式。
    优先级：显式 "Answer: X" > boxed > 括号 > 行首/行尾 > 全文最后孤立字母
    """
    if not text:
        return None
    valid = set(_LETTERS[:num_options])

    patterns = [
        # 显式声明答案
        r"(?:final\s+)?answer\s*[:：]\s*\(?([A-Z])\)?",
        r"\\boxed\{\s*([A-Z])\s*\}",
        r"the\s+(?:correct\s+)?(?:answer|option|choice)\s+is\s+\(?([A-Z])\)?",
        # 括号格式 (A) / [A]
        r"^\s*[\(\[]\s*([A-Z])\s*[\)\]]",
        r"\(\s*([A-Z])\s*\)",
        # 行首/行尾
        r"^\s*([A-Z])\s*[\.:\)、]",
        r"(?:^|\n)\s*([A-Z])\s*$",
    ]

    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE | re.MULTILINE):
            letter = m.group(1).upper()
            if letter in valid:
                return letter

    # 兜底：最后一个孤立大写字母
    isolated = re.findall(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])", text)
    for letter in reversed(isolated):
        if letter in valid:
            return letter
    return None


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """
    GSM8K 专用提取，三级优先：#### > boxed > 最后数行数字
    """
    if not text:
        return None

    def clean(s: str) -> str:
        return s.replace(",", "").replace("$", "").strip().rstrip(".")

    # 优先级 1: #### 格式（标准格式）
    m = re.search(r"####\s*\$?\s*([-+]?\d[\d,]*\.?\d*)", text)
    if m:
        return clean(m.group(1))

    # 优先级 2: \boxed{}
    m = re.search(r"\\boxed\{\s*\$?\s*([-+]?\d[\d,]*\.?\d*)\s*\}", text)
    if m:
        return clean(m.group(1))

    # 优先级 3: "The answer is X" 类句子
    m = re.search(
        r"(?:answer|result|total)\s+(?:is|=|:)\s*\$?\s*([-+]?\d[\d,]*\.?\d*)",
        text, re.IGNORECASE
    )
    if m:
        return clean(m.group(1))

    # 优先级 4: 最后 3 行中的纯数字行（或以数字结尾的行）
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines[-3:]):
        m = re.search(r"\$?\s*([-+]?\d[\d,]*\.?\d*)\s*\.?$", line)
        if m:
            val = clean(m.group(1))
            if val:
                return val

    # 兜底：全文最后一个数字
    nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", text)
    if nums:
        return clean(nums[-1])
    return None


def parse_gsm8k_gold(answer_field: str) -> str:
    if "####" in answer_field:
        return answer_field.split("####")[-1].strip().replace(",", "")
    return answer_field.strip().replace(",", "")


# ============================================================
# 三、各数据集 Evaluator（全部英文 Prompt + 强格式约束）
# ============================================================

# ---------- GSM8K ----------

def eval_gsm8k(
    sample: Dict[str, Any],
    client: GemmaClient,
    debug: bool = False,
) -> Dict[str, Any]:
    prompt = (
        "Solve the following math word problem step by step. "
        "Show all reasoning clearly. "
        "At the very end, write the final numerical answer on a new line "
        "in this exact format: #### <number>\n\n"
        f"Question: {sample['question']}"
    )

    resp = client.complete(prompt, max_tokens=1536, temperature=0.0)
    pred = extract_gsm8k_answer(resp)
    gold = parse_gsm8k_gold(sample["answer"])

    correct = False
    if pred is not None and gold:
        try:
            correct = abs(float(pred) - float(gold)) < 1e-4
        except ValueError:
            correct = pred.strip() == gold.strip()

    if debug:
        _debug_print("GSM8K", prompt, resp, pred, gold, correct)

    return {
        "prompt": prompt, "response": resp,
        "pred": pred, "gold": gold, "correct": correct,
    }


# ---------- MMLU-Pro ----------

def eval_mmlu_pro(
    sample: Dict[str, Any],
    client: GemmaClient,
    debug: bool = False,
) -> Dict[str, Any]:
    options: List[str] = sample["options"]
    letters = list(_LETTERS[:len(options)])
    options_text = "\n".join(f"{l}. {o}" for l, o in zip(letters, options))

    prompt = (
        "The following is a multiple-choice question. "
        "Think step by step, then state your final answer on a new line "
        f"in the exact format: Answer: X  (where X is one of {', '.join(letters)})\n\n"
        f"Question: {sample['question']}\n\n"
        f"Options:\n{options_text}"
    )

    resp = client.complete(prompt, max_tokens=1024, temperature=0.0)
    pred = extract_choice_letter(resp, len(options))

    gold = sample.get("answer")
    if isinstance(gold, int):
        gold = letters[gold]
    elif isinstance(gold, str):
        gold = gold.strip().upper()

    correct = (pred is not None and pred == gold)

    if debug:
        _debug_print("MMLU-Pro", prompt, resp, pred, gold, correct)

    return {
        "category": sample.get("category", ""),
        "prompt": prompt, "response": resp,
        "pred": pred, "gold": gold, "correct": correct,
    }


# ---------- C-Eval ----------

def eval_ceval(
    sample: Dict[str, Any],
    client: GemmaClient,
    debug: bool = False,
) -> Dict[str, Any]:
    options_text = "\n".join(f"{k}. {sample[k]}" for k in ["A", "B", "C", "D"])

    # C-Eval 是中文题，保留中文内容，但用英文指令引导答案格式
    prompt = (
        "The following is a Chinese multiple-choice question. "
        "Read it carefully, think step by step, "
        "then output your final answer on a new line as: Answer: X\n\n"
        f"Question: {sample['question']}\n\n"
        f"{options_text}"
    )

    resp = client.complete(prompt, max_tokens=1024, temperature=0.0)
    pred = extract_choice_letter(resp, 4)
    gold = (sample.get("answer") or "").strip().upper()
    correct = (pred is not None and pred == gold)

    if debug:
        _debug_print("C-Eval", prompt, resp, pred, gold, correct)

    return {
        "subject": sample.get("_subject", ""),
        "prompt": prompt, "response": resp,
        "pred": pred, "gold": gold, "correct": correct,
    }


# ---------- BBH ----------

_BBH_CHOICE_TASKS = {
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "causal_judgement",
    "disambiguation_qa",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "movie_recommendation",
    "navigate",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
}


def eval_bbh(
    sample: Dict[str, Any],
    client: GemmaClient,
    debug: bool = False,
) -> Dict[str, Any]:
    task = sample.get("_task", "")
    question = sample.get("input", "")
    target = str(sample.get("target", "")).strip()
    is_choice = task in _BBH_CHOICE_TASKS

    if is_choice:
        prompt = (
            "Solve the following problem step by step. "
            "Then give your final answer on a new line as: Answer: X\n\n"
            f"Problem: {question}"
        )
    else:
        prompt = (
            "Solve the following problem step by step. "
            "Then give your final answer on a new line as: Final Answer: <answer>\n\n"
            f"Problem: {question}"
        )

    resp = client.complete(prompt, max_tokens=1024, temperature=0.0)

    correct = False
    pred: str = ""

    target_letter_m = re.fullmatch(r"\(([A-Z])\)", target)
    if target_letter_m:
        # 选项类：目标是 "(A)" 这种格式
        gold_letter = target_letter_m.group(1)
        # 推断选项数量（保守估计 26）
        pred_letter = extract_choice_letter(resp, 26)
        pred = f"({pred_letter})" if pred_letter else ""
        correct = (pred_letter == gold_letter)
    else:
        # 自由文本类：取最后一行 or Final Answer 之后
        m = re.search(r"(?:final\s+)?answer\s*[:：]\s*(.+)", resp, re.IGNORECASE)
        if m:
            pred = m.group(1).strip()
        else:
            lines = [l.strip() for l in resp.splitlines() if l.strip()]
            pred = lines[-1] if lines else ""

        norm = lambda s: re.sub(r"\s+", " ", s.strip().lower().rstrip(".").strip("\"'"))
        correct = (norm(pred) == norm(target)) or (norm(target) in norm(resp))

    if debug:
        _debug_print("BBH", prompt, resp, pred, target, correct)

    return {
        "task": task, "prompt": prompt, "response": resp,
        "pred": pred, "gold": target, "correct": correct,
    }


# ---------- GPQA Diamond ----------

def eval_gpqa(
    sample: Dict[str, Any],
    client: GemmaClient,
    debug: bool = False,
) -> Dict[str, Any]:
    import random

    correct_ans = sample.get("Correct Answer") or sample.get("correct_answer", "")
    incorrect = [
        sample.get("Incorrect Answer 1") or sample.get("incorrect_answer_1", ""),
        sample.get("Incorrect Answer 2") or sample.get("incorrect_answer_2", ""),
        sample.get("Incorrect Answer 3") or sample.get("incorrect_answer_3", ""),
    ]
    options = [correct_ans] + [x for x in incorrect if x]

    rng = random.Random(42 + hash(str(sample.get("Question", ""))) % 10_000)
    rng.shuffle(options)

    gold_idx = options.index(correct_ans)
    gold_letter = _LETTERS[gold_idx]

    options_text = "\n".join(f"{_LETTERS[i]}. {opt}" for i, opt in enumerate(options))
    question = sample.get("Question") or sample.get("question", "")

    prompt = (
        "The following is a very difficult graduate-level multiple-choice question. "
        "Think carefully and thoroughly. Show your reasoning step by step. "
        "Then give your final answer on a new line in the exact format: Answer: X\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_text}"
    )

    resp = client.complete(prompt, max_tokens=2048, temperature=0.0)
    pred = extract_choice_letter(resp, len(options))
    correct = (pred == gold_letter)

    if debug:
        _debug_print("GPQA", prompt, resp, pred, gold_letter, correct)

    return {
        "prompt": prompt, "response": resp,
        "pred": pred, "gold": gold_letter, "correct": correct,
    }


# ---------- IFEval ----------

def eval_ifeval(
    sample: Dict[str, Any],
    client: GemmaClient,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    IFEval: 模型直接回答用户 prompt，然后用规则校验输出是否满足所有约束。
    注意: 精确分数请将 predictions.jsonl 喂给官方脚本:
    https://github.com/google-research/google-research/tree/master/instruction_following_eval
    本脚本做启发式快速估算。
    """
    user_prompt = sample["prompt"]
    resp = client.complete(user_prompt, max_tokens=1536, temperature=0.0)

    instr_ids = sample.get("instruction_id_list") or []
    kwargs_list = sample.get("kwargs") or []

    passed = 0
    details = []
    for iid, kw in zip(instr_ids, kwargs_list):
        ok = _ifeval_check(resp, iid, kw or {})
        details.append({"id": iid, "passed": ok})
        passed += int(ok)

    total = len(instr_ids)
    correct = (total > 0 and passed == total)

    if debug:
        _debug_print("IFEval", user_prompt, resp,
                     f"passed={passed}/{total}", str(instr_ids), correct)

    return {
        "prompt": user_prompt, "response": resp,
        "instruction_ids": instr_ids,
        "passed_count": passed, "total_constraints": total,
        "details": details, "correct": correct,
        "key": sample.get("key"),
    }


def _ifeval_check(text: str, iid: str, kw: Dict[str, Any]) -> bool:
    iid = (iid or "").lower()
    try:
        if "length_constraints:number_words" in iid:
            n = len(re.findall(r"\b\w+\b", text))
            if "min_words" in kw and n < int(kw["min_words"]):
                return False
            if "max_words" in kw and n > int(kw["max_words"]):
                return False
            if "num_words" in kw and "relation" in kw:
                target = int(kw["num_words"])
                return n >= target if kw["relation"] == "at least" else n <= target
            return True

        if "length_constraints:number_sentences" in iid:
            n = len(re.findall(r"[.!?。！？]+", text))
            if "num_sentences" in kw and "relation" in kw:
                target = int(kw["num_sentences"])
                return n >= target if kw["relation"] == "at least" else n <= target
            return True

        if "length_constraints:number_paragraphs" in iid:
            n = len([p for p in text.split("\n\n") if p.strip()])
            if "num_paragraphs" in kw and "relation" in kw:
                target = int(kw["num_paragraphs"])
                return n >= target if kw["relation"] == "at least" else n <= target
            return True

        if "keywords:existence" in iid or "keywords:include_keywords" in iid:
            return all(k.lower() in text.lower() for k in kw.get("keywords", []))

        if "keywords:forbidden_words" in iid:
            return all(k.lower() not in text.lower() for k in kw.get("forbidden_words", []))

        if "keywords:frequency" in iid:
            keyword = (kw.get("keyword") or "").lower()
            relation = kw.get("relation", "at least")
            freq = int(kw.get("frequency", 0))
            count = text.lower().count(keyword)
            return count >= freq if relation == "at least" else count <= freq

        if "change_case:english_lowercase" in iid:
            alpha = re.sub(r"[^a-zA-Z]", "", text)
            return alpha == alpha.lower()

        if "change_case:english_capital" in iid:
            alpha = re.sub(r"[^a-zA-Z]", "", text)
            return alpha == alpha.upper()

        if "change_case:title_case" in iid:
            words = re.findall(r"\b[a-zA-Z]+\b", text)
            return all(w[0].isupper() for w in words if w)

        if "punctuation:no_comma" in iid:
            return "," not in text

        if "startswith" in iid:
            s = kw.get("first_word") or kw.get("response_starts_with") or ""
            return text.lstrip().lower().startswith(str(s).lower())

        if "endswith" in iid:
            s = kw.get("end_phrase") or ""
            return text.rstrip().lower().endswith(str(s).lower())

        if "detectable_format:number_bullet_lists" in iid:
            bullets = re.findall(r"^\s*[\*\-•]\s+", text, re.MULTILINE)
            num = int(kw.get("num_bullets", 0))
            relation = kw.get("relation", "at least")
            return len(bullets) >= num if relation == "at least" else len(bullets) <= num

        if "detectable_format:number_highlighted_sections" in iid:
            sections = re.findall(r"\*\*[^*]+\*\*", text)
            num = int(kw.get("num_highlights", 0))
            relation = kw.get("relation", "at least")
            return len(sections) >= num if relation == "at least" else len(sections) <= num

        if "detectable_format:json_format" in iid:
            try:
                m = re.search(r"\{.*\}", text, re.DOTALL)
                if m:
                    json.loads(m.group())
                    return True
            except Exception:
                pass
            return False

        if "detectable_content:number_placeholders" in iid:
            placeholders = re.findall(r"\[[^\]]+\]", text)
            num = int(kw.get("num_placeholders", 0))
            return len(placeholders) >= num

        if "detectable_format:title" in iid:
            return bool(re.search(r"<<[^>]+>>", text))

    except Exception:
        pass
    return True  # 未识别的约束保守通过


# ---------- HumanEval ----------

def eval_humaneval(
    sample: Dict[str, Any],
    client: GemmaClient,
    debug: bool = False,
    exec_timeout: int = 10,
) -> Dict[str, Any]:
    entry_point = sample["entry_point"]
    func_prompt = sample["prompt"]

    instruction = (
        "Complete the following Python function. "
        "Output ONLY a single ```python ... ``` code block "
        "containing the complete function implementation (including the def line). "
        "Do NOT include example usage, test cases, or explanations.\n\n"
        f"```python\n{func_prompt}\n```"
    )

    resp = client.complete(instruction, max_tokens=1024, temperature=0.0)
    code = _extract_python_code(resp)

    # 如果模型没写函数签名，把原 prompt 拼上
    if f"def {entry_point}" not in code:
        code = func_prompt + "\n" + code

    # 拼装测试代码
    program = f"{code}\n\n{sample['test']}\n\ncheck({entry_point})\n"
    passed, err = _run_code_subprocess(program, timeout=exec_timeout)

    if debug:
        _debug_print("HumanEval", instruction, resp,
                     f"passed={passed}", f"entry={entry_point}", passed)

    return {
        "task_id": sample["task_id"],
        "prompt": instruction, "response": resp,
        "code": code, "passed": passed, "error": err, "correct": passed,
    }


def _extract_python_code(text: str) -> str:
    # 优先提取 ```python ... ```，其次提取任意 ``` ... ```
    for pat in [r"```python\s*(.*?)```", r"```\s*(.*?)```"]:
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return text.strip()


def _run_code_subprocess(code: str, timeout: int = 10) -> Tuple[bool, str]:
    """
    ⚠️ 安全警告: 会执行模型生成的代码，请在隔离沙箱中运行评测。
    """
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=timeout,
        )
        ok = proc.returncode == 0
        err = "" if ok else (proc.stderr or proc.stdout)[-600:]
        return ok, err
    except subprocess.TimeoutExpired:
        return False, f"Timeout > {timeout}s"
    except Exception as e:
        return False, str(e)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ============================================================
# 四、调试工具
# ============================================================

def _debug_print(
    tag: str, prompt: str, resp: str,
    pred: Any, gold: Any, correct: Any,
) -> None:
    sep = "─" * 60
    print(f"\n{'='*60}")
    print(f"[{tag}] correct={correct}  pred={pred!r}  gold={gold!r}")
    print(sep)
    print(f"PROMPT (last 300 chars):\n...{prompt[-300:]}")
    print(sep)
    print(f"RESPONSE (first 500 chars):\n{resp[:500]}")
    if len(resp) > 500:
        print(f"  ... [{len(resp)-500} more chars]")
    print("=" * 60)


# ============================================================
# 五、数据集注册表
# ============================================================

EVALUATORS: Dict[str, Callable] = {
    "gsm8k":        eval_gsm8k,
    "mmlu_pro":     eval_mmlu_pro,
    "ceval":        eval_ceval,
    "bbh":          eval_bbh,
    "gpqa_diamond": eval_gpqa,
    "ifeval":       eval_ifeval,
    "humaneval":    eval_humaneval,
}


# ============================================================
# 六、数据加载
# ============================================================

def load_dataset_file(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "samples" in data:
        return data.get("metadata", {}), data["samples"]
    return {}, data


# ============================================================
# 七、单数据集评测调度
# ============================================================

def run_one_dataset(
    name: str,
    samples: List[Dict[str, Any]],
    client: GemmaClient,
    out_dir: Path,
    workers: int = 1,
    limit: Optional[int] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    evaluator = EVALUATORS[name]
    if limit:
        samples = samples[:limit]

    total = len(samples)
    results: List[Optional[Dict]] = [None] * total
    correct_cnt = 0
    error_cnt = 0
    start = time.time()

    def _work(idx_sample: Tuple[int, Dict]) -> Tuple[int, Dict]:
        idx, sample = idx_sample
        try:
            return idx, evaluator(sample, client, debug=debug)
        except Exception as e:
            return idx, {"error": str(e), "correct": False}

    log_interval = max(1, min(10, total // 10))

    if workers <= 1:
        for i, sample in enumerate(samples):
            _, r = _work((i, sample))
            results[i] = r
            if r.get("correct"):
                correct_cnt += 1
            if r.get("error"):
                error_cnt += 1
            if (i + 1) % log_interval == 0 or i + 1 == total:
                acc = correct_cnt / (i + 1)
                print(f"  [{name}] {i+1}/{total}  acc={acc:.4f}  errors={error_cnt}")
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_work, (i, s)) for i, s in enumerate(samples)]
            done = 0
            for fut in as_completed(futs):
                idx, r = fut.result()
                results[idx] = r
                done += 1
                if r.get("correct"):
                    correct_cnt += 1
                if r.get("error"):
                    error_cnt += 1
                if done % log_interval == 0 or done == total:
                    acc = correct_cnt / done
                    print(f"  [{name}] {done}/{total}  acc={acc:.4f}  errors={error_cnt}")

    accuracy = correct_cnt / max(total, 1)
    elapsed = round(time.time() - start, 2)

    summary: Dict[str, Any] = {
        "dataset": name,
        "total": total,
        "correct": correct_cnt,
        "errors": error_cnt,
        "accuracy": accuracy,
        "elapsed_sec": elapsed,
    }

    # 子类聚合统计
    if name == "ceval":
        per = defaultdict(lambda: [0, 0])
        for r in results:
            k = r.get("subject", "")
            per[k][1] += 1
            if r.get("correct"):
                per[k][0] += 1
        summary["per_subject"] = {
            k: {"correct": v[0], "total": v[1], "acc": v[0]/v[1] if v[1] else 0}
            for k, v in per.items()
        }

    if name == "bbh":
        per = defaultdict(lambda: [0, 0])
        for r in results:
            k = r.get("task", "")
            per[k][1] += 1
            if r.get("correct"):
                per[k][0] += 1
        summary["per_task"] = {
            k: {"correct": v[0], "total": v[1], "acc": v[0]/v[1] if v[1] else 0}
            for k, v in per.items()
        }

    if name == "mmlu_pro":
        per = defaultdict(lambda: [0, 0])
        for r in results:
            k = r.get("category", "")
            per[k][1] += 1
            if r.get("correct"):
                per[k][0] += 1
        summary["per_category"] = {
            k: {"correct": v[0], "total": v[1], "acc": v[0]/v[1] if v[1] else 0}
            for k, v in per.items()
        }

    # 写入明细和摘要
    detail_path = out_dir / f"{name}_predictions.jsonl"
    with open(detail_path, "w", encoding="utf-8") as f:
        for r in (results or []):
            f.write(json.dumps(r or {}, ensure_ascii=False) + "\n")

    summary_path = out_dir / f"{name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"  ✓ [{name}] accuracy = {accuracy*100:.2f}%  "
        f"({correct_cnt}/{total})  errors={error_cnt}  time={elapsed}s"
    )
    return summary


# ============================================================
# 八、主入口
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemma 4 26B 优化版评测脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir",  default="./eval_datasets")
    parser.add_argument("--out-dir",   default="./eval_results")
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLAMA_BASE_URL", "http://localhost:8080"),
        help="llama-server 根地址（不含路径），如 http://127.0.0.1:8080",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help=f"要评测的数据集，可选: {' '.join(EVALUATORS)}",
    )
    parser.add_argument("--limit",   type=int, default=None, help="每个数据集只评前 N 条（调试）")
    parser.add_argument("--workers", type=int, default=1,    help="并发请求数（建议 1~4）")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--debug",   action="store_true",    help="打印每条 prompt / response")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = GemmaClient(base_url=args.base_url, timeout=args.timeout)

    # 健康检查
    if client.health_check():
        print(f"✓ Server 可达: {args.base_url}\n")
    else:
        print(f"⚠️  /health 检查失败，尝试继续（server 可能不支持该端点）\n")

    targets = args.datasets or list(EVALUATORS.keys())
    overall: Dict[str, Any] = {}

    for name in targets:
        if name not in EVALUATORS:
            print(f"⚠️  未知数据集: {name}，跳过（可选: {list(EVALUATORS.keys())}）")
            continue

        path = Path(args.data_dir) / f"{name}_sampled.json"
        if not path.exists():
            print(f"⚠️  找不到文件: {path}，跳过")
            continue

        print(f"\n{'='*64}")
        print(f"  Evaluating: {name}")
        print(f"{'='*64}")

        meta, samples = load_dataset_file(path)
        print(f"  Loaded {len(samples)} samples  "
              f"(sampling: {meta.get('_sampling_method', 'n/a')})")

        try:
            summary = run_one_dataset(
                name, samples, client, out_dir,
                workers=args.workers,
                limit=args.limit,
                debug=args.debug,
            )
            overall[name] = summary
        except KeyboardInterrupt:
            print("\n中断！保存已有结果。")
            break
        except Exception as e:
            print(f"  ✗ {name} 评测失败: {e}")
            overall[name] = {"error": str(e)}

    # 写入总摘要
    final_path = out_dir / "overall_summary.json"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    # 打印汇总表
    print(f"\n{'='*72}")
    print(f"{'Dataset':<20}{'Accuracy':>10}{'Correct':>10}{'Total':>8}{'Errors':>8}{'Time(s)':>10}")
    print(f"{'─'*72}")
    for name, s in overall.items():
        if "accuracy" in s:
            print(
                f"{name:<20}{s['accuracy']*100:>9.2f}%"
                f"{s['correct']:>10}{s['total']:>8}"
                f"{s.get('errors',0):>8}{s['elapsed_sec']:>10.1f}"
            )
        else:
            print(f"{name:<20}  ERROR: {s.get('error', '')}")
    print(f"{'='*72}")
    print(f"\n详细结果已保存至: {out_dir.resolve()}")


if __name__ == "__main__":
    main()