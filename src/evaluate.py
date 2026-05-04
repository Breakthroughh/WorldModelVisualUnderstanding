import re
from collections import defaultdict

# Priority 1: explicit <answer> tag  →  <answer>A</answer>  or  <answer>A. text</answer>
_TAG_RE = re.compile(r"<answer>\s*([A-E])", re.IGNORECASE)

# Priority 2: explicit declaration    →  "The answer is A" / "Answer: B."
_DECL_RE = re.compile(
    r"(?:the\s+answer\s+is|my\s+answer\s+is|answer\s*:)\s*([A-E])\.?",
    re.IGNORECASE,
)

# Priority 3: letter at the start of a line or end of text (last match wins)
_LINE_RE = re.compile(r"^\s*([A-E])[\.\):]?\s*$", re.IGNORECASE | re.MULTILINE)

# Priority 4: any A-E letter anywhere (catches "!D", "! C", "!D!" from Qwen2)
_ANY_RE = re.compile(r"([A-E])", re.IGNORECASE)


def extract_answer(text: str) -> str | None:
    """
    Extract a single A-E letter from model output.
    Returns the uppercase letter, or None if nothing is found.
    """
    m = _TAG_RE.search(text)
    if m:
        return m.group(1).upper()

    m = _DECL_RE.search(text)
    if m:
        return m.group(1).upper()

    matches = _LINE_RE.findall(text)
    if matches:
        return matches[-1].upper()

    # Fallback: any A-E letter in the output (handles short outputs like "!D")
    matches = _ANY_RE.findall(text)
    if matches:
        return matches[-1].upper()

    return None


def compute_metrics(results: list[dict]) -> dict:
    """
    Compute overall accuracy and per-setting accuracy from a list of result dicts.

    Each dict must have: predicted (str|None), gt_answer (str), setting (str).
    """
    overall = {"correct": 0, "total": 0}
    by_setting: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in results:
        pred = r["predicted"]
        gt = r["gt_answer"].upper() if r["gt_answer"] else ""
        is_correct = pred is not None and pred == gt

        overall["total"] += 1
        overall["correct"] += int(is_correct)

        s = r.get("setting", "unknown")
        by_setting[s]["total"] += 1
        by_setting[s]["correct"] += int(is_correct)

    def _acc(d: dict) -> float:
        return d["correct"] / d["total"] if d["total"] else 0.0

    return {
        "overall": {"accuracy": _acc(overall), **overall},
        "by_setting": {
            s: {"accuracy": _acc(v), **v}
            for s, v in sorted(by_setting.items())
        },
    }


def print_metrics(metrics: dict) -> None:
    ov = metrics["overall"]
    print(f"\n{'='*40}")
    print(f"  Overall accuracy : {ov['accuracy']:.3f}  ({ov['correct']}/{ov['total']})")
    print(f"{'='*40}")
    print(f"  {'Setting':<18}  {'Acc':>6}  Correct/Total")
    print(f"  {'-'*38}")
    for setting, m in metrics["by_setting"].items():
        print(f"  {setting:<18}  {m['accuracy']:>6.3f}  {m['correct']}/{m['total']}")
    print(f"{'='*40}\n")
