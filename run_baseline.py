"""
Baseline evaluation of LLaVA-OneVision-7B on MindCube.

Usage
-----
python run_baseline.py \
    --data_path  data/raw/MindCube_tinybench.jsonl \
    --image_root data/ \
    --output_path outputs/baseline_results.jsonl

For a quick smoke-test on 10 examples:
python run_baseline.py \
    --data_path  data/raw/MindCube_tinybench.jsonl \
    --image_root data/ \
    --max_samples 10 \
    --output_path outputs/smoke_test.jsonl
1
Data setup (run once)
---------------------
    pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('MLL-Lab/MindCube', repo_type='dataset', local_dir='data_download')
"
    # Then extract data_download/data.zip into the data/ directory so that:
    #   data/raw/MindCube_tinybench.jsonl
    #   data/raw/MindCube_train.jsonl
    #   data/other_all_image/...
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.dataset import MindCubeDataset
from src.evaluate import compute_metrics, extract_answer, print_metrics
from src.model import LLaVAOneVision


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLaVA-OneVision baseline on MindCube")
    p.add_argument(
        "--data_path",
        required=True,
        help="Path to a MindCube JSONL file (e.g. data/raw/MindCube_tinybench.jsonl)",
    )
    p.add_argument(
        "--image_root",
        required=True,
        help="Root directory for images (image paths in JSONL are relative to this)",
    )
    p.add_argument(
        "--output_path",
        default="outputs/baseline_results.jsonl",
        help="Where to write per-sample results (JSONL)",
    )
    p.add_argument(
        "--model_id",
        default="lmms-lab/llava-onevision-qwen2-7b-ov",
        help="HuggingFace model ID for LLaVA-OneVision",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit evaluation to the first N samples (useful for debugging)",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max tokens to generate per sample",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ dataset
    print(f"Loading dataset: {args.data_path}")
    dataset = MindCubeDataset(args.data_path, args.image_root, args.max_samples)
    print(f"  {len(dataset)} samples")

    # ------------------------------------------------------------------ model
    model = LLaVAOneVision(args.model_id)

    # ------------------------------------------------------------------ eval loop
    results: list[dict] = []

    with open(out_path, "w") as f_out:
        for idx in tqdm(range(len(dataset)), desc="Evaluating"):
            sample = dataset[idx]

            try:
                raw_output = model.generate(
                    sample["images"],
                    sample["prompt"],
                    args.max_new_tokens,
                )
                predicted = extract_answer(raw_output)
                error = None
            except Exception as exc:
                raw_output = ""
                predicted = None
                error = str(exc)
                tqdm.write(f"[WARN] sample {sample['id']} failed: {exc}")

            gt = sample["gt_answer"].upper() if sample["gt_answer"] else ""
            result = {
                "id": sample["id"],
                "setting": sample["setting"],
                "gt_answer": gt,
                "predicted": predicted,
                "correct": predicted is not None and predicted == gt,
                "raw_output": raw_output,
                **({"error": error} if error else {}),
            }
            results.append(result)
            f_out.write(json.dumps(result) + "\n")

    # ------------------------------------------------------------------ metrics
    metrics = compute_metrics(results)
    print_metrics(metrics)

    metrics_path = out_path.with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    unanswered = sum(1 for r in results if r["predicted"] is None)
    print(f"Unanswered (no letter extracted): {unanswered}/{len(results)}")
    print(f"Results  → {out_path}")
    print(f"Metrics  → {metrics_path}")


if __name__ == "__main__":
    main()
