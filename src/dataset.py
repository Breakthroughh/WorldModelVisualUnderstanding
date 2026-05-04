import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

_TASK_HEADER = (
    "[Task]\n"
    "Your task is to analyze the spatial arrangement of objects in the scene by "
    "examining the provided images, which show the scene from different viewpoints.\n\n"
    "[Answer Instruction]\n"
    "Provide ONE correct answer by selecting from the options in the question. "
    "Wrap your answer in <answer> tags, e.g., <answer>A</answer>.\n\n"
    "[Question]\n"
)

# Settings present in MindCube
_KNOWN_SETTINGS = {"around", "among", "rotation", "translation"}


class MindCubeDataset(Dataset):
    """
    Loads MindCube from a raw JSONL file.

    Confirmed fields (from MindCube_tinybench.jsonl / MindCube_train.jsonl):
        id        (str)       – e.g. "among_group693_q1_5_2"; prefix encodes setting
        question  (str)       – full question text including A/B/C/D options
        gt_answer (str)       – ground-truth letter, e.g. "C"
        images    (list[str]) – paths relative to image_root, e.g.
                                "other_all_image/among/shoe_216/front_007.jpg"
        category  (list[str]) – fine-grained tags (not used for eval)
        type      (str)       – frame count type, e.g. "1_frame"
        meta_info (list)      – object/relation metadata (not used for eval)
    """

    def __init__(self, jsonl_path: str, image_root: str, max_samples: int | None = None):
        self.image_root = Path(image_root)
        self.samples: list[dict] = []

        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
                    if max_samples and len(self.samples) >= max_samples:
                        break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        record = self.samples[idx]
        return {
            "id": record["id"],
            "images": self._load_images(record["images"]),
            "prompt": _TASK_HEADER + record["question"],
            "gt_answer": record["gt_answer"],
            "setting": self._setting_from_id(record["id"]),
        }

    def _load_images(self, rel_paths: list[str]) -> list[Image.Image]:
        images = []
        for rel in rel_paths:
            path = self.image_root / rel
            if not path.exists():
                raise FileNotFoundError(
                    f"Image not found: {path}\n"
                    f"Check that --image_root points to the MindCube data/ directory "
                    f"(should contain other_all_image/)."
                )
            images.append(Image.open(path).convert("RGB"))
        return images

    @staticmethod
    def _setting_from_id(sample_id: str) -> str:
        """Extract setting from the id prefix, e.g. 'among_group693_q1' → 'among'."""
        prefix = sample_id.split("_")[0].lower()
        return prefix if prefix in _KNOWN_SETTINGS else "unknown"
