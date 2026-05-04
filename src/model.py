import torch
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
import copy

# Use the lmms-lab checkpoint (LLaVA-NeXT format, loaded via llava package).
_DEFAULT_MODEL_ID = "lmms-lab/llava-onevision-qwen2-7b-ov"
_CONV_TEMPLATE = "qwen_1_5"


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LLaVAOneVision:
    """
    Thin wrapper around LLaVA-OneVision-7B for multi-image VQA inference.

    Loading strategy (MPS / CPU):
      load_pretrained_model is called with device_map="cpu" to avoid the
      hardcoded vision_tower.to(device="cuda") call in llava/model/builder.py.
      The model is then moved to the best available device (MPS or CPU) after
      loading. On CUDA, device_map="auto" is used for tensor-parallel placement.

    Greedy decoding (temperature=0, do_sample=False) for reproducible baselines.
    """

    def __init__(self, model_id: str = _DEFAULT_MODEL_ID):
        self.device = _best_device()

        if self.device == "cuda":
            builder_device_map = "auto"
            dtype = torch.float16
        elif self.device == "mps":
            builder_device_map = {"": "mps"}
            dtype = torch.float16
        else:
            builder_device_map = "cpu"
            dtype = torch.float32

        print(f"Loading {model_id} (target device={self.device}, dtype={dtype}) ...")
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_id,
            model_base=None,
            model_name="llava_qwen",
            device_map=builder_device_map,
            attn_implementation="eager",
        )

        if self.device not in ("cuda", "mps"):
            self.model = self.model.to(dtype=dtype)

        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        images: list[Image.Image],
        prompt: str,
        max_new_tokens: int = 256,
    ) -> str:
        """
        Run greedy inference for one sample.

        Args:
            images:         List of PIL images (2-4 views for MindCube).
            prompt:         Task prompt text (image tokens are prepended here).
            max_new_tokens: Generation budget; 256 is plenty for A/B/C/D answers.

        Returns:
            Raw decoded model output (before answer extraction).
        """
        image_prefix = "".join(DEFAULT_IMAGE_TOKEN + "\n" for _ in images)
        full_prompt = image_prefix + prompt

        conv = copy.deepcopy(conv_templates[_CONV_TEMPLATE])
        conv.append_message(conv.roles[0], full_prompt)
        conv.append_message(conv.roles[1], None)
        formatted = conv.get_prompt()

        model_dtype = next(self.model.parameters()).dtype
        image_tensors = process_images(images, self.image_processor, self.model.config)
        image_tensors = [t.to(dtype=model_dtype, device=self.device) for t in image_tensors]
        image_sizes = [img.size for img in images]

        input_ids = (
            tokenizer_image_token(formatted, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.device)
        )

        output_ids = self.model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=max_new_tokens,
        )

        generated = output_ids[0][input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
