"""GPU-side model runner: base (optionally 4-bit) + LoRA adapter, blind generation.

This is the only inference module that holds weights. It loads a base model in
fp16 (or 4-bit NF4 for QLoRA), optionally attaches a trained LoRA adapter, and
greedily generates a recommendation for a held-out dilemma. The native ``<think>``
scratchpad is disabled via ``spec.chat_template_kwargs`` so the only reasoning a
response shows is what the adapter learned, and no system message ever carries
the reversibility principle -- the probe must be blind.
"""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import PROBE_MAX_NEW_TOKENS, ModelSpec


def _quant_config():
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


class ModelRunner:
    """Loaded base (+ optional adapter) with a blind generation helper."""

    def __init__(self, spec: ModelSpec, tokenizer, model, device: str, adapter: str | None):
        """Wrap a loaded tokenizer/model pair (use :meth:`load` instead)."""
        self.spec = spec
        self.tok = tokenizer
        self.model = model
        self.device = device
        self.adapter = adapter

    @classmethod
    def load(
        cls,
        spec: ModelSpec,
        adapter_dir: str | Path | None = None,
        device: str = "cuda",
    ) -> "ModelRunner":
        """Load ``spec`` (4-bit if configured) and optionally attach an adapter.

        Args:
            spec: The base model to load.
            adapter_dir: Optional path to a trained LoRA adapter to attach.
            device: Torch device string.

        Returns:
            A ready :class:`ModelRunner`.
        """
        tok = AutoTokenizer.from_pretrained(spec.hf_id)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        kwargs: dict = {
            "attn_implementation": spec.attn_implementation,
            "dtype": torch.float16,
        }
        if spec.load_in_4bit:
            kwargs["quantization_config"] = _quant_config()
            kwargs["device_map"] = {"": device}
            model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **kwargs).to(device)

        adapter_name: str | None = None
        if adapter_dir is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_dir))
            adapter_name = str(adapter_dir)
        model.eval()
        return cls(spec, tok, model, device, adapter_name)

    def format(self, prompt: str) -> str:
        """Render a user prompt through the chat template (no principle, no <think>).

        Honors ``spec.chat_template_kwargs`` (e.g. ``enable_thinking=False``) and a
        minimal ``spec.system`` used solely to disable thinking traces.
        """
        messages = []
        if self.spec.system:
            messages.append({"role": "system", "content": self.spec.system})
        messages.append({"role": "user", "content": prompt})
        return self.tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.spec.chat_template_kwargs,
        )

    @torch.no_grad()
    def generate_with_meta(
        self, prompt: str, max_new_tokens: int = PROBE_MAX_NEW_TOKENS
    ) -> tuple[str, bool]:
        """Greedily generate, returning the text and whether it hit the token budget.

        ``truncated`` is True iff generation stopped on ``max_new_tokens`` rather than
        EOS. This matters under ``WHY_ORDER="reason_first"``: the scored action sits at
        the END of the answer, so a truncated response can drop the very action the
        judge reads -- the diagnostic flags when ``PROBE_MAX_NEW_TOKENS`` is too small.

        Args:
            prompt: The held-out dilemma (formatted internally; no principle added).
            max_new_tokens: Generation budget.

        Returns:
            ``(text, truncated)`` -- the decoded completion (special tokens stripped,
            whitespace-collapsed) and the budget-hit flag.
        """
        enc = self.tok(self.format(prompt), return_tensors="pt").to(self.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
        )
        new_ids = out[0, enc["input_ids"].shape[1] :]
        truncated = bool(new_ids.shape[0] >= max_new_tokens)
        text = self.tok.decode(new_ids, skip_special_tokens=True)
        return " ".join(text.split()), truncated

    def generate(self, prompt: str, max_new_tokens: int = PROBE_MAX_NEW_TOKENS) -> str:
        """Greedily generate a recommendation for ``prompt`` (text only)."""
        text, _ = self.generate_with_meta(prompt, max_new_tokens)
        return text

    def unload(self) -> None:
        """Free GPU memory held by this runner."""
        del self.model
        torch.cuda.empty_cache()
