"""QLoRA supervised fine-tuning: one LoRA adapter per (arm, seed).

The two arms are trained **identically but for the target text**: same base
model, same seed, same LoRA spec, same prompts, same example order. Only the
completion differs -- the bare action (``demos``) vs the action plus reversibility
reasoning (``why``). That single difference is the experiment's independent
variable, so everything else is pinned here.

Examples are built in TRL's prompt/completion form (the dilemma rendered through
the chat template as the prompt, the target as the completion) so the loss falls
on the completion only. Heavy imports (torch, trl, peft, datasets) are deferred
into the functions that need them, keeping the CLI and pure tests light.
"""

from pathlib import Path

from . import config
from .config import ModelSpec
from .data import Dilemma, ResponsePair


def adapter_dir(spec_key: str, arm: str, seed: int) -> Path:
    """Return the on-disk location of the adapter for ``(spec_key, arm, seed)``."""
    return config.RESULTS_DIR / "adapters" / spec_key / f"{arm}-seed{seed}"


def _format_prompt(tok, spec: ModelSpec, text: str) -> str:
    """Render a dilemma as the user turn through the chat template (no principle)."""
    messages = []
    if spec.system:
        messages.append({"role": "system", "content": spec.system})
    messages.append({"role": "user", "content": text})
    return tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **spec.chat_template_kwargs,
    )


def build_examples(
    tok,
    spec: ModelSpec,
    dilemmas: list[Dilemma],
    pairs: dict[str, ResponsePair],
    arm: str,
) -> list[dict]:
    """Build prompt/completion training rows for one ``arm``.

    Args:
        tok: The base model's tokenizer (for the chat template).
        spec: The base model spec.
        dilemmas: Train-split dilemmas (order is preserved => deterministic).
        pairs: Response pairs keyed by ``dilemma_id``.
        arm: ``"demos"`` or ``"why"``.

    Returns:
        A list of ``{"prompt": ..., "completion": ...}`` rows. Identical across
        arms except for the completion text.

    Raises:
        ValueError: If a dilemma has no matching response pair.
    """
    rows: list[dict] = []
    for item in dilemmas:
        pair = pairs.get(item.id)
        if pair is None:
            raise ValueError(f"no response pair for dilemma {item.id!r}")
        rows.append(
            {
                "prompt": _format_prompt(tok, spec, item.text),
                "completion": pair.target(arm),
            }
        )
    # Truncation guard: SFTConfig.max_length silently clips, which under
    # reason-first would drop the action at the END of `why`. Warn loudly if any
    # built sequence exceeds the budget so we can raise max_seq_len rather than
    # train on a clipped target.
    lengths = [len(tok(r["prompt"] + r["completion"])["input_ids"]) for r in rows]
    over = sum(n > config.LORA.max_seq_len for n in lengths)
    if over:
        print(
            f"[sft] WARNING {arm}: {over}/{len(rows)} sequences exceed "
            f"max_seq_len={config.LORA.max_seq_len} (longest {max(lengths)} tok) -- the "
            f"action at the end of `why` may be clipped; raise max_seq_len."
        )
    return rows


def train_adapter(
    spec: ModelSpec,
    dilemmas: list[Dilemma],
    pairs: dict[str, ResponsePair],
    arm: str,
    seed: int,
    *,
    force: bool = False,
) -> Path:
    """Train and save one LoRA adapter for ``(spec, arm, seed)``.

    Resumable: if the adapter directory already exists and ``force`` is False, the
    existing adapter is returned untouched.

    Args:
        spec: Base model to fine-tune.
        dilemmas: Train-split dilemmas.
        pairs: Response pairs keyed by ``dilemma_id``.
        arm: ``"demos"`` or ``"why"``.
        seed: Random seed (pinned identically across arms).
        force: Retrain even if the adapter already exists.

    Returns:
        Path to the saved adapter directory.
    """
    import shutil
    import tempfile

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    out = adapter_dir(spec.key, arm, seed)
    if out.exists() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    # Trainer scratch (logs, intermediate state) goes to a temp dir, never inside
    # the published adapter directory.
    scratch = Path(tempfile.mkdtemp(prefix=f"sft-{spec.key}-{arm}-{seed}-"))

    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Train in bf16, not fp16. Qwen3.5 is bf16-native, and fp16 mixed precision
    # needs a GradScaler whose unscale op has no bf16 CUDA kernel on Pascal
    # ("_amp_foreach_non_finite_check_and_unscale_cuda not implemented for
    # BFloat16"). bf16 needs no loss scaler, so that path is never taken; bf16's
    # wide exponent also makes it the safer choice for a frozen-base QLoRA fit.
    model_kwargs: dict = {"attn_implementation": spec.attn_implementation, "dtype": torch.bfloat16}
    if spec.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = {"": "cuda"}
    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **model_kwargs)
    if not spec.load_in_4bit:
        model = model.to("cuda")

    peft_config = LoraConfig(
        r=config.LORA.r,
        lora_alpha=config.LORA.alpha,
        lora_dropout=config.LORA.dropout,
        target_modules=list(config.LORA.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Build the PEFT model here (rather than letting SFTTrainer wrap it) so we
    # control the trainable-param dtype. Qwen3.5 is natively bf16, so the LoRA
    # adapters land in bf16 -- but the fp16 AMP GradScaler (the right precision on
    # Pascal) cannot unscale bf16 grads ("..unscale_cuda not implemented for
    # BFloat16"). Cast the trainable LoRA params (tiny; NOT the vocab-sized
    # lm_head) to fp32: GradScaler-safe and numerically the standard QLoRA recipe.
    from peft import get_peft_model

    model = get_peft_model(model, peft_config)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    rows = build_examples(tok, spec, dilemmas, pairs, arm)
    dataset = Dataset.from_list(rows)

    sft_config = SFTConfig(
        output_dir=str(scratch),
        per_device_train_batch_size=config.LORA.batch_size,
        gradient_accumulation_steps=config.LORA.grad_accum,
        num_train_epochs=config.LORA.epochs,
        learning_rate=config.LORA.learning_rate,
        warmup_ratio=config.LORA.warmup_ratio,
        max_length=config.LORA.max_seq_len,
        completion_only_loss=True,
        logging_steps=10,
        save_strategy="no",
        optim="paged_adamw_8bit",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=seed,
        data_seed=seed,
        report_to=[],
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,  # already a PeftModel; do not re-wrap
        args=sft_config,
        train_dataset=dataset,
        processing_class=tok,
    )
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[sft] {spec.key} {arm} seed{seed}: peak GPU {peak:.2f} GB")
    trainer.model.save_pretrained(str(out))
    tok.save_pretrained(str(out))

    del trainer, model
    torch.cuda.empty_cache()
    shutil.rmtree(scratch, ignore_errors=True)
    return out
