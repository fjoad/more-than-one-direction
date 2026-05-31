# run_neel_ablate_llama3_11dirs.py (or single Jupyter cell)
# Neel-style refusal ablation on Llama-3-8B-Instruct across 11 splits
#
# - Model: meta-llama/Meta-Llama-3-8B-Instruct (TransformerLens)
# - Direction: HR–BC, resid_pre, layer=12, pos=-5
# - Train dir on data/train_subset64_working.json (32 HR, 32 BC)
# - Evaluate on data/refusal_13splits/*.json
#       EXCLUDING CocoNot_all.json and SorryBench_all.json
# - For each split, generate ablated responses for all prompts and save:
#       generations/neel_ablate_llama3_11dirs/<split_name>/neel_ablate.json

import json, gc, random, functools
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import einops
from tqdm import tqdm
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
SEED = 123
random.seed(SEED)
torch.manual_seed(SEED)

MODEL_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32

# Direction training file (64 prompts: 32 harmful, 32 unharmful)
NEEL_TRAIN_JSON = Path("data/train_subset64_working.json")

# Refusal splits
SPLITS_DIR    = Path("data") / "refusal_13splits"
EXCLUDE_FILES = {"CocoNot_all.json", "SorryBench_all.json"}

# Output root
OUT_ROOT = Path("generations") / "neel_ablate_llama3_11dirs_L12pos-5"

# Direction location (Llama-3)
NEEL_LAYER = 12     # resid_pre layer index
NEEL_POS   = -5     # position index (near EOT region)

MAX_LEN   = 256
MAX_NEW   = 64
GEN_BATCH = 8

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm().clamp_min(1e-12)

def load_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), f"{path} must contain a list."
    return data

# Llama-3 chat template (Meta docs style)
LLAMA3_CHAT_TEMPLATE = """<|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

def tokenize_instructions_llama3_chat(
    instructions: List[str],
    hf_tok: AutoTokenizer,
    max_length: int = MAX_LEN,
) -> torch.Tensor:
    prompts = [
        LLAMA3_CHAT_TEMPLATE.format(instruction=inst)
        for inst in instructions
    ]
    toks = hf_tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).input_ids.to(model.cfg.device)
    return toks

def cache_resid_pre_layer_pos(
    prompts: List[str],
    layer: int,
    pos_slice: int,
    hf_tok: AutoTokenizer,
    desc: str = "cache",
) -> torch.Tensor:
    """
    Cache resid_pre at a single (layer, pos_slice) for prompts.
    Returns [N, d_model] on CPU float32.
    """
    hook_name = utils.get_act_name("resid_pre", layer)
    chunks: List[torch.Tensor] = []

    if not prompts:
        return torch.empty(0, model.cfg.d_model, dtype=torch.float32)

    for i in tqdm(range(0, len(prompts), GEN_BATCH), desc=f"Cache {desc}", unit="batch"):
        batch_prompts = prompts[i : i + GEN_BATCH]
        toks = tokenize_instructions_llama3_chat(batch_prompts, hf_tok, max_length=MAX_LEN)
        with torch.no_grad():
            # no steering hooks while building the direction
            with model.hooks(fwd_hooks=[]):
                _, cache = model.run_with_cache(
                    toks,
                    names_filter=lambda n: n == hook_name,
                    pos_slice=pos_slice,
                    stop_at_layer=layer + 1,
                )
        acts = cache[hook_name].squeeze(1).to("cpu", dtype=torch.float32).contiguous()
        chunks.append(acts)
        del cache, toks, acts
        clear_cuda()

    return torch.cat(chunks, dim=0)

def _generate_with_hooks(
    toks: torch.Tensor,
    max_tokens_generated: int,
    hf_tok: AutoTokenizer,
    fwd_hooks: List[Tuple[str, callable]],
) -> List[str]:
    """
    Manual greedy decode with TransformerLens hooks.
    """
    B, T = toks.shape
    all_toks = torch.zeros(
        (B, T + max_tokens_generated),
        dtype=torch.long,
        device=toks.device,
    )
    all_toks[:, :T] = toks

    for i in range(max_tokens_generated):
        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks):
                logits = model(all_toks[:, : T + i])
                next_tokens = logits[:, -1, :].argmax(dim=-1)
                all_toks[:, T + i] = next_tokens

    gen_suffix = all_toks[:, T:]
    out = hf_tok.batch_decode(gen_suffix, skip_special_tokens=True)
    return out

def generate_for_records(
    records: List[Dict[str, Any]],
    hf_tok: AutoTokenizer,
    fwd_hooks: List[Tuple[str, callable]],
    desc: str,
) -> List[str]:
    prompts = [r["prompt"] for r in records]
    generations: List[str] = []
    for i in tqdm(range(0, len(prompts), GEN_BATCH), desc=desc, unit="batch"):
        batch_prompts = prompts[i : i + GEN_BATCH]
        toks = tokenize_instructions_llama3_chat(batch_prompts, hf_tok, max_length=MAX_LEN)
        outs = _generate_with_hooks(
            toks,
            max_tokens_generated=MAX_NEW,
            hf_tok=hf_tok,
            fwd_hooks=fwd_hooks,
        )
        generations.extend(outs)
        del toks, outs
        clear_cuda()
    return generations

# ─────────────────────────────────────────────
# Neel-style ablation hooks (residual projection out)
# ─────────────────────────────────────────────
def direction_ablation_hook(
    activation: torch.Tensor,
    hook: HookPoint,
    direction: torch.Tensor,
):
    d = direction.to(activation.device, dtype=activation.dtype)
    proj = einops.einsum(
        activation,
        d.view(-1, 1),
        "... d_act, d_act single -> ... single",
    ) * d
    return activation - proj

def hooks_ablation_all_layers(direction: torch.Tensor) -> List[Tuple[str, callable]]:
    layers = list(range(model.cfg.n_layers))
    act_points = ["resid_pre", "resid_mid", "resid_post"]
    hook_fn = functools.partial(direction_ablation_hook, direction=direction)

    hooks: List[Tuple[str, callable]] = []
    for l in layers:
        for act_name in act_points:
            hook_name = utils.get_act_name(act_name, l)
            hooks.append((hook_name, hook_fn))
    return hooks

# ─────────────────────────────────────────────
# Load model & tokenizer
# ─────────────────────────────────────────────
print("=== Loading Meta-Llama-3-8B-Instruct model ===")
model = HookedTransformer.from_pretrained_no_processing(
    MODEL_PATH,
    device=DEVICE,
    n_devices=torch.cuda.device_count() if DEVICE == "cuda" else 1,
    dtype=DTYPE,
)
model.eval()

hf_tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
hf_tok.padding_side = "left"
if hf_tok.pad_token is None:
    hf_tok.pad_token = hf_tok.eos_token
hf_tok.truncation_side = "left"

print("Model & tokenizer loaded.")
clear_cuda()

# ─────────────────────────────────────────────
# 1) Build Neel HR–BC direction (L12, pos=-5)
# ─────────────────────────────────────────────
print("\n=== Building Neel HR–BC direction (Llama3, L12, pos=-5) ===")
neel_recs = load_json_list(NEEL_TRAIN_JSON)
if len(neel_recs) < 64:
    raise RuntimeError(f"{NEEL_TRAIN_JSON} expected at least 64 rows.")

hr32_prompts = [r["prompt"] for r in neel_recs[:32]]
bc32_prompts = [r["prompt"] for r in neel_recs[32:64]]

acts_hr32 = cache_resid_pre_layer_pos(
    hr32_prompts, layer=NEEL_LAYER, pos_slice=NEEL_POS, hf_tok=hf_tok, desc="Neel HR32"
)
acts_bc32 = cache_resid_pre_layer_pos(
    bc32_prompts, layer=NEEL_LAYER, pos_slice=NEEL_POS, hf_tok=hf_tok, desc="Neel BC32"
)

dir_neel = unit(acts_hr32.mean(dim=0) - acts_bc32.mean(dim=0)).to(
    model.cfg.device, dtype=model.cfg.dtype
)

print("\n=== Sanity check: HR vs BC projections at chosen layer/pos ===")
d_vec = dir_neel.to(acts_hr32.device, dtype=acts_hr32.dtype)
proj_hr = (acts_hr32 @ d_vec).detach().cpu()
proj_bc = (acts_bc32 @ d_vec).detach().cpu()
print(f"HR projections: mean={proj_hr.mean().item():.4f}, std={proj_hr.std().item():.4f}")
print(f"BC projections: mean={proj_bc.mean().item():.4f}, std={proj_bc.std().item():.4f}")
print(f"Separation (HR_mean - BC_mean) = {(proj_hr.mean()-proj_bc.mean()).item():.4f}")

print("\n=== Sanity check: ablation actually removes projection on one HR example ===")
hook_name = utils.get_act_name("resid_pre", NEEL_LAYER)
one_prompt = [hr32_prompts[0]]
toks_single = tokenize_instructions_llama3_chat(one_prompt, hf_tok, max_length=MAX_LEN)

with torch.no_grad():
    with model.hooks(fwd_hooks=[]):
        _, cache_no = model.run_with_cache(
            toks_single,
            names_filter=lambda n: n == hook_name,
            pos_slice=NEEL_POS,
            stop_at_layer=NEEL_LAYER + 1,
        )
acts_no = cache_no[hook_name].squeeze(0).squeeze(0)
proj_before = torch.dot(
    acts_no.to(d_vec.device, dtype=d_vec.dtype),
    d_vec,
).item()

hooks_neel_ablate = hooks_ablation_all_layers(dir_neel)
with torch.no_grad():
    with model.hooks(fwd_hooks=hooks_neel_ablate):
        _, cache_yes = model.run_with_cache(
            toks_single,
            names_filter=lambda n: n == hook_name,
            pos_slice=NEEL_POS,
            stop_at_layer=NEEL_LAYER + 1,
        )
acts_yes = cache_yes[hook_name].squeeze(0).squeeze(0)
proj_after = torch.dot(
    acts_yes.to(d_vec.device, dtype=d_vec.dtype),
    d_vec,
).item()

print(f"projection BEFORE ablate: {proj_before:.6f}")
print(f"projection AFTER  ablate: {proj_after:.6f}")

del acts_hr32, acts_bc32, cache_no, cache_yes, toks_single
clear_cuda()
print("Neel HR–BC direction built.")

# ─────────────────────────────────────────────
# 2) Run Neel-style ablation on 11 splits
# ─────────────────────────────────────────────
print("\n=== Running Neel-style ablation on splits (excluding CocoNot_all & SorryBench_all) ===")
split_files_all = sorted(SPLITS_DIR.glob("*.json"))
split_files = [p for p in split_files_all if p.name not in EXCLUDE_FILES]

print("Using splits:")
for p in split_files:
    print(f"  - {p.name}")

OUT_ROOT.mkdir(parents=True, exist_ok=True)

for split_path in split_files:
    split_name = split_path.stem
    print(f"\n--- Split: {split_name} ---")
    records = load_json_list(split_path)
    print(f"  {len(records)} prompts in this split.")

    split_out_dir = OUT_ROOT / split_name
    split_out_dir.mkdir(parents=True, exist_ok=True)

    out_path = split_out_dir / "neel_ablate.json"
    if out_path.exists():
        print(f"  [skip] {out_path} already exists")
        continue

    gens_ablate = generate_for_records(
        records,
        hf_tok=hf_tok,
        fwd_hooks=hooks_neel_ablate,
        desc=f"{split_name} – Neel ABLATE (Llama3)",
    )

    # Save in the same format as Qwen / Gemma scripts
    assert len(records) == len(gens_ablate)
    out_recs: List[Dict[str, Any]] = []
    for rec, gen in zip(records, gens_ablate):
        out_recs.append(
            {
                "prompt": rec["prompt"],
                "prompt_harm_label": rec["prompt_harm_label"],
                "was_refusal": int(rec["was_refusal"]),
                "model_response": gen,
            }
        )

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_recs, f, ensure_ascii=False, indent=2)

    del gens_ablate, out_recs, records
    clear_cuda()
    print(f"  ✓ Finished Neel ABLATE for split {split_name}. Outputs in {split_out_dir}")

print("\n✓ All 11 splits done. Results written under:")
print(f"  {OUT_ROOT.resolve()}")
