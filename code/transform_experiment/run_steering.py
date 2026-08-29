#!/usr/bin/env python
"""
Transform experiment — MAIN (steer + generate). Gemma-2-9B-IT.
For each transformed pool: build its HR-BC direction (L20/pos-2), steer the shared
HR-50 test set at alpha=100 (paper-canonical saturation), and write generations.

RECOVERABLE / CONTINUABLE:
 - processes one pool at a time; writes results/<pool>_steered.json as soon as that pool finishes
 - skips a pool whose output already exists (re-run resumes where it stopped)
 - also emits a baseline (alpha=0) once, reused for all pools

Outputs (paths):
  directions:  exp1_v10/transform_experiment/directions/<pool>/{d.pt,manifest.json}
  baseline:    exp1_v10/transform_experiment/results/_baseline_unsteered.json
  per pool:    exp1_v10/transform_experiment/results/<pool>_steered.json
Each result row: {test_prompt, baseline_response, steered_response, pool, alpha}
"""
import json, os, time, gc
from pathlib import Path
from typing import List, Tuple, Callable
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]   # <repo>/code/<area>/script.py -> <repo>

EXP = REPO
HERE = REPO / "data/transform_experiment"
POOLS = HERE / "pools"
DIRS  = HERE / "directions";  DIRS.mkdir(parents=True, exist_ok=True)
RES   = REPO / "results/transform_experiment"; RES.mkdir(parents=True, exist_ok=True)
MODEL_PATH = "google/gemma-2-9b-it"
LAYER, POS, ALPHA = 20, -2, 100.0
GEN_BATCH, MAX_LEN, MAX_NEW = 8, 128, 50

POOL_ORDER = ["safety2incomplete", "safety2humanizing", "safety2unsupported",
              "incomplete2safety", "humanizing2safety", "unsupported2safety"]

TEST = json.load(open(REPO / "data/controlled_test_set/harmful_refusals_50.json"))
TEST_PROMPTS = [r["prompt"] for r in TEST]

def _ts(): return time.strftime("%H:%M:%S")
def log(*a): print(f"[{_ts()}]", *a, flush=True)

log(f"loading {MODEL_PATH} sharded across {torch.cuda.device_count()} GPU(s) ...")
# TL self-shards via from_pretrained_no_processing (the working v10 pattern).
# Do NOT pass a pre-sharded hf_model / device_map — that causes a cuda/cpu device clash in fold_value_biases.
model = HookedTransformer.from_pretrained_no_processing(
    MODEL_PATH, device="cuda", n_devices=torch.cuda.device_count(), dtype=torch.float16,
)
model.eval()
hf_tok = model.tokenizer
hf_tok.padding_side = "left"
if hf_tok.pad_token is None: hf_tok.pad_token = hf_tok.eos_token
N_LAYERS, D_MODEL = model.cfg.n_layers, model.cfg.d_model
# EOS set (mandatory multi-EOS per project rule)
_EOT = ["<end_of_turn>", "<eos>", "<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"]
EOS_IDS = set()
if hf_tok.eos_token_id is not None: EOS_IDS.add(hf_tok.eos_token_id)
for t in _EOT:
    tid = hf_tok.convert_tokens_to_ids(t)
    if tid is not None and tid >= 0 and tid != hf_tok.unk_token_id: EOS_IDS.add(tid)
EOS_T = torch.tensor(sorted(EOS_IDS), dtype=torch.long)
PAD_ID = hf_tok.pad_token_id if hf_tok.pad_token_id is not None else hf_tok.eos_token_id
log(f"loaded. n_layers={N_LAYERS} d_model={D_MODEL} EOS={sorted(EOS_IDS)}")

def clear(): gc.collect(); torch.cuda.empty_cache()
def unit(x): return x / x.norm()

def tok_chat(instrs):
    chats = [hf_tok.apply_chat_template([{"role":"user","content":s}], add_generation_prompt=True, tokenize=False) for s in instrs]
    return hf_tok(chats, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").input_ids.to(model.cfg.device)

def generate(toks, fwd_hooks=()):
    B, T = toks.shape
    all_toks = torch.zeros((B, T+MAX_NEW), dtype=torch.long, device=toks.device); all_toks[:, :T] = toks
    eos = EOS_T.to(toks.device); done = torch.zeros(B, dtype=torch.bool, device=toks.device)
    for i in range(MAX_NEW):
        with torch.inference_mode(), model.hooks(fwd_hooks=fwd_hooks):
            logits = model(all_toks[:, :T+i])
            nxt = logits[:, -1, :].argmax(dim=-1).to(all_toks.device)  # unembed may be on last shard
            nxt = torch.where(done, torch.full_like(nxt, PAD_ID), nxt)
            all_toks[:, T+i] = nxt
            done = done | torch.isin(nxt, eos)
            if done.all(): break
    return hf_tok.batch_decode(all_toks[:, T:], skip_special_tokens=True)

def gen_prompts(prompts, fwd_hooks, desc):
    out = []
    for i in tqdm(range(0, len(prompts), GEN_BATCH), desc=desc, leave=False):
        toks = tok_chat(prompts[i:i+GEN_BATCH]); out.extend(generate(toks, fwd_hooks)); del toks; clear()
    return out

def cache_resid(prompts):
    hook = utils.get_act_name("resid_pre", LAYER); chunks = []
    for i in tqdm(range(0, len(prompts), GEN_BATCH), desc="cache", leave=False):
        toks = tok_chat(prompts[i:i+GEN_BATCH])
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=lambda n: n == hook,
                                            pos_slice=POS, stop_at_layer=LAYER+1)
        chunks.append(cache[hook].squeeze(1).to("cpu", dtype=torch.float32)); del toks, cache; clear()
    return torch.cat(chunks, 0)

def build_direction(pool):
    H = json.load(open(POOLS/pool/"H.json")); B = json.load(open(POOLS/pool/"B.json"))
    d = unit(cache_resid([r["prompt"] for r in H]).mean(0) - cache_resid([r["prompt"] for r in B]).mean(0))
    od = DIRS/pool; od.mkdir(parents=True, exist_ok=True); torch.save(d, od/"d.pt")
    json.dump({"pool":pool,"layer":LAYER,"pos":POS,"n_h":len(H),"n_b":len(B)}, open(od/"manifest.json","w"), indent=2)
    return d

def steer_hooks(d, alpha):
    dd = d.to(torch.float16).cuda()
    def hk(act, hook): return act + alpha * dd.to(act.dtype).to(act.device)
    return [(utils.get_act_name("resid_pre", LAYER), hk)]

# ---- baseline (once, reused) ----
base_path = RES/"_baseline_unsteered.json"
if base_path.exists():
    base = json.load(open(base_path)); log("baseline exists, reuse")
else:
    log("generating baseline (alpha=0) ...")
    base = gen_prompts(TEST_PROMPTS, (), "baseline")
    json.dump([{"test_prompt":p,"baseline_response":r} for p,r in zip(TEST_PROMPTS, base)],
              open(base_path,"w"), ensure_ascii=False, indent=1)
    log(f"baseline -> {base_path}")
base_map = {r["test_prompt"]: r["baseline_response"] for r in json.load(open(base_path))}

# ---- per pool (recoverable, skip-existing, write-as-you-go) ----
for pool in POOL_ORDER:
    outp = RES/f"{pool}_steered.json"
    if outp.exists():
        log(f"[skip] {pool} already done -> {outp}"); continue
    log(f"=== {pool}: building direction ===")
    d = build_direction(pool)
    log(f"=== {pool}: steering HR-50 @ alpha={ALPHA} ===")
    steered = gen_prompts(TEST_PROMPTS, steer_hooks(d, ALPHA), f"steer:{pool}")
    rows = [{"test_prompt":p, "baseline_response":base_map[p], "steered_response":s, "pool":pool, "alpha":ALPHA}
            for p,s in zip(TEST_PROMPTS, steered)]
    json.dump(rows, open(outp,"w"), ensure_ascii=False, indent=1)
    log(f"    -> WROTE {outp}  ({len(rows)} rows)")

log("ALL DONE. results in " + str(RES))
