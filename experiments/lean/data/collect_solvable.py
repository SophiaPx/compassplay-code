# Written for CompassPlay.
import argparse, json, os, time

from experiments.lean.support.prompts import get_deepseek_prover_v2_prompt, extract_proof_deepseek_v2

MODEL = os.environ.get("MODEL_PATH", "deepseek-ai/DeepSeek-Prover-V2-7B")


def user_prompt(th):
    return get_deepseek_prover_v2_prompt(header="import Mathlib\n", theorem=th)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new", type=int, default=768)
    ap.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    ap.add_argument("--n_lean", type=int, default=8)
    ap.add_argument("--gpu_mem_util", type=float, default=0.9)
    ap.add_argument("--max_model_len", type=int, default=4096)
    args = ap.parse_args()
    from transformers import set_seed
    seed = int(os.environ["COMPASSPLAY_SEED"])
    set_seed(seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    def log(*a): print(f"[shard {args.start}-{args.end}]", *a, flush=True)

    from datasets import load_dataset
    ds = load_dataset("LukeBailey181Pub/D_3k")["train"]
    idxs = list(range(args.start, args.end))
    fout = open(args.out, "a")
    t0 = time.time()
    n_solv = 0

    def write_solved(ti, th, correct):
        nonlocal n_solv
        if correct:
            fout.write(json.dumps({"idx": ti, "theorem": th, "correct": correct,
                                   "solve_rate": len(correct) / args.k}) + "\n")
            fout.flush(); n_solv += 1

    if args.backend == "vllm":
        from vllm import LLM, SamplingParams
        from experiments.lean.data.lean_verify import LeanPool
        llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem_util,
                  max_model_len=args.max_model_len, trust_remote_code=True, seed=seed)
        vtok = llm.get_tokenizer()
        prompts = [vtok.apply_chat_template([{"role": "user", "content": user_prompt(ds[ti]["theorem"])}],
                                            add_generation_prompt=True, tokenize=False) for ti in idxs]
        log(f"ready (vllm): generating {len(prompts)} prompts x n={args.k} ...")
        sp = SamplingParams(n=args.k, temperature=0.8, top_p=0.95, max_tokens=args.max_new)
        outs = llm.generate(prompts, sp)
        log(f"generation done in {time.time()-t0:.0f}s; flattening + verifying ...")
        flat = []
        for ti, o in zip(idxs, outs):
            th = ds[ti]["theorem"]
            for comp in o.outputs:
                p = extract_proof_deepseek_v2(comp.text)
                if "NO_CODE_FOUND" in p:
                    continue
                flat.append((ti, th, p))
        pool = LeanPool(n_workers=args.n_lean)
        res = pool.verify_batch([th.rstrip() + "\n" + p for _, th, p in flat], timeout=100)
        correct = {ti: [] for ti in idxs}
        for (ti, _, p), r in zip(flat, res):
            if r.ok:
                correct[ti].append(p)
        for ti in idxs:
            write_solved(ti, ds[ti]["theorem"], correct[ti])
        pool.close()
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from experiments.lean.data.lean_verify import LeanREPL
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
        model.eval()
        repl = LeanREPL()
        log("ready (hf)")
        for ti in idxs:
            th = ds[ti]["theorem"]
            inp = tok.apply_chat_template([{"role": "user", "content": user_prompt(th)}],
                                          add_generation_prompt=True, return_tensors="pt").to("cuda:0")
            try:
                with torch.no_grad():
                    out = model.generate(inp, max_new_tokens=args.max_new, do_sample=True, temperature=0.8,
                                         top_p=0.95, num_return_sequences=args.k, pad_token_id=tok.eos_token_id)
                texts = tok.batch_decode(out[:, inp.shape[1]:], skip_special_tokens=True)
            except Exception as e:
                log(f"thm{ti} gen err {repr(e)[:60]}"); continue
            correct = []
            for g in texts:
                p = extract_proof_deepseek_v2(g)
                if "NO_CODE_FOUND" in p:
                    continue
                if repl.verify(th.rstrip() + "\n" + p, timeout=100).ok:
                    correct.append(p)
            write_solved(ti, th, correct)
            if ti % 5 == 0:
                log(f"thm{ti} solv={n_solv} elapsed={time.time()-t0:.0f}s")
        repl.close()

    fout.close()
    log(f"DONE {n_solv} solvable in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
