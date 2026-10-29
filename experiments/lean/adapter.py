"""SGS proof loss, local references, and statement novelty for CompassPlay."""

import logging
import os
import re
from typing import List

import torch

from method.alignment import gradient_statistics, cosine_from_statistics
from method.rewards import alignment_reward

from experiments.lean.data.types import Conjecture
from experiments.lean.generation.config import ProverConfig
from experiments.lean.support.prompts import get_deepseek_prover_v2_prompt

logger = logging.getLogger(__name__)


def _solver_nll(model, tok, device, header: str, theorem: str, proof_body: str,
                max_len: int) -> "torch.Tensor":
    prompt = get_deepseek_prover_v2_prompt(header=header, theorem=theorem)
    pids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt",
    )[0].to(device)
    cids = tok(proof_body, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
    if cids.numel() > max_len:
        cids = cids[:max_len]
    ids = torch.cat([pids, cids]).unsqueeze(0)
    labels = ids.clone()
    labels[:, : pids.shape[0]] = -100
    return model(ids, labels=labels).loss


def _backward_grad(model, tok, device, header, theorem, proofs, max_len):
    model.zero_grad(set_to_none=True)
    for p in proofs:
        (_solver_nll(model, tok, device, header, theorem, p, max_len) / len(proofs)).backward()


def _stmt_tokens(s: str) -> set:
    return set(re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", s or ""))


def _novelty_to_seed(conjecture: str, seed_theorem: str) -> float:
    a, b = _stmt_tokens(conjecture), _stmt_tokens(seed_theorem)
    if not a and not b:
        return 1.0
    return 1.0 - len(a & b) / max(1, len(a | b))


def compute_gradient_align_rewards(
    conjectures: List[Conjecture],
    prover_config: ProverConfig,
    device: str,
    wandb_run=None,
    iteration: int = 0,
) -> List[float]:
    if not conjectures:
        return []

    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert prover_config is not None, "prover_config is required for the gradient-alignment reward"

    max_proofs = int(os.environ.get("SGS_GRAD_MAX_PROOFS", "4"))
    max_len = int(os.environ.get("SGS_GRAD_MAX_LEN", "2048"))
    novelty_mode = os.environ.get("SGS_NOVELTY", "jaccard")
    assert novelty_mode in ("jaccard", "none"), f"SGS_NOVELTY must be jaccard or none, got {novelty_mode}"

    logger.info(
        f"[grad-align] loading solver {prover_config.model_name} on {device} "
        f"for {len(conjectures)} conjectures (max_proofs={max_proofs}, max_len={max_len}, "
        f"novelty={novelty_mode})"
    )
    model = AutoModelForCausalLM.from_pretrained(
        prover_config.model_name, torch_dtype=torch.bfloat16
    ).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    tok = AutoTokenizer.from_pretrained(prover_config.model_name)

    rewards: List[float] = []
    novs: List[float] = []
    n_valid = 0
    for c in conjectures:
        correct = [p.proof_str for p in c.proofs if p.is_correct][:max_proofs]
        if not correct or c.seed_proof is None:
            rewards.append(0.0)
            continue

        _backward_grad(model, tok, device, c.header, c.seed_theorem,
                       [c.seed_proof.proof_str], max_len)
        gref = [
            p.grad.detach().to(torch.bfloat16).clone() if p.grad is not None else None
            for p in model.parameters()
        ]

        _backward_grad(model, tok, device, c.header, c.conjecture, correct, max_len)

        pairs = ((p.grad, gr) for p, gr in zip(model.parameters(), gref)
                 if p.grad is not None and gr is not None)
        cos = cosine_from_statistics(gradient_statistics(pairs)).item()
        if novelty_mode == "jaccard":
            nov = _novelty_to_seed(c.conjecture, c.seed_theorem)
            novs.append(nov)
            rewards.append(alignment_reward(cos, nov))
        else:
            rewards.append(alignment_reward(cos))
        n_valid += 1
        del gref

    model.zero_grad(set_to_none=True)
    del model, tok
    torch.cuda.empty_cache()

    vals = torch.tensor(rewards, dtype=torch.float64)
    nov_mean = sum(novs) / len(novs) if novs else 1.0
    logger.info(
        f"[grad-align] iter {iteration}: {n_valid}/{len(conjectures)} rewarded | "
        f"novelty={novelty_mode} nov_mean={nov_mean:.4f} | "
        f"reward mean={vals.mean():.4f} std={vals.std(unbiased=False):.4f} "
        f"min={vals.min():.4f} max={vals.max():.4f}"
    )
    if wandb_run is not None:
        wandb_run.log({
            "conjectures/grad_align_reward_mean": float(vals.mean()),
            "conjectures/grad_align_reward_std": float(vals.std(unbiased=False)),
            "conjectures/grad_align_novelty_mean": nov_mean,
            "conjectures/grad_align_n_valid": n_valid,
            "conjectures/grad_align_n_total": len(conjectures),
            "iteration": iteration,
        })
    return rewards
