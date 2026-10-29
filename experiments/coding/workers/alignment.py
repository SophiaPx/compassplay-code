# CompassPlay integration with AZR and verl.
import os

import torch
import torch.distributed as dist

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import tensordict_utils as tu
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def rowmean_ce_loss(model_output, data, dp_group=None):
    log_probs = model_output["log_probs"]
    log_probs = no_padding_2_padding(log_probs, data)
    mask = data["response_mask"].to(log_probs.dtype)
    per_row = -(log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
    return per_row.sum(), {}


def _local(t):
    return t.to_local() if hasattr(t, "to_local") else t


from method.alignment import dot as _dot, squared_norm as _self_dot, cosine_from_statistics
from method.rewards import alignment_reward


def _flat_layout(params):
    offs, tot = [], 0
    for p in params:
        n = _local(p).numel()
        offs.append((tot, n))
        tot += n
    return offs, tot


def _pack_grads_into(params, layout, out):
    for p, (o, n) in zip(params, layout):
        g = p.grad
        if g is None:
            out[o : o + n].zero_()
        else:
            out[o : o + n].copy_(_local(g).detach().reshape(-1))
    return out


def _token_chunks(lo, hi, row_lens, budget):
    if budget <= 1:
        return [(i, i + 1) for i in range(lo, hi)]
    out, start, acc = [], lo, 0
    for i in range(lo, hi):
        li = int(row_lens[i])
        if i > start and acc + li > budget:
            out.append((start, i))
            start, acc = i, 0
        acc += li
    if start < hi:
        out.append((start, hi))
    return out


_REF_MICRO_TOK = int(os.environ.get("COMPASSPLAY_REF_MICRO_TOK", "32768"))


def _rank0(*a):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*a, flush=True)


class AlignmentWorker(ActorRolloutRefWorker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from experiments.coding.workers.rocm import apply_and_log

        apply_and_log()

        self._ga_cache_key = None
        self._ga_cache_val = None
        self._ga_gs_buf = None

    @staticmethod
    def _weight_version(engine):
        if not hasattr(engine, "_compassplay_weight_version"):
            engine._compassplay_weight_version = 0
            _orig = engine.optimizer_step

            def _counted(*a, **kw):
                engine._compassplay_weight_version += 1
                return _orig(*a, **kw)

            engine.optimizer_step = _counted
        return engine._compassplay_weight_version

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def compute_gradient_alignment(self, data: DataProto) -> DataProto:
        engine = self.actor.engine
        sp_size = getattr(engine, "ulysses_sequence_parallel_size", 1) or 1
        assert sp_size == 1, f"gradient alignment requires ulysses_sequence_parallel_size=1, got {sp_size}"

        n_ref = int(data.meta_info["n_ref"])
        n_rows = len(data)
        assert 0 < n_ref < n_rows, f"n_ref={n_ref}, n_rows={n_rows}"
        temperature = float(data.meta_info.get("temperature", 1.0))
        row_lens = data.batch["attention_mask"].sum(dim=-1).tolist()

        def _rows_loss(a: int, b: int):
            td = data[a:b].to_tensordict()
            td = left_right_2_no_padding(td)
            tu.assign_non_tensor(
                td,
                temperature=temperature,
                calculate_entropy=False,
                compute_loss=True,
                use_remove_padding=self.actor.model_config.get("use_remove_padding", False),
                use_fused_kernels=self.actor.engine_config.use_fused_kernels,
            )
            loss, _ = engine.forward_step(td, loss_function=rowmean_ce_loss, forward_only=False)
            return loss

        with engine.train_mode():
            params = [p for p in engine.module.parameters() if p.requires_grad]
            device = _local(params[0]).device
            layout, flat_n = _flat_layout(params)

            wver = self._weight_version(engine)
            _ids = data.batch["input_ids"][:n_ref]
            _attn = data.batch["attention_mask"][:n_ref]
            _resp = data.batch["responses"][:n_ref]
            _rmask = data.batch["response_mask"][:n_ref]
            ref_fp = (
                int((_ids * _attn).sum()),
                int(_attn.sum()),
                int((_resp * _rmask).sum()),
                int(_rmask.sum()),
                n_ref,
            )
            cache_key = (wver, ref_fp)
            if self._ga_cache_key == cache_key:
                g_ref = self._ga_cache_val
            else:
                engine.optimizer_zero_grad()
                for a, b in _token_chunks(0, n_ref, row_lens, _REF_MICRO_TOK):
                    (_rows_loss(a, b) / n_ref).backward()
                flat32 = _pack_grads_into(
                    params, layout, torch.empty(flat_n, dtype=torch.float32, device=device)
                )
                g_ref = flat32.to(torch.bfloat16)
                del flat32
                engine.optimizer_zero_grad()
                self._ga_cache_val = None
                self._ga_cache_key = cache_key
                self._ga_cache_val = g_ref

            g_ref_f32 = g_ref.to(torch.float32)
            rn_local = _self_dot(g_ref_f32)

            if self._ga_gs_buf is None or self._ga_gs_buf.numel() != flat_n:
                self._ga_gs_buf = torch.empty(flat_n, dtype=torch.float32, device=device)
            gs = self._ga_gs_buf

            cos_out = []
            for i in range(n_ref, n_rows):
                engine.optimizer_zero_grad()
                loss_i = _rows_loss(i, i + 1)
                loss_i.backward()
                del loss_i
                _pack_grads_into(params, layout, gs)
                trip = torch.stack([_dot(gs, g_ref_f32).double(), _self_dot(gs), rn_local])
                if dist.is_initialized():
                    dist.all_reduce(trip, op=dist.ReduceOp.SUM)
                cos = alignment_reward(cosine_from_statistics(trip))
                cos_out.append(cos.to(torch.float32))

            engine.optimizer_zero_grad()
            del g_ref_f32

        _rank0(f"[GA] n_ref={n_ref} n_synth={n_rows - n_ref} weight_version={wver}")
        cos = torch.stack(cos_out) if cos_out else torch.zeros(0, device=device)
        return DataProto.from_dict(tensors={"cos": cos.detach().cpu()})
