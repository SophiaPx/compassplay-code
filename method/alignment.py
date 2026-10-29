"""Gradient statistics shared by local and sharded solver implementations."""

import torch


def dot(a, b):
    """FP32 inner product, chunked to support vectors larger than BLAS int32."""
    a, b = a.reshape(-1).float(), b.reshape(-1).float()
    limit = 2**31 - 1
    result = torch.zeros((), dtype=torch.float32, device=a.device)
    for start in range(0, a.numel(), limit):
        result += torch.dot(a[start:start + limit], b[start:start + limit])
    return result


def squared_norm(vector):
    # Compute the norm in FP32 and square in FP64, as in the coding worker.
    return torch.linalg.vector_norm(vector.float()).double().square()


def gradient_statistics(pairs):
    """Sum (dot, candidate norm squared, reference norm squared) over blocks.

    Each pair contains matching candidate/reference gradient blocks. Adapters
    handle missing gradients and distributed reduction before normalization.
    """
    statistics = None
    for candidate, reference in pairs:
        gs, gr = candidate.reshape(-1).float(), reference.reshape(-1).float()
        block = torch.stack((dot(gs, gr), dot(gs, gs), dot(gr, gr)))
        statistics = block if statistics is None else statistics + block
    if statistics is None:
        raise ValueError("At least one gradient pair is required")
    return statistics


def cosine_from_statistics(statistics, eps=1e-12):
    """Normalize globally summed statistics; zero gradients give reward zero."""
    inner, candidate_sq, reference_sq = statistics.unbind()
    return inner / (candidate_sq.sqrt() * reference_sq.sqrt() + eps)
