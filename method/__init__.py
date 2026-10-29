"""CompassPlay: gradient alignment rewards for self-play proposers."""

from .alignment import cosine_from_statistics, gradient_statistics
from .rewards import alignment_reward

__all__ = ["cosine_from_statistics", "gradient_statistics", "alignment_reward"]
