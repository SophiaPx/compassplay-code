# Source: https://github.com/LukeBailey181/sgs/blob/64d2de468a0ddf222d5836248bea1d239a253f49/sgs/models/guide/__init__.py; modified for CompassPlay.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional

from experiments.lean.data.types import Conjecture
from experiments.lean.generation.config import ProverConfig

# ------------------------------------------------------------
# Guide classes
# ------------------------------------------------------------


@dataclass
class GuideConfig:
    prover_config: ProverConfig
    batch_size: int = 8
    # If you are training the guide
    lr: float = 1e-5
    num_evals: int = 10
    num_epochs: int = 4
    max_grad_norm: float = 1.0
    guide_load_path: Optional[str] = None
    guide_save_path: Optional[str] = None
    guide_model_path: Optional[str] = None


class Guide(ABC):
    def __init__(
        self,
        guide_config: GuideConfig,
    ):
        """
        Every guide is based on a prover model.
        Prover models will always be saved to disc. So we have
        all the information we need about it from the ProverConfig.

        Args:
            prover_model_config: The configuration for the prover model
        """
        self.prover_model_config = guide_config.prover_config
        self.guide_config = guide_config

    @abstractmethod
    def review(self, conjectures: List[Conjecture]) -> List[Dict]:
        # The review function may or may not use the conjecture.seed_proof
        # We return a (possibly empty) list of all the things to log to wandb
        ...
