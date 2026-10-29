# Source: https://github.com/LukeBailey181/sgs/blob/64d2de468a0ddf222d5836248bea1d239a253f49/sgs/verification/types.py
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class VerificationOutput:
    verdict: bool
    output: Dict[str, Any]
    system_error: bool = (
        False  # If verification failed due to some error outside of verification
    )
    # ^ This is not perfectly supported right now, for local verification it is
