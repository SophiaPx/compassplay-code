"""CompassPlay's signed cosine reward, optionally weighted by novelty."""


def alignment_reward(cosine, novelty=1.0):
    # Preserve negative alignment. Eligibility and batch normalization belong
    # to the base training loop; this function introduces neither clipping nor
    # an additional difficulty factor.
    return cosine * novelty
