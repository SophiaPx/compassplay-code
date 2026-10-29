# CompassPlay integration with AZR and verl.
from verl.experimental.agent_loop import AgentLoopManager


class SelfPlayAgentLoopManager(AgentLoopManager):

    def __init__(self, *args, **kwargs):
        kwargs["reward_loop_worker_handles"] = None
        if len(args) >= 4:
            args = args[:3]
        super().__init__(*args, **kwargs)
        assert self.reward_loop_worker_handles is None, "AZR does not use verl's inline reward loop"
